# -----------------------------------------------------------------------------
# 02_models.R - decision tree vs logistic regression, step by step
#
#   Step 1-2  baseline models and their generalisation error (10-fold CV)
#   Step 3    parameter tuning: complexity vs accuracy
#   Step 4    attribute selection
#   Step 5    attribute construction (new features)
#   + a random forest benchmark, and one final check on an untouched test set
#
# Protocol: 20% of customers are locked away as a test set before anything is
# fitted. Every choice below is made with cross-validation on the other 80%.
# -----------------------------------------------------------------------------

source("R/00_setup.R")
suppressPackageStartupMessages(library(glmnet))

d <- add_features(load_churn())
d$many_service_calls <- factor(d$custserv_calls >= 4, levels = c(FALSE, TRUE),
                               labels = c("no", "yes"))

set.seed(2024)
test_idx <- unlist(lapply(split(seq_len(nrow(d)), d$churn),
                          function(i) sample(i, round(0.2 * length(i)))))
train <- d[-test_idx, ]
test  <- d[test_idx, ]
folds <- stratified_folds(train$churn, k = 10, seed = 1)

# --- Model wrappers: fit on a data frame, return churn probabilities ---------
tree_model <- function(vars, cp = 0.01, maxdepth = 30, minbucket = 7) {
  function(tr, te) {
    f <- rpart(reformulate(vars, "churn"), data = tr, method = "class",
               control = rpart.control(cp = cp, maxdepth = maxdepth,
                                       minbucket = minbucket, xval = 0))
    predict(f, te, type = "prob")[, "yes"]
  }
}
logit_model <- function(vars) {
  function(tr, te) {
    f <- suppressWarnings(glm(reformulate(vars, "churn"), data = tr, family = binomial))
    suppressWarnings(predict(f, te, type = "response"))
  }
}
lasso_model <- function(vars, s = "lambda.1se") {
  function(tr, te) {
    x  <- model.matrix(reformulate(vars), tr)[, -1]
    xt <- model.matrix(reformulate(vars), te)[, -1]
    f <- cv.glmnet(x, tr$churn, family = "binomial", nfolds = 5)
    as.numeric(predict(f, xt, s = s, type = "response"))
  }
}
forest_model <- function(vars) {
  function(tr, te) {
    f <- ranger(reformulate(vars, "churn"), data = tr, num.trees = 500,
                probability = TRUE, seed = 7)
    predict(f, te)$predictions[, "yes"]
  }
}

# Out-of-fold predictions: every training customer is predicted by a model
# that did not see them
cv_predict <- function(model, data = train, k_folds = folds) {
  p <- numeric(nrow(data))
  for (k in sort(unique(k_folds))) {
    p[k_folds == k] <- model(data[k_folds != k, ], data[k_folds == k, ])
  }
  p
}

results <- list()
record <- function(step, name, prob) {
  m <- class_metrics(prob, train$churn)
  results[[length(results) + 1]] <<- data.frame(step = step, model = name, t(round(m, 4)))
  invisible(prob)
}

sink("output/02_models.txt", split = TRUE)
cat(sprintf("Train: %d customers (%d churners) | Test: %d customers (%d churners)\n\n",
            nrow(train), sum(train$churn == "yes"), nrow(test), sum(test$churn == "yes")))

raw_vars <- c("state", "account_length", "area_code", "intl_plan", "vmail_plan",
              "vmail_messages", "day_mins", "day_calls", "day_charge", "eve_mins",
              "eve_calls", "eve_charge", "night_mins", "night_calls", "night_charge",
              "intl_mins", "intl_calls", "intl_charge", "custserv_calls")

# --- Step 1-2: baselines ----------------------------------------------------
record("0 reference", "Always predict 'no churn'", rep(0, nrow(train)))
record("1-2 baseline", "Decision tree (rpart defaults)", cv_predict(tree_model(raw_vars)))
record("1-2 baseline", "Logistic regression (all attributes)", cv_predict(logit_model(raw_vars)))

# --- Step 3: tune the tree - complexity vs accuracy -------------------------
grid <- expand.grid(cp = c(0.001, 0.003, 0.01, 0.03), maxdepth = c(3, 4, 5, 6, 8, 12),
                    minbucket = c(5, 15))
grid$cv_accuracy <- NA; grid$leaves <- NA
for (i in seq_len(nrow(grid))) {
  g <- grid[i, ]
  p <- cv_predict(tree_model(base_vars, g$cp, g$maxdepth, g$minbucket))
  grid$cv_accuracy[i] <- mean((p >= 0.5) == (train$churn == "yes"))
  f <- rpart(reformulate(base_vars, "churn"), data = train, method = "class",
             control = rpart.control(cp = g$cp, maxdepth = g$maxdepth,
                                     minbucket = g$minbucket, xval = 0))
  grid$leaves[i] <- sum(f$frame$var == "<leaf>")
}
best_acc <- max(grid$cv_accuracy)
# Complexity rule: the smallest tree within 0.5 percentage points of the best
ok <- grid[grid$cv_accuracy >= best_acc - 0.005, ]
chosen_tree <- ok[order(ok$leaves, -ok$cv_accuracy), ][1, ]
cat("Tree tuning grid (top 8 by CV accuracy):\n")
print(head(grid[order(-grid$cv_accuracy, grid$leaves), ], 8), row.names = FALSE)
cat(sprintf("\nBest CV accuracy %.4f. Chosen (smallest tree within 0.5 pp): cp=%g, maxdepth=%d, minbucket=%d, %d leaves, accuracy %.4f\n\n",
            best_acc, chosen_tree$cp, chosen_tree$maxdepth, chosen_tree$minbucket,
            chosen_tree$leaves, chosen_tree$cv_accuracy))
write.csv(grid, "output/02_tree_tuning_grid.csv", row.names = FALSE)

p_complex <- ggplot(grid, aes(leaves, cv_accuracy)) +
  geom_point(colour = "#4a5568", alpha = 0.7) +
  geom_point(data = chosen_tree, colour = "#c53030", size = 3.5) +
  geom_hline(yintercept = best_acc - 0.005, linetype = "dashed") +
  annotate("text", x = max(grid$leaves), y = best_acc - 0.005, vjust = 1.5, hjust = 1,
           size = 3.2, label = "best accuracy - 0.5 pp") +
  scale_x_log10() + scale_y_continuous(labels = scales::percent) +
  labs(title = "Decision tree: complexity vs cross-validated accuracy",
       subtitle = "Red = chosen tree: the simplest one within 0.5 pp of the best",
       x = "Number of leaves (log scale)", y = "10-fold CV accuracy")
save_plot(p_complex, "03_tree_complexity")

tuned_tree <- tree_model(base_vars, chosen_tree$cp, chosen_tree$maxdepth, chosen_tree$minbucket)
record("3 tuned", "Decision tree (tuned)", cv_predict(tuned_tree))
record("3 tuned", "Logistic regression (lasso, lambda.1se)", cv_predict(lasso_model(raw_vars)))

# --- Step 4: attribute selection --------------------------------------------
# Removed by reasoning: phone (identifier), 4 x charge (= tariff x minutes),
# state (51 levels, ~8 churners per state in training -> noise)
record("4 selection", "Logistic regression (14 attributes, no state/charges)",
       cv_predict(logit_model(base_vars)))
# Then backward stepwise selection by AIC, refit inside each fold
step_logit <- function(tr, te) {
  full <- glm(reformulate(base_vars, "churn"), data = tr, family = binomial)
  f <- step(full, direction = "backward", trace = 0)
  predict(f, te, type = "response")
}
record("4 selection", "Logistic regression (stepwise AIC)", cv_predict(step_logit))
step_full <- step(glm(reformulate(base_vars, "churn"), data = train, family = binomial),
                  direction = "backward", trace = 0)
cat("Stepwise AIC keeps:", paste(attr(terms(step_full), "term.labels"), collapse = ", "), "\n")
cat("Dropped:", paste(setdiff(base_vars, attr(terms(step_full), "term.labels")), collapse = ", "), "\n\n")

# --- Step 5: attribute construction -----------------------------------------
lr_vars <- c(attr(terms(step_full), "term.labels"))
oof_tree <- record("5 construction", "Decision tree + totals/ratios", cv_predict(
  tree_model(eng_vars, chosen_tree$cp, chosen_tree$maxdepth, chosen_tree$minbucket)))
record("5 construction", "Logistic regression + totals/ratios",
       cv_predict(logit_model(unique(c(lr_vars, "total_charge", "intl_ratio", "avg_call_mins")))))
# The tree finds threshold effects on its own (high total charge, 4+ service
# calls, international plan with few calls). A logistic regression only sees
# straight lines, so the non-linearity has to be written into the formula:
# a spline on total_charge, a 4+ service-call flag (from the EDA) and
# interactions. No cut-off values are hand-picked.
lr_nonlinear <- function(tr, te) {
  f <- suppressWarnings(glm(churn ~ intl_plan * (intl_calls + intl_mins) + vmail_plan +
                              splines::ns(total_charge, 4) * many_service_calls,
                            data = tr, family = binomial))
  suppressWarnings(predict(f, te, type = "response"))
}
oof_lr <- record("5 construction", "Logistic regression + spline(total_charge) x 4+ calls + interactions",
       cv_predict(lr_nonlinear))

# --- Benchmark ---------------------------------------------------------------
oof_rf <- record("benchmark", "Random forest (500 trees, all features)",
       cv_predict(forest_model(eng_vars)))

res <- do.call(rbind, results)
cat("Cross-validated performance on the training set (threshold 0.5):\n")
print(res, row.names = FALSE)
write.csv(res, "output/02_cv_results.csv", row.names = FALSE)

# --- Final, one-time check on the locked test set ---------------------------
final_tree_vars <- eng_vars
final <- list(
  "Decision tree (tuned + new features)" = tree_model(final_tree_vars, chosen_tree$cp,
                                                      chosen_tree$maxdepth, chosen_tree$minbucket),
  "Logistic regression (non-linear terms)" = lr_nonlinear,
  "Random forest (benchmark)" = forest_model(eng_vars)
)
test_prob <- lapply(final, function(m) m(train, test))
test_res <- do.call(rbind, lapply(names(final), function(n) {
  p <- test_prob[[n]]
  data.frame(model = n, t(round(class_metrics(p, test$churn), 4)))
}))
cat("\nUntouched test set (667 customers, threshold 0.5):\n")
print(test_res, row.names = FALSE)
write.csv(test_res, "output/02_test_results.csv", row.names = FALSE)
sink()

saveRDS(list(train = train, test = test, folds = folds, chosen_tree = chosen_tree,
             lr_vars = lr_vars, tree_vars = final_tree_vars,
             oof = list(tree = oof_tree, logit = oof_lr, forest = oof_rf),
             test_prob = setNames(test_prob, c("tree", "logit", "forest"))),
        "output/02_setup.rds")

# --- Figure: the journey -----------------------------------------------------
plot_res <- res[res$step != "0 reference", ]
plot_res$model <- factor(plot_res$model, levels = rev(plot_res$model))
long <- rbind(data.frame(model = plot_res$model, metric = "Recall (churners caught)", value = plot_res$recall),
              data.frame(model = plot_res$model, metric = "Precision", value = plot_res$precision),
              data.frame(model = plot_res$model, metric = "Accuracy", value = plot_res$accuracy))
long$metric <- factor(long$metric, levels = c("Accuracy", "Precision", "Recall (churners caught)"))
long$family <- ifelse(grepl("^Decision", long$model), "Decision tree",
                      ifelse(grepl("^Logistic", long$model), "Logistic regression", "Random forest"))
p_steps <- ggplot(long, aes(value, model, fill = family)) +
  geom_col(width = 0.7) +
  scale_fill_manual(values = c("Decision tree" = "#2b6cb0", "Logistic regression" = "#c53030",
                               "Random forest" = "#4a5568")) +
  geom_text(aes(label = scales::percent(value, accuracy = 1)), hjust = -0.1, size = 3) +
  facet_wrap(~metric) +
  scale_x_continuous(labels = scales::percent, limits = c(0, 1.22), breaks = c(0, 0.5, 1)) +
  labs(title = "From baseline to final model (10-fold CV, threshold 0.5)",
       x = NULL, y = NULL, fill = NULL) +
  theme(legend.position = "top")
save_plot(p_steps, "04_model_steps", width = 10, height = 4.8)
