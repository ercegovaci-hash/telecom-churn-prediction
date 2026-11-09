# -----------------------------------------------------------------------------
# 03_costs.R - step 6: making expensive mistakes rarer
#
# Missing a churner (false negative) costs the customer's future revenue.
# Flagging a loyal customer (false positive) costs a retention offer.
# Assumption used below (illustrative, change COST_FN / COST_FP for a real case):
#   a missed churner costs 5x as much as an unnecessary retention offer.
# Two strategies are compared, as in the RapidMiner project:
#   (a) moving the decision threshold on predicted probabilities
#   (b) cost-sensitive training (rpart loss matrix, the idea behind MetaCost)
# -----------------------------------------------------------------------------

source("R/00_setup.R")
s <- readRDS("output/02_setup.rds")
train <- s$train; test <- s$test; ct <- s$chosen_tree

COST_FN <- 5
COST_FP <- 1

cost_of <- function(prob, y, t) {
  cm <- confusion(prob, y, t)
  COST_FN * cm["no", "yes"] + COST_FP * cm["yes", "no"]
}

thresholds <- seq(0.05, 0.95, by = 0.05)
sweep <- do.call(rbind, lapply(c("tree", "logit"), function(m) {
  do.call(rbind, lapply(thresholds, function(t) {
    met <- class_metrics(s$oof[[m]], train$churn, t)
    data.frame(model = m, threshold = t, cost = cost_of(s$oof[[m]], train$churn, t),
               precision = met[["precision"]], recall = met[["recall"]])
  }))
}))
write.csv(sweep, "output/03_threshold_sweep.csv", row.names = FALSE)

best_t <- sapply(c("tree", "logit"), function(m) {
  x <- sweep[sweep$model == m, ]
  x$threshold[which.min(x$cost)]
})

# (b) cost-sensitive tree: rpart loss matrix, rows = actual, cols = predicted
cost_tree <- function(tr, te) {
  f <- rpart(reformulate(s$tree_vars, "churn"), data = tr, method = "class",
             parms = list(loss = matrix(c(0, COST_FN, COST_FP, 0), 2)),
             control = rpart.control(cp = ct$cp, maxdepth = ct$maxdepth,
                                     minbucket = ct$minbucket, xval = 0))
  predict(f, te, type = "class") == "yes"
}
oof_cost_tree <- logical(nrow(train))
for (k in sort(unique(s$folds))) {
  oof_cost_tree[s$folds == k] <- cost_tree(train[s$folds != k, ], train[s$folds == k, ])
}
as_prob <- function(flag) as.numeric(flag)

sink("output/03_costs.txt", split = TRUE)
cat(sprintf("Cost assumption: missed churner = %d, unnecessary offer = %d\n", COST_FN, COST_FP))
cat(sprintf("Theory: for calibrated probabilities the cost-optimal threshold is %d/(%d+%d) = %.2f\n\n",
            COST_FP, COST_FP, COST_FN, COST_FP / (COST_FP + COST_FN)))

summ <- function(label, prob, y, t) {
  m <- class_metrics(prob, y, t)
  data.frame(strategy = label, threshold = t, cost = cost_of(prob, y, t),
             accuracy = round(m[["accuracy"]], 4), precision = round(m[["precision"]], 4),
             recall = round(m[["recall"]], 4))
}
cv_tab <- rbind(
  summ("Tree, threshold 0.5",                    s$oof$tree,  train$churn, 0.5),
  summ("Tree, cost-optimal threshold",           s$oof$tree,  train$churn, best_t[["tree"]]),
  summ("Tree trained with loss matrix",          as_prob(oof_cost_tree), train$churn, 0.5),
  summ("Logistic regr., threshold 0.5",          s$oof$logit, train$churn, 0.5),
  summ("Logistic regr., cost-optimal threshold", s$oof$logit, train$churn, best_t[["logit"]])
)
cat("Cross-validated (training set):\n"); print(cv_tab, row.names = FALSE)

test_cost_tree <- as_prob(cost_tree(train, test))
test_tab <- rbind(
  summ("Tree, threshold 0.5",                    s$test_prob$tree,  test$churn, 0.5),
  summ("Tree, cost-optimal threshold",           s$test_prob$tree,  test$churn, best_t[["tree"]]),
  summ("Tree trained with loss matrix",          test_cost_tree,    test$churn, 0.5),
  summ("Logistic regr., threshold 0.5",          s$test_prob$logit, test$churn, 0.5),
  summ("Logistic regr., cost-optimal threshold", s$test_prob$logit, test$churn, best_t[["logit"]])
)
cat("\nUntouched test set (thresholds chosen on training CV):\n"); print(test_tab, row.names = FALSE)
write.csv(rbind(cbind(data = "cv", cv_tab), cbind(data = "test", test_tab)),
          "output/03_cost_results.csv", row.names = FALSE)

# How much do the tree's leaves leave room for a threshold? (pure leaves -> little)
cat("\nShare of tree out-of-fold probabilities that are exactly 0 or 1:",
    round(mean(s$oof$tree %in% c(0, 1)), 3), "\n")
sink()

# --- Figure ------------------------------------------------------------------
lab <- c(tree = "Decision tree", logit = "Logistic regression")
sweep$model_label <- lab[sweep$model]
opt <- do.call(rbind, lapply(split(sweep, sweep$model), function(x) x[which.min(x$cost), ]))
p <- ggplot(sweep, aes(threshold, cost, colour = model_label)) +
  geom_line(linewidth = 0.9) + geom_point(size = 1.4) +
  geom_point(data = opt, size = 3.5, shape = 21, fill = "white", stroke = 1.4) +
  geom_vline(xintercept = 0.5, linetype = "dotted") +
  annotate("text", x = 0.51, y = max(sweep$cost), label = "default 0.5", hjust = 0, size = 3.2) +
  scale_colour_manual(values = c("Decision tree" = "#2b6cb0", "Logistic regression" = "#c53030")) +
  labs(title = "Total misclassification cost by decision threshold",
       subtitle = sprintf("Missed churner = %d, unnecessary offer = %d (10-fold CV, training set). Circles = minimum",
                          COST_FN, COST_FP),
       x = "Predict 'churn' if probability >=", y = "Total cost", colour = NULL) +
  theme(legend.position = "top")
save_plot(p, "05_cost_by_threshold")
