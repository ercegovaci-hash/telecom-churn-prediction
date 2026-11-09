# -----------------------------------------------------------------------------
# 04_interpret.R - steps 7-9: show the models, compare them, turn them into actions
# -----------------------------------------------------------------------------

source("R/00_setup.R")
s <- readRDS("output/02_setup.rds")
train <- s$train; ct <- s$chosen_tree
d <- add_features(load_churn())

sink("output/04_interpret.txt", split = TRUE)

# --- Step 7a: the decision tree ----------------------------------------------
tree <- rpart(reformulate(s$tree_vars, "churn"), data = train, method = "class",
              control = rpart.control(cp = ct$cp, maxdepth = ct$maxdepth,
                                      minbucket = ct$minbucket, xval = 0))
cat("Decision tree (fitted on the training set):\n")
print(tree, digits = 3)
cat("\nVariable importance (rpart):\n")
print(round(100 * tree$variable.importance / sum(tree$variable.importance), 1))

# Readable rules: one line per leaf
leaf_rules <- function(fit) {
  frame <- fit$frame
  leaves <- as.integer(rownames(frame)[frame$var == "<leaf>"])
  paths <- path.rpart(fit, nodes = leaves, print.it = FALSE)
  data.frame(
    rule = sapply(paths, function(p) paste(p[-1], collapse = " AND ")),
    customers = frame$n[frame$var == "<leaf>"],
    churn_rate = round(100 * frame$yval2[frame$var == "<leaf>", 5], 1),
    prediction = ifelse(frame$yval[frame$var == "<leaf>"] == 2, "CHURN", "stay"),
    row.names = NULL
  )
}
rules <- leaf_rules(tree)
rules <- rules[order(-rules$churn_rate), ]
cat("\nTree as rules (training set):\n"); print(rules, row.names = FALSE)
write.csv(rules, "output/04_tree_rules.csv", row.names = FALSE)

svglite::svglite("figures/06_decision_tree.svg", width = 11, height = 6.5, bg = "white")
plot(partykit::as.party(tree), gp = grid::gpar(fontsize = 9),
     tp_args = list(fill = c(pal[["yes"]], pal[["no"]]), id = FALSE),
     ip_args = list(id = FALSE), ep_args = list(digits = 1))
invisible(dev.off())

# --- Step 7b: logistic regression parameters ---------------------------------
# The interpretable (linear) model from step 4: odds ratios per meaningful unit
logit <- glm(reformulate(s$lr_vars, "churn"), data = train, family = binomial)
co <- summary(logit)$coefficients
units <- c(intl_planyes = 1, vmail_planyes = 1, vmail_messages = 10, day_mins = 30,
           day_calls = 10, eve_mins = 30, night_mins = 30, intl_mins = 1,
           intl_calls = 1, custserv_calls = 1)
unit_label <- c(intl_planyes = "has international plan", vmail_planyes = "has voice-mail plan",
                vmail_messages = "+10 voice-mail messages", day_mins = "+30 day minutes",
                day_calls = "+10 day calls", eve_mins = "+30 evening minutes",
                night_mins = "+30 night minutes", intl_mins = "+1 international minute",
                intl_calls = "+1 international call", custserv_calls = "+1 customer-service call")
ci <- suppressMessages(confint.default(logit))
or <- data.frame(term = names(units), change = unit_label[names(units)],
                 coefficient = round(co[names(units), "Estimate"], 4),
                 p_value = signif(co[names(units), "Pr(>|z|)"], 2),
                 odds_ratio = round(exp(co[names(units), "Estimate"] * units), 2),
                 or_low = round(exp(ci[names(units), 1] * units), 2),
                 or_high = round(exp(ci[names(units), 2] * units), 2), row.names = NULL)
or <- or[order(-or$odds_ratio), ]
cat("\nLogistic regression (stepwise AIC model) - odds ratios:\n"); print(or, row.names = FALSE)
write.csv(or, "output/04_logit_odds_ratios.csv", row.names = FALSE)

or$change <- factor(or$change, levels = rev(or$change))
p_or <- ggplot(or, aes(odds_ratio, change)) +
  geom_vline(xintercept = 1, linetype = "dashed") +
  geom_errorbarh(aes(xmin = or_low, xmax = or_high), height = 0.25, colour = "#4a5568") +
  geom_point(size = 2.6, colour = ifelse(or$odds_ratio > 1, pal[["yes"]], "#2b6cb0")) +
  scale_x_log10(breaks = c(0.1, 0.25, 0.5, 1, 2, 4, 8)) +
  labs(title = "Logistic regression: how each factor changes the odds of churn",
       subtitle = "Odds ratio with 95% CI (log scale). Right of 1 = more churn, left = less",
       x = "Odds ratio", y = NULL)
save_plot(p_or, "07_odds_ratios", height = 4.4)

# --- Step 8: do the two models agree? ----------------------------------------
rf <- ranger(reformulate(s$tree_vars, "churn"), data = train, num.trees = 500,
             importance = "permutation", seed = 7)
imp <- sort(rf$variable.importance, decreasing = TRUE)
cat("\nRandom forest permutation importance (third opinion):\n")
print(round(head(imp, 10) / max(imp), 3))

# Where the linear model cannot follow the tree: service calls x spending
seg <- with(d, data.frame(calls = ifelse(custserv_calls >= 4, "4+ calls", "0-3 calls"),
                          spend = cut(total_charge, c(-Inf, 56.7, 74, Inf),
                                      labels = c("charge < 56.7", "56.7-74", "charge >= 74")),
                          churn = churn))
tab <- aggregate(churn ~ calls + spend, data = seg,
                 FUN = function(x) c(customers = length(x), churn_rate = round(100 * mean(x == "yes"), 1)))
tab <- do.call(data.frame, tab)
names(tab) <- c("service_calls", "monthly_charge", "customers", "churn_rate_pct")
cat("\nChurn rate by service calls x total charge (all 3,333 customers):\n")
print(tab, row.names = FALSE)

# --- Step 9: segments for retention actions (all customers) -------------------
segments <- with(d, list(
  "High spenders without voice-mail (charge >= 74, no VM plan)" =
    total_charge >= 74.04 & vmail_plan == "no",
  "Unhappy low spenders (4+ service calls, charge < 56.7)" =
    custserv_calls >= 4 & total_charge < 56.71,
  "International plan, few or long intl calls (<3 calls or >= 13.1 min)" =
    intl_plan == "yes" & (intl_calls < 3 | intl_mins >= 13.1)
))
seg_tab <- do.call(rbind, lapply(names(segments), function(n) {
  m <- segments[[n]]
  data.frame(segment = n, customers = sum(m),
             share_of_base = round(100 * mean(m), 1),
             churn_rate = round(100 * mean(d$churn[m] == "yes"), 1),
             share_of_all_churners = round(100 * sum(d$churn[m] == "yes") / sum(d$churn == "yes"), 1))
}))
any_seg <- Reduce(`|`, segments)
cat("\nRetention segments (all 3,333 customers):\n"); print(seg_tab, row.names = FALSE)
cat(sprintf("\nTogether: %d customers (%.1f%% of the base) contain %.1f%% of all churners.\n",
            sum(any_seg), 100 * mean(any_seg),
            100 * sum(d$churn[any_seg] == "yes") / sum(d$churn == "yes")))
write.csv(seg_tab, "output/04_segments.csv", row.names = FALSE)
sink()
