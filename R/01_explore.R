# -----------------------------------------------------------------------------
# 01_explore.R - what the data looks like before any model
# -----------------------------------------------------------------------------

source("R/00_setup.R")
d <- load_churn()

sink("output/01_explore.txt", split = TRUE)
cat("Customers:", nrow(d), " Attributes:", ncol(d) - 1, "\n")
print(table(d$churn))
cat(sprintf("Churn rate: %.1f%%  -> a model that predicts 'no churn' for everyone is %.1f%% accurate\n",
            100 * mean(d$churn == "yes"), 100 * mean(d$churn == "no")))

# Data quality checks that drive the attribute selection later
cat("\nUnique phone numbers:", length(unique(d$phone)), "of", nrow(d),
    "-> phone is an identifier, not a predictor\n")
for (p in c("day", "eve", "night", "intl")) {
  f <- lm(d[[paste0(p, "_charge")]] ~ d[[paste0(p, "_mins")]])
  cat(sprintf("%-5s charge = %.4f x minutes (R^2 = %.6f) -> charge adds no information\n",
              p, coef(f)[2], summary(f)$r.squared))
}

rate <- function(x) {
  t <- table(x, d$churn)
  data.frame(level = rownames(t), customers = rowSums(t),
             churn_rate = round(100 * t[, "yes"] / rowSums(t), 1), row.names = NULL)
}
cat("\nChurn rate (%) by international plan:\n"); print(rate(d$intl_plan))
cat("\nChurn rate (%) by voice-mail plan:\n");     print(rate(d$vmail_plan))
cat("\nChurn rate (%) by calls to customer service:\n"); print(rate(d$custserv_calls))
cat("\nMean day minutes by churn:\n"); print(round(tapply(d$day_mins, d$churn, mean), 1))
sink()

# --- Figures -----------------------------------------------------------------
cs <- rate(pmin(d$custserv_calls, 6))
cs$level <- factor(cs$level, labels = c(0:5, "6+"))
p1 <- ggplot(cs, aes(level, churn_rate)) +
  geom_col(fill = ifelse(as.integer(cs$level) >= 5, pal["yes"], pal["no"]), width = 0.7) +
  geom_text(aes(label = paste0(churn_rate, "%")), vjust = -0.4, size = 3.4) +
  geom_text(aes(y = 2, label = paste0("n=", customers)), colour = "white", size = 3) +
  geom_hline(yintercept = 100 * mean(d$churn == "yes"), linetype = "dashed") +
  annotate("text", x = 7.45, y = 100 * mean(d$churn == "yes") + 3,
           label = "average 14.5%", hjust = 1, size = 3.3) +
  labs(title = "Churn jumps from the 4th call to customer service",
       x = "Calls to customer service", y = "Churn rate (%)") +
  scale_y_continuous(limits = c(0, 75))
save_plot(p1, "01_churn_by_service_calls")

p2 <- ggplot(d, aes(day_mins, fill = churn)) +
  geom_histogram(binwidth = 10, position = "identity", alpha = 0.75, colour = "white") +
  scale_fill_manual(values = pal, labels = c("stayed", "churned")) +
  labs(title = "Heavy daytime users churn far more often",
       subtitle = "Above 260 day minutes, more than half of customers churn (vs 14.5% overall)",
       x = "Day minutes per month", y = "Customers", fill = NULL) +
  theme(legend.position = "top")
save_plot(p2, "02_day_minutes")
