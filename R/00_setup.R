# -----------------------------------------------------------------------------
# 00_setup.R - packages, data preparation and evaluation helpers
# -----------------------------------------------------------------------------

required <- c("rpart", "ranger", "pROC", "ggplot2", "svglite", "scales")
missing  <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing) > 0) {
  stop("Install missing packages first: install.packages(c(",
       paste0('"', missing, '"', collapse = ", "), "))")
}
suppressPackageStartupMessages({
  library(rpart)
  library(ranger)
  library(pROC)
  library(ggplot2)
})

options(width = 200)
dir.create("figures", showWarnings = FALSE)
dir.create("output",  showWarnings = FALSE)

# --- Data --------------------------------------------------------------------
# churn.csv: 3,333 telecom customers, 20 attributes, target "Churn?"
load_churn <- function(path = "data/churn.csv") {
  raw <- read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
  d <- data.frame(
    state          = factor(raw[["State"]]),
    account_length = raw[["Account Length"]],
    area_code      = factor(raw[["Area Code"]]),
    phone          = raw[["Phone"]],
    intl_plan      = factor(raw[["Int'l Plan"]], levels = c("no", "yes")),
    vmail_plan     = factor(raw[["VMail Plan"]], levels = c("no", "yes")),
    vmail_messages = raw[["VMail Message"]],
    day_mins       = raw[["Day Mins"]],
    day_calls      = raw[["Day Calls"]],
    day_charge     = raw[["Day Charge"]],
    eve_mins       = raw[["Eve Mins"]],
    eve_calls      = raw[["Eve Calls"]],
    eve_charge     = raw[["Eve Charge"]],
    night_mins     = raw[["Night Mins"]],
    night_calls    = raw[["Night Calls"]],
    night_charge   = raw[["Night Charge"]],
    intl_mins      = raw[["Intl Mins"]],
    intl_calls     = raw[["Intl Calls"]],
    intl_charge    = raw[["Intl Charge"]],
    custserv_calls = raw[["CustServ Calls"]],
    churn          = factor(ifelse(raw[["Churn?"]] == "True.", "yes", "no"),
                            levels = c("no", "yes"))
  )
  d
}

# Engineered features (the same ideas as in the RapidMiner project)
add_features <- function(d) {
  d$total_mins   <- d$day_mins + d$eve_mins + d$night_mins + d$intl_mins
  d$total_calls  <- d$day_calls + d$eve_calls + d$night_calls + d$intl_calls
  d$total_charge <- d$day_charge + d$eve_charge + d$night_charge + d$intl_charge
  d$avg_call_mins <- ifelse(d$total_calls > 0, d$total_mins / d$total_calls, 0)
  d$intl_ratio   <- ifelse(d$total_mins > 0, d$intl_mins / d$total_mins, 0)
  d
}

# Attribute sets
# - phone is a unique customer identifier: never a predictor
# - the four *_charge columns are exact linear functions of *_mins (tariff x minutes)
base_vars <- c("account_length", "area_code", "intl_plan", "vmail_plan",
               "vmail_messages", "day_mins", "day_calls", "eve_mins", "eve_calls",
               "night_mins", "night_calls", "intl_mins", "intl_calls", "custserv_calls")
eng_vars  <- c(base_vars, "total_mins", "total_calls", "total_charge",
               "avg_call_mins", "intl_ratio")

# --- Splits ------------------------------------------------------------------
stratified_folds <- function(y, k, seed) {
  set.seed(seed)
  folds <- integer(length(y))
  for (lv in levels(y)) {
    idx <- which(y == lv)
    folds[idx] <- sample(rep_len(seq_len(k), length(idx)))
  }
  folds
}

# --- Metrics -----------------------------------------------------------------
# prob = predicted probability of churn, y = factor(no, yes)
confusion <- function(prob, y, threshold = 0.5) {
  pred <- factor(ifelse(prob >= threshold, "yes", "no"), levels = c("no", "yes"))
  table(predicted = pred, actual = y)
}

class_metrics <- function(prob, y, threshold = 0.5) {
  cm <- confusion(prob, y, threshold)
  tp <- cm["yes", "yes"]; fp <- cm["yes", "no"]
  fn <- cm["no", "yes"];  tn <- cm["no", "no"]
  precision <- if (tp + fp > 0) tp / (tp + fp) else NA
  recall    <- tp / (tp + fn)
  c(accuracy  = (tp + tn) / sum(cm),
    precision = precision,
    recall    = recall,
    f1        = if (is.na(precision) || precision + recall == 0) NA
                else 2 * precision * recall / (precision + recall),
    roc_auc   = as.numeric(pROC::auc(pROC::roc(y, prob, levels = c("no", "yes"),
                                               direction = "<", quiet = TRUE))),
    pr_auc    = pr_auc(prob, y))
}

# Area under the precision-recall curve (step-wise / average precision)
pr_auc <- function(prob, y) {
  o <- order(prob, decreasing = TRUE)
  hits <- as.numeric(y[o] == "yes")
  tp <- cumsum(hits)
  precision <- tp / seq_along(tp)
  sum(precision * hits) / sum(hits)
}

save_plot <- function(plot, name, width = 8, height = 4.2) {
  ggsave(file.path("figures", paste0(name, ".svg")), plot,
         width = width, height = height, device = svglite::svglite, bg = "white")
  invisible(plot)
}

theme_set(theme_minimal(base_size = 11) +
            theme(plot.title = element_text(face = "bold"),
                  panel.grid.minor = element_blank(),
                  plot.background = element_rect(fill = "white", colour = NA)))
pal <- c(no = "#a0aec0", yes = "#c53030")
