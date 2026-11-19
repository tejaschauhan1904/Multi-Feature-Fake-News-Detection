# ==============================================================================
# 08_model_evaluation.R
# Multi-Feature Fake News Detection Using R Language
# Purpose: Rigorous evaluation on held-out test data for all 3 models.
# ==============================================================================

suppressPackageStartupMessages({
  library(Matrix)
  library(glmnet)
  library(randomForest)
  library(e1071)
  library(pROC)
})

cat("========================================================\n")
cat(" [08] Evaluating Machine Learning Models on Test Set\n")
cat("========================================================\n")

get_project_root <- function() {
  candidates <- c(getwd(), file.path(getwd(), ".."), "d:/Tejas/College/Sem_5/R Language/R_PBL_project")
  for (cand in candidates) {
    if (file.exists(file.path(cand, "dataset", ".eval_cache_temp.rds"))) {
      return(normalizePath(cand, winslash = "/"))
    }
  }
  return(normalizePath(getwd(), winslash = "/"))
}

PROJECT_ROOT <- get_project_root()
eval_cache_path <- file.path(PROJECT_ROOT, "dataset", ".eval_cache_temp.rds")

if (!file.exists(eval_cache_path)) {
  stop("[ERROR] Evaluation cache not found. Please run 07_model_training.R first.")
}

cached <- readRDS(eval_cache_path)
cv_lasso        <- cached$cv_lasso
rf_model        <- cached$rf_model
nb_model        <- cached$nb_model
X_test_combined <- cached$X_test_combined
rf_test_df      <- cached$rf_test_df
y_test          <- cached$y_test
y_test_factor   <- cached$y_test_factor

cat(sprintf("Evaluating on %d held-out test samples (Fake: %d, Real: %d)...\n",
            length(y_test), sum(y_test == 0), sum(y_test == 1)))

# Evaluation helper function
# Positive class = 1 (Real News), Negative class = 0 (Fake News)
calc_metrics <- function(y_true, y_pred, y_prob, model_name) {
  tp <- sum(y_true == 1 & y_pred == 1)
  tn <- sum(y_true == 0 & y_pred == 0)
  fp <- sum(y_true == 0 & y_pred == 1)
  fn <- sum(y_true == 1 & y_pred == 0)
  
  acc   <- (tp + tn) / (tp + tn + fp + fn)
  prec  <- ifelse((tp + fp) > 0, tp / (tp + fp), 0)
  rec   <- ifelse((tp + fn) > 0, tp / (tp + fn), 0) # Sensitivity
  spec  <- ifelse((tn + fp) > 0, tn / (tn + fp), 0)
  f1    <- ifelse((prec + rec) > 0, 2 * (prec * rec) / (prec + rec), 0)
  
  roc_obj <- tryCatch({
    pROC::roc(response = y_true, predictor = as.numeric(y_prob), quiet = TRUE)
  }, error = function(e) NULL)
  
  auc_val <- if (!is.null(roc_obj)) as.numeric(pROC::auc(roc_obj)) else NA
  
  list(
    metrics = data.frame(
      Model = model_name,
      Accuracy = round(acc, 4),
      Precision = round(prec, 4),
      Recall = round(rec, 4),
      Specificity = round(spec, 4),
      F1_Score = round(f1, 4),
      ROC_AUC = round(auc_val, 4),
      TP = tp,
      TN = tn,
      FP = fp,
      FN = fn,
      stringsAsFactors = FALSE
    ),
    roc = roc_obj,
    y_prob = y_prob,
    y_pred = y_pred
  )
}

# ------------------------------------------------------------------------------
# 1. Evaluate Logistic Regression (LASSO)
# ------------------------------------------------------------------------------
cat("Evaluating Model 1: Logistic Regression with LASSO...\n")
prob_lr <- as.vector(predict(cv_lasso, newx = X_test_combined, s = "lambda.1se", type = "response"))
pred_lr <- ifelse(prob_lr >= 0.5, 1, 0)
eval_lr <- calc_metrics(y_test, pred_lr, prob_lr, "Logistic Regression (LASSO)")

# ------------------------------------------------------------------------------
# 2. Evaluate Random Forest
# ------------------------------------------------------------------------------
cat("Evaluating Model 2: Random Forest...\n")
prob_rf_matrix <- predict(rf_model, newdata = rf_test_df, type = "prob")
prob_rf <- prob_rf_matrix[, "Real"]
pred_rf_labels <- predict(rf_model, newdata = rf_test_df, type = "response")
pred_rf <- ifelse(pred_rf_labels == "Real", 1, 0)
eval_rf <- calc_metrics(y_test, pred_rf, prob_rf, "Random Forest")

# ------------------------------------------------------------------------------
# 3. Evaluate Naive Bayes
# ------------------------------------------------------------------------------
cat("Evaluating Model 3: Naive Bayes...\n")
prob_nb_matrix <- predict(nb_model, newdata = rf_test_df, type = "raw")
prob_nb <- prob_nb_matrix[, "Real"]
pred_nb_labels <- predict(nb_model, newdata = rf_test_df, type = "class")
pred_nb <- ifelse(pred_nb_labels == "Real", 1, 0)
eval_nb <- calc_metrics(y_test, pred_nb, prob_nb, "Naive Bayes")

# Combine metrics
all_metrics_df <- rbind(eval_lr$metrics, eval_rf$metrics, eval_nb$metrics)
cat("\n================ Evaluation Summary ================\n")
print(all_metrics_df[, c("Model", "Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC")])
cat("====================================================\n\n")

# Save results to CSV
eval_csv_path <- file.path(PROJECT_ROOT, "results", "evaluation_metrics.csv")
write.csv(all_metrics_df, eval_csv_path, row.names = FALSE)
cat(sprintf("Saved: %s\n", eval_csv_path))

# Save Confusion Matrix Breakdown
cm_df <- data.frame(
  Model = rep(all_metrics_df$Model, each = 4),
  Category = rep(c("True Positive (Real as Real)", "True Negative (Fake as Fake)",
                   "False Positive (Fake as Real)", "False Negative (Real as Fake)"), 3),
  Count = c(
    eval_lr$metrics$TP, eval_lr$metrics$TN, eval_lr$metrics$FP, eval_lr$metrics$FN,
    eval_rf$metrics$TP, eval_rf$metrics$TN, eval_rf$metrics$FP, eval_rf$metrics$FN,
    eval_nb$metrics$TP, eval_nb$metrics$TN, eval_nb$metrics$FP, eval_nb$metrics$FN
  )
)
cm_csv_path <- file.path(PROJECT_ROOT, "results", "confusion_matrix.csv")
write.csv(cm_df, cm_csv_path, row.names = FALSE)
cat(sprintf("Saved: %s\n", cm_csv_path))

# Cache evaluation objects for visualization script
eval_export <- list(
  metrics_df = all_metrics_df,
  cm_df = cm_df,
  y_test = y_test,
  eval_lr = eval_lr,
  eval_rf = eval_rf,
  eval_nb = eval_nb
)
saveRDS(eval_export, file.path(PROJECT_ROOT, "dataset", ".eval_results_temp.rds"))

cat("[08] Model evaluation complete.\n\n")
