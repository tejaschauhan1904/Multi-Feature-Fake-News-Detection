# ==============================================================================
# 09_model_comparison.R
# Multi-Feature Fake News Detection Using R Language
# Purpose: Compare models side-by-side, determine best model, generate report.
# ==============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(jsonlite)
})

cat("========================================================\n")
cat(" [09] Model Comparison and Selection\n")
cat("========================================================\n")

get_project_root <- function() {
  candidates <- c(getwd(), file.path(getwd(), ".."), "d:/Tejas/College/Sem_5/R Language/R_PBL_project")
  for (cand in candidates) {
    if (file.exists(file.path(cand, "results", "evaluation_metrics.csv"))) {
      return(normalizePath(cand, winslash = "/"))
    }
  }
  return(normalizePath(getwd(), winslash = "/"))
}

PROJECT_ROOT <- get_project_root()
eval_csv_path <- file.path(PROJECT_ROOT, "results", "evaluation_metrics.csv")

if (!file.exists(eval_csv_path)) {
  stop("[ERROR] Evaluation metrics CSV not found. Please run 08_model_evaluation.R first.")
}

metrics_df <- read.csv(eval_csv_path, stringsAsFactors = FALSE)

# Rank models based primarily on F1-Score and ROC-AUC
ranked_df <- metrics_df %>%
  mutate(Composite_Score = (F1_Score * 0.6) + (ROC_AUC * 0.4)) %>%
  arrange(desc(Composite_Score))

best_model_name <- ranked_df$Model[1]
best_f1  <- ranked_df$F1_Score[1]
best_auc <- ranked_df$ROC_AUC[1]
best_acc <- ranked_df$Accuracy[1]

cat(sprintf("Best Performing Model: %s\n", best_model_name))
cat(sprintf("  -> Accuracy: %0.2f%% | F1-Score: %0.4f | ROC-AUC: %0.4f\n\n",
            best_acc * 100, best_f1, best_auc))

# Save enhanced model comparison table
comp_csv_path <- file.path(PROJECT_ROOT, "results", "model_comparison.csv")
write.csv(ranked_df, comp_csv_path, row.names = FALSE)
cat(sprintf("Saved: %s\n", comp_csv_path))

# Save JSON summary for Python Streamlit dashboard
comp_summary <- list(
  best_model = best_model_name,
  best_accuracy = best_acc,
  best_f1_score = best_f1,
  best_roc_auc = best_auc,
  ranked_models = ranked_df$Model,
  model_count = nrow(metrics_df),
  evaluation_date = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
  methodology_note = paste(
    "Logistic Regression (LASSO) evaluated over full 4,000 TF-IDF vocabulary + 32 numerical features;",
    "Random Forest and Naive Bayes evaluated over top 50 TF-IDF features + 32 numerical features to preserve memory.",
    "Best model selected via balanced F1-score and ROC-AUC criteria."
  )
)

json_path <- file.path(PROJECT_ROOT, "results", "reports", "model_comparison_summary.json")
write_json(comp_summary, json_path, pretty = TRUE, auto_unbox = TRUE)
cat(sprintf("Saved: %s\n", json_path))

# Generate text summary report
report_text <- sprintf(
"================================================================================
MULTI-FEATURE FAKE NEWS DETECTION: MODEL COMPARISON REPORT
Generated: %s
================================================================================

1. EXECUTIVE SUMMARY
--------------------------------------------------------------------------------
Three supervised machine learning algorithms were trained and evaluated on 
stratified held-out test data (80:20 train-test ratio, seed = 42).

Best Performing Model: %s
- Accuracy:  %0.2f%%
- Precision: %0.4f
- Recall:    %0.4f
- F1-Score:  %0.4f
- ROC-AUC:   %0.4f

2. FULL MODEL PERFORMANCE COMPARISON
--------------------------------------------------------------------------------
%-30s | %-8s | %-9s | %-8s | %-8s | %-8s
%s
",
format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
best_model_name,
best_acc * 100,
ranked_df$Precision[1],
ranked_df$Recall[1],
best_f1,
best_auc,
"Model", "Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC",
paste(rep("-", 80), collapse = "")
)

for (i in seq_len(nrow(ranked_df))) {
  row <- ranked_df[i, ]
  report_text <- paste0(report_text, sprintf(
    "%-30s | %-8.4f | %-9.4f | %-8.4f | %-8.4f | %-8.4f\n",
    row$Model, row$Accuracy, row$Precision, row$Recall, row$F1_Score, row$ROC_AUC
  ))
}

report_text <- paste0(report_text, "
3. ARCHITECTURAL & FEATURE REPRESENTATION NOTES
--------------------------------------------------------------------------------
- Model 1 (Logistic Regression with LASSO):
  Trained on combined sparse matrix comprising 4,000 TF-IDF unigrams/bigrams
  plus 32 standardized numerical features (sentiment, readability, linguistic,
  and statistical). LASSO (L1 regularization) automatically pruned uninformative
  terms, achieving superior generalization and fast inference.

- Model 2 (Random Forest):
  Ensemble of 100 decision trees. In order to avoid memory exhaustion and high
  tree-construction latency from 4,000 dense features, Random Forest utilized all
  32 numerical features combined with the top 50 highest-variance TF-IDF terms.

- Model 3 (Naive Bayes):
  Probabilistic classifier with Laplace smoothing (alpha = 1). Assumes conditional
  independence between terms and numerical dimensions. Serves as a strong fast
  baseline.

4. SELECTION CRITERIA
--------------------------------------------------------------------------------
Model selection prioritizes F1-score and ROC-AUC rather than accuracy alone, ensuring
balanced sensitivity across both Fake and Real classes without positive-class bias.
================================================================================
")

report_path <- file.path(PROJECT_ROOT, "results", "reports", "model_summary.txt")
writeLines(report_text, report_path)
cat(sprintf("Saved: %s\n", report_path))

cat("[09] Model comparison complete.\n\n")
