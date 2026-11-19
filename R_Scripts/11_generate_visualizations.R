# ==============================================================================
# 11_generate_visualizations.R
# Multi-Feature Fake News Detection Using R Language
# Purpose: Generate high-resolution academic charts using ggplot2.
# ==============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(pROC)
  library(randomForest)
  library(jsonlite)
})

cat("========================================================\n")
cat(" [11] Generating Evaluation and Analysis Charts\n")
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
charts_dir   <- file.path(PROJECT_ROOT, "results", "charts")
if (!dir.exists(charts_dir)) dir.create(charts_dir, recursive = TRUE)

eval_csv_path <- file.path(PROJECT_ROOT, "results", "evaluation_metrics.csv")
cm_csv_path   <- file.path(PROJECT_ROOT, "results", "confusion_matrix.csv")

if (!file.exists(eval_csv_path)) {
  stop("[ERROR] Evaluation metrics not found. Please run 08_model_evaluation.R first.")
}

metrics_df <- read.csv(eval_csv_path, stringsAsFactors = FALSE)
cm_df      <- read.csv(cm_csv_path, stringsAsFactors = FALSE)

# Color Palette: Professional Academic Theme
navy_blue   <- "#1E3A8A"
cyan_teal   <- "#0284C7"
emerald_grn <- "#059669"
crimson_red <- "#DC2626"
slate_gray  <- "#475569"

# ------------------------------------------------------------------------------
# 1. Model Comparison Bar Chart
# ------------------------------------------------------------------------------
cat("Plotting 1: Model Comparison Bar Chart...\n")
metrics_long <- data.frame(
  Model = rep(metrics_df$Model, 4),
  Metric = rep(c("Accuracy", "Precision", "Recall", "F1_Score"), each = nrow(metrics_df)),
  Score = c(metrics_df$Accuracy, metrics_df$Precision, metrics_df$Recall, metrics_df$F1_Score)
)

p1 <- ggplot(metrics_long, aes(x = Model, y = Score, fill = Metric)) +
  geom_bar(stat = "identity", position = position_dodge(0.8), width = 0.7) +
  geom_text(aes(label = sprintf("%0.3f", Score)),
            position = position_dodge(0.8), vjust = -0.4, size = 3.2, fontface = "bold") +
  scale_fill_manual(values = c("Accuracy" = "#1E3A8A", "Precision" = "#0284C7", "Recall" = "#059669", "F1_Score" = "#F59E0B")) +
  scale_y_continuous(limits = c(0, 1.1), breaks = seq(0, 1, 0.2)) +
  labs(
    title = "Machine Learning Model Performance Comparison",
    subtitle = "Multi-Feature Fake News Detection Evaluation Metrics on Held-out Test Set",
    x = "Model Architecture",
    y = "Score (0 - 1.0)",
    fill = "Evaluation Metric"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14, color = "#0F172A"),
    plot.subtitle = element_text(color = "#475569", size = 10, margin = ggplot2::margin(b = 10)),
    legend.position = "top",
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(face = "bold", size = 10)
  )

ggsave(file.path(charts_dir, "model_comparison_bar.png"), p1, width = 9, height = 5.5, dpi = 300)

# ------------------------------------------------------------------------------
# 2. Confusion Matrices Heatmap
# ------------------------------------------------------------------------------
cat("Plotting 2: Confusion Matrices...\n")
cm_matrix_data <- data.frame(
  Model = rep(metrics_df$Model, each = 4),
  Actual = rep(c("Real (1)", "Real (1)", "Fake (0)", "Fake (0)"), nrow(metrics_df)),
  Predicted = rep(c("Real (1)", "Fake (0)", "Real (1)", "Fake (0)"), nrow(metrics_df)),
  Count = c(
    metrics_df$TP[1], metrics_df$FN[1], metrics_df$FP[1], metrics_df$TN[1],
    metrics_df$TP[2], metrics_df$FN[2], metrics_df$FP[2], metrics_df$TN[2],
    metrics_df$TP[3], metrics_df$FN[3], metrics_df$FP[3], metrics_df$TN[3]
  )
)

p2 <- ggplot(cm_matrix_data, aes(x = Predicted, y = Actual, fill = Count)) +
  geom_tile(color = "white", linewidth = 1.2) +
  geom_text(aes(label = format(Count, big.mark = ",")), size = 5, fontface = "bold", color = "white") +
  scale_fill_gradient(low = "#93C5FD", high = "#1E3A8A") +
  facet_wrap(~ Model, scales = "free") +
  labs(
    title = "Confusion Matrix Heatmaps by Model",
    subtitle = "Evaluation counts on held-out test articles",
    x = "Predicted Class",
    y = "Ground Truth Class"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 14, color = "#0F172A"),
    strip.text = element_text(face = "bold", size = 11),
    legend.position = "none"
  )

ggsave(file.path(charts_dir, "confusion_matrices.png"), p2, width = 10, height = 4.5, dpi = 300)

# ------------------------------------------------------------------------------
# 3. ROC Curves
# ------------------------------------------------------------------------------
cat("Plotting 3: Receiver Operating Characteristic (ROC) Curves...\n")
eval_temp_path <- file.path(PROJECT_ROOT, "dataset", ".eval_results_temp.rds")
if (file.exists(eval_temp_path)) {
  eval_data <- readRDS(eval_temp_path)
  
  roc_df_list <- list()
  models_to_plot <- list(
    list(name = "Logistic Regression (LASSO)", eval = eval_data$eval_lr, col = "#1E3A8A"),
    list(name = "Random Forest", eval = eval_data$eval_rf, col = "#059669"),
    list(name = "Naive Bayes", eval = eval_data$eval_nb, col = "#F59E0B")
  )
  
  for (m in models_to_plot) {
    if (!is.null(m$eval$roc)) {
      df_r <- data.frame(
        FPR = 1 - m$eval$roc$specificities,
        TPR = m$eval$roc$sensitivities,
        Model = sprintf("%s (AUC = %0.4f)", m$name, as.numeric(auc(m$eval$roc)))
      )
      roc_df_list[[length(roc_df_list) + 1]] <- df_r
    }
  }
  
  roc_all <- do.call(rbind, roc_df_list)
  
  p3 <- ggplot(roc_all, aes(x = FPR, y = TPR, color = Model)) +
    geom_line(linewidth = 1.1) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "#94A3B8") +
    scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
    scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
    scale_color_manual(values = c("#1E3A8A", "#059669", "#F59E0B")) +
    labs(
      title = "Receiver Operating Characteristic (ROC) Curves",
      subtitle = "True Positive Rate vs False Positive Rate across Decision Thresholds",
      x = "False Positive Rate (1 - Specificity)",
      y = "True Positive Rate (Sensitivity)",
      color = "Model & AUC"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", size = 14, color = "#0F172A"),
      legend.position = c(0.7, 0.25),
      legend.background = element_rect(fill = alpha("white", 0.8), color = "#CBD5E1")
    )
  
  ggsave(file.path(charts_dir, "roc_curves.png"), p3, width = 8, height = 6, dpi = 300)
}

# ------------------------------------------------------------------------------
# 4. Feature Importance Chart (from Random Forest & LASSO)
# ------------------------------------------------------------------------------
cat("Plotting 4: Feature Importance Chart...\n")
rf_path <- file.path(PROJECT_ROOT, "models", "random_forest_model.rds")
if (file.exists(rf_path)) {
  rf_mod <- readRDS(rf_path)
  imp <- importance(rf_mod)
  imp_df <- data.frame(
    Feature = rownames(imp),
    Importance = imp[, "MeanDecreaseGini"]
  ) %>%
    arrange(desc(Importance)) %>%
    head(15)
  
  # Format feature names cleanly
  imp_df$Feature_Clean <- gsub("_", " ", imp_df$Feature)
  imp_df$Feature_Clean <- tools::toTitleCase(imp_df$Feature_Clean)
  
  p4 <- ggplot(imp_df, aes(x = reorder(Feature_Clean, Importance), y = Importance)) +
    geom_bar(stat = "identity", fill = "#0284C7", width = 0.7) +
    coord_flip() +
    labs(
      title = "Top 15 Most Discriminative Features",
      subtitle = "Gini Impurity Decrease in Random Forest Ensemble",
      x = "Feature Name",
      y = "Mean Decrease Gini"
    ) +
    theme_minimal(base_size = 11) +
    theme(
      plot.title = element_text(face = "bold", size = 13, color = "#0F172A"),
      axis.text.y = element_text(face = "bold")
    )
  
  ggsave(file.path(charts_dir, "feature_importance.png"), p4, width = 8, height = 5.5, dpi = 300)
}

# ------------------------------------------------------------------------------
# 5. Dataset Distribution Chart
# ------------------------------------------------------------------------------
cat("Plotting 5: Dataset Distribution Chart...\n")
dist_path <- file.path(PROJECT_ROOT, "results", "reports", "dataset_cleaning_summary.json")
if (file.exists(dist_path)) {
  clean_rep <- fromJSON(dist_path)
  dist_df <- data.frame(
    Class = c("Fake News", "Real News"),
    Count = c(clean_rep$fake_articles, clean_rep$real_articles)
  )
  
  p5 <- ggplot(dist_df, aes(x = Class, y = Count, fill = Class)) +
    geom_bar(stat = "identity", width = 0.5) +
    geom_text(aes(label = sprintf("%s articles\n(%0.1f%%)",
                                  format(Count, big.mark = ","),
                                  (Count / sum(Count)) * 100)),
              vjust = -0.3, fontface = "bold", size = 4) +
    scale_fill_manual(values = c("Fake News" = "#DC2626", "Real News" = "#059669")) +
    scale_y_continuous(limits = c(0, max(dist_df$Count) * 1.2)) +
    labs(
      title = "Dataset Class Balance After Cleaning & Deduplication",
      subtitle = sprintf("Total Cleaned Corpus: %s Articles", format(sum(dist_df$Count), big.mark = ",")),
      x = "News Category",
      y = "Article Count"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", size = 14, color = "#0F172A"),
      legend.position = "none"
    )
  
  ggsave(file.path(charts_dir, "dataset_distribution.png"), p5, width = 7, height = 5, dpi = 300)
}

# ------------------------------------------------------------------------------
# 6. Sentiment Distribution Chart
# ------------------------------------------------------------------------------
cat("Plotting 6: Sentiment Distribution Chart...\n")
feat_cache_path <- file.path(PROJECT_ROOT, "dataset", ".features_cache_temp.rds")
if (file.exists(feat_cache_path)) {
  cached_f <- readRDS(feat_cache_path)
  sample_f <- cached_f$train_feats
  sample_f$Class <- ifelse(sample_f$label == 1, "Real News", "Fake News")
  
  p6 <- ggplot(sample_f, aes(x = sentiment_score, fill = Class)) +
    geom_histogram(bins = 40, alpha = 0.7, position = "identity") +
    scale_fill_manual(values = c("Fake News" = "#DC2626", "Real News" = "#059669")) +
    scale_x_continuous(limits = c(-20, 25)) +
    labs(
      title = "Lexicon Sentiment Score Distribution by Class",
      subtitle = "Sentiment Score = Positive Words - Negative Words",
      x = "Net Sentiment Score",
      y = "Article Frequency",
      fill = "Class"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", size = 14, color = "#0F172A"),
      legend.position = "top"
    )
  
  ggsave(file.path(charts_dir, "sentiment_distribution.png"), p6, width = 8, height = 5, dpi = 300)
  
  # 7. Readability Comparison Boxplot
  cat("Plotting 7: Readability Comparison Boxplot...\n")
  p7 <- ggplot(sample_f, aes(x = Class, y = flesch_reading_ease, fill = Class)) +
    geom_boxplot(width = 0.4, outlier.alpha = 0.3) +
    scale_fill_manual(values = c("Fake News" = "#DC2626", "Real News" = "#059669")) +
    labs(
      title = "Flesch Reading Ease Comparison",
      subtitle = "Higher scores indicate easier reading; lower scores indicate complex academic/technical prose",
      x = "News Category",
      y = "Flesch Reading Ease Score"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", size = 14, color = "#0F172A"),
      legend.position = "none"
    )
  
  ggsave(file.path(charts_dir, "readability_comparison.png"), p7, width = 7, height = 5, dpi = 300)
}

cat("[11] All visualizations generated successfully in results/charts/.\n\n")
