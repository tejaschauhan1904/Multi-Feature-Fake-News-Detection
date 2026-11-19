# ==============================================================================
# 07_model_training.R
# Multi-Feature Fake News Detection Using R Language
# Purpose: Train Logistic Regression (LASSO), Random Forest, and Naive Bayes models.
# ==============================================================================

suppressPackageStartupMessages({
  library(Matrix)
  library(glmnet)
  library(randomForest)
  library(e1071)
})

cat("========================================================\n")
cat(" [07] Training Machine Learning Models\n")
cat("========================================================\n")

get_project_root <- function() {
  candidates <- c(getwd(), file.path(getwd(), ".."), "d:/Tejas/College/Sem_5/R Language/R_PBL_project")
  for (cand in candidates) {
    if (file.exists(file.path(cand, "dataset", ".matrix_cache_temp.rds"))) {
      return(normalizePath(cand, winslash = "/"))
    }
  }
  return(normalizePath(getwd(), winslash = "/"))
}

PROJECT_ROOT <- get_project_root()
matrix_cache_path <- file.path(PROJECT_ROOT, "dataset", ".matrix_cache_temp.rds")

if (!file.exists(matrix_cache_path)) {
  stop("[ERROR] Matrix cache not found. Please run 06_tfidf_vectorization.R first.")
}

cached <- readRDS(matrix_cache_path)
X_train_combined <- cached$X_train_combined
X_test_combined  <- cached$X_test_combined
y_train          <- cached$y_train
y_test           <- cached$y_test
num_train_scaled <- cached$num_train_scaled
num_test_scaled  <- cached$num_test_scaled
dtm_train_tfidf  <- cached$dtm_train_tfidf
dtm_test_tfidf   <- cached$dtm_test_tfidf
feature_cols     <- cached$feature_cols

cat(sprintf("Training dimensions: %d rows x %d features\n", nrow(X_train_combined), ncol(X_train_combined)))
cat(sprintf("Class distribution: Fake (0) = %d | Real (1) = %d\n", sum(y_train == 0), sum(y_train == 1)))

# ------------------------------------------------------------------------------
# MODEL 1: Logistic Regression with LASSO (Primary Model)
# ------------------------------------------------------------------------------
cat("\n[1/3] Training Logistic Regression with LASSO (glmnet)...\n")
set.seed(42)
# 5-fold cross-validation to select optimal regularization parameter lambda
start_time_lr <- Sys.time()
cv_lasso <- cv.glmnet(
  x = X_train_combined,
  y = y_train,
  family = "binomial",
  alpha = 1, # L1 penalty (LASSO)
  nfolds = 5,
  type.measure = "auc",
  standardize = FALSE # Features already scaled/normalized
)
elapsed_lr <- round(as.numeric(difftime(Sys.time(), start_time_lr, units = "secs")), 2)
cat(sprintf("  -> Optimal lambda (min): %0.5f | lambda (1se): %0.5f (Trained in %s s)\n",
            cv_lasso$lambda.min, cv_lasso$lambda.1se, elapsed_lr))

saveRDS(cv_lasso, file.path(PROJECT_ROOT, "models", "logistic_regression_model.rds"))
cat("  -> Saved: models/logistic_regression_model.rds\n")

# ------------------------------------------------------------------------------
# MODEL 2: Random Forest
# ------------------------------------------------------------------------------
cat("\n[2/3] Training Random Forest Classifier...\n")
# Documented feature-selection for Random Forest:
# Standard Random Forest requires a dense matrix; 4000 dense TF-IDF columns causes memory explosion.
# Therefore, we combine all 32 scaled multi-category features with top 50 highest-variance TF-IDF words.
tfidf_vars <- apply(dtm_train_tfidf, 2, var)
top_50_terms_idx <- order(tfidf_vars, decreasing = TRUE)[1:50]
top_terms_names <- colnames(dtm_train_tfidf)[top_50_terms_idx]

dtm_train_top50 <- as.matrix(dtm_train_tfidf[, top_50_terms_idx])
dtm_test_top50  <- as.matrix(dtm_test_tfidf[, top_50_terms_idx])

# Create combined dense training frame for tree/probabilistic models
rf_train_df <- as.data.frame(cbind(num_train_scaled, dtm_train_top50))
rf_test_df  <- as.data.frame(cbind(num_test_scaled, dtm_test_top50))

# Sanitize column names for formula compatibility
colnames(rf_train_df) <- make.names(colnames(rf_train_df), unique = TRUE)
colnames(rf_test_df)  <- make.names(colnames(rf_test_df), unique = TRUE)

y_train_factor <- as.factor(ifelse(y_train == 1, "Real", "Fake"))
y_test_factor  <- as.factor(ifelse(y_test == 1, "Real", "Fake"))

set.seed(42)
start_time_rf <- Sys.time()
rf_model <- randomForest(
  x = rf_train_df,
  y = y_train_factor,
  ntree = 100,
  mtry = floor(sqrt(ncol(rf_train_df))),
  importance = TRUE
)
elapsed_rf <- round(as.numeric(difftime(Sys.time(), start_time_rf, units = "secs")), 2)
cat(sprintf("  -> Random Forest trained with 100 trees over %d features (Trained in %s s)\n",
            ncol(rf_train_df), elapsed_rf))

saveRDS(rf_model, file.path(PROJECT_ROOT, "models", "random_forest_model.rds"))
cat("  -> Saved: models/random_forest_model.rds\n")

# ------------------------------------------------------------------------------
# MODEL 3: Naive Bayes
# ------------------------------------------------------------------------------
cat("\n[3/3] Training Naive Bayes Classifier (e1071)...\n")
start_time_nb <- Sys.time()
nb_model <- naiveBayes(
  x = rf_train_df,
  y = y_train_factor,
  laplace = 1
)
elapsed_nb <- round(as.numeric(difftime(Sys.time(), start_time_nb, units = "secs")), 2)
cat(sprintf("  -> Naive Bayes trained with Laplace smoothing = 1 (Trained in %s s)\n", elapsed_nb))

saveRDS(nb_model, file.path(PROJECT_ROOT, "models", "naive_bayes_model.rds"))
cat("  -> Saved: models/naive_bayes_model.rds\n")

# ------------------------------------------------------------------------------
# Save Model Metadata
# ------------------------------------------------------------------------------
model_metadata <- list(
  trained_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
  positive_class = "Real (1)",
  negative_class = "Fake (0)",
  training_samples = nrow(X_train_combined),
  test_samples = nrow(X_test_combined),
  models = list(
    logistic_regression = list(
      algorithm = "glmnet LASSO Logistic Regression",
      features_used = "Full TF-IDF (4,000 terms) + 32 Multi-category Numerical Features",
      total_features = ncol(X_train_combined),
      training_time_secs = elapsed_lr,
      lambda_min = cv_lasso$lambda.min,
      lambda_1se = cv_lasso$lambda.1se
    ),
    random_forest = list(
      algorithm = "Breiman's Random Forest",
      features_used = "32 Multi-category Numerical Features + Top 50 TF-IDF Terms",
      total_features = ncol(rf_train_df),
      trees = 100,
      training_time_secs = elapsed_rf,
      top_terms = top_terms_names
    ),
    naive_bayes = list(
      algorithm = "Gaussian Naive Bayes (Laplace = 1)",
      features_used = "32 Multi-category Numerical Features + Top 50 TF-IDF Terms",
      total_features = ncol(rf_train_df),
      training_time_secs = elapsed_nb,
      laplace = 1
    )
  ),
  rf_feature_names = colnames(rf_train_df),
  top_50_terms_idx = top_50_terms_idx
)

saveRDS(model_metadata, file.path(PROJECT_ROOT, "models", "model_metadata.rds"))
cat("Saved: models/model_metadata.rds\n")

# Cache test representations and models for evaluation script
saveRDS(list(
  cv_lasso = cv_lasso,
  rf_model = rf_model,
  nb_model = nb_model,
  X_test_combined = X_test_combined,
  rf_test_df = rf_test_df,
  y_test = y_test,
  y_test_factor = y_test_factor
), file.path(PROJECT_ROOT, "dataset", ".eval_cache_temp.rds"))

cat("[07] Machine learning model training complete.\n\n")
