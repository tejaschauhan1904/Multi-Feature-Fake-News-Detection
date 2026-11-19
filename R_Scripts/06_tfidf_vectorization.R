# ==============================================================================
# 06_tfidf_vectorization.R
# Multi-Feature Fake News Detection Using R Language
# Purpose: Fit TF-IDF on train data, transform test data, combine with numerical features.
# ==============================================================================

suppressPackageStartupMessages({
  library(text2vec)
  library(Matrix)
  library(dplyr)
  library(stringr)
})

cat("========================================================\n")
cat(" [06] TF-IDF Vectorization & Feature Integration\n")
cat("========================================================\n")

SOURCED_BY_PIPELINE <- TRUE
get_project_root <- function() {
  candidates <- c(getwd(), file.path(getwd(), ".."), "d:/Tejas/College/Sem_5/R Language/R_PBL_project")
  for (cand in candidates) {
    if (file.exists(file.path(cand, "dataset", ".features_cache_temp.rds"))) {
      return(normalizePath(cand, winslash = "/"))
    }
  }
  return(normalizePath(getwd(), winslash = "/"))
}

PROJECT_ROOT <- get_project_root()
source(file.path(PROJECT_ROOT, "R_Scripts", "04_text_preprocessing.R"))
source(file.path(PROJECT_ROOT, "R_Scripts", "05_feature_extraction.R"))

features_cache_path <- file.path(PROJECT_ROOT, "dataset", ".features_cache_temp.rds")
if (!file.exists(features_cache_path)) {
  stop("[ERROR] Features cache not found. Please run 05_feature_extraction.R first.")
}

cached <- readRDS(features_cache_path)
train_df    <- cached$train_df
train_feats <- cached$train_feats
test_df     <- cached$test_df
test_feats  <- cached$test_feats

cat(sprintf("Training samples: %d | Test samples: %d\n", nrow(train_df), nrow(test_df)))

# ------------------------------------------------------------------------------
# 1. Clean text for TF-IDF
# ------------------------------------------------------------------------------
cat("Cleaning combined text for TF-IDF (unigrams + bigrams)...\n")
train_clean_text <- clean_text_for_nlp(train_df$combined_text, remove_stopwords = TRUE)
test_clean_text  <- clean_text_for_nlp(test_df$combined_text, remove_stopwords = TRUE)

# ------------------------------------------------------------------------------
# 2. Build Vocabulary and TF-IDF Transformer on Training Data Only (No Leakage!)
# ------------------------------------------------------------------------------
cat("Tokenizing and creating vocabulary on training set...\n")
it_train <- itoken(train_clean_text,
                   tokenizer = word_tokenizer,
                   ids = train_df$id,
                   progressbar = FALSE)

vocab <- create_vocabulary(it_train, ngram = c(1L, 2L))

# Prune vocabulary to avoid overfitting and keep memory optimal
# Retain up to 4,000 most informative terms
pruned_vocab <- prune_vocabulary(vocab,
                                 term_count_min = 5L,
                                 doc_proportion_max = 0.5,
                                 doc_proportion_min = 0.002,
                                 vocab_term_max = 4000L)

cat(sprintf("Raw vocabulary size: %d | Pruned vocabulary size: %d\n",
            nrow(vocab), nrow(pruned_vocab)))

vectorizer <- vocab_vectorizer(pruned_vocab)

# Document-Term Matrix
dtm_train <- create_dtm(it_train, vectorizer)

# Fit TF-IDF model on TRAIN DTM
tfidf_model <- TfIdf$new(norm = "l2", sublinear_tf = TRUE)
dtm_train_tfidf <- tfidf_model$fit_transform(dtm_train)

# Transform TEST DTM using already fitted vectorizer and tfidf_model
it_test <- itoken(test_clean_text,
                  tokenizer = word_tokenizer,
                  ids = test_df$id,
                  progressbar = FALSE)
dtm_test <- create_dtm(it_test, vectorizer)
dtm_test_tfidf <- tfidf_model$transform(dtm_test)

cat(sprintf("TF-IDF Train Matrix: %d docs x %d terms\n", nrow(dtm_train_tfidf), ncol(dtm_train_tfidf)))
cat(sprintf("TF-IDF Test Matrix:  %d docs x %d terms\n", nrow(dtm_test_tfidf), ncol(dtm_test_tfidf)))

# ------------------------------------------------------------------------------
# 3. Scale and Standardize Numerical Features
# ------------------------------------------------------------------------------
# Exclude non-feature columns (id, label)
feature_cols <- setdiff(colnames(train_feats), c("id", "label"))
num_train <- as.matrix(train_feats[, feature_cols])
num_test  <- as.matrix(test_feats[, feature_cols])

# Calculate mean and standard deviation from TRAIN ONLY
feature_means <- colMeans(num_train, na.rm = TRUE)
feature_sds   <- apply(num_train, 2, sd, na.rm = TRUE)
# Prevent division by zero
feature_sds[feature_sds == 0 | is.na(feature_sds)] <- 1

scale_matrix <- function(mat, means, sds) {
  scaled <- sweep(mat, 2, means, "-")
  scaled <- sweep(scaled, 2, sds, "/")
  scaled[is.na(scaled)] <- 0
  scaled[is.infinite(scaled)] <- 0
  return(scaled)
}

num_train_scaled <- scale_matrix(num_train, feature_means, feature_sds)
num_test_scaled  <- scale_matrix(num_test, feature_means, feature_sds)

# ------------------------------------------------------------------------------
# 4. Integrate TF-IDF and Numerical Features into Combined Sparse Matrix
# ------------------------------------------------------------------------------
cat("Combining sparse TF-IDF matrix with scaled numerical features...\n")
sparse_num_train <- Matrix(num_train_scaled, sparse = TRUE)
sparse_num_test  <- Matrix(num_test_scaled, sparse = TRUE)

X_train_combined <- cbind(dtm_train_tfidf, sparse_num_train)
X_test_combined  <- cbind(dtm_test_tfidf, sparse_num_test)

y_train <- train_df$label
y_test  <- test_df$label

cat(sprintf("Final Integrated Train Matrix: %d rows x %d features\n",
            nrow(X_train_combined), ncol(X_train_combined)))
cat(sprintf("Final Integrated Test Matrix:  %d rows x %d features\n",
            nrow(X_test_combined), ncol(X_test_combined)))

# ------------------------------------------------------------------------------
# 5. Save Preprocessor Objects and Vectorizer
# ------------------------------------------------------------------------------
tfidf_vectorizer_bundle <- list(
  vectorizer = vectorizer,
  pruned_vocab = pruned_vocab,
  tfidf_model = tfidf_model,
  ngram = c(1, 2)
)

saveRDS(tfidf_vectorizer_bundle, file.path(PROJECT_ROOT, "models", "tfidf_vectorizer.rds"))
cat("Saved: models/tfidf_vectorizer.rds\n")

preprocessing_bundle <- list(
  feature_cols = feature_cols,
  feature_means = feature_means,
  feature_sds = feature_sds,
  stopwords_count = length(STOPWORDS_EN),
  positive_words_count = length(SENTIMENT_LEX$positive),
  negative_words_count = length(SENTIMENT_LEX$negative),
  tfidf_feature_count = ncol(dtm_train_tfidf),
  num_feature_count = length(feature_cols),
  total_feature_count = ncol(X_train_combined),
  created_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S")
)

saveRDS(preprocessing_bundle, file.path(PROJECT_ROOT, "models", "preprocessing_objects.rds"))
cat("Saved: models/preprocessing_objects.rds\n")

# Save combined matrices for modular model training
matrix_cache <- list(
  X_train_combined = X_train_combined,
  X_test_combined  = X_test_combined,
  y_train = y_train,
  y_test  = y_test,
  num_train_scaled = num_train_scaled,
  num_test_scaled  = num_test_scaled,
  dtm_train_tfidf  = dtm_train_tfidf,
  dtm_test_tfidf   = dtm_test_tfidf,
  feature_cols     = feature_cols
)

saveRDS(matrix_cache, file.path(PROJECT_ROOT, "dataset", ".matrix_cache_temp.rds"))
cat("[06] TF-IDF vectorization and feature integration complete.\n\n")
