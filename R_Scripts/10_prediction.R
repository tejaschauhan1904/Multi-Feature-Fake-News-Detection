# ==============================================================================
# 10_prediction.R
# Multi-Feature Fake News Detection Using R Language
# Purpose: JSON-based prediction interface for Python Streamlit integration.
# Interface: Rscript 10_prediction.R <input_json> <output_json>
# ==============================================================================

# Auto-configure local r_libs if present in project
for (cand in c(getwd(), file.path(getwd(), ".."), "d:/Tejas/College/Sem_5/R Language/R_PBL_project")) {
  local_rlib <- file.path(cand, "r_libs")
  if (dir.exists(local_rlib)) {
    .libPaths(c(normalizePath(local_rlib, winslash = "/"), .libPaths()))
    break
  }
}

suppressPackageStartupMessages({
  library(jsonlite)
  library(Matrix)
  library(glmnet)
  library(randomForest)
  library(e1071)
  library(text2vec)
  library(stringr)
})

# Find project root
get_project_root <- function() {
  candidates <- c(getwd(), file.path(getwd(), ".."), "d:/Tejas/College/Sem_5/R Language/R_PBL_project")
  for (cand in candidates) {
    if (file.exists(file.path(cand, "models", "preprocessing_objects.rds"))) {
      return(normalizePath(cand, winslash = "/"))
    }
  }
  return(normalizePath(getwd(), winslash = "/"))
}

PROJECT_ROOT <- get_project_root()

# Source text preprocessing and feature extraction modules
SOURCED_BY_PIPELINE <- TRUE
source(file.path(PROJECT_ROOT, "R_Scripts", "04_text_preprocessing.R"))
source(file.path(PROJECT_ROOT, "R_Scripts", "05_feature_extraction.R"))

# Error response helper
write_error_exit <- function(msg, out_path = NULL) {
  err_payload <- list(
    status = "error",
    message = msg,
    timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  )
  if (!is.null(out_path)) {
    write_json(err_payload, out_path, pretty = TRUE, auto_unbox = TRUE)
  } else {
    cat(toJSON(err_payload, pretty = TRUE, auto_unbox = TRUE), "\n")
  }
  quit(status = 1)
}

# Main prediction function
predict_article <- function(title, text, selected_model_name = "auto") {
  title <- ifelse(is.null(title) || is.na(title), "", as.character(title))
  text  <- ifelse(is.null(text) || is.na(text), "", as.character(text))
  
  if (nchar(str_squish(title)) == 0 && nchar(str_squish(text)) == 0) {
    stop("Both title and text are empty. Please provide article content.")
  }
  
  # Check required models and preprocessors
  preproc_path <- file.path(PROJECT_ROOT, "models", "preprocessing_objects.rds")
  vectorizer_path <- file.path(PROJECT_ROOT, "models", "tfidf_vectorizer.rds")
  meta_path <- file.path(PROJECT_ROOT, "models", "model_metadata.rds")
  
  if (!file.exists(preproc_path) || !file.exists(vectorizer_path)) {
    stop("Trained models or preprocessor objects not found. Please train models first using main.R.")
  }
  
  preproc <- readRDS(preproc_path)
  tfidf_bundle <- readRDS(vectorizer_path)
  metadata <- if (file.exists(meta_path)) readRDS(meta_path) else list()
  
  # Step 1: Extract 32 Multi-category features from raw text
  num_feats_df <- extract_single_article_features(title, text)
  
  # Step 2: Scale numerical features using training parameters
  feature_cols <- preproc$feature_cols
  # Align columns
  aligned_num <- as.matrix(num_feats_df[, feature_cols, drop = FALSE])
  aligned_num[is.na(aligned_num)] <- 0
  
  scaled_num <- sweep(aligned_num, 2, preproc$feature_means, "-")
  scaled_num <- sweep(scaled_num, 2, preproc$feature_sds, "/")
  scaled_num[is.na(scaled_num)] <- 0
  scaled_num[is.infinite(scaled_num)] <- 0
  
  # Step 3: Clean text & Generate TF-IDF representation
  combined_text <- paste(title, text, sep = " ")
  cleaned_text  <- clean_text_for_nlp(combined_text, remove_stopwords = TRUE)
  
  it_live <- itoken(cleaned_text, tokenizer = word_tokenizer, progressbar = FALSE)
  dtm_live <- create_dtm(it_live, tfidf_bundle$vectorizer)
  dtm_live_tfidf <- tfidf_bundle$tfidf_model$transform(dtm_live)
  
  # Extract top article keywords based on TF-IDF
  nonzero_terms <- which(as.numeric(dtm_live_tfidf[1, ]) > 0)
  top_keywords <- list()
  if (length(nonzero_terms) > 0) {
    term_names <- colnames(dtm_live_tfidf)[nonzero_terms]
    term_scores <- as.numeric(dtm_live_tfidf[1, nonzero_terms])
    ord <- order(term_scores, decreasing = TRUE)
    top_n <- min(8, length(ord))
    top_keywords <- as.list(setNames(round(term_scores[ord[1:top_n]], 4), term_names[ord[1:top_n]]))
  }
  
  # Combine into sparse matrix for Logistic Regression
  sparse_num <- Matrix(scaled_num, sparse = TRUE)
  X_combined_live <- cbind(dtm_live_tfidf, sparse_num)
  
  # Determine model to use
  # Options: "logistic_regression", "random_forest", "naive_bayes", "auto"
  chosen_model <- tolower(selected_model_name)
  if (chosen_model %in% c("auto", "logistic", "logistic_regression", "lasso")) {
    chosen_model <- "logistic_regression"
  } else if (chosen_model %in% c("rf", "random_forest", "randomforest")) {
    chosen_model <- "random_forest"
  } else if (chosen_model %in% c("nb", "naive_bayes", "naivebayes")) {
    chosen_model <- "naive_bayes"
  } else {
    chosen_model <- "logistic_regression"
  }
  
  real_prob <- 0.5
  model_display_name <- "Logistic Regression (LASSO)"
  
  if (chosen_model == "logistic_regression") {
    model_path <- file.path(PROJECT_ROOT, "models", "logistic_regression_model.rds")
    if (!file.exists(model_path)) stop("Logistic regression model file missing.")
    lr_model <- readRDS(model_path)
    real_prob <- as.numeric(predict(lr_model, newx = X_combined_live, s = "lambda.1se", type = "response"))[1]
    model_display_name <- "Logistic Regression (LASSO)"
  } else if (chosen_model %in% c("random_forest", "naive_bayes")) {
    # Prepare dense input: scaled numerical features + top 50 TF-IDF features
    top_50_idx <- metadata$top_50_terms_idx
    dtm_top50 <- as.matrix(dtm_live_tfidf[, top_50_idx, drop = FALSE])
    dense_df <- as.data.frame(cbind(scaled_num, dtm_top50))
    colnames(dense_df) <- metadata$rf_feature_names
    
    if (chosen_model == "random_forest") {
      model_path <- file.path(PROJECT_ROOT, "models", "random_forest_model.rds")
      if (!file.exists(model_path)) stop("Random Forest model file missing.")
      rf_model <- readRDS(model_path)
      rf_probs <- predict(rf_model, newdata = dense_df, type = "prob")
      real_prob <- as.numeric(rf_probs[1, "Real"])
      model_display_name <- "Random Forest Classifier"
    } else {
      model_path <- file.path(PROJECT_ROOT, "models", "naive_bayes_model.rds")
      if (!file.exists(model_path)) stop("Naive Bayes model file missing.")
      nb_model <- readRDS(model_path)
      nb_probs <- predict(nb_model, newdata = dense_df, type = "raw")
      real_prob <- as.numeric(nb_probs[1, "Real"])
      model_display_name <- "Naive Bayes Classifier"
    }
  }
  
  # Ensure real_prob is bounded in [0, 1]
  real_prob <- max(0.0001, min(0.9999, real_prob))
  fake_prob <- 1.0 - real_prob
  
  prediction_class <- ifelse(real_prob >= 0.5, "REAL NEWS", "FAKE NEWS")
  predicted_label  <- ifelse(real_prob >= 0.5, 1, 0)
  class_confidence <- ifelse(predicted_label == 1, real_prob, fake_prob)
  
  # Format extracted feature dictionary for UI inspection
  features_dict <- as.list(num_feats_df[1, ])
  
  list(
    status = "success",
    prediction = prediction_class,
    label = predicted_label,
    real_news_probability = round(real_prob, 4),
    fake_news_probability = round(fake_prob, 4),
    confidence = round(class_confidence, 4),
    confidence_percent = sprintf("%0.2f%%", class_confidence * 100),
    model_used = model_display_name,
    model_key = chosen_model,
    extracted_features = features_dict,
    top_keywords = top_keywords,
    sentiment_breakdown = list(
      sentiment_score = num_feats_df$sentiment_score,
      positive_words = num_feats_df$pos_words_count,
      negative_words = num_feats_df$neg_words_count,
      sentiment_balance = round(num_feats_df$sentiment_balance, 3)
    ),
    readability = list(
      flesch_reading_ease = round(num_feats_df$flesch_reading_ease, 1),
      avg_words_per_sentence = round(num_feats_df$avg_words_per_sentence, 1),
      avg_sentence_length_chars = round(num_feats_df$avg_sentence_length_chars, 1),
      total_words = num_feats_df$word_count,
      total_sentences = num_feats_df$sentence_count
    ),
    disclaimer = "This result is a machine learning classification based on patterns learned from the training dataset. It does not independently verify the factual accuracy of the article.",
    timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  )
}

# ------------------------------------------------------------------------------
# CLI Execution Handler
# ------------------------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)

if (length(args) >= 2) {
  input_path  <- args[1]
  output_path <- args[2]
  
  if (!file.exists(input_path)) {
    write_error_exit(sprintf("Input file '%s' not found.", input_path), output_path)
  }
  
  # Read input JSON
  in_data <- tryCatch({
    fromJSON(input_path)
  }, error = function(e) {
    write_error_exit(paste("Failed to parse input JSON:", e$message), output_path)
  })
  
  title <- ifelse("title" %in% names(in_data), in_data$title, "")
  text  <- ifelse("text" %in% names(in_data), in_data$text, "")
  model <- ifelse("model" %in% names(in_data), in_data$model, "auto")
  
  result <- tryCatch({
    predict_article(title, text, model)
  }, error = function(e) {
    write_error_exit(paste("Prediction error:", e$message), output_path)
  })
  
  # Write output JSON
  write_json(result, output_path, pretty = TRUE, auto_unbox = TRUE)
  
} else if (length(args) == 0) {
  # Self-test demonstration
  cat("Running self-test on sample article...\n")
  sample_title <- "NASA Announces Successful Launch of Next-Generation Climate Satellite"
  sample_text  <- "The National Aeronautics and Space Administration successfully launched its advanced Earth-observing satellite from Cape Canaveral today. The mission aims to provide global measurements of ocean winds and weather patterns to enhance scientific research into atmospheric dynamics."
  
  res <- predict_article(sample_title, sample_text, "logistic_regression")
  cat(toJSON(res, pretty = TRUE, auto_unbox = TRUE), "\n")
} else {
  write_error_exit("Invalid usage. Expected: Rscript 10_prediction.R <input_json> <output_json>")
}
