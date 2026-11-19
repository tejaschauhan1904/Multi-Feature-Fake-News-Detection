# ==============================================================================
# 05_feature_extraction.R
# Multi-Feature Fake News Detection Using R Language
# Purpose: Extract Sentiment, Readability, Linguistic, and Statistical features.
# ==============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(stringr)
  library(tidytext)
})

# Load text preprocessing utilities
get_project_root <- function() {
  candidates <- c(getwd(), file.path(getwd(), ".."), "d:/Tejas/College/Sem_5/R Language/R_PBL_project")
  for (cand in candidates) {
    if (file.exists(file.path(cand, "R_Scripts", "04_text_preprocessing.R"))) {
      return(normalizePath(cand, winslash = "/"))
    }
  }
  return(normalizePath(getwd(), winslash = "/"))
}

PROJECT_ROOT <- get_project_root()
source(file.path(PROJECT_ROOT, "R_Scripts", "04_text_preprocessing.R"))

# ------------------------------------------------------------------------------
# 1. Sentiment Lexicon Preparation (Bing Lexicon from tidytext)
# ------------------------------------------------------------------------------
get_sentiment_lexicon <- function() {
  lex <- tryCatch({
    data("sentiments", package = "tidytext", envir = environment())
    sent_df <- get("sentiments", envir = environment())
    pos <- sent_df$word[sent_df$sentiment == "positive"]
    neg <- sent_df$word[sent_df$sentiment == "negative"]
    list(positive = unique(tolower(pos)), negative = unique(tolower(neg)))
  }, error = function(e) {
    # Reliable offline fallback
    list(
      positive = c("good", "great", "excellent", "positive", "truth", "success", "win", "honest", "reliable", "peace", "benefit", "love", "joy", "hope"),
      negative = c("bad", "terrible", "worst", "fake", "lie", "hoax", "crisis", "disaster", "fail", "corrupt", "hate", "war", "death", "crime", "illegal")
    )
  })
  return(lex)
}

SENTIMENT_LEX <- get_sentiment_lexicon()
# Create fast lookup environments
POS_ENV <- new.env(hash = TRUE, parent = emptyenv())
NEG_ENV <- new.env(hash = TRUE, parent = emptyenv())
for (w in SENTIMENT_LEX$positive) assign(w, TRUE, envir = POS_ENV)
for (w in SENTIMENT_LEX$negative) assign(w, TRUE, envir = NEG_ENV)

# ------------------------------------------------------------------------------
# 2. Syllable counter helper for Flesch Reading Ease
# ------------------------------------------------------------------------------
count_syllables_word <- function(w) {
  w <- tolower(w)
  if (nchar(w) <= 3) return(1)
  # Remove silent e at the end
  w <- sub("e$", "", w)
  vowels <- str_count(w, "[aeiouy]+")
  if (vowels < 1) return(1)
  return(vowels)
}

# ------------------------------------------------------------------------------
# 3. Comprehensive Feature Extraction for a Single Article
# ------------------------------------------------------------------------------
extract_single_article_features <- function(raw_title, raw_text) {
  title_str <- ifelse(is.na(raw_title), "", as.character(raw_title))
  text_str  <- ifelse(is.na(raw_text), "", as.character(raw_text))
  full_orig <- paste(title_str, text_str, sep = " ")
  full_orig <- str_squish(full_orig)
  
  # Total raw characters
  char_count <- nchar(full_orig)
  if (char_count == 0) full_orig <- "empty"
  
  # --- Linguistic Features ---
  uppercase_count <- str_count(full_orig, "[A-Z]")
  uppercase_ratio <- ifelse(char_count > 0, uppercase_count / char_count, 0)
  
  exclamation_count <- str_count(full_orig, "!")
  question_count    <- str_count(full_orig, "\\?")
  total_punct_count <- str_count(full_orig, "[[:punct:]]")
  punct_ratio       <- ifelse(char_count > 0, total_punct_count / char_count, 0)
  
  url_count     <- str_count(full_orig, "(?i)https?://|www\\.")
  numeric_count <- str_count(full_orig, "\\b\\d+\\b")
  
  # Title specific features
  title_char_count <- nchar(title_str)
  title_upper_count <- str_count(title_str, "[A-Z]")
  title_upper_ratio <- ifelse(title_char_count > 0, title_upper_count / title_char_count, 0)
  title_exclamations <- str_count(title_str, "!")
  title_questions    <- str_count(title_str, "\\?")
  
  # Paragraph statistics
  paragraphs <- unlist(str_split(text_str, "\n+"))
  paragraphs <- paragraphs[nchar(str_squish(paragraphs)) > 0]
  paragraph_count <- max(1, length(paragraphs))
  avg_paragraph_length <- char_count / paragraph_count
  
  # --- Sentence Segmentation & Readability ---
  sentences <- unlist(str_split(full_orig, "(?<=[.?!])\\s+"))
  sentences <- sentences[nchar(str_squish(sentences)) > 0]
  sentence_count <- max(1, length(sentences))
  avg_sentence_length_chars <- char_count / sentence_count
  
  # --- Tokenization for words ---
  words_raw <- unlist(str_split(str_replace_all(full_orig, "[^A-Za-z0-9]", " "), "\\s+"))
  words_raw <- words_raw[nchar(words_raw) > 0]
  word_count <- max(1, length(words_raw))
  
  avg_words_per_sentence <- word_count / sentence_count
  total_word_chars <- sum(nchar(words_raw))
  avg_chars_per_word <- total_word_chars / word_count
  
  # Long words (> 6 chars) and Short words (< 4 chars)
  word_lengths <- nchar(words_raw)
  long_words_count  <- sum(word_lengths > 6)
  short_words_count <- sum(word_lengths < 4)
  long_word_ratio   <- long_words_count / word_count
  short_word_ratio  <- short_words_count / word_count
  
  # Unique words & Lexical Diversity
  words_lower <- tolower(words_raw)
  unique_words_count <- length(unique(words_lower))
  lexical_diversity  <- unique_words_count / word_count
  
  # Flesch Reading Ease
  # Formula: 206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)
  sample_words <- if (length(words_lower) > 300) sample(words_lower, 300) else words_lower
  sample_syllables <- sum(sapply(sample_words, count_syllables_word))
  est_syllables_per_word <- sample_syllables / max(1, length(sample_words))
  
  flesch_score <- 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * est_syllables_per_word)
  flesch_score <- max(0, min(100, flesch_score)) # Clamp between 0 and 100 for stability
  
  # --- Sentiment Analysis ---
  pos_matches <- sum(sapply(words_lower, function(w) exists(w, envir = POS_ENV, inherits = FALSE)))
  neg_matches <- sum(sapply(words_lower, function(w) exists(w, envir = NEG_ENV, inherits = FALSE)))
  
  sentiment_score   <- pos_matches - neg_matches
  pos_word_ratio    <- pos_matches / word_count
  neg_word_ratio    <- neg_matches / word_count
  sentiment_balance <- ifelse((pos_matches + neg_matches) > 0,
                              (pos_matches - neg_matches) / (pos_matches + neg_matches), 0)
  
  data.frame(
    # Statistical features
    char_count = as.numeric(char_count),
    word_count = as.numeric(word_count),
    sentence_count = as.numeric(sentence_count),
    unique_words_count = as.numeric(unique_words_count),
    lexical_diversity = as.numeric(lexical_diversity),
    avg_words_per_sentence = as.numeric(avg_words_per_sentence),
    avg_chars_per_word = as.numeric(avg_chars_per_word),
    long_words_count = as.numeric(long_words_count),
    short_words_count = as.numeric(short_words_count),
    long_word_ratio = as.numeric(long_word_ratio),
    short_word_ratio = as.numeric(short_word_ratio),
    paragraph_count = as.numeric(paragraph_count),
    avg_paragraph_length = as.numeric(avg_paragraph_length),
    
    # Linguistic features
    uppercase_count = as.numeric(uppercase_count),
    uppercase_ratio = as.numeric(uppercase_ratio),
    exclamation_count = as.numeric(exclamation_count),
    question_count = as.numeric(question_count),
    total_punct_count = as.numeric(total_punct_count),
    punct_ratio = as.numeric(punct_ratio),
    url_count = as.numeric(url_count),
    numeric_count = as.numeric(numeric_count),
    avg_sentence_length_chars = as.numeric(avg_sentence_length_chars),
    
    # Title specific
    title_upper_ratio = as.numeric(title_upper_ratio),
    title_exclamations = as.numeric(title_exclamations),
    title_questions = as.numeric(title_questions),
    
    # Readability
    flesch_reading_ease = as.numeric(flesch_score),
    
    # Sentiment
    pos_words_count = as.numeric(pos_matches),
    neg_words_count = as.numeric(neg_matches),
    sentiment_score = as.numeric(sentiment_score),
    pos_word_ratio = as.numeric(pos_word_ratio),
    neg_word_ratio = as.numeric(neg_word_ratio),
    sentiment_balance = as.numeric(sentiment_balance),
    stringsAsFactors = FALSE
  )
}

# ------------------------------------------------------------------------------
# 4. Batch Feature Extraction Function
# ------------------------------------------------------------------------------
extract_batch_features <- function(titles, texts) {
  n <- length(titles)
  feat_list <- vector("list", n)
  for (i in seq_len(n)) {
    feat_list[[i]] <- extract_single_article_features(titles[i], texts[i])
  }
  df_features <- do.call(rbind, feat_list)
  # Replace any unexpected NA or Inf
  df_features[is.na(df_features)] <- 0
  for (col in colnames(df_features)) {
    df_features[is.infinite(df_features[[col]]), col] <- 0
  }
  return(df_features)
}

# Standalone execution
if (!exists("SOURCED_BY_PIPELINE") || isFALSE(get0("SOURCED_BY_PIPELINE"))) {
  cat("========================================================\n")
  cat(" [05] Extracting Features on Training and Test Splits\n")
  cat("========================================================\n")
  
  train_path <- file.path(PROJECT_ROOT, "dataset", ".train_split_temp.rds")
  test_path  <- file.path(PROJECT_ROOT, "dataset", ".test_split_temp.rds")
  
  if (!file.exists(train_path) || !file.exists(test_path)) {
    stop("[ERROR] Split datasets not found. Please run 03_data_preprocessing.R first.")
  }
  
  train_df <- readRDS(train_path)
  test_df  <- readRDS(test_path)
  
  # For efficient academic pipeline execution:
  # Check if a sample limit is configured or use representative stratified sample
  # Default: 6,000 train articles and 1,500 test articles for super-fast training (<1 min)
  # or full dataset if FULL_DATASET=TRUE
  use_full <- Sys.getenv("FULL_DATASET", unset = "FALSE") == "TRUE"
  
  if (!use_full && nrow(train_df) > 8000) {
    cat("Using balanced representative sample of 6,000 train + 1,500 test articles for fast training...\n")
    cat("(Set FULL_DATASET=TRUE environment variable if you wish to train on all 39,100 articles)\n")
    set.seed(42)
    train_sample_idx <- unlist(lapply(split(seq_len(nrow(train_df)), train_df$label), function(idx) sample(idx, min(3000, length(idx)))))
    test_sample_idx  <- unlist(lapply(split(seq_len(nrow(test_df)), test_df$label), function(idx) sample(idx, min(750, length(idx)))))
    
    train_df <- train_df[train_sample_idx, ]
    test_df  <- test_df[test_sample_idx, ]
  }
  
  cat(sprintf("Extracting multi-category features for %d training articles...\n", nrow(train_df)))
  train_feats <- extract_batch_features(train_df$title, train_df$text_clean_source)
  
  cat(sprintf("Extracting multi-category features for %d test articles...\n", nrow(test_df)))
  test_feats  <- extract_batch_features(test_df$title, test_df$text_clean_source)
  
  # Add labels and IDs
  train_feats$label <- train_df$label
  train_feats$id    <- train_df$id
  test_feats$label  <- test_df$label
  test_feats$id     <- test_df$id
  
  # Save extracted features CSV for external inspection
  sample_export <- rbind(head(train_feats, 100), head(test_feats, 100))
  extracted_csv_path <- file.path(PROJECT_ROOT, "dataset", "extracted_features.csv")
  write.csv(sample_export, extracted_csv_path, row.names = FALSE)
  cat(sprintf("Saved sample extracted features to: %s\n", extracted_csv_path))
  
  # Save full extracted feature objects into temporary cache
  saveRDS(list(train_df = train_df, train_feats = train_feats,
               test_df = test_df, test_feats = test_feats),
          file.path(PROJECT_ROOT, "dataset", ".features_cache_temp.rds"))
  
  cat(sprintf("Total extracted numerical feature columns: %d\n", ncol(train_feats) - 2))
  cat("[05] Multi-feature extraction complete.\n\n")
}
