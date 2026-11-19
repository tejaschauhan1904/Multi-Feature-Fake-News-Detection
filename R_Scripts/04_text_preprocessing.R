# ==============================================================================
# 04_text_preprocessing.R
# Multi-Feature Fake News Detection Using R Language
# Purpose: Modular, reusable text cleaning, stopword removal, and tokenization.
# ==============================================================================

suppressPackageStartupMessages({
  library(stringr)
  library(tm)
})

# Standard English Stopwords list
get_stopwords <- function() {
  sw <- tryCatch({
    tm::stopwords("en")
  }, error = function(e) {
    c("i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
      "he", "him", "his", "she", "her", "it", "its", "they", "them", "their",
      "what", "which", "who", "whom", "this", "that", "these", "those", "am",
      "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
      "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but",
      "if", "or", "because", "as", "until", "while", "of", "at", "by", "for",
      "with", "about", "against", "between", "into", "through", "during", "before",
      "after", "above", "below", "to", "from", "up", "down", "in", "out", "on",
      "off", "over", "under", "again", "further", "then", "once", "here", "there",
      "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
      "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same",
      "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should")
  })
  return(unique(tolower(sw)))
}

STOPWORDS_EN <- get_stopwords()

#' Clean text for NLP and TF-IDF
#' Performs: Lowercase conversion, HTML removal, URL removal, punctuation cleaning,
#' number handling, whitespace normalization, and optional stopword removal.
clean_text_for_nlp <- function(text_vector, remove_stopwords = TRUE, remove_numbers = TRUE) {
  if (is.null(text_vector) || length(text_vector) == 0) return(character(0))
  
  # Ensure character type
  txt <- as.character(text_vector)
  txt[is.na(txt)] <- ""
  
  # 1. Lowercase conversion
  txt <- tolower(txt)
  
  # 2. HTML tag removal
  txt <- str_replace_all(txt, "<[^>]+>", " ")
  
  # 3. URL handling (strip http/https/www links)
  txt <- str_replace_all(txt, "https?://\\S+|www\\.\\S+", " ")
  
  # 4. Remove email addresses
  txt <- str_replace_all(txt, "\\S+@\\S+", " ")
  
  # 5. Number handling
  if (remove_numbers) {
    txt <- str_replace_all(txt, "\\b\\d+\\b", " ")
    txt <- str_replace_all(txt, "\\d+", " ")
  }
  
  # 6. Punctuation handling (keep standard letters and spaces)
  txt <- str_replace_all(txt, "[^a-z\\s]", " ")
  
  # 7. Whitespace normalization
  txt <- str_squish(txt)
  
  # 8. Stopword removal (vectorized)
  if (remove_stopwords) {
    # Fast regex stopword boundary removal
    # Or word splitting
    txt <- sapply(txt, function(s) {
      if (nchar(s) == 0) return("")
      words <- unlist(str_split(s, "\\s+"))
      words <- words[!words %in% STOPWORDS_EN & nchar(words) > 1]
      paste(words, collapse = " ")
    }, USE.NAMES = FALSE)
  }
  
  return(unname(txt))
}

#' Tokenize into Unigrams
tokenize_unigrams <- function(cleaned_text) {
  if (nchar(str_squish(cleaned_text)) == 0) return(character(0))
  words <- unlist(str_split(cleaned_text, "\\s+"))
  words[nchar(words) > 1]
}

#' Tokenize into Bigrams
tokenize_bigrams <- function(cleaned_text) {
  unigrams <- tokenize_unigrams(cleaned_text)
  if (length(unigrams) < 2) return(character(0))
  bigrams <- paste(unigrams[-length(unigrams)], unigrams[-1], sep = " ")
  return(bigrams)
}

# Standalone execution verification
if (!exists("SOURCED_BY_PIPELINE")) {
  cat("========================================================\n")
  cat(" [04] Text Preprocessing Module Verification\n")
  cat("========================================================\n")
  
  sample_input <- "BREAKING NEWS! Government announces a new policy with $100M budget. Visit https://news.example.com"
  cat(sprintf("Original text:  '%s'\n", sample_input))
  
  cleaned <- clean_text_for_nlp(sample_input, remove_stopwords = TRUE)
  cat(sprintf("Cleaned text:   '%s'\n", cleaned))
  
  unigrams <- tokenize_unigrams(cleaned)
  cat(sprintf("Unigrams:       %s\n", paste(unigrams, collapse = ", ")))
  
  bigrams <- tokenize_bigrams(cleaned)
  cat(sprintf("Bigrams:        %s\n", paste(bigrams, collapse = " | ")))
  
  cat("[04] Text preprocessing module ready.\n\n")
}
