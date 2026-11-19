# ==============================================================================
# 03_data_preprocessing.R
# Multi-Feature Fake News Detection Using R Language
# Purpose: Clean text data, deduplicate, strip publisher leakage, create stratified split.
# ==============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(stringr)
  library(caret)
  library(jsonlite)
})

cat("========================================================\n")
cat(" [03] Data Preprocessing & Leakage Prevention\n")
cat("========================================================\n")

get_project_root <- function() {
  candidates <- c(getwd(), file.path(getwd(), ".."), "d:/Tejas/College/Sem_5/R Language/R_PBL_project")
  for (cand in candidates) {
    if (file.exists(file.path(cand, "dataset", ".raw_combined_temp.rds"))) {
      return(normalizePath(cand, winslash = "/"))
    }
  }
  return(normalizePath(getwd(), winslash = "/"))
}

PROJECT_ROOT <- get_project_root()
raw_rds_path <- file.path(PROJECT_ROOT, "dataset", ".raw_combined_temp.rds")

if (!file.exists(raw_rds_path)) {
  stop("[ERROR] Raw data cache not found. Please run 02_data_loading.R first.")
}

df <- readRDS(raw_rds_path)
initial_rows <- nrow(df)
cat(sprintf("Loaded raw articles: %d\n", initial_rows))

# Step 1: Normalize whitespace and trim
df$title <- str_squish(df$title)
df$text <- str_squish(df$text)

# Step 2: Filter out empty articles or whitespace-only articles
df <- df %>% filter(nchar(title) > 3 | nchar(text) > 10)
rows_after_empty_removal <- nrow(df)
cat(sprintf("Articles after removing empty/stub records: %d (removed %d)\n",
            rows_after_empty_removal, initial_rows - rows_after_empty_removal))

# Step 3: Remove exact duplicate titles and contents
# Removing duplicates avoids data leakage between train and test
df <- df %>% distinct(title, text, .keep_all = TRUE)
rows_after_exact_dedup <- nrow(df)
cat(sprintf("Articles after exact deduplication: %d (removed %d duplicates)\n",
            rows_after_exact_dedup, rows_after_empty_removal - rows_after_exact_dedup))

# Step 4: Address publisher-specific artifacts & leakage (e.g. "WASHINGTON (Reuters) - ")
# Real news in the ISOT dataset frequently starts with "<CITY> (Reuters) - " or "(Reuters) -"
clean_publisher_artifacts <- function(txt) {
  # Strip leading Reuters datelines: e.g. "WASHINGTON (Reuters) - " or "(Reuters) - "
  cleaned <- str_replace(txt, "(?i)^[A-Z\\s,/.-]+?\\s*\\(Reuters\\)\\s*[-—–:]\\s*", "")
  cleaned <- str_replace(cleaned, "(?i)^\\(Reuters\\)\\s*[-—–:]?\\s*", "")
  cleaned <- str_replace(cleaned, "(?i)^Reuters\\s*[-—–:]\\s*", "")
  return(str_squish(cleaned))
}

cat("Removing publisher datelines and format artifacts to prevent trivial model leakage...\n")
df$text_clean_source <- clean_publisher_artifacts(df$text)

# Re-check for empty texts after stripping dateline
df <- df %>% filter(nchar(text_clean_source) > 10 | nchar(title) > 5)

# Step 5: Construct combined text for NLP feature extraction and TF-IDF
# Keep original title and text intact for readability/linguistic calculations
df$combined_text <- paste(df$title, df$text_clean_source, sep = " ")

# Keep a clean ID for each record
df$id <- seq_len(nrow(df))

final_rows <- nrow(df)
fake_count <- sum(df$label == 0)
real_count <- sum(df$label == 1)

cat(sprintf("Final cleaned articles: %d (Fake: %d, Real: %d, Balance: %0.2f%% Real)\n",
            final_rows, fake_count, real_count, (real_count / final_rows) * 100))

# Step 6: Stratified Train-Test Split (80:20)
set.seed(42)
train_indices <- createDataPartition(df$label, p = 0.80, list = FALSE)

train_df <- df[train_indices, ]
test_df <- df[-train_indices, ]

cat(sprintf("Stratified Train set: %d articles (Fake: %d, Real: %d)\n",
            nrow(train_df), sum(train_df$label == 0), sum(train_df$label == 1)))
cat(sprintf("Stratified Test set:  %d articles (Fake: %d, Real: %d)\n",
            nrow(test_df), sum(test_df$label == 0), sum(test_df$label == 1)))

# Step 7: Save cleaned dataset summary and files
cleaned_csv_path <- file.path(PROJECT_ROOT, "dataset", "cleaned_news.csv")

# Save a lightweight version to CSV (title, text_clean_source, label, class_name)
cat("Saving cleaned_news.csv...\n")
write.csv(df %>% select(id, title, text = text_clean_source, subject, date, label, class_name),
          cleaned_csv_path, row.names = FALSE)
cat(sprintf("Saved: %s (%0.2f MB)\n", cleaned_csv_path, file.info(cleaned_csv_path)$size / (1024 * 1024)))

# Save train/test splits as RDS for downstream modular scripts
saveRDS(train_df, file.path(PROJECT_ROOT, "dataset", ".train_split_temp.rds"))
saveRDS(test_df, file.path(PROJECT_ROOT, "dataset", ".test_split_temp.rds"))

# Cleaning summary report
cleaning_summary <- list(
  initial_records = initial_rows,
  cleaned_records = final_rows,
  duplicates_removed = initial_rows - final_rows,
  fake_articles = fake_count,
  real_articles = real_count,
  train_split_size = nrow(train_df),
  test_split_size = nrow(test_df),
  train_ratio = 0.80,
  random_seed = 42,
  leakage_mitigation = "Publisher datelines removed, stratified split strictly isolated",
  timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S")
)

write_json(cleaning_summary, file.path(PROJECT_ROOT, "results", "reports", "dataset_cleaning_summary.json"),
           pretty = TRUE, auto_unbox = TRUE)

cat("[03] Data preprocessing complete.\n\n")
