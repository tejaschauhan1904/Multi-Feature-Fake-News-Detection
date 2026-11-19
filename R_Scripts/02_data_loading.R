# ==============================================================================
# 02_data_loading.R
# Multi-Feature Fake News Detection Using R Language
# Purpose: Load Fake.csv and True.csv, assign ground-truth labels, and merge.
# ==============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(jsonlite)
})

cat("========================================================\n")
cat(" [02] Loading Raw Datasets\n")
cat("========================================================\n")

get_project_root <- function() {
  candidates <- c(getwd(), file.path(getwd(), ".."), "d:/Tejas/College/Sem_5/R Language/R_PBL_project")
  for (cand in candidates) {
    if (file.exists(file.path(cand, "dataset", "Fake.csv"))) {
      return(normalizePath(cand, winslash = "/"))
    }
  }
  return(normalizePath(getwd(), winslash = "/"))
}

PROJECT_ROOT <- get_project_root()
fake_path <- file.path(PROJECT_ROOT, "dataset", "Fake.csv")
true_path <- file.path(PROJECT_ROOT, "dataset", "True.csv")

cat("Reading Fake.csv...\n")
df_fake <- read.csv(fake_path, stringsAsFactors = FALSE, encoding = "UTF-8")
cat(sprintf("Loaded Fake news records: %d\n", nrow(df_fake)))

cat("Reading True.csv...\n")
df_true <- read.csv(true_path, stringsAsFactors = FALSE, encoding = "UTF-8")
cat(sprintf("Loaded True news records: %d\n", nrow(df_true)))

# Verify expected columns
expected_cols <- c("title", "text", "subject", "date")
check_cols <- function(df, name) {
  missing_cols <- setdiff(expected_cols, colnames(df))
  if (length(missing_cols) > 0) {
    stop(sprintf("[ERROR] Dataset %s is missing expected columns: %s", name, paste(missing_cols, collapse = ", ")))
  }
}
check_cols(df_fake, "Fake.csv")
check_cols(df_true, "True.csv")

# Assign labels: Fake = 0, Real = 1
df_fake$label <- 0
df_fake$class_name <- "Fake"

df_true$label <- 1
df_true$class_name <- "Real"

# Combine both datasets
df_raw <- rbind(df_fake, df_true)
cat(sprintf("Combined raw dataset total rows: %d\n", nrow(df_raw)))

# Check missing values
missing_summary <- list(
  title_na = sum(is.na(df_raw$title) | df_raw$title == ""),
  text_na = sum(is.na(df_raw$text) | df_raw$text == ""),
  subject_na = sum(is.na(df_raw$subject) | df_raw$subject == ""),
  date_na = sum(is.na(df_raw$date) | df_raw$date == "")
)

cat("Missing or empty field summary before cleaning:\n")
print(missing_summary)

# Subject distribution
subject_dist <- as.list(table(df_raw$subject))

# Save dataset summary JSON for quick dashboard rendering
dataset_summary <- list(
  total_raw_articles = nrow(df_raw),
  raw_fake_count = nrow(df_fake),
  raw_real_count = nrow(df_true),
  columns = colnames(df_raw),
  missing_values = missing_summary,
  subject_distribution = subject_dist,
  created_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S")
)

summary_json_path <- file.path(PROJECT_ROOT, "results", "reports", "dataset_raw_summary.json")
write_json(dataset_summary, summary_json_path, pretty = TRUE, auto_unbox = TRUE)
cat(sprintf("Saved raw dataset summary to: %s\n", summary_json_path))

# Save raw combined dataframe into temp RDS for modular pipeline passing
raw_rds_path <- file.path(PROJECT_ROOT, "dataset", ".raw_combined_temp.rds")
saveRDS(df_raw, raw_rds_path)
cat(sprintf("Temporarily cached combined data to: %s\n", raw_rds_path))

cat("[02] Data loading complete.\n\n")
