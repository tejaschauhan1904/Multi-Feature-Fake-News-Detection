# ==============================================================================
# 01_check_environment.R
# Multi-Feature Fake News Detection Using R Language
# Purpose: Check R version, required R packages, directories, and datasets
# ==============================================================================

cat("========================================================\n")
cat(" [01] Checking System Environment and Dependencies\n")
cat("========================================================\n")

# Detect project root directory
get_project_root <- function() {
  candidates <- c(
    getwd(),
    file.path(getwd(), ".."),
    "d:/Tejas/College/Sem_5/R Language/R_PBL_project"
  )
  for (cand in candidates) {
    if (file.exists(file.path(cand, "dataset", "Fake.csv")) ||
        file.exists(file.path(cand, "dataset", "True.csv"))) {
      return(normalizePath(cand, winslash = "/"))
    }
  }
  return(normalizePath(getwd(), winslash = "/"))
}

PROJECT_ROOT <- get_project_root()
cat(sprintf("Project Root: %s\n", PROJECT_ROOT))

# Ensure required directories exist
required_dirs <- c(
  file.path(PROJECT_ROOT, "dataset"),
  file.path(PROJECT_ROOT, "R_Scripts"),
  file.path(PROJECT_ROOT, "Python_App"),
  file.path(PROJECT_ROOT, "models"),
  file.path(PROJECT_ROOT, "results"),
  file.path(PROJECT_ROOT, "results", "charts"),
  file.path(PROJECT_ROOT, "results", "reports"),
  file.path(PROJECT_ROOT, "logs"),
  file.path(PROJECT_ROOT, "docs")
)

for (d in required_dirs) {
  if (!dir.exists(d)) {
    dir.create(d, recursive = TRUE, showWarnings = FALSE)
    cat(sprintf("Created missing directory: %s\n", d))
  }
}

# Check R version
r_ver <- paste(R.version$major, R.version$minor, sep = ".")
cat(sprintf("R Version: %s (%s)\n", r_ver, R.version$version.string))

# Required R packages
required_packages <- c(
  "dplyr",
  "stringr",
  "tidytext",
  "text2vec",
  "tm",
  "caret",
  "randomForest",
  "e1071",
  "pROC",
  "glmnet",
  "Matrix",
  "jsonlite",
  "ggplot2"
)

missing_packages <- c()
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    missing_packages <- c(missing_packages, pkg)
  }
}

if (length(missing_packages) > 0) {
  cat("\n[WARNING] The following required packages are missing:\n")
  cat(paste(" -", missing_packages, collapse = "\n"), "\n\n")
  cat("Please install them using install.packages(c(...))\n")
  quit(status = 1)
} else {
  cat("[SUCCESS] All required R packages are installed and available.\n")
}

# Check dataset presence
fake_path <- file.path(PROJECT_ROOT, "dataset", "Fake.csv")
true_path <- file.path(PROJECT_ROOT, "dataset", "True.csv")

if (!file.exists(fake_path) || !file.exists(true_path)) {
  stop("[ERROR] Original dataset files (Fake.csv / True.csv) not found in dataset directory!")
}

cat(sprintf("[SUCCESS] Dataset files verified:\n  - Fake.csv (%0.2f MB)\n  - True.csv (%0.2f MB)\n",
            file.info(fake_path)$size / (1024 * 1024),
            file.info(true_path)$size / (1024 * 1024)))

# Log check
log_file <- file.path(PROJECT_ROOT, "logs", "application.log")
cat(sprintf("[%s] Environment check completed successfully.\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S")),
    file = log_file, append = TRUE)

cat("[01] Environment verification complete.\n\n")
