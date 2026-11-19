# ==============================================================================
# main.R
# Multi-Feature Fake News Detection Using R Language
# Master Execution Pipeline Orchestrator
# ==============================================================================

cat("\n")
cat("================================================================================\n")
cat("      MULTI-FEATURE FAKE NEWS DETECTION USING R LANGUAGE (ACADEMIC PBL)         \n")
cat("================================================================================\n")
cat(sprintf("Execution started at: %s\n\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S")))

pipeline_start_time <- Sys.time()

# Determine project root safely
get_project_root <- function() {
  candidates <- c(
    getwd(),
    file.path(getwd(), ".."),
    "d:/Tejas/College/Sem_5/R Language/R_PBL_project"
  )
  for (cand in candidates) {
    if (file.exists(file.path(cand, "dataset", "Fake.csv")) ||
        file.exists(file.path(cand, "R_Scripts", "01_check_environment.R"))) {
      return(normalizePath(cand, winslash = "/"))
    }
  }
  return(normalizePath(getwd(), winslash = "/"))
}

PROJECT_ROOT <- get_project_root()
setwd(PROJECT_ROOT)
cat(sprintf("Active Workspace Root: %s\n\n", PROJECT_ROOT))

# Define pipeline stages
stages <- list(
  list(step = "01", script = "01_check_environment.R", desc = "System Environment & Dependency Verification"),
  list(step = "02", script = "02_data_loading.R", desc = "Raw Dataset Loading & Integrity Validation"),
  list(step = "03", script = "03_data_preprocessing.R", desc = "Data Cleaning, Deduplication & Stratified Splitting"),
  list(step = "04", script = "04_text_preprocessing.R", desc = "NLP Text Normalization & Tokenization Setup"),
  list(step = "05", script = "05_feature_extraction.R", desc = "Multi-Category Feature Engineering"),
  list(step = "06", script = "06_tfidf_vectorization.R", desc = "TF-IDF Vectorization & Matrix Integration"),
  list(step = "07", script = "07_model_training.R", desc = "Machine Learning Model Training (LASSO, RF, NB)"),
  list(step = "08", script = "08_model_evaluation.R", desc = "Rigorous Held-out Test Evaluation"),
  list(step = "09", script = "09_model_comparison.R", desc = "Model Comparison & Best Model Selection"),
  list(step = "11", script = "11_generate_visualizations.R", desc = "Academic Chart & Heatmap Generation")
)

execute_stage <- function(item) {
  cat("--------------------------------------------------------------------------------\n")
  cat(sprintf("[STAGE %s] %s\n", item$step, item$desc))
  cat(sprintf("Running: R_Scripts/%s\n", item$script))
  cat("--------------------------------------------------------------------------------\n")
  
  script_path <- file.path(PROJECT_ROOT, "R_Scripts", item$script)
  if (!file.exists(script_path)) {
    stop(sprintf("Fatal error: Script '%s' not found!", script_path))
  }
  
  stage_start <- Sys.time()
  
  # Run script in a clean child environment
  tryCatch({
    source(script_path, local = new.env(parent = globalenv()))
    stage_elapsed <- round(as.numeric(difftime(Sys.time(), stage_start, units = "secs")), 2)
    cat(sprintf("[STAGE %s SUCCESS] Completed in %s seconds.\n\n", item$step, stage_elapsed))
  }, error = function(e) {
    cat(sprintf("\n[PIPELINE ERROR] Stage %s failed with message:\n%s\n\n", item$step, e$message))
    quit(status = 1)
  })
}

# Run all stages sequentially
for (st in stages) {
  execute_stage(st)
}

total_duration <- round(as.numeric(difftime(Sys.time(), pipeline_start_time, units = "mins")), 2)

cat("================================================================================\n")
cat("                END-TO-END R PIPELINE COMPLETED SUCCESSFULLY                    \n")
cat("================================================================================\n")
cat(sprintf("Total Pipeline Runtime: %0.2f minutes\n", total_duration))
cat(sprintf("Models saved to:        %s/models/\n", PROJECT_ROOT))
cat(sprintf("Results saved to:       %s/results/\n", PROJECT_ROOT))
cat(sprintf("Charts saved to:        %s/results/charts/\n", PROJECT_ROOT))
cat("\nYou can now launch the Python Streamlit Frontend:\n")
cat("  streamlit run Python_App/app.py\n")
cat("Or double-click run_app.bat\n\n")
