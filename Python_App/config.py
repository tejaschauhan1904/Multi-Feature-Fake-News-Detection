"""
config.py
Multi-Feature Fake News Detection Using R Language
Configuration and Path Resolution Module
"""

import os
import sys
import shutil
from pathlib import Path

# Resolve Project Root safely
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

# Core Directories
DATASET_DIR = PROJECT_ROOT / "dataset"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
CHARTS_DIR = RESULTS_DIR / "charts"
REPORTS_DIR = RESULTS_DIR / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"
R_SCRIPTS_DIR = PROJECT_ROOT / "R_Scripts"
ASSETS_DIR = CURRENT_DIR / "assets"
R_LIBS_DIR = PROJECT_ROOT / "r_libs"

# R Scripts
R_PREDICTION_SCRIPT = R_SCRIPTS_DIR / "10_prediction.R"
R_MAIN_SCRIPT = PROJECT_ROOT / "main.R"

# Model Artifacts
LOGISTIC_MODEL_PATH = MODELS_DIR / "logistic_regression_model.rds"
RANDOM_FOREST_MODEL_PATH = MODELS_DIR / "random_forest_model.rds"
NAIVE_BAYES_MODEL_PATH = MODELS_DIR / "naive_bayes_model.rds"
TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.rds"
PREPROCESSING_OBJECTS_PATH = MODELS_DIR / "preprocessing_objects.rds"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.rds"

# Results & Reports
EVALUATION_METRICS_PATH = RESULTS_DIR / "evaluation_metrics.csv"
MODEL_COMPARISON_PATH = RESULTS_DIR / "model_comparison.csv"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.csv"
CLEANING_SUMMARY_JSON = REPORTS_DIR / "dataset_cleaning_summary.json"
MODEL_COMPARISON_JSON = REPORTS_DIR / "model_comparison_summary.json"


def find_rscript_executable() -> str:
    """
    Locates the Rscript executable on Windows or POSIX systems.
    Supports environment variable override, PATH search, and standard Windows directories.
    """
    # 1. Environment variable override
    env_rscript = os.environ.get("RSCRIPT_PATH")
    if env_rscript and os.path.isfile(env_rscript):
        return env_rscript

    # 2. Check system PATH
    which_rscript = shutil.which("Rscript") or shutil.which("Rscript.exe")
    if which_rscript:
        return which_rscript

    # 3. Common Windows R installation directories
    candidate_patterns = [
        r"C:\Program Files\R\R-4.6.1\bin\Rscript.exe",
        r"C:\Program Files\R\R-4.6.1\bin\x64\Rscript.exe",
        r"C:\Program Files\R\R-4.*\bin\Rscript.exe",
        r"C:\Program Files\R\R-4.*\bin\x64\Rscript.exe",
        r"C:\Program Files (x86)\R\R-*\bin\Rscript.exe",
    ]

    import glob
    for pat in candidate_patterns:
        matches = glob.glob(pat)
        if matches:
            # Sort descending to pick newest version
            matches.sort(reverse=True)
            for m in matches:
                if os.path.isfile(m):
                    return m

    # Fallback to plain name
    return "Rscript"


RSCRIPT_PATH = find_rscript_executable()

# Supported Machine Learning Models in UI
AVAILABLE_MODELS = {
    "Logistic Regression (LASSO) [Recommended]": "logistic_regression",
    "Random Forest Classifier": "random_forest",
    "Naive Bayes Classifier": "naive_bayes",
}
