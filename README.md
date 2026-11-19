# Multi-Feature Fake News Detection Using R Language

> **College Semester 5 Project-Based Learning (PBL) Project**  
> **Backend & Machine Learning:** R Programming Language (v4.6.1)  
> **Frontend & User Interface:** Python Streamlit (v1.58.0)  
> **Inter-Process Integration:** Native Subprocess Execution with File-Based JSON Exchange  

---

## 📌 Executive Summary

**Multi-Feature Fake News Detection** is an academic machine learning system engineered to classify news articles as **REAL NEWS** or **FAKE NEWS** by synthesizing multiple complementary characteristics rather than relying solely on raw vocabulary.

The project implements a complete, modular **R backend pipeline** for data ingestion, deduplication, publisher dateline leakage mitigation, stratified sampling, multi-category feature engineering (sentiment, readability, linguistic markers, and statistical distributions), sparse TF-IDF n-gram vectorization, cross-validated machine learning model training, and evaluation on held-out test data.

The frontend is implemented in **Python Streamlit**, providing an interactive web dashboard with real-time inference, multi-category feature inspection, comparative performance charts, and dataset analytics—communicating with R via isolated subprocess calls.

---

## 🚀 Key Features

1. **R-Centric Machine Learning Backend:**
   - 100% of data cleaning, feature engineering, vectorization, training, and prediction runs in native R.
   - 3 Supervised ML Architectures: **Logistic Regression with LASSO** (`glmnet`), **Random Forest** (`randomForest`), and **Naive Bayes** (`e1071`).
2. **Multi-Category Feature Extraction (32 Dimensions):**
   - **Lexicon Sentiment:** Positive/negative word counts, net score, and polarity balance using Bing Lexicon.
   - **Readability Statistics:** Flesch Reading Ease score, words per sentence, and characters per word.
   - **Linguistic Markers:** Uppercase ratio, exclamation mark count, question mark count, punctuation density, URLs, and numeric tokens.
   - **Statistical Distributions:** Lexical diversity (type-token ratio), word length distributions, and paragraph metrics.
3. **High-Dimensional TF-IDF Representation:**
   - Unigram and bigram vocabulary extraction using `text2vec`, pruned to 4,000 terms and stored as sparse `dgCMatrix` representations.
4. **Data Leakage Mitigation:**
   - Removal of 5,798 exact duplicate stories to prevent test-set contamination.
   - Regex-based removal of publisher datelines (e.g., `WASHINGTON (Reuters) -`) to eliminate shortcut learning.
   - Stratified 80:20 train-test partition with fixed random seed (42).
5. **Python Streamlit Interactive UI:**
   - 6 Multi-page tabs: Dashboard, Fake News Detection, Feature Analysis, Model Performance, Dataset Analysis, and About Project.
   - Subprocess-driven live prediction with model selection and instant demo article loaders.
   - Academic disclaimer stating that classifications reflect learned statistical patterns rather than independent factual verification.

---

## 🛠️ Technology Stack

| Layer | Technology | Key Libraries / Modules |
| :--- | :--- | :--- |
| **Backend & Machine Learning** | R Language (v4.6.1) | `glmnet`, `randomForest`, `e1071`, `text2vec`, `tm`, `tidytext`, `caret`, `pROC`, `Matrix`, `ggplot2`, `jsonlite` |
| **Frontend & UI** | Python (v3.14) | `streamlit`, `pandas`, `plotly`, `matplotlib`, `numpy`, `subprocess` |
| **Data Storage & Models** | Serialized RDS & CSV | `models/*.rds`, `results/*.csv`, `dataset/*.csv` |

---

## 📂 Project Folder Structure

```
Multi_Feature_Fake_News_Detection/
│
├── dataset/
│   ├── Fake.csv                      # Original raw Fake news dataset (23,481 records)
│   ├── True.csv                      # Original raw True news dataset (21,417 records)
│   ├── cleaned_news.csv              # Deduplicated & cleaned corpus (39,100 records)
│   └── extracted_features.csv        # Sample extracted 32-feature matrix
│
├── R_Scripts/
│   ├── 01_check_environment.R        # Environment & dependency verification
│   ├── 02_data_loading.R             # Dataset loading & schema validation
│   ├── 03_data_preprocessing.R       # Deduplication, artifact removal & stratified split
│   ├── 04_text_preprocessing.R       # Reusable text cleaning, tokenization & stopwords
│   ├── 05_feature_extraction.R       # Multi-category feature extraction (32 features)
│   ├── 06_tfidf_vectorization.R      # text2vec TF-IDF & sparse matrix integration
│   ├── 07_model_training.R           # Model training (LASSO, Random Forest, Naive Bayes)
│   ├── 08_model_evaluation.R         # Held-out test set evaluation & confusion matrices
│   ├── 09_model_comparison.R         # Comparative ranking & summary report generation
│   ├── 10_prediction.R               # JSON-based CLI prediction interface for Python
│   └── 11_generate_visualizations.R  # High-resolution ggplot2 academic charts
│
├── Python_App/
│   ├── app.py                        # Streamlit main application entrypoint
│   ├── config.py                     # Path configuration & model definitions
│   ├── r_integration.py              # Subprocess bridge & JSON exchange layer
│   ├── requirements.txt              # Python dependencies
│   │
│   ├── components/
│   │   ├── sidebar.py                # Navigation & system health monitor
│   │   ├── dashboard.py              # Executive KPI dashboard & charts
│   │   ├── news_detection.py         # Live inference & demo loader
│   │   ├── feature_analysis.py       # 5-domain feature inspection tabs
│   │   ├── model_performance.py      # Metrics tables, confusion heatmaps & ROC
│   │   ├── dataset_analysis.py       # Corpus insights & leakage mitigation audit
│   │   └── about.py                  # Problem statement, architecture & viva info
│   │
│   └── assets/
│       └── style.css                 # Dark navy academic theme stylesheet
│
├── models/
│   ├── logistic_regression_model.rds # Trained cv.glmnet LASSO model
│   ├── random_forest_model.rds       # Trained 100-tree randomForest model
│   ├── naive_bayes_model.rds         # Trained e1071 Naive Bayes model
│   ├── tfidf_vectorizer.rds          # Fitted text2vec vectorizer & vocabulary
│   ├── preprocessing_objects.rds     # Z-score scaling parameters & column indices
│   └── model_metadata.rds            # Model hyperparameters & training metadata
│
├── results/
│   ├── evaluation_metrics.csv        # Test evaluation metrics table
│   ├── model_comparison.csv          # Ranked model comparison table
│   ├── confusion_matrix.csv          # Confusion matrix counts
│   ├── charts/                       # High-resolution ggplot2 visual charts
│   └── reports/                      # Summary JSONs and text reports
│
├── logs/
│   └── application.log               # Execution logs and subprocess traces
│
├── docs/                             # Complete academic documentation suite
│   ├── README.md
│   ├── MODELS_EXPLAINED_SIMPLE.md    # ⭐ Simple, student-friendly guide for College PBL Viva
│   ├── PROJECT_OVERVIEW.md
│   ├── INSTALLATION_GUIDE.md
│   ├── EXECUTION_GUIDE.md
│   ├── PROJECT_WORKFLOW.md
│   ├── SYSTEM_ARCHITECTURE.md
│   ├── TECH_STACK.md
│   ├── DATASET_DOCUMENTATION.md
│   ├── DATA_PREPROCESSING.md
│   ├── FEATURE_ENGINEERING.md
│   ├── MACHINE_LEARNING_MODELS.md
│   ├── MODEL_EVALUATION.md
│   ├── PYTHON_R_INTEGRATION.md
│   ├── USER_GUIDE.md
│   ├── TROUBLESHOOTING.md
│   ├── VIVA_QUESTIONS.md             # 35 comprehensive viva questions & answers
│   └── FUTURE_SCOPE.md
│
├── main.R                            # Master R pipeline orchestrator
├── run_app.bat                       # One-click Windows batch launcher
├── .gitignore
└── requirements.txt
```

---

## ⚡ Quick Start Guide (Windows)

### 1. Prerequisites
- **R (v4.2+)** installed (default: `C:\Program Files\R\R-4.6.1\bin\Rscript.exe`).
- **Python (v3.10+)** installed with pip.

### 2. One-Click Launch
Double click **`run_app.bat`** in the project root. The launcher will automatically verify dependencies, offer to train models if missing, and start the Streamlit server.

### 3. Manual Step-by-Step Execution
```powershell
# Step A: Install Python dependencies
python -m pip install -r requirements.txt

# Step B: Execute the full R training pipeline
& "C:\Program Files\R\R-4.6.1\bin\Rscript.exe" main.R

# Step C: Launch the Python Streamlit web dashboard
streamlit run Python_App/app.py
```
Open your browser at `http://localhost:8501`.

---

## 📊 Empirical Performance Summary

Evaluated on **1,500 held-out test articles** (750 Fake, 750 Real):

| Model Architecture | Accuracy | Precision | Recall | Specificity | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (LASSO) [Best]** | **99.53%** | **99.47%** | **99.60%** | **99.47%** | **0.9953** | **0.9983** |
| **Random Forest Classifier** | **99.33%** | **99.07%** | **99.60%** | **99.07%** | **0.9934** | **0.9999** |
| **Naive Bayes Classifier** | **89.60%** | **93.93%** | **84.67%** | **94.53%** | **0.8906** | **0.9459** |

---

## ⚖️ Project Limitations & Ethical Notice

1. **Statistical Correlation vs Factual Reality:** The machine learning models evaluate stylistic, lexical, and structural correlations learned from labelled datasets. They cannot query live knowledge bases to independently verify real-world facts.
2. **Domain Focus:** The models are trained on English-language political and world news articles (ISOT dataset). Application to non-English languages or specialized scientific prose may exhibit reduced predictive performance.
