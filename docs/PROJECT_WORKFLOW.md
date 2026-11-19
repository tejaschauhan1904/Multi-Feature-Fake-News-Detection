# Project Workflow: Multi-Feature Fake News Detection

This document provides a thorough technical breakdown of the complete end-to-end data and machine learning workflow.

---

## High-Level Workflow Pipeline

```
+---------------------------------------------------------------------------------------+
| 1. DATASET INGESTION                                                                  |
|    - Load Fake.csv (23,481 records) & True.csv (21,417 records)                       |
|    - Assign Ground Truth Labels: Fake = 0, Real = 1                                   |
+-------------------------------------------+-------------------------------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------+
| 2. DATA PREPROCESSING & DATA LEAKAGE MITIGATION                                       |
|    - Remove empty articles & whitespace normalization                                 |
|    - Exact & Near Deduplication: Eliminates 5,798 duplicates (Final: 39,100 records)  |
|    - Publisher Dateline Stripping: Purge 'WASHINGTON (Reuters) -' leakage             |
|    - Stratified Partitioning: 80% Train (31,280) : 20% Test (7,820) (Seed = 42)       |
+-------------------------------------------+-------------------------------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------+
| 3. MULTI-CATEGORY FEATURE ENGINEERING                                                 |
|    - Sentiment Analysis: Lexicon-based scoring using Bing Lexicon                     |
|    - Readability Metrics: Flesch Reading Ease, sentence & word counts                 |
|    - Linguistic Features: Uppercase ratio, exclamation & question marks               |
|    - Statistical Distribution: Lexical diversity (type-token ratio), word lengths     |
+-------------------------------------------+-------------------------------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------+
| 4. TF-IDF VECTORIZATION & MATRIX INTEGRATION                                          |
|    - Clean text (lowercase, stopword removal, URL/HTML stripping)                     |
|    - text2vec unigram & bigram vocabulary fitting on TRAIN ONLY (4,000 terms)        |
|    - Standardize 32 numerical features using TRAIN mean & standard deviation          |
|    - Column-bind into integrated sparse matrix: [TF-IDF (4000) | Scaled Num (32)]     |
+-------------------------------------------+-------------------------------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------+
| 5. MACHINE LEARNING MODEL TRAINING                                                    |
|    - Model 1: glmnet Binomial Logistic Regression with LASSO (alpha = 1, 5-fold CV)   |
|    - Model 2: randomForest (100 trees over 32 numerical + top 50 TF-IDF features)     |
|    - Model 3: e1071 Naive Bayes with Laplace smoothing = 1                            |
+-------------------------------------------+-------------------------------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------+
| 6. MODEL EVALUATION & COMPARISON                                                      |
|    - Evaluate all 3 models on held-out test data (unseen articles)                    |
|    - Accuracy, Precision, Recall, Specificity, F1-Score, ROC-AUC, Confusion Matrix   |
|    - Select best model (Logistic Regression LASSO: 99.53% Accuracy, 0.9953 F1)        |
+-------------------------------------------+-------------------------------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------+
| 7. MODEL ARTIFACT SERIALIZATION                                                       |
|    - Save models to models/*.rds                                                      |
|    - Save preprocessing parameters to models/preprocessing_objects.rds                |
|    - Save TF-IDF bundle to models/tfidf_vectorizer.rds                                |
+-------------------------------------------+-------------------------------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------+
| 8. PYTHON-R SUBPROCESS INTEGRATION                                                    |
|    - Python writes article text to temporary JSON                                     |
|    - Subprocess executes: Rscript R_Scripts/10_prediction.R in.json out.json          |
|    - R parses JSON, extracts features, computes probability, writes response JSON     |
+-------------------------------------------+-------------------------------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------+
| 9. PYTHON STREAMLIT DASHBOARD                                                         |
|    - Multi-page responsive UI: Dashboard, Detection, Features, Models, Dataset       |
|    - Live analytical rendering with interactive Plotly visual charts                  |
+---------------------------------------------------------------------------------------+
```

---

## Detailed Stage Analysis

### Stage 1: Data Ingestion (`02_data_loading.R`)
Reads raw CSV files `dataset/Fake.csv` and `dataset/True.csv`. Assigns binary label column `label` (0 = Fake, 1 = Real). Confirms presence of mandatory schema columns: `title`, `text`, `subject`, `date`. Computes raw counts and missing-value summaries.

### Stage 2: Preprocessing & Leakage Mitigation (`03_data_preprocessing.R`)
1. **Deduplication:** Many public fake news datasets contain duplicated news stories republished across multiple sites. Deduplication removes 5,798 duplicate records, ensuring that no training article also appears in the test partition.
2. **Publisher Artifact Purging:** Reuters wire stories consistently contained datelines such as `WASHINGTON (Reuters) -`. If left uncleaned, classifiers learn that the token `"Reuters"` denotes real news. A regular expression strips these tags prior to training.
3. **Stratification:** Splits 39,100 clean articles into 80% train and 20% test partitions using `caret::createDataPartition()` with seed 42, preserving exact class balance.

### Stage 3: Feature Engineering (`05_feature_extraction.R`)
Extracts 32 continuous features across four domains. Preserves the original raw text with capitalization and punctuation intact for linguistic and readability analysis.

### Stage 4: TF-IDF & Matrix Integration (`06_tfidf_vectorization.R`)
Constructs an unigram and bigram vocabulary on the training text using `text2vec`. Filters terms with document frequencies below 0.002 or above 0.50, and caps the vocabulary at 4,000 terms. Fits the TF-IDF transformer on the training set and transforms both train and test matrices. Scales the 32 numerical features via z-score normalization and binds them to the sparse TF-IDF matrix using `Matrix::cbind()`.

### Stage 5: Model Training (`07_model_training.R`)
Trains three distinct machine learning classifiers:
- **Logistic Regression with LASSO:** Uses `cv.glmnet` with L1 penalty to shrink uninformative coefficients to zero.
- **Random Forest:** Trains an ensemble of 100 decision trees over all 32 numerical features plus the top 50 highest-variance TF-IDF words.
- **Naive Bayes:** Trains a probabilistic classifier with Laplace smoothing = 1.

### Stage 6: Evaluation & Reporting (`08_model_evaluation.R`, `09_model_comparison.R`)
Calculates comprehensive performance metrics on the held-out test set: Accuracy, Precision, Recall, Specificity, F1-Score, and ROC-AUC. Generates confusion matrices and comparative reports.

### Stage 7: Visualization Generation (`11_generate_visualizations.R`)
Generates 7 high-resolution academic charts using `ggplot2` and saves them to `results/charts/`.

### Stage 8 & 9: Python-R Integration & Streamlit UI (`Python_App/`)
Streamlit provides a clean web UI where users can paste articles, select the backend model, trigger inference via `r_integration.py`, and inspect interactive Plotly charts and multi-feature breakdowns.
