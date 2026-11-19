# Technology Stack: Multi-Feature Fake News Detection

This document provides a comprehensive justification of every software framework, programming language, and package utilized in the project.

---

## 1. Backend & Machine Learning Technologies (R Language)

| Technology / Library | Version | Category | Justification & Role in Project |
| :--- | :--- | :--- | :--- |
| **R Language** | `4.6.1` | Core Environment | Statistical computing environment chosen as the primary backend for rigorous data preprocessing, statistical feature extraction, and machine learning model training. |
| **`dplyr`** | `1.2.1` | Data Manipulation | High-performance grammar of data manipulation used for article filtering, deduplication, grouping, and feature restructuring. |
| **`stringr`** | `1.6.0` | String Operations | Consistent wrapper around stringi for regex matching, whitespace squishing, punctuation counting, and dateline cleaning. |
| **`text2vec`** | `0.6.6` | NLP & Vectorization | High-performance C++ based text vectorization package. Used to generate unigram and bigram vocabularies and sparse TF-IDF matrices efficiently without memory explosion. |
| **`tm`** | `0.7-20` | Text Mining | Core text mining framework providing standard English stopword lists and text normalization utilities. |
| **`tidytext`** | `0.4.3` | Sentiment Analysis | Provides tidy data principles for NLP and supplies the integrated Bing sentiment lexicon for positive and negative word scoring. |
| **`caret`** | `7.1-0` | Machine Learning | Classification and Regression Training framework used for stratified train-test splitting (`createDataPartition`) and confusion matrix validation. |
| **`glmnet`** | `5.2` | Regularized Regression | High-speed C-backed package for generalized linear models. Fits Binomial Logistic Regression with LASSO (L1 penalty) and cross-validation (`cv.glmnet`). Serves as the primary model. |
| **`randomForest`** | `4.7-1.2` | Ensemble Learning | Implements Breiman's Random Forest algorithm. Evaluates an ensemble of 100 decision trees over 32 numerical features and top 50 TF-IDF terms. |
| **`e1071`** | `1.7-17` | Probabilistic ML | Provides the Naive Bayes classifier with Laplace smoothing, modeling conditional feature probabilities as a strong baseline. |
| **`Matrix`** | `1.7-5` | Sparse Linear Algebra | Enables memory-efficient sparse matrix representations (`dgCMatrix`) and fast column-binding (`cbind`) between 4,000 TF-IDF features and 32 continuous numerical features. |
| **`pROC`** | `1.19.1` | Evaluation | Specialized tool for constructing Receiver Operating Characteristic (ROC) curves and computing the Area Under the Curve (AUC). |
| **`ggplot2`** | `4.0.3` | Academic Visualization | Grammar of Graphics implementation used to generate publication-quality PNG charts, confusion matrix heatmaps, and boxplots in `results/charts/`. |
| **`jsonlite`** | `2.0.0` | IPC Data Serialization | High-speed JSON parser and serializer facilitating structured data exchange between Python and R. |

---

## 2. Frontend & User Interface Technologies (Python)

| Technology / Library | Version | Category | Justification & Role in Project |
| :--- | :--- | :--- | :--- |
| **Python** | `3.14.6` | Scripting Environment | Acts as the glue language orchestrating the user interface, subprocess invocation, and client dashboard. |
| **`streamlit`** | `1.58.0` | Web Framework | Lightweight, reactive, pure-Python web framework enabling rapid creation of interactive data dashboards without requiring complex web servers (React/Node.js/Django). |
| **`pandas`** | `3.0.2` | Data Analysis | Manages evaluation metric tables, feature vectors, and dataset summaries in memory for dynamic rendering in Streamlit. |
| **`plotly`** | `6.6.0` | Interactive Charting | Renders interactive JavaScript-powered charts (donut distribution, comparative bar charts, sentiment gauges, confusion heatmaps) within Streamlit. |
| **`matplotlib`** | `3.10.8` | Static Plotting | Provides baseline image rendering and visual support. |
| **`numpy`** | `2.4.4` | Numerical Computing | Mathematical array manipulations and probability scaling. |
| **`subprocess`** | Built-in | Inter-process Comm | Native Python library for spawning the `Rscript.exe` process with isolated arguments, timeouts, and standard I/O redirection. |

---

## 3. Technology Integration Rationale

### Why R for Machine Learning?
R was explicitly engineered for statistical data analysis, linear algebra, and data science. Packages like `glmnet` and `text2vec` deliver performance comparable to compiled C/Fortran routines, and R's native statistical summaries and RDS serialization provide reproducibility for academic coursework.

### Why Python Streamlit for Frontend?
While R has Shiny, Streamlit provides modern reactive rendering, superior asynchronous execution handling, clean separation of backend logic from presentation, and seamless Plotly integration without requiring heavy R Shiny server setups.
