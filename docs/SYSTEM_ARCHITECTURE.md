# System Architecture: Multi-Feature Fake News Detection

This document specifies the technical architecture, component responsibilities, inter-process communication protocol, and data flows of the system.

---

## 1. Architectural Overview

The system employs a decoupled, polyglot architecture:
- **Backend & Compute Engine:** R Programming Language (v4.6.1)
- **Frontend & Presentation Layer:** Python (v3.14) + Streamlit (v1.58)
- **IPC Protocol:** Standard Subprocess Execution with File-Based JSON Data Exchange

```mermaid
graph TD
    subgraph Frontend["Python Streamlit Frontend (Port 8501)"]
        UI["Web Dashboard UI<br/>(Dashboard, Detection, Features, Models, Dataset)"]
        Bridge["r_integration.py<br/>(Subprocess Executor & Temp File Manager)"]
        Config["config.py<br/>(Path Resolution & Model Registry)"]
    end

    subgraph IPC["Inter-Process Communication Layer"]
        InJSON["Temporary Input JSON<br/>{title, text, model}"]
        OutJSON["Temporary Output JSON<br/>{prediction, probability, features, keywords}"]
    end

    subgraph Backend["R Backend Engine (Rscript.exe)"]
        PredScript["10_prediction.R<br/>(Prediction CLI & Feature Pipeline)"]
        TextPrep["04_text_preprocessing.R<br/>(Cleaning & Tokenization)"]
        FeatExt["05_feature_extraction.R<br/>(Sentiment, Readability, Linguistics)"]
        TFIDF["models/tfidf_vectorizer.rds<br/>(text2vec Fitted Transformer)"]
        Scaler["models/preprocessing_objects.rds<br/>(Z-Score Normalization Weights)"]
        
        subgraph MLModels["Trained Machine Learning Models"]
            M1["models/logistic_regression_model.rds<br/>(glmnet cv.glmnet LASSO)"]
            M2["models/random_forest_model.rds<br/>(randomForest 100 Trees)"]
            M3["models/naive_bayes_model.rds<br/>(e1071 Naive Bayes)"]
        end
    end

    UI --> Bridge
    Bridge --> Config
    Bridge -->|Write payload| InJSON
    InJSON --> PredScript
    PredScript --> TextPrep
    PredScript --> FeatExt
    PredScript --> TFIDF
    PredScript --> Scaler
    PredScript --> MLModels
    PredScript -->|Write response| OutJSON
    OutJSON --> Bridge
    Bridge --> UI
```

---

## 2. Component Responsibilities

### A. R Language Responsibilities (The Brain)
1. **Data Preprocessing:** Cleaning raw text, stripping publisher tags, eliminating duplicates, and performing stratified sampling.
2. **Feature Extraction:** Lexicon sentiment matching, syllable and sentence counting, uppercase/punctuation extraction, and statistical summarization.
3. **TF-IDF Transformation:** Constructing sparse document-term matrices using `text2vec`.
4. **Machine Learning Model Training:** Model optimization with cross-validation using `glmnet`, `randomForest`, and `e1071`.
5. **Model Evaluation:** Computing Accuracy, Precision, Recall, F1-Score, and ROC-AUC on held-out test data.
6. **Inference Execution:** Standardized prediction CLI (`10_prediction.R`) executing all preprocessing and model scoring identically to training.

### B. Python Streamlit Responsibilities (The Face)
1. **User Interface:** Delivering an interactive, responsive dashboard.
2. **User Input Handling:** Capturing news headlines and articles, validating inputs, and offering preset demonstration samples.
3. **Process Orchestration:** Locating the `Rscript.exe` executable, spawning isolated subprocesses, and safely writing/reading temporary JSON files.
4. **Data Visualization:** Rendering Plotly interactive bar charts, donut charts, gauge meters, and confusion matrices.
5. **Cross-Page State Management:** Persisting inference results in `st.session_state` so users can seamlessly inspect features across tabs.

---

## 3. Data Flow Pipelines

### A. Model Training Flow (Offline Execution)
```mermaid
sequenceDiagram
    autonumber
    participant D as dataset/Fake.csv & True.csv
    participant P as 03_data_preprocessing.R
    participant F as 05_feature_extraction.R
    participant T as 06_tfidf_vectorization.R
    participant M as 07_model_training.R
    participant E as 08_model_evaluation.R
    participant S as models/*.rds

    D->>P: Load 44,898 raw records
    P->>P: Deduplicate (39,100 clean) & Stratified Split (80:20)
    P->>F: Pass Train (31,280) & Test (7,820)
    F->>F: Extract 32 Multi-category Features
    F->>T: Pass Text & Numerical Vectors
    T->>T: Fit 4,000 TF-IDF Unigram/Bigram Vocabulary on Train
    T->>T: Bind Sparse TF-IDF & Scaled Numerical Features
    T->>M: Supply Integrated Train & Test Matrices
    M->>S: Train & Serialize glmnet, randomForest, e1071
    M->>E: Pass Predictions on Held-Out Test Set
    E->>E: Calculate Accuracy, F1, ROC-AUC, Confusion Matrix
```

### B. Live Inference Flow (Real-Time Prediction)
```mermaid
sequenceDiagram
    autonumber
    actor User as Web User
    participant Streamlit as Python Streamlit UI
    participant Bridge as r_integration.py
    participant R as Rscript 10_prediction.R
    participant Models as models/*.rds

    User->>Streamlit: Enters Article Title & Text -> Clicks 'Analyze News'
    Streamlit->>Bridge: Calls predict_news_article(title, text, model)
    Bridge->>Bridge: Generates input_xxxx.json in Temp Directory
    Bridge->>R: Spawns subprocess: Rscript 10_prediction.R in.json out.json
    R->>Models: Loads preprocessors, vectorizer, & model
    R->>R: Preprocesses text, extracts 32 features, computes TF-IDF
    R->>R: Evaluates model probabilities & extracts top keywords
    R->>Bridge: Writes output_xxxx.json
    Bridge->>Bridge: Reads JSON & Cleans up temporary files
    Bridge->>Streamlit: Returns prediction dictionary
    Streamlit->>User: Displays Result Badge, Probability, Metrics & Keyword Pills
```

---

## 4. Security and Isolation Principles
1. **Isolated File Names:** Temporary files use high-resolution timestamps and process IDs (`fakenews_<timestamp>_<pid>_in.json`) to prevent collisions during concurrent requests.
2. **Automatic Cleanup:** Temporary files are safely deleted in Python `finally:` blocks even if errors occur.
3. **Execution Timeouts:** Subprocess calls enforce a strict 45-second execution timeout to prevent runaway processes.
4. **Sandboxed Library Loading:** Local dependencies are isolated in `r_libs/`, ensuring execution succeeds in restricted sandbox and enterprise environments.
