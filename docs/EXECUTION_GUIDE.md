# Execution Guide: Multi-Feature Fake News Detection

This guide provides step-by-step instructions for executing the R backend machine learning pipeline, verifying model generation, and launching the Python Streamlit dashboard.

---

## 1. Quick Launch (Recommended)

The easiest way to execute the project on Windows is using the included batch launcher:

1. Double-click `run_app.bat` in the project root directory, or execute it from PowerShell:
   ```cmd
   .\run_app.bat
   ```
2. The launcher performs:
   - Verification of the `Rscript` engine.
   - Verification of Python 3.10+.
   - Verification of Python dependencies.
   - Verification of trained model files in `models/`.
   - Automatic prompt to execute `main.R` if models are not yet trained.
   - Launching the Streamlit dashboard at `http://localhost:8501`.

---

## 2. Manual Execution Workflow

If you prefer executing each component manually or via terminal, follow the steps below.

### Step A: Execute the R Machine Learning Pipeline
To run the end-to-end data processing, feature extraction, TF-IDF vectorization, model training, evaluation, and chart generation:

```powershell
& "C:\Program Files\R\R-4.6.1\bin\Rscript.exe" main.R
```

#### Individual Stage Execution:
You can also run any pipeline script independently:
```powershell
# 1. Check environment
& "C:\Program Files\R\R-4.6.1\bin\Rscript.exe" R_Scripts/01_check_environment.R

# 2. Ingest raw datasets
& "C:\Program Files\R\R-4.6.1\bin\Rscript.exe" R_Scripts/02_data_loading.R

# 3. Preprocess data, deduplicate, and create stratified splits
& "C:\Program Files\R\R-4.6.1\bin\Rscript.exe" R_Scripts/03_data_preprocessing.R

# 4. Extract 32 multi-category features
& "C:\Program Files\R\R-4.6.1\bin\Rscript.exe" R_Scripts/05_feature_extraction.R

# 5. Fit TF-IDF on train set and combine features
& "C:\Program Files\R\R-4.6.1\bin\Rscript.exe" R_Scripts/06_tfidf_vectorization.R

# 6. Train Logistic Regression, Random Forest, and Naive Bayes
& "C:\Program Files\R\R-4.6.1\bin\Rscript.exe" R_Scripts/07_model_training.R

# 7. Evaluate on held-out test data
& "C:\Program Files\R\R-4.6.1\bin\Rscript.exe" R_Scripts/08_model_evaluation.R

# 8. Generate comparative summary report
& "C:\Program Files\R\R-4.6.1\bin\Rscript.exe" R_Scripts/09_model_comparison.R

# 9. Generate high-resolution ggplot2 charts
& "C:\Program Files\R\R-4.6.1\bin\Rscript.exe" R_Scripts/11_generate_visualizations.R
```

---

## 3. Verifying Model Generation

After training, verify that the following serialized RDS objects exist in the `models/` directory:

| Filename | Description | Approximate Size |
| :--- | :--- | :--- |
| `models/logistic_regression_model.rds` | Cross-validated glmnet LASSO object | ~180 KB |
| `models/random_forest_model.rds` | 100-tree randomForest ensemble | ~1.8 MB |
| `models/naive_bayes_model.rds` | e1071 Naive Bayes probabilistic model | ~35 KB |
| `models/tfidf_vectorizer.rds` | text2vec vectorizer & vocabulary | ~120 KB |
| `models/preprocessing_objects.rds` | Scalers, means, standard deviations | ~15 KB |
| `models/model_metadata.rds` | Model parameters, positive class info | ~10 KB |

Check their presence via PowerShell:
```powershell
Get-ChildItem -Path models\*.rds
```

---

## 4. Testing Live Inference via R CLI

Before launching the web frontend, you can verify that the JSON prediction interface is functioning correctly:

```powershell
& "C:\Program Files\R\R-4.6.1\bin\Rscript.exe" R_Scripts/10_prediction.R
```
This triggers the self-test mode, outputting structured JSON with predicted class, model-estimated probability, and extracted features.

---

## 5. Launching the Python Streamlit Frontend

To launch the web interface:

```powershell
streamlit run Python_App/app.py
```

Your default web browser will automatically open to:
`http://localhost:8501`

### Useful Streamlit Flags:
```powershell
# Run on a custom port:
streamlit run Python_App/app.py --server.port 8502

# Run without automatically opening browser:
streamlit run Python_App/app.py --server.headless true
```
