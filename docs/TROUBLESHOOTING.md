# Troubleshooting Guide: Multi-Feature Fake News Detection

This guide addresses common technical issues encountered during setup, training, and execution on Windows machines.

---

## 1. Rscript Not Found / Execution Errors

### Symptom:
`'Rscript' is not recognized as an internal or external command` or `Rscript executable could not be found`.

### Root Cause:
The R binary directory has not been added to your Windows system `PATH` environment variable.

### Solution:
1. Verify the location of your R installation. Default is `C:\Program Files\R\R-4.6.1\bin\Rscript.exe`.
2. Set the `RSCRIPT_PATH` environment variable in your terminal:
   ```cmd
   set RSCRIPT_PATH=C:\Program Files\R\R-4.6.1\bin\Rscript.exe
   ```
   Or in PowerShell:
   ```powershell
   $env:RSCRIPT_PATH = "C:\Program Files\R\R-4.6.1\bin\Rscript.exe"
   ```
3. Alternatively, open Windows Settings -> System Properties -> Environment Variables -> Edit `Path` -> Add `C:\Program Files\R\R-4.6.1\bin`.

---

## 2. R Package Access Denied in Sandbox / Missing Packages

### Symptom:
`Access is denied: C:\Users\<Username>\AppData\Local\R\win-library\4.6` or `there is no package called 'jsonlite'`.

### Root Cause:
Standard execution sandboxes restrict access to user AppData folders outside the workspace directory.

### Solution:
This project includes a bundled workspace library `r_libs/` inside the project root.
1. `Python_App/config.py` and `Python_App/r_integration.py` automatically set `R_LIBS_USER` and `R_LIBS` to point to `r_libs/`.
2. In R scripts, `.libPaths()` automatically prepends `r_libs/`.
3. To reinstall packages into `r_libs/` manually:
   ```powershell
   & "C:\Program Files\R\R-4.6.1\bin\Rscript.exe" -e ".libPaths('d:/Tejas/College/Sem_5/R Language/R_PBL_project/r_libs'); install.packages(c('jsonlite', 'glmnet', 'randomForest', 'e1071', 'text2vec', 'tm', 'tidytext', 'caret', 'pROC', 'ggplot2'), repos='https://cloud.r-project.org')"
   ```

---

## 3. Model Files Missing

### Symptom:
Streamlit shows: `Machine learning model files are missing: Logistic Regression Model (logistic_regression_model.rds)... Please run the R training pipeline first`.

### Root Cause:
The R machine learning training pipeline has not been executed yet.

### Solution:
Run the master training script from PowerShell:
```powershell
& "C:\Program Files\R\R-4.6.1\bin\Rscript.exe" main.R
```
This will automatically execute data loading, preprocessing, feature extraction, TF-IDF vectorization, model training, and save all `.rds` objects into `models/`.

---

## 4. Dataset Files Not Found

### Symptom:
`[ERROR] Original dataset files (Fake.csv / True.csv) not found in dataset directory!`

### Root Cause:
The CSV files are not located in the `dataset/` subfolder of your workspace.

### Solution:
Ensure `Fake.csv` and `True.csv` are placed directly inside `dataset/`:
```
dataset/
  ├── Fake.csv
  └── True.csv
```
Verify via PowerShell:
```powershell
Test-Path dataset/Fake.csv
Test-Path dataset/True.csv
```

---

## 5. Python Dependencies Missing / Streamlit Not Starting

### Symptom:
`ModuleNotFoundError: No module named 'streamlit'` or `'streamlit' is not recognized`.

### Root Cause:
Python packages are not installed in your active Python environment.

### Solution:
Install all dependencies using pip:
```powershell
python -m pip install -r requirements.txt
```
If Python is not in PATH, use the full path to `python.exe`:
```powershell
& "C:\Users\<Username>\AppData\Local\Programs\Python\Python314\python.exe" -m pip install -r requirements.txt
```

---

## 6. Subprocess Timeout / JSON Communication Errors

### Symptom:
`RIntegrationError: Prediction timed out after 45 seconds in R backend.`

### Root Cause:
R process hung or was waiting for interactive user input.

### Solution:
1. Verify that `10_prediction.R` runs cleanly in non-interactive CLI mode:
   ```powershell
   & "C:\Program Files\R\R-4.6.1\bin\Rscript.exe" R_Scripts/10_prediction.R
   ```
2. Check `logs/application.log` to view any R warnings or diagnostic messages.
3. Verify that your temporary folder (`%TEMP%`) has write permissions.

---

## 7. Memory Overhead During TF-IDF Generation

### Symptom:
R terminates with `Error: cannot allocate vector of size...` during vectorization.

### Root Cause:
Attempting to create a dense matrix of 40,000 documents by 50,000 vocabulary terms.

### Solution:
1. `06_tfidf_vectorization.R` utilizes `text2vec` sparse matrices (`dgCMatrix`) and vocabulary pruning (`term_count_min = 5`, `vocab_term_max = 4000L`).
2. Sparse matrices only store non-zero indices, reducing RAM consumption from ~8 GB to under 200 MB.
