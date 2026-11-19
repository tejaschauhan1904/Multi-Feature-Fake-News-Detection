# Installation Guide: Multi-Feature Fake News Detection

This guide walks through configuring your Windows development environment to run both the R machine learning pipeline and the Python Streamlit user interface.

---

## 1. Prerequisites

### A. R Programming Environment
1. Download **R (version 4.2 or higher)** from CRAN:
   - Official Windows installer: [https://cran.r-project.org/bin/windows/base/](https://cran.r-project.org/bin/windows/base/)
2. Run the installer with default settings (installed by default to `C:\Program Files\R\R-4.x.x`).
3. **Verify R installation via PowerShell or Command Prompt:**
   ```powershell
   & "C:\Program Files\R\R-4.6.1\bin\Rscript.exe" --version
   # Expected output: Rscript (R) version 4.6.1 (...)
   ```
4. *(Optional but Recommended)* Add `C:\Program Files\R\R-4.6.1\bin` to your Windows system `PATH` environment variable.

### B. Python Environment
1. Download **Python (version 3.10 to 3.14)** from:
   - Official installer: [https://www.python.org/downloads/](https://www.python.org/downloads/)
2. **Important:** During installation, check the box **"Add Python to PATH"**.
3. **Verify Python and pip:**
   ```powershell
   python --version
   python -m pip --version
   ```

---

## 2. Installing Required R Packages

Launch Windows PowerShell or Command Prompt and execute the following command to install all necessary backend packages directly from CRAN:

```powershell
& "C:\Program Files\R\R-4.6.1\bin\Rscript.exe" -e "install.packages(c('dplyr', 'stringr', 'tidytext', 'text2vec', 'tm', 'caret', 'randomForest', 'e1071', 'pROC', 'glmnet', 'Matrix', 'jsonlite', 'ggplot2'), repos='https://cloud.r-project.org')"
```

### Verification of R Packages
Run the environment check script to confirm all packages load cleanly:
```powershell
& "C:\Program Files\R\R-4.6.1\bin\Rscript.exe" R_Scripts/01_check_environment.R
```
Expected output:
```
[SUCCESS] All required R packages are installed and available.
[SUCCESS] Dataset files verified: Fake.csv, True.csv
```

---

## 3. Installing Python Dependencies

The Python frontend uses Streamlit, Plotly, Pandas, and Matplotlib. Install them using the root `requirements.txt`:

```powershell
python -m pip install -r requirements.txt
```

### Dependencies Installed:
| Package | Minimum Version | Purpose |
| :--- | :--- | :--- |
| `streamlit` | 1.30.0+ | Web application dashboard and reactive UI |
| `pandas` | 2.0.0+ | Data processing and tabular visualization |
| `plotly` | 5.18.0+ | Interactive publication-quality charts |
| `matplotlib`| 3.8.0+ | Supplementary plotting support |
| `numpy` | 1.24.0+ | Numerical matrix utilities |
| `requests` | 2.31.0+ | Network communication helper |

### Verification of Python Packages
```powershell
python -c "import streamlit, pandas, plotly, matplotlib, numpy; print('All Python dependencies verified successfully!')"
```

---

## 4. Local R Library Isolation (`r_libs`)
To ensure that R operates smoothly inside secure execution sandboxes without Windows AppData permission issues, this project bundles its installed packages into the workspace directory `r_libs/`. 

Both `R_Scripts/10_prediction.R` and `Python_App/r_integration.py` automatically detect and load `r_libs/` via `.libPaths()` and environment variables (`R_LIBS_USER`).
