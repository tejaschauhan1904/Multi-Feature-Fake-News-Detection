@echo off
setlocal EnableDelayedExpansion

:: Ensure we are running from the directory where this script is located
cd /d "%~dp0"

title Multi-Feature Fake News Detection - Launcher
color 0B

echo ================================================================================
echo        MULTI-FEATURE FAKE NEWS DETECTION USING R LANGUAGE (ACADEMIC PBL)
echo ================================================================================
echo.

:: 1. Detect Rscript executable
echo [1/5] Verifying R Installation...
set "RSCRIPT_CMD="

where Rscript >nul 2>nul
if %ERRORLEVEL% equ 0 (
    set "RSCRIPT_CMD=Rscript"
) else (
    if exist "C:\Program Files\R\R-4.6.1\bin\Rscript.exe" (
        set "RSCRIPT_CMD=C:\Program Files\R\R-4.6.1\bin\Rscript.exe"
    ) else if exist "C:\Program Files\R\R-4.6.1\bin\x64\Rscript.exe" (
        set "RSCRIPT_CMD=C:\Program Files\R\R-4.6.1\bin\x64\Rscript.exe"
    )
)

if "%RSCRIPT_CMD%"=="" (
    echo [ERROR] Rscript executable could not be found!
    echo Please install R or add Rscript.exe to your system PATH.
    pause
    exit /b 1
)
echo  - Found R Engine: %RSCRIPT_CMD%

:: 2. Detect Python
echo [2/5] Verifying Python Installation...
python --version >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.10+ from python.org and add it to PATH.
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('python --version') do echo  - Found Python: %%i

:: 3. Verify Python Packages
echo [3/5] Verifying Python Dependencies...
python -c "import streamlit, pandas, plotly, matplotlib" >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [WARNING] Some Python dependencies appear to be missing.
    echo Installing from requirements.txt...
    python -m pip install -r requirements.txt
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to install Python dependencies.
        pause
        exit /b 1
    )
) else (
    echo  - All required Python libraries are installed.
)

:: 4. Check Trained Models
echo [4/5] Checking Machine Learning Model RDS Artifacts...
set "MODELS_MISSING=0"
if not exist "%~dp0models\logistic_regression_model.rds" set "MODELS_MISSING=1"
if not exist "%~dp0models\preprocessing_objects.rds" set "MODELS_MISSING=1"
if not exist "%~dp0models\tfidf_vectorizer.rds" set "MODELS_MISSING=1"

if "%MODELS_MISSING%"=="0" (
    echo  - Trained R models detected and validated.
    goto launch_app
)

echo.
echo [NOTICE] Trained R machine learning models were not found in 'models\'.
echo Would you like to execute the full R training pipeline now?
set "RUN_TRAIN=N"
set /p "RUN_TRAIN=Enter choice [Y/N]: "

if /i "%RUN_TRAIN%"=="Y" goto run_training
if /i "%RUN_TRAIN%"=="YES" goto run_training

echo Skipping training. Note: Live predictions will remain unavailable until models are trained.
goto launch_app

:run_training
echo.
echo Starting R Training Pipeline main.R...
"%RSCRIPT_CMD%" main.R
if %ERRORLEVEL% neq 0 (
    echo [ERROR] R Training Pipeline encountered an error. Please check the logs.
    pause
    exit /b 1
)
echo.
echo [SUCCESS] Training complete. Proceeding to launch dashboard...

:launch_app
:: 5. Launch Streamlit Application
echo.
echo [5/5] Launching Streamlit Dashboard...
echo Starting server on local port (default: 8501)...
echo Press Ctrl+C in this terminal window to stop the application.
echo.

set "STREAMLIT_CONFIG_DIR=%~dp0.streamlit"
python -m streamlit run "%~dp0Python_App\app.py" --browser.gatherUsageStats false

pause
