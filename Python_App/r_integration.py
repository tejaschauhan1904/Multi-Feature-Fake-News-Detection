"""
r_integration.py
Multi-Feature Fake News Detection Using R Language
Subprocess Bridge between Python Streamlit and R Backend
"""

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List

try:
    from config import (
        RSCRIPT_PATH,
        R_PREDICTION_SCRIPT,
        LOGISTIC_MODEL_PATH,
        RANDOM_FOREST_MODEL_PATH,
        NAIVE_BAYES_MODEL_PATH,
        TFIDF_VECTORIZER_PATH,
        PREPROCESSING_OBJECTS_PATH,
        LOGS_DIR,
        R_LIBS_DIR,
    )
except ImportError:
    from Python_App.config import (
        RSCRIPT_PATH,
        R_PREDICTION_SCRIPT,
        LOGISTIC_MODEL_PATH,
        RANDOM_FOREST_MODEL_PATH,
        NAIVE_BAYES_MODEL_PATH,
        TFIDF_VECTORIZER_PATH,
        PREPROCESSING_OBJECTS_PATH,
        LOGS_DIR,
        R_LIBS_DIR,
    )


class RIntegrationError(Exception):
    """Custom exception raised when R subprocess execution fails."""
    pass


def check_r_environment() -> Tuple[bool, str]:
    """
    Checks if Rscript is callable and returns its version.
    """
    if not RSCRIPT_PATH or not os.path.exists(RSCRIPT_PATH):
        return False, f"Rscript binary not found at '{RSCRIPT_PATH}'. Please verify R installation."

    try:
        res = subprocess.run(
            [RSCRIPT_PATH, "--version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        version_str = res.stdout.strip() or res.stderr.strip()
        if res.returncode == 0 or "R scripting front-end" in version_str or "version" in version_str.lower():
            return True, version_str
        return False, f"Rscript returned error: {version_str}"
    except Exception as e:
        return False, str(e)


def check_models_presence() -> Tuple[bool, List[str]]:
    """
    Verifies that all trained R model files and preprocessing artifacts exist.
    """
    required_files = [
        ("Logistic Regression Model", LOGISTIC_MODEL_PATH),
        ("Random Forest Model", RANDOM_FOREST_MODEL_PATH),
        ("Naive Bayes Model", NAIVE_BAYES_MODEL_PATH),
        ("TF-IDF Vectorizer", TFIDF_VECTORIZER_PATH),
        ("Preprocessing Objects", PREPROCESSING_OBJECTS_PATH),
    ]

    missing = []
    for label, path in required_files:
        if not path.exists():
            missing.append(f"{label} ({path.name})")

    return len(missing) == 0, missing


def predict_news_article(
    title: str,
    text: str,
    model_name: str = "auto",
    timeout: int = 45
) -> Dict[str, Any]:
    """
    Executes the R prediction pipeline via subprocess.
    Passes article text via a temporary JSON file and receives structured analytical output.
    """
    # 1. Validate inputs
    clean_title = (title or "").strip()
    clean_text = (text or "").strip()

    if not clean_title and not clean_text:
        raise RIntegrationError("Both article title and text are empty. Please enter an article.")

    # 2. Check model artifacts
    models_ok, missing_models = check_models_presence()
    if not models_ok:
        raise RIntegrationError(
            f"Machine learning model files are missing:\n" +
            "\n".join(f" - {m}" for m in missing_models) +
            "\nPlease run the R training pipeline first (e.g., execute main.R)."
        )

    # 3. Check R script presence
    if not R_PREDICTION_SCRIPT.exists():
        raise RIntegrationError(f"Prediction script not found at {R_PREDICTION_SCRIPT}")

    # 4. Create safe isolated temporary files for input and output JSON
    temp_dir = tempfile.gettempdir()
    unique_id = f"fakenews_{int(time.time() * 1000)}_{os.getpid()}"
    input_file_path = os.path.join(temp_dir, f"{unique_id}_in.json")
    output_file_path = os.path.join(temp_dir, f"{unique_id}_out.json")

    input_payload = {
        "title": clean_title,
        "text": clean_text,
        "model": model_name,
    }

    try:
        with open(input_file_path, "w", encoding="utf-8") as f:
            json.dump(input_payload, f, ensure_ascii=False)

        cmd = [
            RSCRIPT_PATH,
            str(R_PREDICTION_SCRIPT),
            input_file_path,
            output_file_path,
        ]

        # Configure environment to include workspace r_libs
        env = os.environ.copy()
        if R_LIBS_DIR.exists():
            env["R_LIBS_USER"] = str(R_LIBS_DIR)
            env["R_LIBS"] = str(R_LIBS_DIR)

        # Execute R script in subprocess
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=env,
        )

        # Check stderr for any critical errors
        stderr_output = proc.stderr.strip() if proc.stderr else ""
        if stderr_output:
            log_path = LOGS_DIR / "application.log"
            try:
                with open(log_path, "a", encoding="utf-8") as lf:
                    lf.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] R subprocess log:\n{stderr_output}\n")
            except Exception:
                pass

        if proc.returncode != 0:
            err_msg = stderr_output or proc.stdout.strip() or f"Process exited with code {proc.returncode}"
            # Check if an error JSON was written
            if os.path.exists(output_file_path):
                try:
                    with open(output_file_path, "r", encoding="utf-8") as f:
                        err_json = json.load(f)
                        if "message" in err_json:
                            err_msg = err_json["message"]
                except Exception:
                    pass
            raise RIntegrationError(f"R Prediction Error: {err_msg}")

        # Verify output JSON exists
        if not os.path.exists(output_file_path):
            raise RIntegrationError(
                f"R prediction finished but no output JSON was produced.\n"
                f"Process Output: {proc.stdout}\nErrors: {stderr_output}"
            )

        with open(output_file_path, "r", encoding="utf-8") as f:
            result_data = json.load(f)

        if result_data.get("status") == "error":
            raise RIntegrationError(result_data.get("message", "Unknown error returned by R backend."))

        return result_data

    except subprocess.TimeoutExpired:
        raise RIntegrationError(f"Prediction timed out after {timeout} seconds in R backend.")
    except json.JSONDecodeError as jde:
        raise RIntegrationError(f"Failed to parse R prediction response JSON: {str(jde)}")
    except Exception as e:
        if not isinstance(e, RIntegrationError):
            raise RIntegrationError(f"Subprocess communication failed: {str(e)}")
        raise
    finally:
        # Secure cleanup of temporary files
        for p in (input_file_path, output_file_path):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass
