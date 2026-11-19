# Python-R Integration: Subprocess & JSON Exchange Architecture

This document describes the technical implementation of the integration layer connecting the Python Streamlit user interface with the R machine learning backend.

---

## 1. Architectural Motivation

A key requirement of this project is strict separation between computational responsibilities:
- **R Backend:** Responsible for data preprocessing, linguistic feature extraction, TF-IDF vectorization, model training, and probability estimation. Machine learning is **not** reimplemented in Python.
- **Python Frontend:** Responsible for the reactive web dashboard, user input capture, Plotly charts, and user guidance.

By coupling Python and R via native operating system subprocesses and JSON files, we achieve:
1. **Zero Memory Bleed:** R and Python run in independent memory address spaces.
2. **Crash Resilience:** An unexpected error or timeout in R does not crash the Python Streamlit web server.
3. **No Heavy Daemons:** Eliminates the need for REST APIs (FastAPI/Flask/Plumber) or database servers.

---

## 2. Communication Protocol Specification

The bridge module `Python_App/r_integration.py` executes the following protocol:

```
[Python Streamlit]
       |
       | 1. Writes payload to temp file: /tmp/fakenews_<timestamp>_<pid>_in.json
       v
[File System: Input JSON]
       |
       | 2. Spawns: Rscript 10_prediction.R in.json out.json (with timeout)
       v
[Rscript 10_prediction.R]
       |
       | 3. Reads in.json, extracts features, computes TF-IDF & model prediction
       | 4. Writes response to /tmp/fakenews_<timestamp>_<pid>_out.json
       v
[File System: Output JSON]
       |
       | 5. Python reads out.json & deletes temporary files in finally: block
       v
[Python Streamlit UI]
```

### A. Input JSON Schema:
```json
{
  "title": "NASA Announces Successful Satellite Launch",
  "text": "The National Aeronautics and Space Administration successfully deployed its next-generation satellite...",
  "model": "logistic_regression"
}
```

### B. Output JSON Schema:
```json
{
  "status": "success",
  "prediction": "REAL NEWS",
  "label": 1,
  "real_news_probability": 0.8743,
  "fake_news_probability": 0.1257,
  "confidence": 0.8743,
  "confidence_percent": "87.43%",
  "model_used": "Logistic Regression (LASSO)",
  "extracted_features": {
    "char_count": 349,
    "word_count": 46,
    "sentence_count": 2,
    "lexical_diversity": 0.8913,
    "flesch_reading_ease": 48.2,
    "sentiment_score": 4,
    "uppercase_ratio": 0.0573
  },
  "top_keywords": {
    "advanced": 1.1182,
    "earth": 1.1088,
    "space": 1.0666,
    "launch": 1.0349
  },
  "disclaimer": "This result is a machine learning classification based on patterns learned from the training dataset. It does not independently verify the factual accuracy of the article.",
  "timestamp": "2026-11-17 22:32:54"
}
```

---

## 3. Subprocess Execution & Error Handling

In `Python_App/r_integration.py`, the execution function `predict_news_article()` implements defensive safeguards:

```python
env = os.environ.copy()
if R_LIBS_DIR.exists():
    env["R_LIBS_USER"] = str(R_LIBS_DIR)
    env["R_LIBS"] = str(R_LIBS_DIR)

proc = subprocess.run(
    [RSCRIPT_PATH, str(R_PREDICTION_SCRIPT), in_path, out_path],
    capture_output=True,
    text=True,
    timeout=45,
    check=False,
    env=env,
)
```

### Handled Error Scenarios:
1. **Rscript Not Found:** Config module searches PATH and common Windows installation paths (`C:\Program Files\R\R-4.6.1\bin\Rscript.exe`).
2. **Missing Model Files:** Checks if `models/*.rds` exist before invoking R, raising clear instructions if training is required.
3. **Empty Input Text:** Validates text before calling R.
4. **Execution Timeout:** Enforces a 45-second cap via `subprocess.TimeoutExpired`.
5. **Temporary File Collisions:** File paths incorporate microsecond timestamps and system process IDs (`os.getpid()`).
6. **Graceful Cleanup:** Temporary files are safely deleted in `finally:` blocks even on unexpected exceptions.
