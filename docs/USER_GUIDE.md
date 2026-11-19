# User Guide: Multi-Feature Fake News Detection Dashboard

This guide provides end-to-end instructions for operating each page of the Python Streamlit web dashboard.

---

## 1. Launching the Application
Execute `run_app.bat` or run:
```powershell
streamlit run Python_App/app.py
```
Open your web browser at `http://localhost:8501`.

---

## 2. Navigating Dashboard Pages

Use the **left sidebar** to switch between the 6 application sections.

### Page 1: 🏠 Dashboard (Executive Overview)
- **Top Metrics:** View total cleaned articles (39,100), class balance percentage, best model name, and feature dimensionality (4,032 features).
- **Interactive Visuals:** 
  - Hover over the **Class Balance Donut Chart** to see precise counts of Real and Fake articles.
  - Review the **Accuracy & F1-Score Chart** comparing all three algorithms.
- **Workflow Pipeline:** Review the 6-stage end-to-end system architecture.

---

### Page 2: 🔍 Fake News Detection (Live Inference)
This is the core interactive testing environment.

#### How to Analyze an Article:
1. **Load a Demo Article (Optional):** Click the dropdown at the top to auto-fill sample Real or Fake news articles.
2. **Or Enter Custom Text:**
   - **Headline (Optional):** Enter the article title (e.g., `"Senate Approves Fiscal Budget"`).
   - **Body Text (Required):** Paste the complete article text.
3. **Select Model (Sidebar):** Choose between Logistic Regression (LASSO), Random Forest, or Naive Bayes.
4. **Click "⚡ Analyze News Article":**
   - The system passes the article to the R backend.
   - Within 1–2 seconds, the result card appears.
5. **Interpret the Results:**
   - **Green Card (✅ REAL NEWS):** The article displays structural, linguistic, and lexical patterns consistent with verified journalism.
   - **Red Card (🚨 FAKE NEWS):** The article exhibits stylistic, sentiment, and vocabulary traits typical of fabricated reporting.
   - **Confidence Percentage:** The posterior probability estimated by the model.
   - **Feature Highlights:** Check the 5 metrics below the card (Words, Characters, Net Sentiment, Readability, and Uppercase %).
   - **Top Keywords:** Inspect the salient TF-IDF vocabulary tokens identified in the text.

---

### Page 3: 📊 Feature Analysis (Detailed Breakdown)
Dive deeper into the features extracted from your analyzed article:
- **Tab 1: Sentiment Analysis:** Net sentiment score, positive/negative word counts, and sentiment gauge.
- **Tab 2: Readability Analysis:** Flesch Reading Ease score, words per sentence, and structural readability interpretation.
- **Tab 3: Linguistic Markers:** Table of uppercase ratios, exclamation marks, question marks, and punctuation counts.
- **Tab 4: Statistical Properties:** Lexical diversity (type-token ratio), word lengths, and paragraph metrics.
- **Tab 5: TF-IDF & Keywords:** Horizontal bar chart of the highest-weighted vocabulary terms in the article.
- **Raw Feature Table:** Expand the bottom accordion to inspect the complete 32-dimensional continuous numerical vector.

---

### Page 4: 📈 Model Performance (Comparative Evaluation)
- **Best Model Card:** Summarizes the top-performing model selected via balanced F1-score and ROC-AUC.
- **Full Metrics Table:** Compare Accuracy, Precision, Recall, Specificity, F1, and AUC side-by-side.
- **Interactive Confusion Matrices:** Inspect the True Positive, True Negative, False Positive, and False Negative counts for each architecture.
- **High-Resolution Academic Charts:** View the publication-quality ROC curves, feature importance bar charts, and comparative performance plots generated in R using `ggplot2`.

---

### Page 5: 📁 Dataset Analysis (Corpus Insights)
- **Cleaning & Deduplication Audit:** Review the removal of 5,798 duplicate records and publisher dateline leakage mitigation.
- **Corpus Distribution Visuals:** View the cleaned class distribution and sentiment distributions.
- **Stylistic Signatures:** Read the empirical linguistic distinctions observed between fake and real news during training.

---

### Page 6: ℹ️ About Project (Academic Documentation)
- Review the project problem statement, objectives, technology stack, limitations, and future scope.
