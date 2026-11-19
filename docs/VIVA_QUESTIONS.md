# Academic Viva Voce Questions & Answers: Multi-Feature Fake News Detection

This document provides 35 comprehensive viva voce examination questions and in-depth academic answers covering R language concepts, machine learning algorithms, NLP, feature engineering, and system integration.

---

## Category 1: R Language & System Concepts

### Q1: Why was the R programming language chosen for the machine learning backend of this project instead of doing everything in Python?
**Answer:** R was explicitly designed for statistical modeling, exploratory data analysis, and scientific computing. Packages like `glmnet` provide Fortran/C-optimized implementations of coordinate descent for regularized generalized linear models, while `text2vec` delivers fast memory-efficient text vectorization. In an academic context, R allows clear transparency in mathematical formula modeling, matrix operations (`Matrix`), and data manipulation (`dplyr`).

### Q2: What is an RDS file in R, and how does it differ from RData?
**Answer:** An RDS (`.rds`) file serializes and stores a single R object to disk using `saveRDS()` and restores it using `readRDS()`. Unlike `.RData` (which saves and restores an entire workspace environment with multiple variables bound to their original names), RDS files allow assigning the loaded object to any variable name in a modular fashion:
```r
model <- readRDS("models/logistic_regression_model.rds")
```

### Q3: What is a sparse matrix (`dgCMatrix`), and why is it essential for TF-IDF in R?
**Answer:** A `dgCMatrix` is a compressed sparse column (CSC) matrix representation provided by the `Matrix` package. In high-dimensional NLP, most words in the vocabulary do not appear in any given document, meaning over 98% of matrix entries are zero. A dense matrix of 6,000 documents $\times$ 4,000 words requires storing 24,000,000 floating-point numbers (~192 MB). A sparse matrix stores only non-zero coordinates and values, reducing memory consumption by over 95%.

### Q4: How is project-root detection handled dynamically across different environments in R?
**Answer:** Rather than hardcoding absolute paths like `C:\Users\...`, the scripts define a resolver function `get_project_root()` that checks candidate directories (current working directory `getwd()`, parent directory `..`, and workspace paths) for anchor files such as `dataset/Fake.csv` or `main.R`. This guarantees execution works seamlessly across RStudio, command prompt, and cloud environments.

### Q5: What role does `.libPaths()` play in R package isolation?
**Answer:** `.libPaths()` manages the search directories from which R loads installed packages. In this project, `.libPaths()` is prepended with the project's local `r_libs/` directory. This isolates dependencies within the project directory and prevents access-denied errors when running in restricted sandbox environments.

---

## Category 2: Natural Language Processing & TF-IDF

### Q6: What is TF-IDF, and what are its mathematical components?
**Answer:** Term Frequency-Inverse Document Frequency (TF-IDF) is a numerical statistic reflecting how important a word is to a document within a collection. It is the product of two statistics:
1. **Term Frequency (TF):** Measures how frequently term $t$ appears in document $d$:
   $$\text{TF}(t, d) = \frac{f_{t, d}}{\sum_{t'} f_{t', d}}$$
2. **Inverse Document Frequency (IDF):** Penalizes terms that occur ubiquitously across all documents:
   $$\text{IDF}(t, D) = \ln\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$

### Q7: Why did we use sublinear term frequency scaling in `text2vec`?
**Answer:** Sublinear TF scales term frequency logarithmically ($\text{TF} = 1 + \ln(f)$ for $f > 0$). This embodies the principle of diminishing returns: an article mentioning the word `"election"` 20 times is not 20 times more relevant than an article mentioning it twice.

### Q8: What is the difference between unigrams and bigrams? Why include both?
**Answer:** Unigrams are individual words ($n=1$), while bigrams are sequences of two adjacent words ($n=2$). While unigrams capture broad topic matter, bigrams preserve contextual collocations (e.g., `"breaking news"`, `"white house"`, `"mainstream media"`). Clickbait and fake news frequently exhibit distinctive bigram signatures that individual unigrams miss.

### Q9: Why is vocabulary pruning necessary before model training?
**Answer:** The raw vocabulary across 39,100 news articles exceeds 900,000 terms. Most terms are rare typos or hapax legomena. Pruning terms with document frequencies below 0.002 or above 0.50, and capping the vocabulary at 4,000 terms:
1. Prevents extreme matrix dimensionality and memory exhaustion.
2. Eliminates noisy, non-generalizable terms that cause overfitting.
3. Accelerates training and inference latency.

### Q10: Why must the vocabulary and TF-IDF transformer be fitted ONLY on training data?
**Answer:** Fitting the vocabulary or IDF weights on the entire dataset prior to splitting constitutes **data leakage**. It allows information about the word distributions in the test set to leak into the training process, producing artificially optimistic test performance that fails to generalize to real-world data.

---

## Category 3: Feature Engineering & Linguistics

### Q11: What are the 6 categories of features extracted in this project?
**Answer:**
1. **Textual / TF-IDF Features:** 4,000 unigram and bigram document-term weights.
2. **Lexicon Sentiment Features:** Positive words, negative words, net score, and polarity balance.
3. **Readability Features:** Flesch Reading Ease score, words per sentence, and characters per word.
4. **Linguistic Features:** Uppercase character ratios, exclamation marks, question marks, and punctuation counts.
5. **Statistical Features:** Character count, word count, unique vocabulary size, lexical diversity, and word lengths.
6. **Title-Specific Features:** Headline uppercase ratio, title exclamations, and title question marks.

### Q12: How is the Flesch Reading Ease score calculated, and what does it indicate?
**Answer:** The Flesch Reading Ease formula is:
$$\text{Flesch} = 206.835 - 1.015 \left(\frac{\text{total words}}{\text{total sentences}}\right) - 84.6 \left(\frac{\text{total syllables}}{\text{total words}}\right)$$
Higher scores (70–100) denote easy, conversational prose, while lower scores (0–50) denote complex, academic prose. In journalism, real reporting exhibits longer, multi-clause sentences (lower scores), while fake news relies on shorter, simplified sentences designed for viral consumption (higher scores).

### Q13: What is Lexical Diversity (Type-Token Ratio), and how does it distinguish fake from real news?
**Answer:** Lexical Diversity is calculated as the ratio of unique words ($U$) to total words ($W$): $\text{TTR} = U / W$. Professional journalism employs richer, more diverse vocabularies (higher TTR), whereas automated or fabricated stories exhibit repetitious phrasing and lower TTR.

### Q14: How does lexicon-based sentiment analysis work in R using `tidytext`?
**Answer:** The text is tokenized into words and matched against the Bing Liu Sentiment Lexicon, which categorizes 6,786 words as positive or negative. The Net Sentiment Score is computed as:
$$\text{Sentiment Score} = N_{\text{positive}} - N_{\text{negative}}$$
A hash table environment is used in R for $O(1)$ constant-time lookup per token.

### Q15: Does a negative sentiment score automatically indicate fake news?
**Answer:** No. Genuine news reporting frequently covers tragedies, economic downturns, and conflicts, which inherently produce negative sentiment scores. Instead, the model evaluates sentiment extremity and balance alongside linguistic markers and TF-IDF collocations.

### Q16: Why are continuous numerical features standardized using Z-score normalization?
**Answer:** Continuous features vary across vastly different scales (e.g., `char_count` ranges from 200 to 10,000, while `uppercase_ratio` ranges from 0.0 to 1.0). Regularized regression (LASSO) applies an $L_1$ penalty to coefficient magnitudes. If features are unstandardized, features with naturally small scales would be unfairly penalized regardless of their predictive power.

---

## Category 4: Machine Learning Models & Mathematics

### Q17: What is Logistic Regression with LASSO, and how does it perform feature selection?
**Answer:** Binomial logistic regression models class probability via the sigmoid function. LASSO adds an $L_1$ regularization penalty ($\lambda \sum |\beta_j|$) to the cross-entropy loss function. Because the $L_1$ ball has sharp corners at the parameter axes, optimizing under $L_1$ constraint forces less informative coefficients to become exactly zero, automatically performing sparse feature selection.

### Q18: What is the purpose of `cv.glmnet` and 5-fold cross-validation?
**Answer:** `cv.glmnet` partitions the training data into 5 folds, fits LASSO models across a sequence of candidate $\lambda$ values, and evaluates out-of-fold cross-validation error (AUC). It identifies:
- $\lambda_{\text{min}}$: Lambda achieving minimum mean cross-validated error.
- $\lambda_{\text{1se}}$: Largest lambda within one standard error of $\lambda_{\text{min}}$, producing a sparser, more robust model that avoids overfitting.

### Q19: How does the Random Forest algorithm work?
**Answer:** Random Forest builds an ensemble of $B$ de-correlated decision trees using bootstrap aggregation (bagging) and feature sub-sampling. Each tree is trained on a bootstrap sample of the training data. At each candidate split, only a random subset of $m = \lfloor\sqrt{p}\rfloor$ features is considered. Predictions are pooled via majority voting or probability averaging.

### Q20: Why didn't we train Random Forest on the full 4,000 TF-IDF features?
**Answer:** The standard R `randomForest` implementation requires a dense numerical matrix. Converting a 4,000-column sparse matrix into dense format for 6,000 samples creates an immense memory footprint and results in hours of tree construction time. By combining all 32 numerical features with the top 50 highest-variance TF-IDF words (82 total features), Random Forest trained in just 3.17 seconds while achieving 99.33% accuracy.

### Q21: What is Naive Bayes, and what is its "naive" assumption?
**Answer:** Naive Bayes applies Bayes' Theorem to calculate posterior class probabilities $P(y \mid \mathbf{x})$. Its "naive" assumption is that all predictor features $x_1, x_2, \dots, x_p$ are conditionally independent given the class label:
$$P(\mathbf{x} \mid y = c) = \prod_{j=1}^p P(x_j \mid y = c)$$
While this assumption is violated in natural language (words are syntactically and semantically dependent), Naive Bayes remains a fast, effective baseline classifier.

### Q22: What is Laplace smoothing in Naive Bayes?
**Answer:** If a word or feature state never appears with a particular class in the training data, its conditional probability is zero ($P(x_j \mid y) = 0$). Because Naive Bayes multiplies all feature probabilities, a single zero cancels out all other probabilities. Laplace smoothing adds a pseudocount $\alpha = 1$ to the numerator and $K\alpha$ to the denominator to prevent zero-probability collapse.

---

## Category 5: Model Evaluation & Metrics

### Q23: Why is Accuracy alone an inadequate metric for evaluating classification models?
**Answer:** In imbalanced datasets, a trivial majority-class classifier can achieve high accuracy while completely failing to detect the minority class. Even in balanced datasets, accuracy obscures the asymmetric trade-off between False Positives and False Negatives. In fake news detection, False Positives (flagging real news as fake) undermine user trust, while False Negatives (allowing fake news to circulate) spread misinformation.

### Q24: Define Precision and Recall in the context of this project.
**Answer:**
- **Precision:** Of all articles the model predicted as Real News, what proportion was actually Real?
  $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
- **Recall (Sensitivity):** Of all actual Real News articles in the dataset, what proportion did the model correctly identify?
  $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

### Q25: What is the F1-Score, and why is it preferred for model ranking?
**Answer:** The F1-Score is the harmonic mean of Precision and Recall:
$$\text{F}_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
Unlike the arithmetic mean, the harmonic mean penalizes models that sacrifice Recall to achieve high Precision or vice versa. It reaches its maximum only when both metrics are high.

### Q26: What is a Confusion Matrix? Provide the actual results achieved by the best model in this project.
**Answer:** A Confusion Matrix is a tabular summary of prediction outcomes against ground truth. On 1,500 held-out test articles, our best model (Logistic Regression LASSO) achieved:
- **True Positives (Real as Real):** 747
- **True Negatives (Fake as Fake):** 746
- **False Positives (Fake as Real):** 4
- **False Negatives (Real as Fake):** 3
Total misclassifications: only 7 out of 1,500 articles (99.53% Accuracy).

### Q27: What is the Area Under the ROC Curve (ROC-AUC)?
**Answer:** The ROC curve plots the True Positive Rate against the False Positive Rate across all decision thresholds $\tau \in [0, 1]$. The Area Under the Curve (AUC) quantifies the probability that the classifier will rank a randomly chosen positive instance higher than a randomly chosen negative one. Our LASSO model achieved an AUC of **0.9983**, and Random Forest achieved **0.9999**.

---

## Category 6: Data Leakage & Ethical Machine Learning

### Q28: What is data leakage, and what major leakages were mitigated in this dataset?
**Answer:** Data leakage occurs when information from outside the training dataset is used to create the model. In this project, two critical leakages were identified and resolved:
1. **Publisher Datelines:** Real articles opened with `"WASHINGTON (Reuters) - "`. Stripping this artifact prevented models from learning the trivial shortcut that the word `"Reuters"` means real news.
2. **Duplicate Articles:** Removing 5,798 duplicate stories prevented identical articles from appearing in both the training and test sets.

### Q29: Why is stratified splitting necessary?
**Answer:** Stratified sampling (`caret::createDataPartition`) guarantees that both training and testing sets maintain the exact same proportion of Real and Fake articles (54.2% Real, 45.8% Fake) as the original corpus, preventing sampling bias.

### Q30: What is the difference between model-estimated confidence and factual truth?
**Answer:** A machine learning model computes a conditional probability $P(\text{Real} \mid \mathbf{x})$ based on statistical patterns learned from labelled historical training data. The model cannot independently verify factual reality. For example, a completely fabricated article written in immaculate journalistic prose may be classified as Real, while authentic citizen reporting with informal punctuation may be flagged as Fake. The application explicitly displays a disclaimer to clarify this distinction.

---

## Category 7: Python-R Integration & Architecture

### Q31: How does Python communicate with the R backend in this application?
**Answer:** Python executes R scripts as an operating system subprocess via `subprocess.run()`. Python writes user-submitted text to a temporary JSON file (`in.json`), invokes `Rscript.exe R_Scripts/10_prediction.R in.json out.json`, and reads the resulting analytical response from `out.json`.

### Q32: Why use temporary files rather than standard input (stdin) for JSON transfer?
**Answer:** News articles can span thousands of words containing special characters, quotes, and newlines. Passing large strings through shell arguments or standard input can encounter buffer overflow limits and character-encoding corruption on Windows. File-based JSON exchange ensures encoding consistency (UTF-8) and safe process isolation.

### Q33: How does the system prevent temporary file collisions during multiple concurrent predictions?
**Answer:** Temporary file names incorporate high-resolution millisecond timestamps and system process IDs:
```python
unique_id = f"fakenews_{int(time.time() * 1000)}_{os.getpid()}"
```
This guarantees distinct file paths for every request.

### Q34: What happens if the R backend fails or takes too long to execute?
**Answer:** The Python bridge enforces a strict 45-second execution timeout. If R crashes or hangs, Python catches `subprocess.TimeoutExpired` or the non-zero exit code, safely deletes the temporary files in a `finally:` block, and surfaces a user-friendly error message in Streamlit without crashing the web app.

### Q35: Could this system be deployed to a production cloud environment?
**Answer:** Yes. The decoupled architecture can be packaged into a Docker container with an R runtime and Python Streamlit. To scale horizontally, the file-based subprocess can be ported to an asynchronous Redis worker queue where multiple R worker containers consume prediction tasks from Streamlit frontends.
