# Project Overview: Multi-Feature Fake News Detection Using R Language

## 1. Introduction
The rise of digital journalism and algorithmically curated social media feeds has democratized information dissemination, but has simultaneously accelerated the propagation of deceptive, sensationalized, and factually misleading content ("Fake News"). Misinformation influences public health choices, electoral integrity, and economic markets. 

Automated fake news detection is an essential task within Applied Machine Learning and Natural Language Processing (NLP). Traditional computational approaches rely overwhelmingly on basic unigram bags-of-words or simple keyword heuristics. Such simplistic models are prone to domain overfitting, topical drift, and vulnerability to adversarial evasion.

This project, titled **"Multi-Feature Fake News Detection Using R Language"**, formulates a comprehensive multi-dimensional classification system. Rather than assessing text in isolation, it analyzes structural, linguistic, emotional, readability, and statistical properties of news articles alongside n-gram Term Frequency-Inverse Document Frequency (TF-IDF) distributions.

---

## 2. Problem Statement
Given an arbitrary news article containing an optional headline and article body, predict whether the content is statistically characteristic of:
1. **FAKE NEWS (Class 0):** Fabricated, misleading, or heavily distorted reporting.
2. **REAL NEWS (Class 1):** Verified, fact-checked reporting adhering to professional journalistic standards.

### Core Challenges:
- **Lexical Overlap:** Real and fake articles frequently discuss identical subjects (e.g., elections, conflicts, inflation) using overlapping vocabularies.
- **Dataset Artifacts & Leakage:** Datasets often contain source tags (such as Reuters datelines) that can lead to trivial shortcut learning if not purged during preprocessing.
- **Stylistic Signatures:** Fake news writers systematically employ emotional exaggeration, sensationalist punctuation (multiple exclamation marks), hyper-simplified sentence structures, and aggressive capitalization to evoke cognitive urgency.

---

## 3. Project Objectives
1. **R-Centric Analytical Backend:** Implement data loading, data cleaning, text normalization, multi-category feature extraction, TF-IDF vectorization, model training, cross-validation, and performance evaluation natively in the **R programming language**.
2. **Multi-Feature Engineering:** Extract and integrate 32 numerical features across four primary domains:
   - **Lexicon Sentiment:** Net sentiment scores, positive/negative word ratios, and polarity balance.
   - **Readability Statistics:** Flesch Reading Ease score, words per sentence, and characters per word.
   - **Linguistic Markers:** Uppercase ratios, exclamation/question counts, and punctuation frequency.
   - **Statistical Distributions:** Lexical diversity (type-token ratio), word length ratios, and paragraph lengths.
3. **High-Dimensional TF-IDF Representation:** Extract unigram and bigram document-term representations using `text2vec`, pruned to 4,000 informative terms.
4. **Machine Learning Model Comparison:** Train, evaluate, and benchmark three supervised learning algorithms:
   - Logistic Regression with LASSO Regularization (`glmnet`)
   - Random Forest Ensemble (`randomForest`)
   - Naive Bayes Classifier (`e1071`)
5. **Python Streamlit Interactive UI:** Build a responsive web dashboard in Python Streamlit communicating with R via isolated subprocess calls.
6. **Academic Transparency:** Communicate clearly that the model evaluates statistical patterns learned from labelled corpora rather than establishing absolute real-world factual truth.

---

## 4. Scope and Limitations
### In-Scope:
- English-language news articles.
- Textual, linguistic, structural, and sentimental feature analysis.
- Offline model evaluation with cross-validation on held-out test data.
- Modular JSON-based inter-process communication between R and Python.

### Out-of-Scope:
- Fact-checking against dynamic live databases (e.g. real-time web search or Wikipedia knowledge graphs).
- Analysis of audio, video, or image deepfakes.
- Multilingual translation and analysis.
