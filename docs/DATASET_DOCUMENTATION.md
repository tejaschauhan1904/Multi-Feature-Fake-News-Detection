# Dataset Documentation: ISOT Fake & Real News Corpus

This document provides a thorough audit of the primary dataset used for training, validation, and evaluation in this project.

---

## 1. Dataset Origin and Attribution
The dataset utilized is the widely cited **ISOT Fake News Dataset**, compiled by the Information Security and Object Technology (ISOT) Research Lab at the University of Victoria.

- **Real News Source:** Articles collected from Reuters.com between 2016 and 2017, representing authentic, verified reporting.
- **Fake News Source:** Articles flagged as untrustworthy or fabricated by fact-checking organizations (including PolitiFact) and scraped from unreliable websites during 2016–2017.

---

## 2. File Specifications & Schema

The dataset is partitioned into two primary CSV files inside the `dataset/` directory:

| Dataset File | File Size | Raw Record Count | Target Label | Class Designation |
| :--- | :--- | :--- | :--- | :--- |
| `Fake.csv` | 59.88 MB | 23,481 records | `0` | Fake News |
| `True.csv` | 51.10 MB | 21,417 records | `1` | Real News |
| **Combined Raw** | **110.98 MB** | **44,898 records** | - | - |

### Schema Breakdown:
1. `title` *(Character)*: The headline or title of the news story.
2. `text` *(Character)*: The full body text of the article.
3. `subject` *(Character)*: Topical category (e.g., `politicsNews`, `worldnews`, `Government News`, `US_News`, `left-news`).
4. `date` *(Character)*: The original publication date string (e.g., `"December 31, 2017"`).

---

## 3. Data Integrity & Missing Values Audit

Inspection in R (`02_data_loading.R`) confirmed that:
- **Missing (`NA`) Values:** 0 missing values detected across `title`, `text`, `subject`, and `date`.
- **Empty / Stub Strings:** All records contain valid non-empty text strings.

---

## 4. Critical Dataset Biases & Leakage Mitigation

A key academic strength of this project is its explicit identification and mitigation of dataset leakages that commonly compromise naive models.

### A. Deduplication (Removal of 5,798 Duplicate Articles)
In political journalism, syndicated stories or viral conspiracy theories are frequently copied verbatim across dozens of fringe blogs.
- If raw data is split naively without deduplication, identical articles appear in both training and test sets, artificially inflating accuracy.
- Script `03_data_preprocessing.R` performs exact deduplication on `(title, text)`.
- **Outcome:** Removed **5,798 duplicate records**, yielding a strictly unique corpus of **39,100 articles** (17,905 Fake, 21,195 Real).

### B. Publisher Dateline Leakage (The Reuters Artifact)
In the raw `True.csv`, articles typically open with wire dateline headers:
```
WASHINGTON (Reuters) - The U.S. Congress on Tuesday passed a bipartisan tax reform bill...
```
A classifier trained on this uncleaned text will simply learn that the word `"Reuters"` is a shortcut for Real News!

- Script `03_data_preprocessing.R` uses targeted regular expressions to strip leading dateline patterns:
  ```r
  str_replace(txt, "(?i)^[A-Z\\s,/.-]+?\\s*\\(Reuters\\)\\s*[-—–:]\\s*", "")
  ```
- This forces the models to learn genuine semantic, structural, and linguistic patterns rather than publisher trademarks.

### C. Exclusion of `subject` and `date`
Notice that in the raw data, subjects in `Fake.csv` are labelled `Government News` or `left-news`, while in `True.csv` they are labelled `politicsNews`. Using `subject` or `date` as a feature would represent severe data leakage, as real-world unseen articles do not come with labelled dataset categories. Consequently, `subject` and `date` are strictly excluded from model feature representations.
