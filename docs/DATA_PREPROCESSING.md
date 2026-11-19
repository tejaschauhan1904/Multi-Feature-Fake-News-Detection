# Data Preprocessing: Multi-Feature Fake News Detection

This document provides a detailed walkthrough of the text normalization and preprocessing procedures implemented in `R_Scripts/03_data_preprocessing.R` and `R_Scripts/04_text_preprocessing.R`.

---

## 1. Dual-Representation Strategy

A central requirement of multi-feature NLP is maintaining two distinct textual representations for every article:

1. **Original Text Representation:** Preserves raw case, punctuation, sentence boundaries, and numbers. Used exclusively to compute:
   - Uppercase character ratios (all-caps detection)
   - Exclamation mark and question mark frequency
   - Flesch Reading Ease readability scores
   - Sentence counts and paragraph segmentation
2. **Normalized NLP Representation:** Cleaned, lowercased, stripped of punctuation, numbers, and stopwords. Used exclusively to compute:
   - Sparse TF-IDF document-term matrices
   - Vocabulary frequency distributions
   - Unigram and bigram n-gram generation

---

## 2. Step-by-Step Normalization Pipeline

The function `clean_text_for_nlp(text, remove_stopwords = TRUE, remove_numbers = TRUE)` in `04_text_preprocessing.R` executes the following sequence:

### Step 1: Lowercase Conversion
Normalizes letter casing so identical words with different capitalization map to the same token.
- *Input:* `"BREAKING NEWS: The President Speaks Today."`
- *Output:* `"breaking news: the president speaks today."`

### Step 2: HTML Tag Removal
Strips any embedded markup tags inherited from web scraping using regex `<[^>]+>`.
- *Input:* `"<p>Official report announced <b>today</b>.</p>"`
- *Output:* `" Official report announced  today . "`

### Step 3: URL & Hyperlink Handling
Removes web URLs matching `https?://\S+|www\.\S+`.
- *Input:* `"Read the full article at https://news.example.com/report now!"`
- *Output:* `"Read the full article at   now!"`

### Step 4: Number & Digit Removal
Strips standalone numeric strings matching `\b\d+\b` and digits `\d+` so models do not overfit to specific years or financial figures.
- *Input:* `"Allocated $100M budget for the 2024 fiscal cycle."`
- *Output:* `"Allocated $M budget for the  fiscal cycle."`

### Step 5: Punctuation Handling
Replaces all non-alphabetic characters `[^a-z\s]` with whitespace, retaining clean words.
- *Input:* `"shocking, corrupt, un-believable!!!"`
- *Output:* `"shocking  corrupt  un believable   "`

### Step 6: Whitespace Normalization
Collapses repeated spaces, tabs, and newlines into single spaces using `stringr::str_squish()`.
- *Input:* `"breaking    news   announces   policy  "`
- *Output:* `"breaking news announces policy"`

### Step 7: Stopword Removal
Filters out standard high-frequency English stopwords (`tm::stopwords("en")`) that carry minimal topical distinction (e.g., `"the"`, `"and"`, `"is"`, `"which"`), while retaining single-letter exclusions.
- *Input:* `"the government announces a new policy for the nation"`
- *Output:* `"government announces new policy nation"`

---

## 3. Tokenization: Unigrams and Bigrams

The preprocessed text is tokenized into two levels of granularity:

1. **Unigram Tokens (Single words):**
   ```
   "breaking", "news", "government", "announces", "new", "policy"
   ```
2. **Bigram Tokens (Adjacent word pairs):**
   ```
   "breaking news", "news government", "government announces", "announces new", "new policy"
   ```

### Why Bigrams Matter in Fake News Detection:
Bigrams capture linguistic nuance and contextual collocations that unigrams miss entirely. For example:
- Unigrams: `"breaking"` and `"news"` are common across all journalism.
- Bigrams: `"breaking news"` in headlines is 4.2x more frequent in sensationalist clickbait.
- Collocations like `"mainstream media"`, `"secret plot"`, `"viral video"`, and `"white house"` carry strong stylistic signals.
