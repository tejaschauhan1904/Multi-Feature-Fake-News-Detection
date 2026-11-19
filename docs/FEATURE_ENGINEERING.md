# Feature Engineering: Multi-Feature Fake News Detection

This document specifies the mathematical formulations, definitions, and rationales for all six feature categories implemented in `R_Scripts/05_feature_extraction.R` and `R_Scripts/06_tfidf_vectorization.R`.

---

## 1. Overview of the 32 Continuous Numerical Features

| Domain | Feature Identifier | Mathematical Definition / Calculation |
| :--- | :--- | :--- |
| **Statistical** | `char_count` | Total character count of full text: $L = \sum_{i} 1$ |
| **Statistical** | `word_count` | Total tokens in text: $W = \sum_{w} 1$ |
| **Statistical** | `sentence_count` | Number of terminal sentence boundaries: $S = \sum_{s} 1$ |
| **Statistical** | `unique_words_count` | Count of unique lowercase words: $U = |\text{Unique}(W)|$ |
| **Statistical** | `lexical_diversity` | Type-Token Ratio: $\text{TTR} = U / W$ |
| **Statistical** | `avg_words_per_sentence` | Sentence density: $\bar{w}_s = W / S$ |
| **Statistical** | `avg_chars_per_word` | Mean token length: $\bar{c}_w = \left(\sum |w_i|\right) / W$ |
| **Statistical** | `long_words_count` | Words with character length $> 6$ |
| **Statistical** | `short_words_count` | Words with character length $< 4$ |
| **Statistical** | `long_word_ratio` | $\text{Ratio}_{\text{long}} = \text{long\_words} / W$ |
| **Statistical** | `short_word_ratio` | $\text{Ratio}_{\text{short}} = \text{short\_words} / W$ |
| **Statistical** | `paragraph_count` | Count of non-empty newline separated paragraphs $P$ |
| **Statistical** | `avg_paragraph_length` | Mean characters per paragraph: $L / P$ |
| **Linguistic** | `uppercase_count` | Total uppercase characters: $C_{\text{upper}} = \sum [A-Z]$ |
| **Linguistic** | `uppercase_ratio` | Proportion of uppercase characters: $C_{\text{upper}} / L$ |
| **Linguistic** | `exclamation_count` | Total exclamation marks (`!`) |
| **Linguistic** | `question_count` | Total question marks (`?`) |
| **Linguistic** | `total_punct_count` | All punctuation characters matching `[[:punct:]]` |
| **Linguistic** | `punct_ratio` | Proportion of punctuation: $\text{Punct} / L$ |
| **Linguistic** | `url_count` | Links matching `https?://\|www\.` |
| **Linguistic** | `numeric_count` | Count of standalone numeric tokens matching `\b\d+\b` |
| **Linguistic** | `avg_sentence_length_chars` | Mean characters per sentence: $L / S$ |
| **Title Specific** | `title_upper_ratio` | Uppercase ratio in title alone: $C_{\text{title\_upper}} / L_{\text{title}}$ |
| **Title Specific** | `title_exclamations` | Exclamation mark count in headline |
| **Title Specific** | `title_questions` | Question mark count in headline |
| **Readability** | `flesch_reading_ease` | Flesch Reading Ease Formula (see Section 3) |
| **Sentiment** | `pos_words_count` | Positive word matches in Bing Lexicon: $N_{\text{pos}}$ |
| **Sentiment** | `neg_words_count` | Negative word matches in Bing Lexicon: $N_{\text{neg}}$ |
| **Sentiment** | `sentiment_score` | Net Sentiment Score: $\text{Score}_{\text{sent}} = N_{\text{pos}} - N_{\text{neg}}$ |
| **Sentiment** | `pos_word_ratio` | $N_{\text{pos}} / W$ |
| **Sentiment** | `neg_word_ratio` | $N_{\text{neg}} / W$ |
| **Sentiment** | `sentiment_balance` | Normalized polarity: $(N_{\text{pos}} - N_{\text{neg}}) / (N_{\text{pos}} + N_{\text{neg}} + \epsilon)$ |

---

## 2. Textual Features & TF-IDF Vectorization

The textual content of each article is transformed into a high-dimensional sparse vector using Term Frequency-Inverse Document Frequency (TF-IDF):

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

### A. Sublinear Term Frequency
To prevent exceptionally long articles from artificially dominating token counts, sublinear term frequency scaling is applied:

$$\text{TF}(t, d) = 1 + \ln(f_{t, d}) \quad \text{for } f_{t, d} > 0$$

### B. Inverse Document Frequency
Calculates the specificity of a term across the entire training corpus $D$:

$$\text{IDF}(t, D) = \ln\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$

### C. N-Gram Integration & Pruning
- Range: Unigrams ($n=1$) and Bigrams ($n=2$).
- Pruning Rules: Minimum term count $\ge 5$, document proportion between $0.002$ and $0.50$.
- Capped at top **4,000 most informative terms** to maintain optimal memory footprint and prevent overfitting.

---

## 3. Readability Formulation (Flesch Reading Ease)

The Flesch Reading Ease index measures text difficulty based on average sentence length and syllable count:

$$\text{Flesch} = 206.835 - 1.015 \left(\frac{\text{total words}}{\text{total sentences}}\right) - 84.6 \left(\frac{\text{total syllables}}{\text{total words}}\right)$$

### Syllable Estimation Heuristic:
A vowel-cluster heuristic counts diphthongs and vowels `[aeiouy]+`, adjusting for silent terminal `'e'`. Scores are bounded within $[0, 100]$:
- **90–100:** Very Easy (5th grade level)
- **60–70:** Standard English (8th–9th grade)
- **30–50:** Difficult (College level)
- **0–30:** Very Difficult / Academic Journal

Empirical observation shows that **Real news** averages **35–50** (complex grammatical structures and professional journalistic prose), while **Fake news** averages **55–75** (shorter, simplified sentences designed for viral reading).

---

## 4. Sentiment Feature Extraction

Sentiment is extracted using the **Bing Liu Sentiment Lexicon** (bundled via `tidytext`), containing 2,005 positive words and 4,781 negative words.

$$\text{Net Sentiment Score} = N_{\text{positive}} - N_{\text{negative}}$$

$$\text{Sentiment Balance} = \frac{N_{\text{positive}} - N_{\text{negative}}}{N_{\text{positive}} + N_{\text{negative}} + 10^{-5}}$$

*Note:* A negative sentiment score does not automatically denote fake news. Instead, the model observes that fake news displays bimodal extremes (either hyperbolic praise or inflammatory outrage), whereas real reporting adheres to more balanced neutral distributions.

---

## 5. Feature Integration & Z-Score Standardization

Before passing features to machine learning algorithms:

1. **Z-Score Normalization:** Continuous numerical features vary widely in scale (e.g. `char_count` $\in [100, 10000]$ vs `uppercase_ratio` $\in [0, 1]$). Each feature $j$ is standardized using the mean $\mu_j$ and standard deviation $\sigma_j$ calculated **strictly on the training partition**:
   $$z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j + \epsilon}$$
2. **Sparse Column Binding:** The standardized continuous feature matrix ($N \times 32$) is converted to a sparse `dgCMatrix` and column-bound to the sparse TF-IDF document-term matrix ($N \times 4000$) using `Matrix::cbind()`:
   $$\mathbf{X}_{\text{integrated}} = [\mathbf{X}_{\text{TF-IDF}} \;|\; \mathbf{Z}_{\text{numerical}}] \in \mathbb{R}^{N \times 4032}$$
