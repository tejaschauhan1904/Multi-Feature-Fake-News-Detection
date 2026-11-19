# Machine Learning Models: Multi-Feature Fake News Detection

This document provides a technical description of the three supervised machine learning algorithms trained in R, their theoretical formulations, hyperparameter tuning, and implementation specifics.

---

## Model 1: Logistic Regression with LASSO (Primary Model)

### 1. Mathematical Formulation
Logistic regression models the posterior probability that a news article $i$ belongs to the **Real News** class ($y_i = 1$) given its combined feature vector $\mathbf{x}_i \in \mathbb{R}^{4032}$:

$$P(y_i = 1 \mid \mathbf{x}_i) = \sigma(\beta_0 + \mathbf{x}_i^T \boldsymbol{\beta}) = \frac{1}{1 + e^{-(\beta_0 + \mathbf{x}_i^T \boldsymbol{\beta})}}$$

To handle the high dimensionality of 4,032 features and prevent collinearity between adjacent bigrams, **LASSO (Least Absolute Shrinkage and Selection Operator, $L_1$ Regularization)** is applied:

$$\min_{\beta_0, \boldsymbol{\beta}} \left\{ -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln \sigma(\beta_0 + \mathbf{x}_i^T \boldsymbol{\beta}) + (1 - y_i) \ln(1 - \sigma(\beta_0 + \mathbf{x}_i^T \boldsymbol{\beta})) \right] + \lambda \sum_{j=1}^p |\beta_j| \right\}$$

### 2. Implementation in R (`glmnet`)
- **Package:** `glmnet::cv.glmnet`
- **Family:** `"binomial"`
- **Alpha:** `1.0` (pure LASSO penalty)
- **Cross-Validation:** 5-fold cross-validation with measure `"auc"`.
- **Lambda Selection:** The optimal regularizer $\lambda_{\text{1se}}$ is selected automatically (the largest $\lambda$ within one standard error of the minimum MSE), enforcing sparsity by setting uninformative word weights strictly to zero.
- **Why it is the Primary Model:** LASSO scales to tens of thousands of sparse features in seconds, avoids overfitting, and produces clean probability estimates.

---

## Model 2: Random Forest Classifier

### 1. Theoretical Formulation
Breiman's Random Forest is an ensemble learning method that constructs a collection of de-correlated decision trees $\{T_b\}_{b=1}^B$ during training. For binary classification:

$$\hat{C}_{\text{rf}}^B(\mathbf{x}) = \text{majority\_vote}\left( \{\hat{C}_b(\mathbf{x})\}_{b=1}^B \right)$$

$$\hat{P}(y = 1 \mid \mathbf{x}) = \frac{1}{B} \sum_{b=1}^B I(\hat{C}_b(\mathbf{x}) = 1)$$

At each split within each tree, a random subset of $m = \lfloor\sqrt{p}\rfloor$ candidate features is evaluated using the Gini Impurity criterion:

$$I_G(t) = 1 - \sum_{k=0}^1 p(k \mid t)^2$$

### 2. Feature Selection Rationale for Random Forest
Standard R `randomForest` requires a dense matrix representation. Allocating a dense $6,000 \times 4,000$ matrix for 100 trees requires immense RAM and can cause tree construction to take hours. 

To solve this responsibly without losing predictive power:
- The Random Forest model is trained on an **optimized representation** combining:
  1. All **32 standardized continuous numerical features** (sentiment, readability, linguistic, statistical).
  2. The **top 50 highest-variance TF-IDF words** extracted from the document-term matrix.
- Total features: 82 features.
- Trees: $B = 100$.
- Result: Trains in **~3.2 seconds** while achieving **99.33% accuracy** on the held-out test set!

---

## Model 3: Naive Bayes Classifier

### 1. Theoretical Formulation
The Naive Bayes classifier is a generative probabilistic model based on Bayes' Theorem:

$$P(y = c \mid \mathbf{x}) = \frac{P(y = c) P(\mathbf{x} \mid y = c)}{P(\mathbf{x})}$$

It enforces the conditional independence assumption that features are mutually independent given the class label:

$$P(\mathbf{x} \mid y = c) = \prod_{j=1}^p P(x_j \mid y = c)$$

For continuous features, Gaussian conditional distributions are modeled:

$$P(x_j \mid y = c) = \frac{1}{\sqrt{2\pi \sigma_{jc}^2}} \exp\left( -\frac{(x_j - \mu_{jc})^2}{2\sigma_{jc}^2} \right)$$

### 2. Implementation in R (`e1071`)
- **Package:** `e1071::naiveBayes`
- **Laplace Smoothing:** Set to $1.0$ to prevent zero-probability collapse on rare tokens.
- **Input Features:** Evaluated over the same 82-feature representation as Random Forest.
- **Role in Project:** Serves as a classical baseline classifier, achieving **89.60% accuracy** and highlighting the relative performance boost achieved by regularized regression and ensemble methods.

---

## Hyperparameter Summary Table

| Model Architecture | R Implementation | Key Hyperparameters | Input Dimensions | Training Time |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | `glmnet::cv.glmnet` | $\alpha = 1$, 5-fold CV, $\lambda_{\text{1se}} = 0.00047$ | 4,032 features (sparse) | ~4.65 seconds |
| **Random Forest** | `randomForest::randomForest` | $B = 100$ trees, $m = \lfloor\sqrt{82}\rfloor = 9$ | 82 features (dense) | ~3.17 seconds |
| **Naive Bayes** | `e1071::naiveBayes` | Laplace $= 1$, Gaussian likelihood | 82 features (dense) | ~0.04 seconds |
