# Machine Learning Comprehensive Syllabus

## Graduate-Level Review and Study Plan

-----

## Course Overview

This syllabus covers statistical foundations and machine learning concepts, providing a structured path from probability and statistics fundamentals through advanced ML algorithms and applications.

-----

# PART I: STATISTICAL FOUNDATIONS

-----

## Module 1: Descriptive Statistics (Week 1)

### Measures of Central Tendency

- Mean (arithmetic, geometric, harmonic)
- Median and mode
- Weighted averages
- When to use each measure

### Measures of Dispersion

- Range and interquartile range (IQR)
- Variance and standard deviation
- Coefficient of variation
- Mean absolute deviation

### Data Distribution Shape

- Skewness (positive, negative, symmetric)
- Kurtosis (leptokurtic, platykurtic, mesokurtic)
- Percentiles and quartiles
- Box plots and outlier detection using IQR

### Data Visualization for Statistics

- Histograms and frequency distributions
- Stem-and-leaf plots
- Q-Q plots for normality assessment

-----

## Module 2: Probability Fundamentals (Week 2)

### Basic Probability Concepts

- Sample spaces and events
- Probability axioms
- Complementary events
- Mutually exclusive vs. independent events

### Probability Rules

- Addition rule (union of events)
- Multiplication rule (intersection of events)
- Conditional probability P(A|B)
- Law of total probability

### Bayes’ Theorem

- Prior, likelihood, and posterior
- Bayesian updating
- Applications in classification
- Naive Bayes foundation

### Combinatorics

- Permutations and combinations
- Counting principles
- Applications in probability calculations

-----

## Module 3: Probability Distributions (Week 3)

### Discrete Distributions

- **Bernoulli Distribution**: Single trial, success/failure
- **Binomial Distribution**: n trials, k successes, probability mass function
- **Poisson Distribution**: Rare events, rate parameter λ
- **Geometric Distribution**: Trials until first success
- **Multinomial Distribution**: Multiple categories

### Continuous Distributions

- **Uniform Distribution**: Equal probability over interval
- **Exponential Distribution**: Time between events
- **Gamma Distribution**: Wait times, shape and rate parameters
- **Beta Distribution**: Probability of probabilities

### The Normal (Gaussian) Distribution

- **Bell Curve Properties**
  - Symmetric about the mean
  - Mean = Median = Mode
  - Asymptotic tails (never touches x-axis)
  - Total area under curve = 1
- **Parameters**
  - μ (mu): Population mean - center of distribution
  - σ (sigma): Population standard deviation - spread
- **Probability Density Function**
  - Formula: f(x) = (1/σ√2π) × e^(-(x-μ)²/2σ²)
  - Interpretation and applications

### Standard Normal Distribution (Z-Distribution)

- μ = 0, σ = 1
- Z-score calculation: Z = (X - μ) / σ
- Z-tables and probability lookup
- Converting between raw scores and Z-scores

-----

## Module 4: The Empirical Rule and Standard Deviation Theorems (Week 4)

### The Empirical Rule (68-95-99.7 Rule)

- **68%** of data falls within ±1 standard deviation of mean
- **95%** of data falls within ±2 standard deviations of mean
- **99.7%** of data falls within ±3 standard deviations of mean
- Applications in quality control and anomaly detection

### Chebyshev’s Theorem

- Applies to ANY distribution (not just normal)
- At least (1 - 1/k²) of data within k standard deviations
  - k=2: At least 75% within ±2σ
  - k=3: At least 89% within ±3σ
- When to use Chebyshev vs. Empirical Rule

### Central Limit Theorem (CLT)

- **Statement**: Sampling distribution of the mean approaches normal as n increases
- Works regardless of population distribution shape
- Requires sufficiently large sample size (typically n ≥ 30)
- Standard error of the mean: SE = σ / √n
- Applications in hypothesis testing and confidence intervals

### Law of Large Numbers

- Sample mean converges to population mean as n → ∞
- Weak vs. strong law
- Relationship to CLT
- Practical implications for sampling

-----

## Module 5: Statistical Inference (Week 5)

### Sampling and Estimation

- Point estimates vs. interval estimates
- Sampling distributions
- Standard error calculations
- Bias and consistency of estimators

### Confidence Intervals

- Interpretation of confidence level
- CI for means (known σ: Z-interval, unknown σ: t-interval)
- CI for proportions
- Margin of error and sample size determination
- Common confidence levels (90%, 95%, 99%)

### Hypothesis Testing Framework

- Null hypothesis (H₀) and alternative hypothesis (H₁)
- One-tailed vs. two-tailed tests
- Test statistics (Z-test, t-test)
- P-values and significance levels (α)
- Type I error (false positive) and Type II error (false negative)
- Statistical power (1 - β)

### Common Statistical Tests

- **One-sample t-test**: Compare sample mean to known value
- **Two-sample t-test**: Compare means of two groups
- **Paired t-test**: Compare related samples
- **Chi-square test**: Categorical data independence
- **ANOVA**: Compare means across multiple groups
- **F-test**: Compare variances

-----

## Module 6: Correlation and Regression Foundations (Week 6)

### Correlation Analysis

- **Pearson Correlation (r)**
  - Measures linear relationship strength
  - Range: -1 to +1
  - Interpretation guidelines
  - Assumptions: linearity, normality, homoscedasticity
- **Spearman Rank Correlation**
  - Non-parametric alternative
  - Based on ranked data
  - Robust to outliers
- **Correlation vs. Causation**
  - Confounding variables
  - Spurious correlations
  - Establishing causality

### Covariance

- Definition and formula
- Relationship to correlation
- Covariance matrices

### Simple Linear Regression

- Least squares method
- Regression coefficients interpretation
- R² (coefficient of determination)
- Residual analysis
- Assumptions of linear regression

-----

# PART II: MATHEMATICAL FOUNDATIONS FOR ML

-----

## Module 7: Linear Algebra Essentials (Week 7)

### Vectors and Matrices

- Vector operations (addition, scalar multiplication, dot product)
- Matrix operations (multiplication, transpose, inverse)
- Identity and diagonal matrices
- Tensor basics

### Eigenvalues and Eigenvectors

- Definition and geometric interpretation
- Characteristic equation
- Applications in PCA and spectral methods

### Matrix Decomposition

- Singular Value Decomposition (SVD)
- LU decomposition
- QR decomposition
- Applications in ML (dimensionality reduction, recommender systems)

-----

## Module 8: Calculus and Optimization (Week 8)

### Differential Calculus

- Derivatives and partial derivatives
- Gradient vectors
- Chain rule (foundation for backpropagation)
- Hessian matrix

### Optimization

- Convex vs. non-convex functions
- Local vs. global minima
- Gradient descent algorithm
- Variants: SGD, Mini-batch, Momentum, Adam, RMSprop
- Learning rate selection and scheduling
- Lagrange multipliers for constrained optimization

### Information Theory

- Entropy: H(X) = -Σ p(x) log p(x)
- Cross-entropy loss
- KL divergence
- Mutual information

-----

# PART III: MACHINE LEARNING

-----

## Module 9: Introduction to Machine Learning (Week 9)

### ML Paradigms

- Supervised learning (classification, regression)
- Unsupervised learning (clustering, dimensionality reduction)
- Semi-supervised learning
- Reinforcement learning overview

### The ML Pipeline

- Data collection and preprocessing
- Feature engineering and selection
- Model training and validation
- Evaluation and deployment

### Bias-Variance Tradeoff

- Underfitting vs. overfitting
- Model complexity and generalization
- Regularization introduction

-----

## Module 10: Supervised Learning - Regression (Week 10)

### Linear Regression (ML Perspective)

- Cost function (MSE)
- Gradient descent for linear regression
- Normal equation (closed-form solution)
- Feature scaling importance

### Regularized Regression

- **Ridge Regression (L2)**: Penalty on squared coefficients
- **Lasso Regression (L1)**: Penalty on absolute coefficients, feature selection
- **Elastic Net**: Combination of L1 and L2
- Regularization parameter (λ) selection

### Polynomial Regression

- Feature transformation
- Overfitting risks
- Cross-validation for degree selection

### Evaluation Metrics

- MSE, RMSE, MAE
- R² and Adjusted R²
- Residual analysis

-----

## Module 11: Supervised Learning - Classification (Week 11)

### Logistic Regression

- Sigmoid function
- Log-odds and decision boundary
- Binary cross-entropy loss
- Gradient descent for logistic regression
- Multiclass: One-vs-Rest, Softmax

### K-Nearest Neighbors (KNN)

- Distance metrics (Euclidean, Manhattan, Minkowski)
- Choosing optimal k
- Curse of dimensionality
- KNN for regression

### Naive Bayes Classifier

- Conditional independence assumption
- Gaussian, Multinomial, Bernoulli variants
- Text classification applications
- Laplace smoothing

### Classification Metrics

- Confusion matrix components (TP, TN, FP, FN)
- Accuracy, precision, recall, F1-score
- ROC curve and AUC
- Precision-recall curve
- Handling class imbalance (SMOTE, class weights)

-----

## Module 12: Tree-Based Methods (Week 12)

### Decision Trees

- Splitting criteria: Information gain, Gini impurity, entropy
- Tree construction algorithms (ID3, C4.5, CART)
- Pruning (pre-pruning, post-pruning)
- Handling categorical and numerical features
- Interpretability advantages

### Random Forest

- Bagging (bootstrap aggregating)
- Feature randomness
- Out-of-bag error estimation
- Feature importance
- Hyperparameters: n_estimators, max_depth, max_features

### Gradient Boosting

- Boosting concept: sequential weak learners
- Gradient descent in function space
- **AdaBoost**: Adaptive boosting
- **GBM**: Gradient boosting machines
- **XGBoost**: Regularization, parallel processing
- **LightGBM**: Histogram-based, leaf-wise growth
- **CatBoost**: Categorical feature handling

### Ensemble Techniques

- Voting classifiers (hard, soft)
- Stacking and blending
- Model diversity importance

-----

## Module 13: Support Vector Machines (Week 13)

### Linear SVM

- Maximum margin classifier
- Support vectors
- Hard margin vs. soft margin (C parameter)
- Hinge loss

### Kernel SVM

- Kernel trick for non-linear boundaries
- RBF (Gaussian) kernel
- Polynomial kernel
- Kernel selection guidelines
- Gamma parameter

### SVM for Regression (SVR)

- Epsilon-insensitive loss
- Applications

-----

## Module 14: Unsupervised Learning - Clustering (Week 14)

### K-Means Clustering

- Algorithm steps
- Convergence and local minima
- K-means++ initialization
- Choosing k: Elbow method, silhouette score
- Mini-batch K-means for large datasets

### Hierarchical Clustering

- Agglomerative (bottom-up)
- Divisive (top-down)
- Linkage methods: Single, complete, average, Ward
- Dendrograms and cluster interpretation

### Density-Based Clustering

- DBSCAN: Core points, border points, noise
- Eps and min_samples parameters
- Advantages for arbitrary shapes
- HDBSCAN for varying densities

### Gaussian Mixture Models

- Soft clustering (probabilistic assignments)
- Expectation-Maximization algorithm
- Model selection: AIC, BIC
- Comparison to K-means

### Clustering Evaluation

- Internal metrics: Silhouette score, Davies-Bouldin index
- External metrics: Adjusted Rand Index, NMI

-----

## Module 15: Dimensionality Reduction (Week 15)

### Principal Component Analysis (PCA)

- Variance maximization perspective
- Eigenvector approach
- SVD approach
- Explained variance ratio
- Choosing number of components
- Kernel PCA for non-linear reduction

### Other Techniques

- **Linear Discriminant Analysis (LDA)**: Supervised reduction
- **t-SNE**: Non-linear visualization (perplexity parameter)
- **UMAP**: Faster alternative to t-SNE
- **Independent Component Analysis (ICA)**: Signal separation

### Feature Selection

- Filter methods (correlation, mutual information)
- Wrapper methods (recursive feature elimination)
- Embedded methods (L1 regularization)

-----

## Module 16: Neural Networks Fundamentals (Week 16)

### Perceptron and MLP

- Single perceptron as linear classifier
- Multi-layer architecture
- Universal approximation theorem

### Activation Functions

- Sigmoid and vanishing gradient problem
- Tanh
- ReLU and variants (Leaky ReLU, ELU, SELU)
- Softmax for multi-class output

### Training Neural Networks

- Forward propagation
- Backpropagation algorithm derivation
- Loss functions: MSE, cross-entropy
- Weight initialization (Xavier, He)

### Regularization Techniques

- L2 regularization (weight decay)
- Dropout
- Batch normalization
- Early stopping

-----

## Module 17: Deep Learning Architectures (Week 17-18)

### Convolutional Neural Networks (CNNs)

- Convolution operation and filters
- Pooling layers (max, average)
- Stride and padding
- Classic architectures: LeNet, AlexNet, VGG, ResNet, Inception
- Transfer learning and fine-tuning

### Recurrent Neural Networks (RNNs)

- Sequential data processing
- Vanishing/exploding gradient problem
- LSTM: Gates (forget, input, output), cell state
- GRU: Simplified gating mechanism
- Bidirectional RNNs
- Sequence-to-sequence models

### Advanced Architectures

- Autoencoders (vanilla, denoising, sparse)
- Variational Autoencoders (VAEs)
- Generative Adversarial Networks (GANs)
- Attention mechanisms
- Transformer architecture basics

-----

## Module 18: Specialized Topics (Week 19-20)

### Time Series Analysis

- Components: Trend, seasonality, cyclical, noise
- Stationarity testing (ADF test)
- Differencing and transformations
- ARIMA models
- Exponential smoothing
- Prophet
- Deep learning for time series (LSTM, Transformer)

### Natural Language Processing

- Text preprocessing: Tokenization, stemming, lemmatization
- Bag of words and TF-IDF
- Word embeddings: Word2Vec (CBOW, Skip-gram), GloVe
- Sentiment analysis
- Topic modeling (LDA)
- Introduction to BERT and transformers for NLP

### Recommender Systems

- Collaborative filtering (user-based, item-based)
- Matrix factorization (SVD, ALS)
- Content-based filtering
- Hybrid approaches
- Cold start problem
- Evaluation: RMSE, precision@k, recall@k

-----

## Module 19: Advanced Topics (Week 21-22)

### Reinforcement Learning

- Markov Decision Processes
- Value functions and Q-values
- Q-learning
- Policy gradient methods
- Deep Q-Networks (DQN) introduction

### Model Interpretability

- Feature importance (permutation, tree-based)
- LIME (Local Interpretable Model-agnostic Explanations)
- SHAP (SHapley Additive exPlanations)
- Partial dependence plots
- Counterfactual explanations

### Handling Real-World Challenges

- Missing data: Imputation strategies (mean, median, KNN, MICE)
- Outlier detection: IQR, Z-score, Isolation Forest
- Imbalanced data: Oversampling (SMOTE), undersampling, class weights
- Data leakage prevention
- Feature engineering best practices

-----

## Module 20: MLOps and Production (Week 23-24)

### Model Deployment

- Model serialization (pickle, joblib, ONNX)
- REST APIs with Flask/FastAPI
- Docker containerization
- Cloud deployment (AWS, GCP, Azure)

### Model Monitoring

- Data drift detection
- Model performance monitoring
- A/B testing
- Retraining strategies

### Ethics and Responsible AI

- Bias detection and mitigation
- Fairness metrics
- Privacy-preserving ML (differential privacy, federated learning)
- Model documentation and transparency

-----

## Recommended Resources

### Textbooks

- “Pattern Recognition and Machine Learning” - Bishop
- “The Elements of Statistical Learning” - Hastie, Tibshirani, Friedman
- “Deep Learning” - Goodfellow, Bengio, Courville
- “Hands-On ML with Scikit-Learn, Keras, TensorFlow” - Géron
- “Introduction to Statistical Learning” - James, Witten, Hastie, Tibshirani

### Online Courses

- Stanford CS229 (Andrew Ng)
- Fast.ai Practical Deep Learning
- MIT 6.S191 Introduction to Deep Learning
- Khan Academy (Statistics foundations)

-----

## 24-Week Study Plan Summary

|Weeks|Part                   |Focus                                                        |
|-----|-----------------------|-------------------------------------------------------------|
|1-6  |Statistical Foundations|Descriptive stats, probability, distributions, CLT, inference|
|7-8  |Math Foundations       |Linear algebra, calculus, optimization                       |
|9-13 |Supervised Learning    |Regression, classification, trees, SVM                       |
|14-15|Unsupervised Learning  |Clustering, dimensionality reduction                         |
|16-18|Deep Learning          |Neural networks, CNNs, RNNs                                  |
|19-20|Specialized Topics     |Time series, NLP, recommenders                               |
|21-22|Advanced Topics        |RL, interpretability, real-world challenges                  |
|23-24|MLOps                  |Deployment, monitoring, ethics                               |

-----

## Capstone Project Ideas

- End-to-end predictive analytics pipeline
- Image classification with interpretability analysis
- NLP sentiment analysis system
- Recommendation engine with A/B testing framework
- Time series forecasting with ensemble methods
- Fraud detection with imbalanced data handling

-----

*Adjust pacing based on your familiarity with each topic. The statistical foundations in Part I are essential for understanding the “why” behind ML algorithms.*