# Systematic Machine Learning & Statistical Arbitrage Factor Models

This repository contains high-performance, vectorized implementations of core statistical learning algorithms. All models are developed inside **Jupyter Notebooks using mathematically generated synthetic datasets** to isolate and evaluate algorithmic performance under controlled statistical parameters.

Rather than treating machine learning architectures as black boxes, this repo emphasizes **fundamental numerical optimization from scratch using NumPy**.


### 1. Unsupervised Regime Classification & State-Space Clustering
* **Module Path:** `src/clustering/`
* **Quantitative Translation:** Partitions time-series data into unobserved, discrete market regimes (e.g., Low Volatility/Mean-Reverting vs. High Volatility/Trending states) to scale trade sizing.
* **Implementation Highlights:**
  * Implements **K-Means Clustering** as a geometric baseline partition mechanism.
  * Upgrades to **Gaussian Mixture Models (GMM)** to execute probabilistic, soft-clustering assignments via the Expectation-Maximization (EM) algorithm. This soft-clustering framework outputs a continuous probability density array, allowing portfolio position sizes to scale smoothly rather than triggering jarring binary portfolio allocations.



### 2. Penalized Regressions for Alpha Signal Selection
* **Module Path:** `src/regressions/`
* **Quantitative Translation:** Blends arrays of highly correlated indicators while mitigating multi-collinearity and overfitting.
* **Implementation Highlights:**
  * Implements **L1 regularization (Lasso)** to enforce sparse weight coefficients, driving uninformative or redundant alpha variables directly to zero for automatic feature selection.
  * Employs **L2 regularization (Ridge)** to shrink cross-sectional exposure weights, stabilizing model variance across volatile data spaces.
  * Formulates bounded **Logistic Regression classifiers** optimizing binary cross-entropy loss via vectorized gradient descent to predict binary directional asset trajectories.


### 3. Dimensionality Reduction & Factor Extraction
* **Module Path:** `src/dimensionality/`
* **Quantitative Translation:** Extracts latent risk factors from high-dimensional asset panels (such as stock universes or yield curves).
* **Implementation Highlights:** 
  * Extracts principal components directly via Singular Value Decomposition (`np.linalg.svd`) on empirical matrix arrays to isolate structural variances.
  * Formulates factor mappings to identify eigenvectors corresponding to **Market (Level), Industry (Slope), and Style (Twist)** dynamics.
  * Extends into non-linear domain spaces using custom RBF Kernel PCA transformations to track structural state shifts when linear projections warp under high volatility.
---


