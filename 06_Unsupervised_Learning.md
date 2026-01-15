# Chapter 6: Unsupervised Learning

Unsupervised learning detects patterns in unlabeled data. There is no $y$ target, only $X$.

---

## 1. Clustering

### K-Means Clustering
Partitions data into $K$ clusters.
*   **Algorithm**:
    1.  Initialize $K$ centroids randomly.
    2.  Assign points to nearest centroid.
    3.  Move centroid to the mean of assigned points.
    4.  Repeat until convergence.

#### Mathematics: Inertia
K-Means minimizes the **Within-Cluster Sum of Squares (WCSS)** or Inertia:
$$ J = \sum_{i=1}^{K} \sum_{x \in C_i} ||x - \mu_i||^2 $$
Where $\mu_i$ is the centroid of cluster $C_i$.

#### Practical Usage: The Elbow Method
How to pick $K$? Plot Inertia vs $K$. The "Elbow" is the point of diminishing returns (where variance reduction slows down).

### DBSCAN (Density-Based)
K-Means fails on non-spherical clusters (e.g., crescents). DBSCAN groups points that are close together (Density).
*   **Pros**: Finds outliers (Noise). No need to specify $K$.
*   **Cons**: Struggles with varying densities.

---

## 2. Dimensionality Reduction

### Principal Component Analysis (PCA)
Reduces dimensions (e.g., 100 features $\to$ 2 features) while keeping the most information.

#### Mathematical Intuition
1.  PCA finds the "Principal Components" (directions of maximum variance).
2.  It uses **Eigenvectors** and **Eigenvalues** of the Covariance Matrix.
    *   **Eigenvector**: The direction of usage.
    *   **Eigenvalue**: The magnitude (amount of variance).
3.  We project data onto the top eigenvectors.

### t-SNE and UMAP
PCA is linear. t-SNE and UMAP are non-linear manifold learning techniques.
*   **Use Case**: Visualization only. They preserve local structure (neighbors stay neighbors).
*   **Warning**: Distances in t-SNE plots are not always meaningful globally.

---

## 3. Association Rule Learning
Discover relationships like "People who buy diapers also buy beer".
*   **Support**: Frequency of itemset.
*   **Confidence**: Likelihood of B given A.
*   **Lift**: Ratio of observed support to expected support using independence.
    $$ Lift(A \to B) = \frac{Confidence(A \to B)}{Support(B)} $$
    *   Lift > 1: Positive correlation.

---
