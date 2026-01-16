# Chapter 6: Unsupervised Learning

In Supervised Learning, we had a teacher (the target $y$). In **Unsupervised Learning**, we are on our own. It is the art of **Discovery**—finding hidden structures, groups, or patterns in data without being told what to look for.

---

## 0. The Business Reality of Unsupervised Learning
Before we dive into algorithms, let's talk about the **elephant in the room**: Unsupervised Learning is risky.

### No Labels = No Accuracy
*   In Supervised Learning, you can measure accuracy. In Unsupervised Learning, you don't have "correct answers."
*   **How do you know if your clustering is good?** You don't. You test it in the real world and see if it makes business sense.

### The Experimental Budget
*   **Example (Banking)**: A bank wants to segment customers into groups for targeted marketing. You run K-Means and get 5 clusters. Are they meaningful? You won't know until the marketing team runs campaigns for each cluster and measures ROI.
*   **The Risk**: You might spend months on clustering, only to find that the segments don't behave differently. The project might have **zero return on investment**.

### When Unsupervised Learning Shines
*   **Customer Segmentation**: Banks, e-commerce, and marketing teams use clustering to find "Emotional Buyers" vs "Price Hunters."
*   **Anomaly Detection**: Finding fraudulent transactions or defective products.
*   **Dimensionality Reduction**: Compressing high-dimensional data (like images or DNA sequences) for visualization or faster processing.

### The Data Decay Problem
Even if you gather behavioral data today, people's behavior changes over time. Your clusters might be outdated in 6 months.

**Critical Insight**: Always include an "Experimental Phase" in your project budget for unsupervised learning. You are exploring, not predicting.

---

## 1. Clustering: The Science of Grouping
Clustering is like walking into a room full of strangers and trying to group them by their hobbies, clothing, or behavior without anyone telling you who they are.

### A. K-Means: The Dance of the Centroids
K-Means is the most popular clustering algorithm. It partitions data into $K$ groups.

**The Intuition (The Dance)**:
1.  **Random Start**: You pick $K$ random people to be "Team Captains" (Centroids).
2.  **Assignment**: Every other person in the room goes to the captain they are closest to.
3.  **The Move**: Each captain looks at their team and moves to the exact center (the average) of where their teammates are standing.
4.  **Repeat**: People might change teams now that captains have moved. This "dance" continues until no one changes teams and captains stop moving.

**The Math (The Tightness Meter)**:
K-Means tries to minimize **Inertia** (Within-Cluster Sum of Squares). It wants the teams to be as "tightly packed" around their captain as possible.
$$ J = \sum_{i=1}^{K} \sum_{x \in C_i} ||x - \mu_i||^2 $$
*   $\mu_i$: The captain (centroid).
*   $||x - \mu_i||^2$: The squared distance of a team member from the captain.

### The Elbow Method: Finding the Right K
How do you choose $K$ (the number of clusters)? The **Elbow Method** helps.

1.  **Run K-Means for K = 1, 2, 3, ..., 10**.
2.  **Plot Inertia** (the "tightness" score) for each K.
3.  **Look for the "Elbow"**: The point where adding more clusters doesn't reduce Inertia much.

**Example**: If Inertia drops sharply from K=1 to K=3, then slowly from K=3 to K=10, the elbow is at **K=3**.

**Practical Tip**: If the elbow is unclear (e.g., two possible points at K=3 and K=5), try both and let the **domain expert** decide. For example, in retail, you might test both 3-segment and 5-segment marketing campaigns and see which performs better.

### When to Use K-Means?
*   **Large datasets**: K-Means is fast and scales well.
*   **Spherical clusters**: Works best when clusters are roughly circular.

### When NOT to Use K-Means?
*   **Weird shapes**: If clusters are crescent-shaped or intertwined, K-Means fails.
*   **Outliers**: A single outlier can pull a centroid far away, distorting the entire cluster.
*   **Categorical data**: K-Means uses distance, which doesn't work well with categorical features (e.g., "Red" vs "Blue"). Use **Frequency Encoding** or **One-Hot Encoding** first.

---

## 2. Hierarchical Clustering: The Family Tree
Hierarchical Clustering builds a **tree of clusters** (called a **Dendrogram**). It doesn't require you to specify K upfront.

### Agglomerative (Bottom-Up)
The most common approach:
1.  **Start**: Every data point is its own cluster.
2.  **Merge**: Find the two closest clusters and combine them.
3.  **Repeat**: Keep merging until you have one giant cluster.
4.  **Cut the Tree**: Draw a horizontal line across the dendrogram to get your desired number of clusters.

### The Dendrogram: Reading the Tree
*   **Height**: The vertical distance between merges shows how "different" the clusters are.
*   **Cutting**: The longest vertical line that doesn't intersect with other merges is a good place to cut.

### When to Use Hierarchical Clustering?
*   **Small to medium datasets**: It's computationally expensive (O(n³)).
*   **Exploratory analysis**: The dendrogram gives you a visual understanding of the data structure.
*   **High-dimensional data**: Works well with complex data like DNA sequences (Bioinformatics).

### Practical Tip: Use K-Means to Find K, Then Use Hierarchical
Some practitioners run K-Means with the Elbow Method to find the optimal K, then use Hierarchical Clustering for better interpretability.

---

## 3. DBSCAN: The Social Circle
### The Algorithm
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) looks for **Density**. It says: "If you have at least `minPoints` neighbors within `epsilon` distance, you are part of a crowd."

### Parameters
1.  **Epsilon (ε)**: The radius around each point. Think of it as "personal space."
2.  **MinPoints**: The minimum number of neighbors required to form a dense region (a cluster).

### Types of Points
*   **Core Point**: Has at least `minPoints` neighbors within `epsilon`.
*   **Border Point**: Within `epsilon` of a core point, but doesn't have enough neighbors to be a core point itself.
*   **Noise (Outlier)**: Not within `epsilon` of any core point. DBSCAN labels these as **-1**.

### When to Use DBSCAN?
*   **Weird shapes**: DBSCAN can find crescent-shaped, spiral, or arbitrary-shaped clusters.
*   **Noisy data**: It naturally handles outliers by labeling them as noise.
*   **Unknown K**: You don't need to specify the number of clusters upfront.

### When NOT to Use DBSCAN?
*   **Varying density**: If some clusters are very dense and others are sparse, DBSCAN struggles.
*   **High-dimensional data**: The "Curse of Dimensionality" makes distances meaningless.
*   **Parameter tuning**: Choosing `epsilon` and `minPoints` is hard. There's no "Elbow Method" for DBSCAN.

### Practical Tip: Start with Domain Knowledge
For `epsilon`, think about the physical meaning. For example, in geospatial data, if you're clustering cities, `epsilon` might be 50 km. For `minPoints`, start with 5 and adjust based on results.

---

## 4. The Kernel Trick for Clustering
Just like in SVMs, you can use the **Kernel Trick** to transform data into a higher-dimensional space where it's easier to cluster.

### The Idea
Imagine you have two intertwined spirals. In 2D, no clustering algorithm can separate them. But if you "lift" the data into 3D (using a kernel), you can slice through them with a plane.

### Kernel K-Means
Instead of clustering in the original space, you:
1.  **Compute the Kernel Matrix**: Use RBF, Polynomial, or other kernels.
2.  **Run K-Means on the Kernel Matrix**.

### Kernel DBSCAN
Similarly, you can apply DBSCAN to the kernel-transformed space.

### When to Use Kernel Clustering?
*   **Non-linearly separable data**: When standard clustering fails.
*   **Complex patterns**: For data with intricate structures.

### Trade-offs
*   **Computational Cost**: Kernel matrices can be huge (O(n²) memory).
*   **Parameter Sensitivity**: Choosing the right kernel and its parameters (e.g., γ for RBF) is critical.

---

## 5. Dimensionality Reduction: The Camera Angle
Imagine a 3D teapot. You want to take a 2D photograph of it. If you take the photo from the top, you just see a circle (you lost the spout and handle info). If you take it from the side, you see the whole shape.

**PCA (Principal Component Analysis)** is the algorithm that finds the "best camera angle" to squeeze 100 dimensions into 2 dimensions while losing as little information as possible.

### The Math: Eigenvectors and Eigenvalues
1.  **Variance**: This is the "information." We want to keep the most spread-out parts of the data.
2.  **Eigenvectors**: These are the directions of the "camera angles."
3.  **Eigenvalues**: This is the "importance" of each angle. A high eigenvalue means that angle captures a lot of the teapot's shape.

**Intuition**: PCA rotates your data so that the first axis (PC1) captures the most "stretch" (variance) in the data.

### t-SNE: The Neighborhood Preserver
While PCA is linear, **t-SNE** (t-Distributed Stochastic Neighbor Embedding) is non-linear. It's designed for **visualization**.

*   **Goal**: Keep similar points close together in the low-dimensional space.
*   **Use Case**: Visualizing high-dimensional data like images or word embeddings.
*   **Warning**: t-SNE is slow and doesn't preserve global structure (only local neighborhoods).

---

## 6. Association Rule Learning: The Supermarket Blueprint
This is how Amazon knows: *"People who bought this also bought..."*

### The Metrics of Relationships
1.  **Support**: How often does this combo happen? (e.g., How many people bought both Milk and Bread?)
2.  **Confidence**: If someone buys Milk, how likely are they to also buy Bread?
3.  **Lift (The "Magic" Factor)**: 
    *   If Bread is very popular, people might buy it anyway. 
    *   **Lift** tells you if the connection is *special*. 
    *   $Lift > 1$ means the items are physically linked in the customer's mind.
    $$ Lift(A \to B) = \frac{Confidence(A \to B)}{Support(B)} $$

---

## 7. Tips and Tricks for Unsupervised Learning
1.  **Always visualize first**: Use PCA or t-SNE to plot your data in 2D before clustering. This gives you intuition about the number of clusters.
2.  **Domain knowledge is king**: In unsupervised learning, there's no "correct answer." Always validate clusters with domain experts.
3.  **Try multiple algorithms**: Run K-Means, Hierarchical, and DBSCAN on the same data. Compare the results and choose the one that makes the most business sense.
4.  **Feature scaling is critical**: K-Means and Hierarchical Clustering use distances, so scale your features (Standardization or Min-Max).
5.  **Elbow Method is a guide, not a rule**: If the elbow is unclear, test multiple K values in production and measure business outcomes (e.g., campaign ROI).
6.  **DBSCAN for outlier detection**: If your goal is to find anomalies (fraud, defects), DBSCAN is excellent because it naturally labels outliers as noise.
7.  **Beware of the Curse of Dimensionality**: In high dimensions (>50 features), distances become meaningless. Use PCA to reduce dimensions first.
8.  **Categorical data needs encoding**: K-Means doesn't work with categorical features. Use Frequency Encoding, One-Hot Encoding, or Target Encoding.
9.  **Silhouette Score for validation**: Use the Silhouette Score to measure how well-separated your clusters are. A score close to 1 is good, close to 0 is bad.
10. **Budget for experimentation**: Unsupervised learning is exploratory. Include time and money for testing clusters in the real world.

---

## 8. Real-World Example: Customer Segmentation
A bank wants to segment customers for targeted marketing.

### Step 1: Data Preparation
*   Features: Age, Income, Spending, Loan History.
*   Scaling: Standardize all features.

### Step 2: Elbow Method
*   Run K-Means for K = 1 to 10.
*   Elbow appears at K = 5.

### Step 3: Interpret Clusters
*   **Cluster 1**: High Income, High Spending → Premium customers.
*   **Cluster 2**: High Income, Low Spending → Savers (target for investment products).
*   **Cluster 3**: Low Income, High Spending → Loan seekers.
*   **Cluster 4**: Middle Income, Middle Spending → Average customers.
*   **Cluster 5**: Low Income, Low Spending → Low engagement.

### Step 4: Business Validation
*   Marketing runs campaigns for each cluster.
*   Measures ROI for each segment.
*   Adjusts strategy based on results.

---
