# Chapter 5: Supervised Learning - Classification

In Regression, we predicted numbers (Price). In **Classification**, we predict labels (Spam vs. Not Spam). While regression finds a line that goes *through* the data, classification finds a boundary that **separates** the data.

---

## 1. Logistic Regression: The Probability Curve
Don't let the name fool you—it's for classification. If we used a standard line ($y = wx + b$) to predict labels, the line would eventually go above 1 or below 0, which makes no sense for probability.

### The Sigmoid "S-Curve"
We take our line and pass it through the **Sigmoid Function**. It squashes any number (from negative infinity to positive infinity) into a range between **0 and 1**.

$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$

*   **If $z$ is a large positive number**: $\sigma(z) \approx 1$.
*   **If $z$ is a large negative number**: $\sigma(z) \approx 0$.
*   **If $z$ is 0**: $\sigma(z) = 0.5$.
*   **Intuition**: The model isn't saying "This is a cat." It's saying "There is an 85% probability this is a cat."

### Threshold Tuning: The Business Decision
By default, we use a threshold of **0.5**. If the probability is above 0.5, we predict "1" (Positive). But this threshold is **not fixed**.

*   **Example 1 (Marketing)**: A business might ask for customers with a **70% probability** of buying. You set the threshold to 0.7.
*   **Example 2 (Medical Screening)**: In cancer detection, you might lower the threshold to **0.3** to catch more cases (even if it means more false alarms).

**Critical Insight**: Logistic Regression is a **probabilistic model**. You can return probabilities instead of hard labels, giving the business flexibility to decide.

### When Does Logistic Regression Fail?
*   **Non-linear Data**: If the decision boundary is curved or circular, Logistic Regression struggles. Unlike SVMs (which have the Kernel Trick), Logistic Regression only has the exponential (Sigmoid) function.
*   **Multiclass Problems**: For more than 2 classes, you need extensions like **Softmax Regression** or **One-vs-Rest**.

### The Loss Function: Cross-Entropy (Log Loss)
Instead of MSE, we use **Cross-Entropy Loss** (also called Log Loss):
$$ J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h(x^{(i)})) + (1 - y^{(i)}) \log(1 - h(x^{(i)})) \right] $$

*   **Why not MSE?** Because the Sigmoid function makes MSE non-convex (multiple local minima). Cross-Entropy is convex, which guarantees Gradient Descent will find the global minimum.

---

## 2. Evaluation Metrics: The Confusion Matrix
In classification, accuracy alone is misleading. We need to understand **where** the model is making mistakes.

### The Confusion Matrix
For a binary classification problem:

|  | **Predicted Positive** | **Predicted Negative** |
| :--- | :--- | :--- |
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

### Key Metrics
1.  **Accuracy**: $\frac{TP + TN}{TP + TN + FP + FN}$ — Overall correctness.
2.  **Precision**: $\frac{TP}{TP + FP}$ — Of all the "Positive" predictions, how many were correct?
    *   **Use Case**: Spam detection. You don't want to mark important emails as spam (minimize FP).
3.  **Recall (Sensitivity)**: $\frac{TP}{TP + FN}$ — Of all the actual "Positives," how many did we catch?
    *   **Use Case**: Cancer detection. You don't want to miss any cancer cases (minimize FN).
4.  **F1 Score**: $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ — The harmonic mean of Precision and Recall.

### The Precision-Recall Trade-off
*   **Increasing Precision** (by raising the threshold) will **decrease Recall**.
*   To improve both, you need more data, better features, or a better model.

### Practical Tip: The Harmonic Mean
Why use the harmonic mean for F1? Because it emphasizes **lower values**. If Precision is 90% but Recall is 10%, the arithmetic mean would be 50%, which is misleading. The harmonic mean (F1) would be much lower (~18%), reflecting the true imbalance.

---

## 3. K-Nearest Neighbors (KNN): The Lazy Learner
KNN is called a **lazy algorithm** because it doesn't "learn" a model during training. It just stores the data and calculates distances at prediction time.

### How It Works
1.  **Store all training data**.
2.  When a new point arrives, calculate the distance to **all** training points.
3.  Find the **K nearest neighbors**.
4.  **Vote**: The majority class among the K neighbors is the prediction.

### Distance Metrics
*   **Euclidean Distance**: $d = \sqrt{\sum (x_i - y_i)^2}$ — Straight-line distance.
*   **Manhattan Distance**: $d = \sum |x_i - y_i|$ — Grid-based distance (like walking in a city).
*   **Minkowski Distance**: Generalization of both (with parameter $p$).

### Why Use Odd K?
To **avoid ties**. If K=3, you can have 2 votes for "Cat" and 1 for "Dog." If K=4, you might have 2 votes each, leading to a tie.

### When to Use KNN?
*   **Non-linear data**: KNN can handle complex decision boundaries.
*   **Small datasets**: It's simple and effective.

### When NOT to Use KNN?
*   **Big data**: Calculating distances to millions of points is extremely slow.
*   **High-dimensional data**: The "Curse of Dimensionality" makes distances meaningless in high dimensions.

### Practical Tip: Feature Scaling is Critical
Since KNN relies on distances, features with larger ranges (e.g., Salary: 20,000-200,000) will dominate features with smaller ranges (e.g., Age: 20-80). **Always scale your features** using Standardization or Min-Max Scaling (from Chapter 2).

---

## 4. Decision Trees: The Messy Room Analogy
A Decision Tree is like a game of **20 Questions**. It asks a series of "Yes/No" questions to narrow down the answer. But how does it know which question to ask first? It looks for the question that reduces **Entropy** (Messiness) the most.

### A. Entropy (The Messiness Meter)
Imagine a room filled with red and blue balls.
*   **Case 1 (Pure)**: All balls are red. Entropy is **0**. (The room is perfectly organized).
*   **Case 2 (Impure)**: Half are red, half are blue. Entropy is **1**. (The room is a total mess).

**The Math**:
$$ H(S) = - \sum_{i=1}^{c} p_i \log_2(p_i) $$

### B. Information Gain (The Cleanup)
The tree calculates the Entropy *before* the split and *after* the split. The difference is the **Information Gain**. The tree chooses the question that gives it the biggest cleanup.

### C. Gini Impurity (The Wrong Guess)
Similar to Entropy but faster to calculate. It represents the probability that a random item would be labeled incorrectly if we guessed based on the distribution of labels in the node.
$$ Gini = 1 - \sum_{i=1}^{c} p_i^2 $$

### D. Overfitting: The Memorization Problem
Decision Trees can grow very deep and memorize the training data (overfitting). To prevent this:
*   **Max Depth**: Limit how deep the tree can grow.
*   **Min Samples Split**: Require a minimum number of samples before splitting a node.
*   **Pruning**: Cut off branches that don't improve performance on validation data.

### Practical Tip: Grid Search for Hyperparameters
Use **Grid Search** (trying many combinations of max_depth, min_samples_split, etc.) to find the best settings for your tree.

---

## 5. Support Vector Machines (SVM): The Wide Street
Imagine two groups of people standing in a field. You want to build a road between them. 
*   **Decision Boundary**: The center line/hyperplane of the road.
*   **Margin**: The width of the road.
*   **Support Vectors**: The people standing right at the edge of the road.

**Goal**: SVM tries to build the **widest street possible**. A wider street means the model is more confident and less likely to misclassify future data.

### The Kernel Trick: Lifting the Data
What if the groups are mixed in a circle? You can't draw a straight line.
*   **Intuition**: Imagine the circle is on a flat sheet of paper. You "lift" the center of the paper up (into 3D). Now you can slide a flat knife (a plane) under the lifted part to separate them. This "lifting" is what a **Kernel** (like RBF) does mathematically.

### When to Use SVM?
*   **High-dimensional data**: SVMs work well when you have many features.
*   **Clear margin of separation**: When the classes are well-separated.

### When NOT to Use SVM?
*   **Large datasets**: Training SVMs is slow (O(n²) to O(n³)).
*   **Noisy data with overlapping classes**: SVMs struggle when there's no clear boundary.

---

## 6. Ensemble Methods: The Wisdom of Crowds
Why use one model when you can use a hundred?

### A. Bagging (Wisdom of Crowds)
*   **Concept**: Train 100 trees on slightly different versions of the data. Let them vote.
*   **Analogy**: If you ask 100 people to guess how many jellybeans are in a jar, the average of their guesses is usually better than any single person's guess.
*   **Example**: **Random Forest**. It reduces overfitting (Variance).

### B. Boosting (Learning from Mistakes)
*   **Concept**: Train one tree. See where it was wrong. Train the next tree specifically to fix those errors.
*   **Analogy**: This is like a student practicing for an exam. They spend most of their time on the questions they got wrong last time.
*   **Example**: **XGBoost, Gradient Boosting**. It reduces underfitting (Bias).

---

## 7. Tips and Tricks for Classification
1.  **Always check class imbalance**: If 95% of your data is "Not Fraud," a model that always predicts "Not Fraud" will have 95% accuracy but is useless. Use **SMOTE** (Synthetic Minority Over-sampling) or adjust class weights.
2.  **Use the right metric**: Don't rely on accuracy alone. Use Precision, Recall, or F1 based on your business goal.
3.  **Threshold tuning for Logistic Regression**: Experiment with different thresholds (0.3, 0.5, 0.7) and plot a **Precision-Recall curve** to find the optimal balance.
4.  **Feature scaling for distance-based models**: KNN and SVM require scaled features. Decision Trees and Random Forests do not.
5.  **Cross-validation**: Always use k-fold cross-validation to ensure your model generalizes well to unseen data.
6.  **Ensemble when possible**: Random Forests and XGBoost almost always outperform single models. Start with them as a baseline.
7.  **Interpretability vs Performance**: Decision Trees are interpretable but less accurate. Neural Networks are accurate but black boxes. Choose based on your needs.

---
