# Chapter 5: Supervised Learning - Classification

Classification predicts a category (e.g., Spam/Not Spam, Cat/Dog).

---

## 1. Logistic Regression
Despite the name, it's for **Classification**. It fits a line (like linear regression) but squashes the output between 0 and 1 Using the **Sigmoid Function**.

### The Sigmoid Function
$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$
*   If $z \to \infty, \sigma(z) \to 1$.
*   If $z \to -\infty, \sigma(z) \to 0$.
*   Decision Boundary: Usually threshold at 0.5.

---

## 2. Decision Trees
A flowchart-like structure. It splits data based on the feature that results in the highest **Information Gain**.

### Mathematics of Splitting
How does the tree choose the "Best" split? It minimizes **Impurity**.

#### 1. Entropy
Measure of randomness/disorder.
$$ H(S) = - \sum_{i=1}^{c} p_i \log_2(p_i) $$
*   0 = Pure node (all same class).
*   1 = High disorder (50/50 mix).

#### 2. Gini Impurity
Another metric, faster to compute (no log).
$$ Gini = 1 - \sum_{i=1}^{c} p_i^2 $$

### Numerical Intuition: Calculating Entropy
Dataset: 4 items: $[+, +, -, -]$.
1.  Total = 4.
2.  $p(+) = 2/4 = 0.5$.
3.  $p(-) = 2/4 = 0.5$.
4.  **Entropy**:
    $$ H = - (0.5 \log_2 0.5 + 0.5 \log_2 0.5) $$
    $$ H = - (0.5(-1) + 0.5(-1)) = -(-1) = 1 $$
    Result: 1 (Maximum Impurity).
    *If we split and get $[+, +]$ in one node, Entropy becomes 0.*

---

## 3. Support Vector Machines (SVM)
SVM finds the **Hyperplane** that separates classes with the maximum **Margin**.

### The Kernel Trick
Linear SVMs fail on circular data (e.g., a donut shape). We key the **Kernel Trick** to project data into higher dimensions where it becomes linearly separable.
*   **RBF Kernel** (Radial Basis Function): Infinite dimensional projection.
*   **Parameters**:
    *   `C`: Penalty for misclassification. High C = Overfitting (Strict). Low C = Smooth boundary.
    *   `Gamma`: Reach of a single training example. High Gamma = Close points have high weight (Complex).

---

## 4. Ensemble Methods

### Bagging (Bootstrap Aggregating)
*   **Concept**: Train multiple models independently on random subsets of data and vote.
*   **Example**: **Random Forest**.
    *   Reduces **Variance** (Overfitting).
    *   Parallelizable (Fast).

### Boosting
*   **Concept**: Train models sequentially. Each model fixes the errors of the previous one.
*   **Example**: **AdaBoost, XGBoost, LightGBM**.
    *   Reduces **Bias** (Underfitting).
    *   Sequential (Slower).
    *   Currently the state-of-the-art for tabular data.

---
