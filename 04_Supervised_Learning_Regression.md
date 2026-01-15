# Chapter 4: Supervised Learning - Regression

## 1. Linear Regression
Regression predicts a continuous value (e.g., Price, Temperature).
The Simplest model is **Linear Regression**: fitting a straight line to data.

### The Hypothesis
$$ h_\theta(x) = \theta_0 + \theta_1 x $$
*   $\theta_0$: Bias (Intercept)
*   $\theta_1$: Weight (Slope)

### The Cost Function (MSE)
We need to measure how "wrong" the line is. We use **Mean Squared Error (MSE)**. We add a $1/2$ factor to make the derivative cleaner (the 2 cancels out).
$$ J(\theta_0, \theta_1) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 $$

---

## 2. Gradient Descent (The Engine of ML)
To find the best $\theta$, we minimize $J(\theta)$. We use **Gradient Descent**: iteratively moving in the direction of steepest descent.

### Mathematical Derivation
We need the partial derivatives of the Cost Function with respect to each parameter.
1.  **For Bias ($\theta_0$)**:
    $$ \frac{\partial J}{\partial \theta_0} = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) $$
2.  **For Weight ($\theta_1$)**:
    $$ \frac{\partial J}{\partial \theta_1} = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x^{(i)} $$

### Update Rule
$$ \theta_j := \theta_j - \alpha \frac{\partial J}{\partial \theta_j} $$
Where $\alpha$ (Alpha) is the **Learning Rate**.
*   If $\alpha$ is too small: Learning is slow.
*   If $\alpha$ is too large: It may overshoot and diverge.

### Numerical Intuition: Use Case
Dataset: One point $(x=1, y=3)$.
Init: $\theta_0=0, \theta_1=1$. Learning Rate $\alpha = 0.1$.
1.  **Prediction**: $h(1) = 0 + 1(1) = 1$.
2.  **Error**: $(h - y) = 1 - 3 = -2$.
3.  **Gradients**:
    *   $\frac{\partial J}{\partial \theta_0} = -2$
    *   $\frac{\partial J}{\partial \theta_1} = -2 \times 1 = -2$
4.  **Updates**:
    *   $\theta_0 := 0 - 0.1(-2) = 0 + 0.2 = 0.2$
    *   $\theta_1 := 1 - 0.1(-2) = 1 + 0.2 = 1.2$
5.  **New Prediction**: $h(1) = 0.2 + 1.2(1) = 1.4$.
    *   We moved from prediction 1.0 to 1.4. We are getting closer to true value 3!

---

## 3. Optimizers: OLS vs Gradient Descent
*   **Gradient Descent**: Iterative. Good for large datasets.
*   **OLS (Ordinary Least Squares)**: Closed-form solution (Normal Equation).
    $$ \theta = (X^T X)^{-1} X^T y $$
    *   **Pros**: Exact solution, no $\alpha$ needed.
    *   **Cons**: Computing inverse $(X^T X)^{-1}$ is $O(n^3)$. Very slow for $>10,000$ features. Use GD for big data.

---

## 4. Evaluation Metrics: MSE vs MAE
*   **MSE (Mean Squared Error)**: Penalizes large errors heavily (due to squaring). Use when outliers are "bad" and should be avoided.
*   **MAE (Mean Absolute Error)**: Linear penalty. More robust to outliers.
*   **$R^2$ Score**: "Accuracy" for regression. $1.0$ is perfect, $0.0$ is baseline (predicting the mean).

---
