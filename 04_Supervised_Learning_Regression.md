# Chapter 4: Supervised Learning - Regression

In classification, we predict "What" (e.g., Cat or Dog?). In **Regression**, we predict "How Much" (e.g., How much will this house cost?). Regression is the art of finding a mathematical relationship between inputs and a continuous numerical output.

---

## 1. Linear Regression: The Straight Line
The most basic form of regression is fitting a straight line through your data points. 

### The Hypothesis ($h_\theta$)
Think of this as the "Model's Formula." 
$$ h_\theta(x) = \theta_0 + \theta_1 x $$

*   **$\theta_0$ (Bias/Intercept)**: Where the line hits the vertical axis. If $x$ is 0, what is our starting guess?
*   **$\theta_1$ (Weight/Slope)**: How much the output ($y$) changes for every 1-unit increase in $x$.

---

## 2. The Cost Function (The "Mistake" Meter)
We need a way to tell the model: *"You are 20% wrong."* We use the **Mean Squared Error (MSE)**.

### Why do we square the errors?
$$ J(\theta_0, \theta_1) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 $$

1.  **Eliminate Negatives**: If one prediction is +5 off and another is -5 off, adding them would give 0 error (perfect), which is a lie! Squaring makes them both +25.
2.  **Punish "Big" Mistakes**: Squaring an error of 2 gives 4. Squaring an error of 10 gives **100**. This forces the model to prioritize fixing large errors.

### MSE vs MAE: Which Loss to Use?
Before we optimize, we need to choose the right "Mistake Meter."

| Metric | Formula | When to Use | Why? |
| :--- | :--- | :--- | :--- |
| **MSE** | $\frac{1}{m} \sum (y - \hat{y})^2$ | Default for most cases | Smooth (differentiable), punishes large errors heavily |
| **MAE** | $\frac{1}{m} \sum |y - \hat{y}|$ | When you have outliers | Robust to outliers, but **not smooth** (sharp corner at 0) |

**Critical Insight**: MAE is not differentiable at 0 (the "sharp edge"). This makes it harder to use with Gradient Descent because the slope is undefined at that point. MSE is smooth everywhere, which is why it's the default.

---

## 3. Two Ways to Find the Best Line
There are two main approaches to finding the optimal parameters.

### A. Gradient Descent (The Iterative Climber)
How does the model find the best $\theta_0$ and $\theta_1$? It uses **Gradient Descent**.

#### The Analogy: The Foggy Mountain
Imagine you are standing on top of a mountain in thick fog. You want to reach the village at the very bottom (the point where Error is 0).
1.  You can't see the village, but you can feel the **slope** of the ground under your feet.
2.  You take a step in the direction where the ground goes **down** most steeply.
3.  You repeat this until the ground is flat—you've reached the bottom!

#### The Math (The Slope)
We use **Partial Derivatives** to calculate that "slope" for each parameter. It tells us how much the Total Error ($J$) changes if we wiggle $\theta$ just a little bit.

1.  **Slope for $\theta_0$**: $\frac{1}{m} \sum (h - y)$
2.  **Slope for $\theta_1$**: $\frac{1}{m} \sum (h - y) \cdot x$

#### The Update Rule
$$ \theta_j := \theta_j - \alpha \cdot (\text{Slope}) $$
*   **$\alpha$ (Learning Rate)**: This is your **Step Size**. 
    *   Small step? You'll reach the bottom eventually, but it takes forever.
    *   Huge step? You might accidentally jump over the valley and land on the opposite mountain!

### B. OLS (Ordinary Least Squares) - The Direct Solution
Instead of taking many small steps, we can solve for the best parameters **directly** using the **Normal Equation**.

#### The Math (Closed-Form Solution)
$$ \theta = (X^T X)^{-1} X^T y $$

*   **Advantage**: You get the answer in **one calculation**. No need to tune learning rate or run many iterations.
*   **Disadvantage**: If you have millions of features, computing $(X^T X)^{-1}$ becomes extremely slow (matrix inversion is expensive).

#### When to Use Which?
*   **Use OLS** if you have a small dataset (< 10,000 rows, < 100 features).
*   **Use Gradient Descent** if you have a large dataset or need to scale to millions of rows.

---

## 4. Numerical Intuition Walkthrough (Gradient Descent)
Let's see the engine in action.
*   **Data**: One point $(x=1, y=3)$.
*   **Current Model**: $\theta_0 = 0, \theta_1 = 1$. (Line is $y = 1x + 0$).
*   **Target**: We want the model to predict 3.

1.  **Predict**: $h(1) = 0 + 1(1) = \mathbf{1}$.
2.  **Error**: $1 - 3 = \mathbf{-2}$. (We are under-predicting).
3.  **Calculate Update** (using $\alpha = 0.1$):
    *   $\theta_0$ update: $0 - [0.1 \times (-2)] = \mathbf{0.2}$.
    *   $\theta_1$ update: $1 - [0.1 \times (-2 \times 1)] = \mathbf{1.2}$.
4.  **New Model**: $y = 1.2x + 0.2$.
5.  **New Predict**: $h(1) = 0.2 + 1.2(1) = \mathbf{1.4}$.

**Observation**: We moved from prediction 1.0 toward 3.0. The "Mistake Meter" (Loss) is shrinking!

### Why is Linear Regression Fast?
With only **two parameters** ($\theta_0, \theta_1$), the fitting process is extremely fast. Even with Gradient Descent, you typically reach the optimal solution in just a few hundred iterations.

---

## 5. Evaluation: How good is our model?
*   **MAE (Mean Absolute Error)**: The average distance between predicted and actual. *"On average, we are off by $500."* Easy for humans to understand.
*   **MSE (Mean Squared Error)**: Used for training because the math (derivatives) is smoother. Penalizes large errors heavily.
*   **$R^2$ Score (Coefficient of Determination)**: 
    *   **1.0**: Perfect model.
    *   **0.0**: As good as just guessing the average every time.
    *   **Negative**: Worse than just guessing the average!

### Practical Tip: $R^2$ as "Accuracy"
Since $R^2$ is between 0 and 1, many people call it the "accuracy" of a regression model. However, this is a simplification. $R^2$ tells you **how much variance** your model explains, not how "correct" it is in absolute terms.

---

## 6. Tips and Tricks for Regression
1.  **Always visualize your data first**: Plot $x$ vs $y$ to see if a straight line makes sense. If the data is curved, you need Polynomial Regression (covered in advanced topics).
2.  **Check for outliers**: A single extreme value can throw off your entire line. Use Z-Score or IQR (from Chapter 2) to detect them.
3.  **Feature Scaling**: If you have multiple features with very different ranges (e.g., Age: 20-80, Salary: 20,000-200,000), scale them first. Otherwise, Gradient Descent will take forever to converge.
4.  **Learning Rate Tuning**: Start with $\alpha = 0.01$ and adjust. If the loss is increasing, your learning rate is too high. If it's decreasing very slowly, it's too low.
5.  **Use OLS for small datasets**: If you have fewer than 10,000 rows, the Normal Equation is faster and more accurate than Gradient Descent.

---
