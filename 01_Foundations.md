# Chapter 1: Foundations of Machine Learning

## 1. Introduction to Machine Learning
Machine Learning (ML) is the science of programming computers to learn from data. Instead of explicitly programming rules (e.g., `if x > 5 then y`), we provide the machine with data and a flexible model, and it learns the rules by minimizing an error function.

### Mathematical Formulation
Formally, a machine learning algorithm is an algorithm that is able to learn from data. Mitchell provided a concise definition:
> "A computer program is said to learn from experience **E** with respect to some class of tasks **T** and performance measure **P**, if its performance at tasks in **T**, as measured by **P**, improves with experience **E**."

In mathematical terms, we are often trying to approximate an unknown function $f: X \to Y$ that maps input features $X$ to targets $Y$. We try to find a hypothesis $h_\theta(x)$ (where $\theta$ are parameters) such that:
$$ h_\theta(x) \approx f(x) $$
We achieve this by minimizing a Loss Function $L(\theta)$:
$$ \theta^* = \arg\min_\theta \sum_{i=1}^{N} L(h_\theta(x^{(i)}), y^{(i)}) $$

---

## 2. Problem Framing
Before writing any code, the most critical step is **Problem Framing**. This involves defining the problem, assessing its business value, and determining the appropriate approach.

### The Framework Approach
1.  **Define the Goal**: What specifically are we trying to predict or categorize?
2.  **Assess ROI (Return on Investment)**:
    *   **Cost**: Compute resources (GPU/TPU cost), Engineering time, Data labeling cost.
    *   **Value**: Revenue increase, Cost saving, Time saving.
    *   *Real World Example*: Implementing an LLM for specific company RAG.
        *   Cost: $100k GPU + Engineering Team.
        *   Value: If it only saves 10 minutes/week for 5 employees, ROI is negative. Drop it.
3.  **Data Availability**: Do we have the data? Is it relevant?
    *   *Domain Conflict*: If data suggests sugar lowers diabetes risk (contradicting medical domain), **trust the domain** and investigate the data for bias/errors.

### Real-World Usage Example: Churn Prediction
*   **Goal**: Predict if a user will subscribe or leave (Churn).
*   **Type**: Binary Classification.
*   **Metric**: Accuracy is not enough. If 95% of users stay, a model predicting "Stay" for everyone has 95% accuracy but is useless. We use **Precision** and **Recall**.

---

## 3. Machine Learning vs. Deep Learning (Deep View)
While often used interchangeably, the distinction determines your toolset.

| Feature | Machine Learning (Traditional) | Deep Learning (DL) |
| :--- | :--- | :--- |
| **Data Size** | Works well with Small-Medium data (<10k - 100k rows) | Requires Large data (Millions of rows) to converge |
| **Hardware** | Runs on CPU | Requires GPU/TPU for matrix operations |
| **Feature Engineering** | **Manual**: You must create `BMI` from `Weight` and `Height` | **Automatic**: Learn features from raw pixels/text |
| **Interpretability** | **High**: (e.g., Decision Tree rules are clear) | **Low**: "Black Box" (Millions of weights) |
| **Data Type** | Structured (Tabular) | Unstructured (Images, Text, Audio) |

### Use Case Heuristics
*   **Tabular Data**: Use **XGBoost, Random Forest, or LightGBM**. They dominate Kaggle competitions for tabular data.
*   **Images/Text**: Use **Deep Learning (CNNs, Transformers)**. They capture spatial/temporal hierarchies.

### Deep Dive: When to use DL for Tabular Data?
Use DL for tabular data only when:
1.  **High Cardinality**: Features with thousands of categories (e.g., UserID). Embeddings can handle this better than One-Hot Encoding.
2.  **Complex Interactions**: When feature $A$ interacts with $B$ effectively only when $C$ is high. Neural Networks capture these non-linearities automatically.

---

## 4. The Learning Process: Intuition
How does a model actually "learn"?

1.  **Initialization**: Start with random weights (random guess).
2.  **Forward Pass (Prediction)**: Pass data through the model to get predictions.
3.  **Loss Calculation**: Calculate how far off predictions are (e.g., $MSE = (y - \hat{y})^2$).
4.  **Backward Pass (Optimization)**: Adjust weights to reduce loss using the gradient (direction of steepest descent).
    *   *Math Intuition*: $\theta_{new} = \theta_{old} - \alpha \cdot \nabla L$
    *   $\alpha$ is the learning rate (step size).

### Numerical Intuition Example
Imagine a simple line $y = wx$. We want to fit point $(x=2, y=10)$.
1.  **Init**: $w=3$. Prediction $\hat{y} = 3 \times 2 = 6$.
2.  **Loss**: $(10 - 6)^2 = 16$.
3.  **Gradient**: Sensitivity of Error to $w$. If we increase $w$, $\hat{y}$ increases, getting closer to 10. Gradient is negative (slope of cost function).
4.  **Update**: Increase $w$ to say 4. $\hat{y} = 4 \times 2 = 8$. Loss = $(10-8)^2 = 4$. Better!

---
