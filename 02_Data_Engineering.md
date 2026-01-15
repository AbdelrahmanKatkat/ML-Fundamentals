# Chapter 2: Data Engineering and Preprocessing

## 1. Data ETL (Extract, Transform, Load)
Data does not come clean. It comes in various forms:
*   **Structured**: Relational databases (SQL tables). Easy to query.
*   **Semi-Structured**: JSON, XML, Logs. Contains tags/markers but no fixed schema.
*   **Unstructured**: Images, Audio, PDF text. Requires massive preprocessing to extract features.

The role of **Data Engineering** is to build pipelines that convert raw data into a usable format ($X, y$) for the model.

---

## 2. Feature Scaling
Machine Learning models that rely on distance (KNN, K-Means, SVM) or gradients (Linear Regression, Neural Nets) **require** scaling. If one feature ranges from 0-1 and another from 0-1000, the gradients will be biased towards the larger feature.

### Standardization (Z-Score Normalization)
Transforms data to have Mean $\mu = 0$ and Standard Deviation $\sigma = 1$.
$$ z = \frac{x - \mu}{\sigma} $$
*   **When to use**: Most algorithms (Linear Regression, Logistic Regression, SVM, Neural Nets).
*   ** robustness**: Less affected by outliers than Min-Max.

### Min-Max Scaling
Rescales data to a fixed range $[0, 1]$.
$$ x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}} $$
*   **When to use**: Image processing (0-255 pixels), or algorithms that need bounded inputs.

### Numerical Intuition Example
Dataset: Age = $[20, 30, 40, 50, 60]$
1.  **Mean ($\mu$)**: $40$
2.  **Std ($\sigma$)**: $\approx 14.14$
3.  **Standardize 60**:
    $$ z = \frac{60 - 40}{14.14} = 1.41 $$
    This means 60 is 1.41 standard deviations above the average.

---

## 3. Outlier Detection
Outliers are extreme values that deviate significantly from other observations. They can be valid (wealth of a billionaire) or errors (sensor glitch).

### Methods
1.  **Z-Score Method**: Any data point with $|z| > 3$ is considered an outlier (covers 99.7% of data in Normal Distribution).
2.  **IQR Method (Robust)**:
    *   $IQR = Q3 - Q1$
    *   Lower Bound = $Q1 - 1.5 \times IQR$
    *   Upper Bound = $Q3 + 1.5 \times IQR$

### Handling Skewness
If data is skewed (long tail), scaling might not be enough.
*   **Log Transformation**: Apply $\log(x+1)$ to compress large values.
*   **Box-Cox Transformation**: Statisticians' tool to force normality.

---

## 4. Categorical Encoding
Models only understand numbers.

### Label Encoding
Assign an integer to each category (e.g., Apple=0, Banana=1, Cat=2).
*   **Problem**: Model thinks $2 > 1$, implies "Cat is greater than Banana".
*   **Use when**: Target variable (y), or Ordinal features (Small, Medium, Large).

### One-Hot Encoding
Create a binary column for each category.
*   **Apple**: $[1, 0, 0]$
*   **Banana**: $[0, 1, 0]$
*   **Cat**: $[0, 0, 1]$
*   **Problem**: **Curse of Dimensionality**. If you have a "City" column with 10,000 cities, you add 10,000 columns (mostly zeros).
*   **Solution**: For high cardinality, use Target Encoding or Embeddings.

---

## 5. Missing Data Handling
1.  **Drop**: If dataset is huge (millions) and missing rows are few (<5%).
2.  **Impute (Mean/Median)**: Fill with average. (Median is better if data has outliers).
3.  **Impute (Model-based)**: Use a KNN or Regression model to predict the missing value based on other features (Most accurate).

---
