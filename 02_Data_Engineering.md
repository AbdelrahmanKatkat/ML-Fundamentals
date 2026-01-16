# Chapter 2: Data Engineering and Preprocessing

In Machine Learning, we have a saying: **"Garbage In, Garbage Out."** No matter how advanced your model is, if your data is messy, biased, or unscaled, your predictions will be poor. Data Engineering is the process of cleaning and preparing data so the model can actually see the patterns.

---

## 1. Data ETL (Extract, Transform, Load)
Raw data lives in many different "homes":
*   **Structured Data**: Like an Excel sheet or SQL table. It has clear columns (Age, Salary). This is the easiest to work with.
*   **Semi-Structured Data**: Like a JSON file or a tweet. It has tags (e.g., `user_name`, `timestamp`) but doesn't fit perfectly into a table.
*   **Unstructured Data**: This is raw info like an image (a grid of pixels) or an audio file. To use this in traditional ML, we must "Transform" it into numbers first.

### The 80/20 Rule of Data Science
Data scientists spend **80% of their time** on data cleaning and preprocessing, and only **20%** on modeling. This chapter is about that critical 80%.

---

## 1.5. Data Quality: The Foundation
Before preprocessing, ask these questions:
1.  **Completeness**: Are there missing values? How many?
2.  **Accuracy**: Is the data correct? (e.g., Age = -5 is clearly wrong)
3.  **Consistency**: Do different sources agree? (e.g., "USA" vs "United States")
4.  **Timeliness**: Is the data up-to-date? (e.g., 2010 census data might be outdated for 2024 predictions)

**Critical Insight**: Always perform **Exploratory Data Analysis (EDA)** before preprocessing. Plot distributions, check for nulls, and understand your data's story.

---

## 2. Feature Scaling: Giving Every Feature a Fair Voice
Imagine you are predicting house prices using:
1.  **Number of Rooms** (Range: 1 to 5)
2.  **Square Footage** (Range: 500 to 5000)

If you don't scale, the model will think "Square Footage" is 1000x more important than "Rooms" just because the numbers are bigger. Scaling puts them on a level playing field.

### A. Standardization (Z-Score)
This centers the data around 0.
*   **Mean ($\mu$)**: The average value.
*   **Standard Deviation ($\sigma$)**: How much the values "spread out" from the average.

**Formula**:
$$ z = \frac{x - \mu}{\sigma} $$

*   **Logic**: It tells you how many "steps" (Standard Deviations) a value is away from the average.
*   **Intuition**: $z=0$ means the value is exactly average. $z=2$ means it's much higher than average.

### B. Min-Max Scaling (Normalization)
This squashes everything into a range between 0 and 1.
**Formula**:
$$ x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}} $$
*   **Intuition**: The smallest value becomes 0, the largest becomes 1, and everything else sits in between. Great for images where pixel values are 0-255.

### When to Use Which?
*   **Standardization**: Use when your data has **outliers** or when using algorithms that assume normally distributed data (e.g., Logistic Regression, SVM).
*   **Min-Max Scaling**: Use when you need a **bounded range** (e.g., neural networks with sigmoid activation, image data).

### Practical Tip: Fit on Training, Transform on Test
**Critical Mistake**: Never fit the scaler on the test set. This causes **data leakage**.
```python
# Correct way
scaler.fit(X_train)  # Learn mean and std from training data only
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use training statistics
```

---

## 3. Outlier Detection: Spotting the "Weird" Data
An outlier is a data point that is so far away from the rest that it might "pull" the model in the wrong direction.

### Analogy: The Billionaire in the Bar
If 10 people in a bar earn $50k/year, the average is $50k. If a billionaire walks in, the **average** salary jumps to $90 Million, but the **median** (the middle person) stays $50k. This is why we use robust methods to find outliers.

### Methods to Find Outliers
1.  **Z-Score Method**: Any point with $|z| > 3$ is usually an update/error.
2.  **IQR Method (Interquartile Range)**:
    *   **Q1**: The 25th percentile (bottom 25% mark).
    *   **Q3**: The 75th percentile (top 25% mark).
    *   **IQR**: $Q3 - Q1$ (The middle 50% of your data).
    *   **The Bound**: Anything outside $Q1 - 1.5 \times IQR$ or $Q3 + 1.5 \times IQR$ is an outlier.

### What to Do with Outliers?
1.  **Remove**: If it's clearly an error (e.g., Age = 200).
2.  **Cap (Winsorize)**: Replace extreme values with the 95th or 99th percentile.
3.  **Transform**: Use log transformation to reduce the impact of extreme values.
4.  **Keep**: If it's a legitimate value (e.g., a billionaire in a salary dataset), keep it but use robust algorithms (e.g., Random Forest, which is less sensitive to outliers).

**Practical Tip**: Always visualize outliers using **box plots** or **scatter plots** before deciding what to do.

---

## 4. Categorical Encoding: Speaking the Machine's Language
Computers don't understand "Red", "Blue", or "Green". They only understand numbers. But how we convert them matters.

### A. Label Encoding (The dangerous way)
Giving each color a number: Red=1, Blue=2, Green=3.
*   **The Trap**: The model might think $3 > 1$, implying "Green is greater than Red." This is only okay for ordered data (e.g., Small=1, Medium=2, Large=3).

### B. One-Hot Encoding (The safe way)
Creating a column for each color.
*   **Red**: $[1, 0, 0]$
*   **Blue**: $[0, 1, 0]$
*   **Green**: $[0, 0, 1]$
*   **Intuition**: Now every color has its own "switch" (Yes/No). No color is "greater" than another.

### C. Target Encoding (The Advanced Way)
For high-cardinality features (e.g., 1000 different cities), One-Hot Encoding creates 1000 columns, which is inefficient.

**Target Encoding** replaces each category with the **mean of the target variable** for that category.

**Example**: Predicting house prices by city.
*   **New York**: Average price = $800k → Encode as 800,000
*   **Detroit**: Average price = $200k → Encode as 200,000

**Warning**: This can cause **overfitting**. Use **cross-validation** or add **smoothing** to prevent this.

### D. Frequency Encoding
Replace each category with **how often it appears** in the dataset.

**Example**: 
*   **Red** appears 100 times → Encode as 100
*   **Blue** appears 50 times → Encode as 50

**Use Case**: Useful for clustering algorithms (like K-Means) that don't work well with One-Hot Encoding.

### When to Use Which Encoding?
| Encoding | When to Use | Pros | Cons |
| :--- | :--- | :--- | :--- |
| **Label Encoding** | Ordinal data (Small, Medium, Large) | Simple, no extra columns | Implies order |
| **One-Hot Encoding** | Nominal data with few categories (<10) | No false relationships | Creates many columns |
| **Target Encoding** | High-cardinality features (>50 categories) | Compact, captures target relationship | Risk of overfitting |
| **Frequency Encoding** | Clustering, high-cardinality | Compact, no target leakage | Loses category meaning |

---

## 5. Handling Missing Data
If a value is missing, you have three choices:
1.  **Delete**: Throw away the row (Only if you have millions of rows and very few misses).
2.  **Fill with Average (Impute)**: Use the Mean or Median.
    *   *Tip*: Use **Median** if you have heavy outliers (like the billionaire example).
3.  **Predict**: Use another AI to "guess" what the missing number should be based on other features (e.g., **KNN Imputation**).
4.  **Flag as Missing**: Create a new binary column `Age_is_missing` (1 if missing, 0 otherwise). This tells the model that "missingness" itself might be informative.

### Practical Tip: Understand Why Data is Missing
*   **Missing Completely at Random (MCAR)**: No pattern (e.g., sensor malfunction). Safe to impute.
*   **Missing at Random (MAR)**: Missing depends on other variables (e.g., older people don't report income). Imputation can work.
*   **Missing Not at Random (MNAR)**: Missing is related to the value itself (e.g., high earners don't report salary). Imputation is risky.

---

## 6. Handling Imbalanced Data
In classification, if 95% of your data is "Not Fraud" and 5% is "Fraud," the model will just predict "Not Fraud" every time and get 95% accuracy.

### Techniques to Handle Imbalance
1.  **Resampling**:
    *   **Oversampling**: Duplicate minority class samples (e.g., **SMOTE** - Synthetic Minority Over-sampling Technique).
    *   **Undersampling**: Remove majority class samples (risky if you have little data).
2.  **Class Weights**: Tell the model to "care more" about the minority class by assigning higher weights.
3.  **Anomaly Detection**: Treat the minority class as "anomalies" and use algorithms like Isolation Forest or One-Class SVM.

**Practical Tip**: Always use **Precision, Recall, and F1 Score** instead of accuracy for imbalanced datasets.

---

## 7. Tips and Tricks for Data Engineering
1.  **Always visualize first**: Use histograms, box plots, and scatter plots to understand your data before preprocessing.
2.  **Check for data leakage**: Never use information from the test set during preprocessing (e.g., fitting scalers).
3.  **Feature engineering is key**: Creating new features (e.g., `Age_Group` from `Age`) can improve model performance more than tuning hyperparameters.
4.  **Document your pipeline**: Keep track of all preprocessing steps so you can reproduce results and apply the same transformations to new data.
5.  **Use pipelines**: In scikit-learn, use `Pipeline` to chain preprocessing and modeling steps. This prevents errors and makes code cleaner.
6.  **Handle categorical data early**: Encode categorical features before scaling, as scaling doesn't make sense for categories.
7.  **Watch for high cardinality**: If a categorical feature has >50 unique values, consider Target Encoding or Frequency Encoding instead of One-Hot.
8.  **Test multiple imputation strategies**: Try mean, median, mode, and KNN imputation, then compare model performance.
9.  **Feature selection**: After preprocessing, use techniques like **Recursive Feature Elimination (RFE)** or **Feature Importance** to remove irrelevant features.
10. **Cross-validation for preprocessing**: When using Target Encoding or other target-dependent techniques, apply them inside cross-validation folds to avoid leakage.

---

## 8. Real-World Example: Preprocessing a Customer Dataset
A bank wants to predict loan defaults.

### Step 1: Exploratory Data Analysis
*   Features: Age, Income, Employment_Type, Credit_Score, Loan_Amount.
*   Missing: 10% of `Income` is missing.
*   Outliers: Some `Income` values are 10x the median.
*   Categorical: `Employment_Type` has 5 categories.

### Step 2: Handle Missing Data
*   Impute `Income` with **median** (robust to outliers).
*   Create `Income_is_missing` flag.

### Step 3: Handle Outliers
*   Cap `Income` at the 99th percentile (Winsorization).

### Step 4: Encode Categorical Data
*   Use **One-Hot Encoding** for `Employment_Type` (only 5 categories).

### Step 5: Scale Features
*   Use **Standardization** for `Age`, `Income`, `Credit_Score`, `Loan_Amount`.

### Step 6: Handle Imbalance
*   Only 5% of loans default. Use **SMOTE** to oversample the minority class.

### Step 7: Train Model
*   Use a **Random Forest** (robust to outliers and doesn't require scaling, but we scaled for consistency).

---
