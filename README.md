
# Fraudulent Claim Detection

This project aims to detect fraudulent insurance claims using machine learning techniques. It involves exploratory data analysis (EDA), feature engineering, and building classification models to accurately identify fraudulent claims.

## 📁 Project Files

- `Fraudulent_Claim_Detection.ipynb`: Full implementation including EDA, model building, evaluation, and feature importance.
- `Fraudulent_Claim_Detection_Starter.ipynb`: Starter notebook with initial data exploration and modeling framework.

## 📊 Problem Statement

The goal is to build a model that can predict whether a claim is fraudulent based on various customer and claim-related attributes.

## 🔧 Workflow

### 1. **Data Preparation**
- Load necessary libraries and the dataset.
- Examine data structure and initial statistics.

### 2. **Data Cleaning**
- Handle missing values.
- Fix data types.
- Remove or handle redundant features.

### 3. **Train-Validation Split**
- Separate features and target (`fraud`).
- Split into training and validation sets.

### 4. **Exploratory Data Analysis (EDA)**
- Univariate and bivariate analysis.
- Correlation matrix and class distribution.
- Insights on key differentiators between fraudulent and non-fraudulent claims.

### 5. **Feature Engineering**
- Handle imbalanced data with resampling.
- Create new features and handle categorical variables.
- One-hot encoding and feature scaling.

### 6. **Model Building**
- Logistic Regression:
  - Baseline model with cutoff tuning.
- Random Forest:
  - Feature selection.
  - Hyperparameter tuning using GridSearchCV.
  - Model interpretation via feature importance.

### 7. **Evaluation**
- Confusion Matrix, Accuracy, Precision, Recall, F1 Score, ROC-AUC.
- Compare performance across models on validation data.

## ✅ Results

- Random Forest achieved higher performance compared to Logistic Regression.
- Key features influencing fraud detection include policy terms, number of vehicles, and claim amounts.

## 📈 Tools & Libraries

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Imbalanced-learn

## 🚀 How to Run

1. Clone the repository
2. Open the Jupyter notebooks
3. Run all cells to execute the pipeline

```bash
git clone https://github.com/yourusername/fraudulent-claim-detection.git
cd fraudulent-claim-detection
jupyter notebook
```

## 📌 Future Improvements

- Experiment with advanced ML models like XGBoost or LightGBM
- Use automated feature selection techniques
- Deploy model using Flask or Streamlit for user interaction
