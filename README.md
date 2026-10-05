# 📊 Customer Churn Prediction

## 📌 Project Overview

Customer Churn Prediction is an end-to-end Machine Learning project that predicts whether a customer is likely to leave a telecommunications service.

The project covers the complete Data Science workflow, starting from data understanding and exploratory data analysis (EDA), followed by feature engineering, preprocessing, model training, hyperparameter tuning, evaluation, and finally deploying the trained model through a Streamlit web application.

---

## 🎯 Business Problem

Customer churn is an important challenge for telecommunications companies.

Identifying customers who are likely to churn can help companies:

- Detect high-risk customers early.
- Improve customer retention strategies.
- Reduce customer acquisition costs.
- Provide targeted offers and services.
- Make data-driven business decisions.

The goal of this project is to build a classification model that predicts whether a customer will churn based on their demographic information, services, contract details, and billing information.

---

## 📊 Dataset

The project uses the **https://www.kaggle.com/datasets/blastchar/telco-customer-churn?utm_source=gemini**.

### Dataset Information

- **Rows:** 7,043
- **Columns:** 21
- **Target Variable:** `Churn`
- **Task:** Binary Classification

The target variable contains two classes:

- `No` → Customer stays
- `Yes` → Customer churns

The dataset contains information about:

- Customer demographics
- Services
- Internet services
- Contract information
- Payment methods
- Monthly charges
- Total charges
- Customer tenure

---

## 🔍 Exploratory Data Analysis

Several EDA techniques were performed to understand the dataset and identify patterns related to customer churn.

### Data Inspection

- Dataset shape
- Data types
- Statistical summary
- Missing values
- Duplicate values
- Unique values

### Data Cleaning

- Converted `TotalCharges` to numeric.
- Handled blank values in `TotalCharges`.
- Verified duplicate records.
- Checked categorical values for consistency.
- Removed `customerID` because it is an identifier and does not provide predictive information.

### Univariate Analysis

The distributions of important numerical and categorical features were analyzed.

### Bivariate Analysis

Relationships between customer churn and important features were explored, including:

- Contract type
- Internet service
- Monthly charges
- Tenure
- Payment method

### Multivariate Analysis

Multivariate relationships were explored using:

- Tenure vs Monthly Charges vs Churn
- Contract vs Internet Service vs Churn
- Numerical correlation heatmap

---

## ⚙️ Feature Engineering

Three additional features were created to improve the representation of customer behavior.

### 1. TenureType

Customers were grouped according to their tenure:

- New
- Short-term
- Medium-term
- Long-term

### 2. NumServices

The total number of subscribed services was calculated from:

- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies

### 3. AvgMonthlyCharges

Average monthly charges were calculated using:

`TotalCharges / tenure`

For customers with zero tenure, `MonthlyCharges` was used instead.

---

## 🧹 Data Preprocessing

The target variable was converted into binary values:

- `No` → `0`
- `Yes` → `1`

The dataset was split into:

- **80% Training**
- **20% Testing**

Stratified splitting was used to preserve the class distribution.

### Numerical Features

Numerical features were standardized using:

`StandardScaler`

### Categorical Features

Categorical features were converted into numerical representations using:

`OneHotEncoder`

with:

`handle_unknown="ignore"`

### ColumnTransformer

`ColumnTransformer` was used to apply different preprocessing techniques to numerical and categorical features.

A Scikit-learn Pipeline was used in the final model to combine preprocessing and model training while avoiding data leakage during cross-validation.

---

## 🤖 Machine Learning Models

Two baseline classification models were trained and evaluated:

### Logistic Regression

Logistic Regression was used as the main baseline classification model.

### Random Forest

Random Forest was used as a tree-based model for comparison.

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

---

## 📈 Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7331 | 0.4983 | 0.7914 | 0.6116 | 0.8420 |
| Random Forest | 0.7800 | 0.6111 | 0.4706 | 0.5317 | 0.8219 |
| Tuned Logistic Regression | 0.7445 | 0.5122 | 0.7834 | 0.6195 | 0.8404 |

---

## 🔧 Hyperparameter Tuning

GridSearchCV was used to optimize the Logistic Regression model.

The following hyperparameters were tuned:

- `C`
- `solver`
- `class_weight`

The optimization metric was:

`F1 Score`

The best model used:

- `C = 0.01`
- `solver = lbfgs`
- `class_weight = balanced`

The tuned Logistic Regression model achieved a cross-validation F1 score of approximately:

**0.6325**

---

## 🏆 Final Model

The final selected model is:

**Tuned Logistic Regression**

It was selected because it provided a good balance between:

- Precision
- Recall
- F1 Score
- ROC-AUC

Recall is particularly important for this problem because failing to identify a customer who is actually going to churn may result in a missed retention opportunity.

### Final Performance

- **Accuracy:** 74.45%
- **Precision:** 51.22%
- **Recall:** 78.34%
- **F1 Score:** 61.95%
- **ROC-AUC:** 84.04%
- **Specificity:** 73.04%

---

## 📊 Model Evaluation

The final model was evaluated using:

### Confusion Matrix

The confusion matrix was used to analyze:

- True Positives
- True Negatives
- False Positives
- False Negatives

### ROC Curve

The ROC curve was used to evaluate the model's ability to distinguish between customers who churn and customers who stay.

### Specificity

Specificity measures the model's ability to correctly identify customers who do not churn.

The final model achieved approximately:

**73.04% Specificity**

---

## 🌐 Streamlit Application

The trained model was deployed using **Streamlit**.

The application allows users to enter customer information and receive:

- Churn prediction
- Churn probability
- Prediction result

The application includes customer information related to:

- Demographics
- Tenure
- Services
- Contract
- Internet service
- Payment method
- Monthly charges
- Total charges

### Application Preview

<img width="1891" height="734" alt="Image1" src="https://github.com/user-attachments/assets/581fabde-3567-474f-ada4-fb846441827f" />


<img width="1920" height="664" alt="Image2" src="https://github.com/user-attachments/assets/178ad04a-ceac-4e08-8152-64bb35fbf7f4" />


<img width="1920" height="397" alt="Image3" src="https://github.com/user-attachments/assets/56ecb24e-e010-41af-baec-55bfac48f5b2" />

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook

---

## Possible future improvements include:

```text
- Trying additional classification algorithms.
- Advanced hyperparameter optimization.
- Feature importance and model explainability using SHAP.
- Threshold optimization based on business requirements.
- Adding customer risk categories.
- Deploying the application online.
- Adding a dashboard for monitoring churn predictions.

---

## 💡Key Insights

```text
The project demonstrates a complete machine learning workflow from raw data to a deployed prediction application.
Important lessons from the project include:
- Customer tenure and contract type are important factors in churn prediction.
- Proper preprocessing is essential when working with mixed numerical and categorical data.
- Recall is an important metric for churn prediction because missing potential churners can reduce retention opportunities.
- Hyperparameter tuning can improve model performance.
- Building a Pipeline helps maintain consistent preprocessing between training and prediction.
🔮 Future Improvements

---

## 📁 Project Structure

```text
Customer-Churn-Prediction/
│
├── Customer_Churn_Prediction.ipynb
├── app.py
├── customer_churn_model.pkl
├── requirements.txt
├── README.md
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
└── screenshots.png
