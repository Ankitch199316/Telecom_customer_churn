# 📉 Telco Customer Churn Prediction: A Data Science Journey

## 🧭 Introduction

As an aspiring data analyst, I embarked on this project to explore how data science can solve real-world business problems. I chose customer churn prediction because it blends data wrangling, exploratory analysis, machine learning, and actionable business strategy.

The goal was to predict which customers are most likely to leave a telecom provider so that retention efforts can be focused effectively.

---

## 🎯 Project Objectives

- Analyze customer data to identify patterns associated with churn.
- Build predictive models to forecast churn likelihood.
- Evaluate model performance and select the most effective one.
- Provide actionable insights to reduce churn rates and inform retention strategy.

---

## 🗂️ Dataset Overview

- **Source:** [Telco Customer Churn - Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Shape:** 7,043 rows × 21 columns
- **Features include:** Customer tenure, contract type, services used, charges, demographics, and churn status.

---

## 🔍 Exploratory Data Analysis (EDA)

EDA helped surface critical relationships:

- **Churn Rate:** ~26.5% overall.
- **Contract Type:** Month-to-month contracts are the most churn-prone.
- **Tenure:** Lower tenure customers churn more often.
- **Payment Method:** Electronic check users show higher churn rates.

**Key Charts:**

- ![output_20_0](https://github.com/user-attachments/assets/eda495c3-2065-411d-af6e-983d54742b22)
- ![output_22_0](https://github.com/user-attachments/assets/0b3d2dc4-cc48-40fe-ac3c-e5d0601e0939)
- ![output_24_0](https://github.com/user-attachments/assets/78170541-6c21-4fe8-991a-5137b9ed0631)
- ![output_27_0](https://github.com/user-attachments/assets/657dfe46-58b8-4f36-aa72-b993a52938b2)
- ![output_29_0](https://github.com/user-attachments/assets/0dc5e16a-d48c-4b53-be70-da2b6778fef5)
- ![output_31_0](https://github.com/user-attachments/assets/8e66bc69-b1d8-4a7b-bb0c-5e5a0b419b8f)
- ![output_88_0](https://github.com/user-attachments/assets/8d5a16f5-c07d-4ef4-9edf-fe443865ec1b)



---

## 🛠️ Data Preprocessing & Feature Engineering

**Missing Values:**
- The `TotalCharges` column contained non-numeric values and nulls.
```python
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
```

**Encoding Categorical Variables:**
```python
df_encoded = pd.get_dummies(df, drop_first=True)
```

**Scaling Numerical Features:**
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_encoded[numerical_features] = scaler.fit_transform(df_encoded[numerical_features])
```

#### Modeling

**Models Tried**
- Logistic Regression
- Random Forest
- XGBoost (Final Model)

**Evaluation Metrics:**
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC

**Model Comparison:**
| Model               | Accuracy | Precision | Recall | F1 Score | AUC  |
|---------------------|----------|-----------|--------|----------|------|
| Logistic Regression | 0.726    | 0.49      | 0.80   | 0.61     | 0.73 |
| Random Forest       | 0.786    | 0.62      | 0.51   | 0.56     | 0.71 |
| XGBoost             | 0.801    | 0.65      | 0.53   | 0.60     | 0.74 |


#### Final Model Selection:
XGBoost was chosen for deployment due to:

- Highest accuracy and ROC AUC
- Balanced precision and recall
- Strong generalization on unseen data

### 🚧 Challenges & Solutions
| Challenge                  | What I Did                                                              |
|----------------------------|-------------------------------------------------------------------------|
| Missing `TotalCharges` values | Used `pd.to_numeric` and median imputation                          |
| Class imbalance            | Used stratified sampling and prioritized recall + AUC                  |
| Feature redundancy         | Identified and dropped multicollinear columns                          |
| Model interpretability     | Compared multiple models and used feature importances                 |


### 📈 Key Insights & Business Recommendations

**📉 Customers with Month-to-Month contracts are ~3× more likely to churn.**
→ Recommend promoting yearly or 2-year plans with discounts.

**🕒 Customers with tenure under 12 months are at highest churn risk.**
→ Onboarding campaigns and early engagement could reduce churn.

**💳 Customers paying via Electronic Check churn at higher rates.**
→ Incentivize switching to auto-payment or card methods.

### 📊 Dashboards

**📌 Dashboard 1**: Churn Summary & Risk Explorer
Includes KPIs, churn scatterplot, contract risk view, and top 100 high-risk customers.

<img width="1219" alt="Screenshot 2025-04-21 at 11 41 45 AM" src="https://github.com/user-attachments/assets/1a05d544-a308-4347-b939-2550c6375572" />

**📌 Dashboard 2**: Model Performance Comparison
Shows Accuracy, Precision, Recall, F1, and AUC for all models.

<img width="1224" alt="Screenshot 2025-04-21 at 11 42 06 AM" src="https://github.com/user-attachments/assets/9343978e-b3c7-4f04-991b-35f0c45ec354" />

### 💻 Code Snippets
```python
# Splitting the data
from sklearn.model_selection import train_test_split

X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

```python
# Training XGBoost
from xgboost import XGBClassifier

xgb = XGBClassifier()
xgb.fit(X_train, y_train)
```

```python
# Evaluating the model
from sklearn.metrics import classification_report

y_pred = xgb.predict(X_test)
print(classification_report(y_test, y_pred))
```
### ✅ Conclusion

This project helped me strengthen my end-to-end data science workflow — from raw data to actionable insights and stakeholder-ready dashboards.

XGBoost delivered the best results and was used to score churn probabilities, which powered an interactive Tableau dashboard for decision-makers.

Looking forward to applying this same mindset and rigor to future analytical challenges!


