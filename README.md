# 📉 Telco Customer Churn Prediction: A Data Science Journey

## 🛍️ Introduction

This project addresses customer churn in the telecom industry by predicting which customers are most likely to leave and translating those predictions into **retention prioritization insights**.

The objective is to support data-driven decision-making by identifying high-risk customers, understanding the drivers behind churn, and enabling teams to focus retention efforts where they are most likely to have impact.

---

## 🌟 Project Objectives

- Identify behavioral and contract-level patterns associated with customer churn.
- Develop a churn risk scoring approach to prioritize retention efforts.
- Select a final model based on business-relevant performance trade-offs.
- Translate analytical findings into practical retention and intervention strategies.

---

## 📂 Dataset Overview

- **Source:** [Telco Customer Churn - Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Shape:** 7,043 rows × 21 columns
- **Features include:** Customer tenure, contract type, services used, charges, demographics, and churn status.

---

## 🔍 Exploratory Analysis & Key Risk Signals

Exploratory analysis was used to identify **high-impact churn risk signals** relevant for retention decision-making:

- **Overall churn rate (~26.5%)** indicates a meaningful retention challenge.
- **Contract type:** Month-to-month customers exhibit significantly higher churn risk, highlighting contract structure as a key lever.
- **Tenure:** Early-tenure customers are substantially more likely to churn, suggesting the need for early engagement strategies.
- **Payment method:** Electronic check users show elevated churn, indicating potential friction in the billing experience.


**Key Charts:**

![output_20_0](https://github.com/user-attachments/assets/e00170b1-d063-454e-8da7-17fa6199205d)

![output_22_0](https://github.com/user-attachments/assets/d08a8fd7-98d9-457a-a46c-f8d4cc9fc789)

![output_24_0](https://github.com/user-attachments/assets/388b346b-5719-49cf-ab4c-4a1a016a158c)

![output_27_0](https://github.com/user-attachments/assets/2643095c-92b5-4530-bd6b-a6da08d5871f)

![output_29_0](https://github.com/user-attachments/assets/303151cd-3f74-458c-9519-89813579f92d)

![output_31_0](https://github.com/user-attachments/assets/1cfacee6-3aa2-48b5-895f-30f752ffd8c2)

![output_88_0](https://github.com/user-attachments/assets/c47ff5dc-a9d8-4cb2-80c0-eb9d17773d9a)

---
## 🧼 Data Preparation & Feature Handling

Data preparation focused on ensuring reliable churn signal extraction and model stability:

- Cleaned and standardized numeric features (e.g., tenure, charges) to support model performance.
- Handled missing and inconsistent values to preserve data integrity.
- Encoded categorical variables for compatibility with tree-based and linear models.
- Removed identifiers and low-signal features to avoid noise and leakage.

Feature handling decisions prioritized interpretability and robustness over excessive feature engineering.

---

## ⚙️ Model Training & Validation

Multiple classification models were evaluated to estimate churn risk, including Logistic Regression, Random Forest, and XGBoost.

Model selection prioritized:
- Predictive performance (ROC-AUC)
- Recall balance for identifying high-risk customers
- Stability across validation splits

XGBoost was selected as the final model due to its balanced performance (ROC-AUC ≈ 0.74) and ability to capture non-linear churn patterns.


---

## 🔍 Key Churn Drivers

Feature importance analysis highlighted the strongest drivers of customer churn:

- **Contract type:** Long-term contracts significantly reduce churn risk.
- **Tenure:** Early-tenure customers are more vulnerable to churn.
- **Monthly charges:** Higher charges correlate with increased churn likelihood.
- **Service type:** Fiber optic internet customers show elevated churn.
- **Payment method:** Electronic check usage is associated with higher churn.

These drivers informed the retention prioritization and recommendations outlined below.

---

## 📊 Model Performance

### Models Tried:
- Logistic Regression
- Random Forest
- XGBoost (Final Model)

### Evaluation Metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC

### Model Comparison:

| Model               | Accuracy | Precision | Recall | F1 Score | AUC  |
|---------------------|----------|-----------|--------|----------|------|
| Logistic Regression | 0.726    | 0.49      | 0.80   | 0.61     | 0.73 |
| Random Forest       | 0.786    | 0.62      | 0.51   | 0.56     | 0.71 |
| **XGBoost**         | **0.801**| **0.65**  | 0.53   | **0.60** | **0.74** |

### Final Model Selection:

XGBoost was chosen for deployment due to:
- Highest accuracy and ROC AUC
- Balanced precision and recall
- Strong generalization on unseen data

---

## 🚧 Challenges & Solutions

| Challenge                  | What I Did                                                              |
|----------------------------|-------------------------------------------------------------------------|
| Missing `TotalCharges` values | Used `pd.to_numeric` and median imputation                          |
| Class imbalance            | Used stratified sampling and prioritized recall + AUC                  |
| Feature redundancy         | Identified and dropped multicollinear columns                          |
| Model interpretability     | Compared multiple models and used feature importances                 |

---

## 📈 Business Implications

This project generated data-backed recommendations to improve customer retention:

- 📉 **Contract Type:** Customers on month-to-month contracts churn significantly more.
  - **Recommendation:** Promote yearly contracts with loyalty discounts.

- 🥒 **Tenure < 6 months = High Risk**
  - **Recommendation:** Target new customers with onboarding engagement and proactive support.

- 💳 **Electronic check users = High churn**
  - **Recommendation:** Encourage customers to switch to auto-payment methods via incentives.

- 🧪 **Senior citizens with fiber plans showed higher risk**
  - **Recommendation:** Tailor technical support and simplify plan offerings for this segment.

---

## 📊 Dashboards

- 📌 **Dashboard 1:** Churn Summary & Risk Explorer  
  Includes KPIs, churn scatterplot, contract risk view, and top 100 high-risk customers.  
  
<img width="1222" alt="Screenshot 2025-04-21 at 12 07 56 PM" src="https://github.com/user-attachments/assets/577ecbd8-426e-4fe7-9b98-88b61d75b6e8" />

- 📌 **Dashboard 2:** Model Performance Comparison  
  Shows Accuracy, Precision, Recall, F1, and AUC for all models.  

<img width="1218" alt="Screenshot 2025-04-21 at 12 08 18 PM" src="https://github.com/user-attachments/assets/fed20483-2fbc-461f-97ab-26f60239531a" />

---

## 💻 Code Snippets

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

---

## 🛠️ How to Reproduce This Project

### 📁 Directory Structure
```
├── data/
│   ├── raw/                  # Original Telco Customer Churn CSV
│   ├── cleaned/              # Preprocessed data files
├── notebooks/
│   ├── Telco_Customer_Churn.ipynb
├── dashboards/
│   ├── churn_summary_dashboard.twbx
│   ├── model_performance_dashboard.twbx
├── model_scores.csv
├── model_comparison.csv
├── README.md
```

### 📦 Requirements
- Python 3.8+
- pandas
- numpy
- scikit-learn
- xgboost
- seaborn
- matplotlib
- jupyter

### ▶️ Steps to Run:
1. Clone this repository
2. Install dependencies via:
   ```
   pip install -r requirements.txt
   ```
3. Run the notebook:  
   ```
   jupyter notebook notebooks/Telco_Customer_Churn.ipynb
   ```

---

## 🚀 Conclusion

This project demonstrates how customer churn prediction can be translated from model outputs into **practical retention prioritization** for a telecom business.

By identifying high-risk customer segments, understanding key churn drivers, and selecting a model based on business-relevant trade-offs, the analysis shows how data can support targeted interventions rather than blanket discounting. The results highlight where retention efforts are likely to be most effective and how analytical insights can directly inform contract strategy, billing optimization, and customer experience improvements.


