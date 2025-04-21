# 📉 Telco Customer Churn Prediction: A Data Science Journey

## 🛍️ Introduction

As an aspiring data analyst, I embarked on this project to explore how data science can solve real-world business problems. I chose customer churn prediction because it blends data wrangling, exploratory analysis, machine learning, and actionable business strategy.

The goal was to predict which customers are most likely to leave a telecom provider so that retention efforts can be focused effectively.

---

## 🌟 Project Objectives

- Analyze customer data to identify patterns associated with churn.
- Build predictive models to forecast churn likelihood.
- Evaluate model performance and select the most effective one.
- Provide actionable insights to reduce churn rates and inform retention strategy.

---

## 📂 Dataset Overview

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

![output_20_0](https://github.com/user-attachments/assets/e00170b1-d063-454e-8da7-17fa6199205d)

![output_22_0](https://github.com/user-attachments/assets/d08a8fd7-98d9-457a-a46c-f8d4cc9fc789)

![output_24_0](https://github.com/user-attachments/assets/388b346b-5719-49cf-ab4c-4a1a016a158c)

![output_27_0](https://github.com/user-attachments/assets/2643095c-92b5-4530-bd6b-a6da08d5871f)

![output_29_0](https://github.com/user-attachments/assets/303151cd-3f74-458c-9519-89813579f92d)

![output_31_0](https://github.com/user-attachments/assets/1cfacee6-3aa2-48b5-895f-30f752ffd8c2)

![output_88_0](https://github.com/user-attachments/assets/c47ff5dc-a9d8-4cb2-80c0-eb9d17773d9a)

---

## 🧼 Data Preprocessing

To prepare the dataset for modeling, the following preprocessing steps were applied:

- **Handled missing values:** The `TotalCharges` column had non-numeric entries. These were converted using `pd.to_numeric` and missing values were imputed using the median.

  ```python
  df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
  df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
  ```

- **Dropped unnecessary columns:** `customerID` was removed as it had no predictive value.

- **Encoded categorical features:** All object-type features were encoded using one-hot encoding.

  ```python
  df_encoded = pd.get_dummies(df, drop_first=True)
  ```

- **Scaled numerical features:** Continuous variables like `tenure`, `MonthlyCharges`, and `TotalCharges` were standardized using `StandardScaler` for better model performance.

  ```python
  scaler = StandardScaler()
  df_encoded[numerical_features] = scaler.fit_transform(df_encoded[numerical_features])
  ```

---

## 🧐 Feature Engineering

Though this project primarily used existing features, the following engineering steps were applied:

- **Binary Transformation:** Converted `SeniorCitizen` from 0/1 integer to a clear Yes/No binary column for consistency.
- **Feature Reassessment:** Removed features with high multicollinearity or no variance (e.g., `customerID`).
- **Planned but Skipped:** Grouping tenure into buckets (0–12 months, 13–24, etc.) was considered, but skipped to preserve numeric continuity for tree-based models.

In future iterations, engineered features like `MonthlyCharge-to-Tenure ratio` or `Contract Length * Monthly Charges` could be explored.

---

## ⚙️ Model Tuning & Cross-Validation

To enhance model generalization:

- **Train-test split** was performed using `stratify=y` to preserve churn proportions.
- **XGBoost** was optimized using basic hyperparameter tuning (grid search was optional due to time constraints).
- **Cross-validation:** A 5-fold cross-validation strategy was considered for Logistic Regression and Random Forest models to avoid overfitting.

```python
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(xgb_model, X, y, cv=5, scoring='roc_auc')
print("Avg CV AUC:", cv_scores.mean())
```

- Final model (XGBoost) was selected for its highest ROC AUC (0.74) and balanced precision-recall tradeoff.

---

## 🔍 Feature Importance

The XGBoost model provides insight into the most influential features in churn prediction:

```python
import matplotlib.pyplot as plt
from xgboost import plot_importance

plot_importance(xgb_model, max_num_features=10)
plt.title("Top 10 Feature Importances")
plt.show()
```

**Top predictors included:**
- MonthlyCharges
- Tenure
- Contract_TwoYear
- InternetService_FiberOptic
- PaymentMethod_ElectronicCheck

These insights directly informed the business recommendations below.

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

## ✅ Conclusion

This project helped me strengthen my end-to-end data science workflow — from raw data to actionable insights and stakeholder-ready dashboards.

XGBoost delivered the best results and was used to score churn probabilities, which powered an interactive Tableau dashboard for decision-makers.

Looking forward to applying this same mindset and rigor to future analytical challenges!

