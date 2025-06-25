# Customer Churn Prediction

This project analyzes a telecommunications dataset to predict whether a customer will churn using a combination of supervised machine learning models. The goal is to support the business in identifying high-risk customers early and reducing churn rates through targeted interventions.


---

## Dataset Overview

- **Train Dataset**: 7,043 rows × 20 columns
- **Test Dataset**: 1,000+ rows
- **Target**: `Churn` (Yes or No)
- **Features** include:
  - Customer account info: `tenure`, `contract`, `monthlycharges`, `totalcharges`
  - Demographic info: `gender`, `seniorcitizen`, `partner`, `dependents`
  - Service usage: `phoneservice`, `internetservice`, `streamingtv`, etc.

---

## Preprocessing Steps

- Converted categorical variables using OneHotEncoding
- Normalized numerical features with MinMaxScaler
- Removed non-contributing columns such as `customerID`
- Balanced the target variable using SMOTE due to class imbalance

---

## Models Used

| Model                   | Notes                                       |
|------------------------|---------------------------------------------|
| Logistic Regression     | Baseline linear classifier                  |
| Random Forest Classifier| Ensemble model with strong performance      |
| Gradient Boosting Classifier | Best performing model overall        |

Hyperparameter tuning was performed using GridSearchCV for optimal results.

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC Score

All models were evaluated on the test set and cross-validated to avoid overfitting.

---

## SHAP Explainability

SHAP values were used to explain predictions of the Gradient Boosting model, providing insights into how each feature contributes to the prediction.

---

## Key Insights

- `Contract`, `Tenure`, and `MonthlyCharges` were among the most important predictors of churn.
- Customers on month-to-month contracts had a significantly higher churn rate.
- The Gradient Boosting Classifier achieved the highest ROC-AUC score, with strong precision-recall tradeoff.

---

## Project Structure

customer-churn-prediction/
├── data/
│ ├── train.csv
│ └── test.csv
├── notebooks/
│ └── customer-churn-prediction.ipynb
├── README.md
├── .gitignore
└── requirements.txt
