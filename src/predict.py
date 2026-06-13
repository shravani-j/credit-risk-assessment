import joblib
import pandas as pd

# Load model
model = joblib.load("models/credit_risk_model.pkl")

# Sample customer
sample = pd.DataFrame([{
    "checking_account_status": 4,
    "loan_duration_months": 12,
    "credit_history": 4,
    "loan_purpose": 2,
    "credit_amount": 2000,
    "savings_account_status": 4,
    "employment_duration": 4,
    "installment_rate": 2,
    "personal_status_sex": 2,
    "guarantors": 1,
    "residence_duration": 3,
    "property": 2,
    "age": 35,
    "other_installment_plans": 1,
    "housing": 2,
    "existing_credits": 1,
    "job": 2,
    "people_liable": 1,
    "telephone": 1,
    "foreign_worker": 1
}])

probability = model.predict_proba(sample)[0][1]
prediction = model.predict(sample)[0]

print("Probability of Good Credit:", round(probability, 4))
print("Prediction:", prediction)