import joblib
import pandas as pd

# Load model
model = joblib.load("models/churn_model.pkl")

# Example single customer input (must match training columns)
sample_input = pd.DataFrame([{
    "tenure": 5,
    "MonthlyCharges": 70,
    "TotalCharges": 350,
    "Contract": "Month-to-month",
    "PaymentMethod": "Electronic check",
    "InternetService": "Fiber optic"
}])

prediction = model.predict(sample_input)
probability = model.predict_proba(sample_input)[0][1]

print("Prediction Script output:")
print("Prediction:", prediction[0])
print("Churn Probability:", round(probability, 3))