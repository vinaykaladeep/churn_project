"""
Production-grade inference script
Loads model from MLflow Model Registry (Production stage)
"""

import mlflow
import mlflow.sklearn
import pandas as pd

# -----------------------------
# MLflow tracking server
# -----------------------------
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# -----------------------------
# Load model from registry
# -----------------------------
MODEL_NAME = "ChurnLogisticRegressionModel"
MODEL_STAGE = "Production"

model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"

print(f"Loading model from MLflow Registry: {model_uri}")

model = mlflow.sklearn.load_model(model_uri)

BUSINESS_THRESHOLD = 0.3


def predict_churn(input_df: pd.DataFrame):
    """
    Predict churn probability and apply business threshold logic.
    Returns structured business output.
    """

    if not isinstance(input_df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    probability = model.predict_proba(input_df)[0][1]

    prediction = 1 if probability >= BUSINESS_THRESHOLD else 0

    recommended_action = (
        "Offer retention discount"
        if prediction == 1
        else "No action needed"
    )

    return {
        "churn_probability": round(float(probability), 3),
        "threshold_used": BUSINESS_THRESHOLD,
        "churn_prediction": int(prediction),
        "recommended_action": recommended_action
    }


if __name__ == "__main__":

    # MUST match training schema exactly
    sample_input = pd.DataFrame([{
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70,
        "TotalCharges": 350
    }])

    result = predict_churn(sample_input)

    print("\nPrediction Output:")
    print(result)