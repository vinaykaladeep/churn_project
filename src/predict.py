import joblib
import pandas as pd

# Load trained model (pipeline with feature engineering inside)
model = joblib.load("models/churn_model.pkl")

BUSINESS_THRESHOLD = 0.3


def predict_churn(input_df: pd.DataFrame):
    """
    Predict churn probability and apply business threshold logic.
    Returns structured business output.
    """

    # Ensure dataframe format
    if not isinstance(input_df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    # Get probability of churn (class 1)
    probability = model.predict_proba(input_df)[0][1]

    # Apply business threshold
    prediction = 1 if probability >= BUSINESS_THRESHOLD else 0

    # Business action logic
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
    # ⚠ MUST match training schema EXACTLY
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

    print("Prediction Script Output:")
    print(result)