from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from src.preprocessing import feature_engineering
from src.predict import predict_churn

# -------------------------
# Load trained model
# -------------------------
model = joblib.load("models/churn_model.pkl")
app = FastAPI(title="Churn Prediction API")

# -------------------------
# Define FULL Input Schema
# (Must match training data EXACTLY)
# -------------------------
class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


# -------------------------
# Prediction Endpoint
# -------------------------
@app.post("/predict")
def predict(customer: Customer):

    try:
        # Convert input to dataframe
        input_df = pd.DataFrame([customer.model_dump()])
        result = predict_churn(input_df)
        return result
        # Apply SAME feature engineering used during training
        # input_df = feature_engineering(input_df)

        # Make prediction
        # prediction = model.predict(input_df)[0]
        # probability = model.predict_proba(input_df)[0][1]
        # BUSINESS_THRESHOLD = 0.3
        # prediction = 1 if probability >= BUSINESS_THRESHOLD else 0
        # recommended_action = (
        #     "Offer retention discount"
        #     if prediction == 1
        #     else "No action needed"
        # )

        # probability = model.predict_proba(input_df)[0][1]

        # return {
        #     "churn_prediction": int(prediction),
        #     "churn_probability": round(float(probability), 3)
        # }

    except Exception as e:
        return {"error": str(e)}