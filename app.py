"""
Production-grade FastAPI service for Churn Prediction
Uses MLflow Model Registry for loading production model
"""

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow
import mlflow.sklearn
import time

from src.predict import predict_churn

# -------------------------------------------------
# MLflow Configuration
# -------------------------------------------------
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME = "ChurnLogisticRegressionModel"
MODEL_STAGE = "Production"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# -------------------------------------------------
# Initialize FastAPI App
# -------------------------------------------------
app = FastAPI(
    title="Churn Prediction API",
    description="Production ML inference service using MLflow Model Registry",
    version="1.0"
)

# -------------------------------------------------
# Global model variable
# (Loaded once when API starts)
# -------------------------------------------------
model = None


# -------------------------------------------------
# Load model on API startup
# -------------------------------------------------
@app.on_event("startup")
def load_model():
    global model

    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    print(f"Loading model from MLflow Registry: {model_uri}")

    model = mlflow.sklearn.load_model(model_uri)

    print("Model loaded successfully.")


# -------------------------------------------------
# FULL Input Schema
# Must match training dataset
# -------------------------------------------------
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


# -------------------------------------------------
# Health Check Endpoint
# Used by load balancers / Kubernetes later
# -------------------------------------------------
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_stage": MODEL_STAGE
    }


# -------------------------------------------------
# Prediction Endpoint
# -------------------------------------------------
@app.post("/predict")
def predict(customer: Customer):

    start_time = time.time()

    try:
        # Convert input into DataFrame
        input_df = pd.DataFrame([customer.model_dump()])

        # Call prediction logic
        result = predict_churn(input_df)

        latency = round(time.time() - start_time, 4)

        response = {
            "prediction": result,
            "model_stage": MODEL_STAGE,
            "latency_seconds": latency
        }

        return response

    except Exception as e:
        return {
            "error": str(e)
        }
    
# ===========
# from fastapi import FastAPI
# from pydantic import BaseModel
# import joblib
# import pandas as pd
# from src.preprocessing import feature_engineering
# from src.predict import predict_churn

# # -------------------------
# # Load trained model
# # -------------------------
# model = joblib.load("models/churn_model.pkl")
# app = FastAPI(title="Churn Prediction API")

# # -------------------------
# # Define FULL Input Schema
# # (Must match training data EXACTLY)
# # -------------------------
# class Customer(BaseModel):
#     gender: str
#     SeniorCitizen: int
#     Partner: str
#     Dependents: str
#     tenure: int
#     PhoneService: str
#     MultipleLines: str
#     InternetService: str
#     OnlineSecurity: str
#     OnlineBackup: str
#     DeviceProtection: str
#     TechSupport: str
#     StreamingTV: str
#     StreamingMovies: str
#     Contract: str
#     PaperlessBilling: str
#     PaymentMethod: str
#     MonthlyCharges: float
#     TotalCharges: float


# # -------------------------
# # Prediction Endpoint
# # -------------------------
# @app.post("/predict")
# def predict(customer: Customer):

#     try:
#         # Convert input to dataframe
#         input_df = pd.DataFrame([customer.model_dump()])
#         result = predict_churn(input_df)
#         return result
#         # Apply SAME feature engineering used during training
#         # input_df = feature_engineering(input_df)

#         # Make prediction
#         # prediction = model.predict(input_df)[0]
#         # probability = model.predict_proba(input_df)[0][1]
#         # BUSINESS_THRESHOLD = 0.3
#         # prediction = 1 if probability >= BUSINESS_THRESHOLD else 0
#         # recommended_action = (
#         #     "Offer retention discount"
#         #     if prediction == 1
#         #     else "No action needed"
#         # )

#         # probability = model.predict_proba(input_df)[0][1]

#         # return {
#         #     "churn_prediction": int(prediction),
#         #     "churn_probability": round(float(probability), 3)
#         # }

#     except Exception as e:
#         return {"error": str(e)}