"""
Main training pipeline for Churn Prediction Project
Production-grade version with MLflow experiment tracking + model registry
"""

from src.data import load_data
from src.preprocessing import split_data
from src.model import train_model
from src.evaluate import evaluate_model
from src.profit import calculate_profit

import config
import joblib
import os
import mlflow
import mlflow.sklearn


def main():

    # ============================================
    # 1️⃣  Set MLflow Experiment
    # ============================================
    mlflow.set_experiment("churn_logistic_regression_experiment")

    # Start MLflow run context
    with mlflow.start_run():

        # ============================================
        # 2️⃣  Load Data
        # ============================================
        df = load_data(config.DATA_PATH)

        X_train, X_test, y_train, y_test = split_data(df)

        # ============================================
        # 3️⃣  Train Model
        # ============================================
        model = train_model(X_train, y_train)
        print("Training complete.")

        # --------------------------------------------
        # Log Model Hyperparameters (IMPORTANT)
        # --------------------------------------------
        # Adjust based on your LogisticRegression params
        mlflow.log_param("model_type", "LogisticRegression")

        if hasattr(model, "C"):
            mlflow.log_param("C", model.C)

        if hasattr(model, "max_iter"):
            mlflow.log_param("max_iter", model.max_iter)

        if hasattr(model, "solver"):
            mlflow.log_param("solver", model.solver)

        # ============================================
        # 4️⃣  Save trained model locally (optional)
        # ============================================
        os.makedirs("models", exist_ok=True)
        model_path = os.path.join("models", "churn_model.pkl")
        joblib.dump(model, model_path)
        print(f"Model saved locally at {model_path}")

        # ============================================
        # 5️⃣  Evaluate Model
        # ============================================
        metrics, cm, probs = evaluate_model(model, X_test, y_test)

        print("\nModel Evaluation Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v}")

        print("\nConfusion Matrix:\n", cm)

        # --------------------------------------------
        # Log Metrics to MLflow
        # --------------------------------------------
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # ============================================
        # 6️⃣  Profit-Based Threshold Optimization
        # ============================================
        results_df = calculate_profit(y_test.values, probs)

        print("\nTop 5 Thresholds by Profit:")
        print(results_df.head())

        best_threshold = results_df.iloc[0]["threshold"]
        best_profit = results_df.iloc[0]["profit"]

        print("\nBest Threshold:", best_threshold)

        # Log business metrics
        mlflow.log_metric("best_threshold", float(best_threshold))
        mlflow.log_metric("best_profit", float(best_profit))

        # ============================================
        # 7️⃣  Log & Register Model in MLflow
        # ============================================
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="ChurnLogisticRegressionModel"
        )

        print("\nModel logged and registered in MLflow.")


if __name__ == "__main__":
    main()