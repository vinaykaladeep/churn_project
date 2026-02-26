from src.data import load_data
from src.preprocessing import split_data
from src.model import train_model
from src.evaluate import evaluate_model
from src.profit import calculate_profit
import config
import joblib
import os

def main():
    df = load_data(config.DATA_PATH)

    X_train, X_test, y_train, y_test = split_data(df)

    model = train_model(X_train, y_train)
    print("Training complete.")

    # -------------------------
    # Save trained model
    # -------------------------
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "churn_model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved at {model_path}")

    metrics, cm, probs = evaluate_model(model, X_test, y_test)
    print("\nModel Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print("\nConfusion Matrix:\n", cm)

    results = calculate_profit(y_test.values, probs)

    # best = sorted(results, key=lambda x: x[1], reverse=True)[0]
    # print("\nBest Threshold:", best)

    results_df = calculate_profit(y_test.values, probs)

    print("\nTop 5 Thresholds by Profit:")
    print(results_df.head())

    best_threshold = results_df.iloc[0]["threshold"]
    print("\nBest Threshold:", best_threshold)

if __name__ == "__main__":
    main()