import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    df = pd.read_csv(path)

    # Drop ID
    df = df.drop("customerID", axis=1)

    # Fix TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()

    # Encode target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df