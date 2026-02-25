from sklearn.model_selection import train_test_split
import pandas as pd
import config

# ---------------------------------------------------
# Step 1: Feature Engineering Function
# ---------------------------------------------------
# This function creates new useful features
# before splitting the dataset.
# ---------------------------------------------------

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:

    # 1️⃣ Tenure Buckets
    # Groups customers based on how long they stayed.
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 6, 12, 24, 60, 100],
        labels=["0-6", "6-12", "12-24", "24-60", "60+"]
    )

    # 2️⃣ Contract Risk Flag
    # Month-to-month customers churn more often.
    df["is_month_to_month"] = (
        df["Contract"] == "Month-to-month"
    ).astype(int)

    # 3️⃣ Payment Risk Flag
    # Electronic check users often churn more.
    df["is_electronic_check"] = (
        df["PaymentMethod"] == "Electronic check"
    ).astype(int)

    # 4️⃣ Interaction Feature
    # Combines monthly charges and tenure.
    # Helps model understand customer value intensity.
    df["monthly_tenure_interaction"] = (
        df["MonthlyCharges"] * df["tenure"]
    )

    return df


# ---------------------------------------------------
# Step 2: Split Function (unchanged name)
# ---------------------------------------------------
# Now we apply feature engineering
# BEFORE splitting into train/test.
# ---------------------------------------------------

def split_data(df: pd.DataFrame):

    # Apply feature engineering first
    df = feature_engineering(df)

    # Separate features and target
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE
    )

    return X_train, X_test, y_train, y_test