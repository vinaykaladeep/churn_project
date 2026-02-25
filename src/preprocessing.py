from sklearn.model_selection import train_test_split
import pandas as pd
import config

def split_data(df: pd.DataFrame):
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE
    )

    return X_train, X_test, y_train, y_test