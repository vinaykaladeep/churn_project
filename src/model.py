from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression


def train_model(X_train, y_train):

    numerical_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
    # categorical_cols = X_train.select_dtypes(include=["object"]).columns
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns

    numeric_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        drop=None
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000))
        ]
    )

    model.fit(X_train, y_train)

    return model