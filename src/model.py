from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import GridSearchCV
from src.preprocessing import feature_engineering


def train_model(X_train, y_train):

    numerical_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns

    numeric_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore"
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("feature_engineering", FunctionTransformer(feature_engineering)),
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=2000))
        ]
    )

    param_grid = [
    # lbfgs supports only l2
    {
        "classifier__solver": ["lbfgs"],
        "classifier__penalty": ["l2"],
        "classifier__C": [0.01, 0.1, 1, 10],
        "classifier__class_weight": [None, "balanced"]
    },
    # liblinear supports l1 and l2
    {
        "classifier__solver": ["liblinear"],
        "classifier__penalty": ["l1", "l2"],
        "classifier__C": [0.01, 0.1, 1, 10],
        "classifier__class_weight": [None, "balanced"]
    }
    ]

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("\nBest Hyperparameters:", grid.best_params_)
    print("Best CV ROC-AUC:", round(grid.best_score_, 4))

    return grid.best_estimator_