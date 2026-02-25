from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def train_model(X_train, y_train):

    # -------------------------
    # Step 1: Identify column types
    # -------------------------
    numerical_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns

    # -------------------------
    # Step 2: Define transformers
    # -------------------------
    numeric_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        drop=None
    )

    # -------------------------
    # Step 3: Column transformer
    # -------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )

    # -------------------------
    # Step 4: Create pipeline with RandomForest
    # -------------------------
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42))
        ]
    )

    # -------------------------
    # Step 5: Define hyperparameter grid
    # -------------------------
    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 10, 20],
        "classifier__min_samples_leaf": [1, 3, 5]
    }

    # -------------------------
    # Step 6: GridSearchCV
    # -------------------------
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring="recall",  # focus on churn detection
        n_jobs=-1
    )

    # -------------------------
    # Step 7: Fit grid search
    # -------------------------
    grid_search.fit(X_train, y_train)

    print("Best Parameters:", grid_search.best_params_)

    # Return best model
    return grid_search.best_estimator_