from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

def evaluate_model(model, X_test, y_test, threshold=0.3):
    """
    Production-grade evaluation:
    - Uses business threshold
    - Returns structured metrics
    """

    # Probabilities
    probs = model.predict_proba(X_test)[:, 1]

    # Apply threshold
    preds = (probs >= threshold).astype(int)

    metrics = {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds), 4),
        "recall": round(recall_score(y_test, preds), 4),
        "f1_score": round(f1_score(y_test, preds), 4),
        "roc_auc": round(roc_auc_score(y_test, probs), 4),
        "threshold_used": threshold
    }

    cm = confusion_matrix(y_test, preds)

    return metrics, cm, probs