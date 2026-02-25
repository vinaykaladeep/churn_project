from sklearn.metrics import confusion_matrix, classification_report

def evaluate_model(model, X_test, y_test, threshold=0.5):
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds)

    return cm, report, probs