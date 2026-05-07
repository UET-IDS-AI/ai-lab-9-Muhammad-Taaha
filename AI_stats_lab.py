"""
AI_stats_lab.py
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# ============================================================
# Question 1: Confusion Matrix, Metrics, and Threshold Effects
# ============================================================

def confusion_matrix_counts(y_true, y_pred):

    TP = np.sum((y_true == 1) & (y_pred == 1))

    FP = np.sum((y_true == 0) & (y_pred == 1))

    FN = np.sum((y_true == 1) & (y_pred == 0))

    TN = np.sum((y_true == 0) & (y_pred == 0))

    return (TP, FP, FN, TN)


def classification_metrics(y_true, y_pred):

    TP, FP, FN, TN = confusion_matrix_counts(
        y_true=y_true,
        y_pred=y_pred
    )

    if (TP + FN) == 0:
        recall = 0.0
    else:
        recall = TP / (TP + FN)

    if (FP + TN) == 0:
        fallout = 0.0
    else:
        fallout = FP / (FP + TN)

    if (TP + FP) == 0:
        precision = 0.0
    else:
        precision = TP / (TP + FP)

    total = TP + FP + FN + TN

    if total == 0:
        accuracy = 0.0
    else:
        accuracy = (TP + TN) / total

    return {
        "recall": recall,
        "fallout": fallout,
        "precision": precision,
        "accuracy": accuracy
    }


def apply_threshold(scores, threshold):

    scores = np.array(scores)

    predictions = np.where(scores >= threshold, 1, 0)

    return predictions


def threshold_metrics_analysis(y_true, scores, thresholds):

    results = []

    for threshold in thresholds:

        predictions = apply_threshold(scores, threshold)

        metrics = classification_metrics(y_true, predictions)

        result = {
            "threshold": threshold,
            "recall": metrics["recall"],
            "fallout": metrics["fallout"],
            "precision": metrics["precision"],
            "accuracy": metrics["accuracy"]
        }

        results.append(result)

    return results


# ============================================================
# Question 2: Train Two Classifiers and Evaluate Them
# ============================================================

def train_two_classifiers(X_train, y_train):

    logistic_model = LogisticRegression(max_iter=1000)

    decision_tree_model = DecisionTreeClassifier(random_state=0)

    logistic_model.fit(X_train, y_train)

    decision_tree_model.fit(X_train, y_train)

    return {
        "logistic_regression": logistic_model,
        "decision_tree": decision_tree_model
    }


def evaluate_classifier(model, X_test, y_test, threshold=0.5):

    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = apply_threshold(probabilities, threshold)

    TP, FP, FN, TN = confusion_matrix_counts(
        y_test,
        predictions
    )

    metrics = classification_metrics(
        y_test,
        predictions
    )

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "recall": metrics["recall"],
        "fallout": metrics["fallout"],
        "precision": metrics["precision"],
        "accuracy": metrics["accuracy"]
    }


def compare_classifiers(
    X_train,
    y_train,
    X_test,
    y_test,
    threshold=0.5
):

    models = train_two_classifiers(X_train, y_train)

    logistic_result = evaluate_classifier(
        models["logistic_regression"],
        X_test,
        y_test,
        threshold
    )

    decision_tree_result = evaluate_classifier(
        models["decision_tree"],
        X_test,
        y_test,
        threshold
    )

    return {
        "logistic_regression": logistic_result,
        "decision_tree": decision_tree_result
    }


if __name__ == "__main__":
    print("Implement all required functions.")