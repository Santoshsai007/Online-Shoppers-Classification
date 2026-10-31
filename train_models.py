"""Train and evaluate the five required classifiers for Assignment 2.

Run:
    python train_models.py

This script creates:
    models/*.joblib
    models/metadata.json
    test_data.csv
    results/model_comparison.csv
    results/classification_reports.json
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, classification_report
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "online_shoppers_intention.csv"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
TARGET = "Revenue"
RANDOM_STATE = 2026
TEST_SIZE = 0.20

# These columns represent measurements/counts and are treated as numeric.
NUMERIC_FEATURES = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay",
]
# Integer-coded identifiers are treated as categorical, not continuous.
CATEGORICAL_FEATURES = [
    "Month", "OperatingSystems", "Browser", "Region",
    "TrafficType", "VisitorType", "Weekend",
]

def build_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ], remainder="drop", sparse_threshold=0)

def calculate_metrics(model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    return {
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC": roc_auc_score(y_test, probabilities),
        "Precision": precision_score(y_test, predictions, average="macro", zero_division=0),
        "Recall": recall_score(y_test, predictions, average="macro", zero_division=0),
        "F1": f1_score(y_test, predictions, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_test, predictions),
    }

def main():
    MODEL_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    data = pd.read_csv(DATA_PATH)
    if TARGET not in data.columns:
        raise ValueError(f"Target column '{TARGET}' is missing.")
    if data[TARGET].isna().any():
        raise ValueError("Target column contains missing values.")

    # The supplied dataset uses Boolean labels. This also handles CSV strings.
    if data[TARGET].dtype != bool:
        mapped = data[TARGET].astype(str).str.strip().str.lower().map(
            {"true": True, "false": False, "1": True, "0": False,
             "yes": True, "no": False}
        )
        if mapped.isna().any():
            raise ValueError("Revenue must contain Boolean-like values (True/False).")
        data[TARGET] = mapped

    expected = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET])
    missing = expected - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    X = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = data[TARGET].astype(bool)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Keep the same split for all models to make the comparison fair.
    test_frame = X_test.copy()
    test_frame[TARGET] = y_test
    test_frame.to_csv(ROOT / "test_data.csv", index=False)

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=2000, solver="lbfgs"),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12, min_samples_leaf=5, random_state=RANDOM_STATE
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7, weights="distance"),
        "Gaussian Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(
            kernel="rbf", C=1.0, probability=True, random_state=RANDOM_STATE
        ),
    }

    comparison = []
    reports = {}
    for display_name, classifier in classifiers.items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ])
        pipeline.fit(X_train, y_train)
        metrics = calculate_metrics(pipeline, X_test, y_test)
        safe_name = {
            "Logistic Regression": "logistic_regression",
            "Decision Tree": "decision_tree",
            "K-Nearest Neighbors": "knn",
            "Gaussian Naive Bayes": "naive_bayes",
            "Support Vector Machine": "svm",
        }[display_name]
        joblib.dump(pipeline, MODEL_DIR / f"{safe_name}.joblib")

        y_pred = pipeline.predict(X_test)
        comparison.append({"Model": display_name, **metrics})
        reports[display_name] = classification_report(
            y_test, y_pred, labels=[False, True],
            target_names=["No purchase", "Purchase"],
            output_dict=True, zero_division=0
        )
        print(f"{display_name}: " + ", ".join(
            f"{key}={value:.4f}" for key, value in metrics.items()
        ))

    comparison_df = pd.DataFrame(comparison)
    comparison_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    with open(RESULTS_DIR / "classification_reports.json", "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)

    metadata = {
        "target": TARGET,
        "features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "positive_class": True,
        "positive_class_label": "Purchase",
        "negative_class_label": "No purchase",
        "metric_averaging": "macro for precision, recall and F1; binary positive-class ROC-AUC",
        "models": {
            "Logistic Regression": "logistic_regression.joblib",
            "Decision Tree": "decision_tree.joblib",
            "K-Nearest Neighbors": "knn.joblib",
            "Gaussian Naive Bayes": "naive_bayes.joblib",
            "Support Vector Machine": "svm.joblib",
        },
    }
    with open(MODEL_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nModel comparison:")
    print(comparison_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved test data ({len(test_frame)} rows), five models and evaluation results.")

if __name__ == "__main__":
    main()
