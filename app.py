"""Streamlit interface for evaluating saved Online Shoppers classifiers."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
TARGET = "Revenue"

st.set_page_config(
    page_title="Online Shopping Purchase Predictor",
    page_icon="🛍️",
    layout="wide",
)

@st.cache_resource
def load_assets():
    metadata_path = MODEL_DIR / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError("Model files are missing. Run train_models.py first.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    models = {
        name: joblib.load(MODEL_DIR / filename)
        for name, filename in metadata["models"].items()
    }
    return metadata, models

def normalize_target(series):
    if series.dtype == bool:
        return series.astype(bool)
    mapped = series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False,
         "yes": True, "no": False}
    )
    if mapped.isna().any():
        raise ValueError("Revenue must contain only True/False (or 1/0, yes/no).")
    return mapped.astype(bool)

st.title("Online Shopping Purchase Intention")
st.write(
    "Evaluate five trained classification models on an uploaded labelled test CSV. "
    "The application loads saved pipelines and does not retrain models."
)

try:
    metadata, models = load_assets()
except Exception as exc:
    st.error(f"Unable to load trained models: {exc}")
    st.info("In the project directory, run `python train_models.py` to create the model files.")
    st.stop()

with st.sidebar:
    st.header("Model configuration")
    selected_model = st.selectbox("Select classifier", list(models.keys()))
    st.caption("All preprocessing is stored inside each saved model pipeline.")
    st.divider()
    st.write("**Target:** Revenue")
    st.write("**Positive class:** Purchase (True)")
    st.write("**Evaluation:** Uploaded labelled test data")

uploaded_file = st.file_uploader("Upload test dataset (CSV)", type=["csv"])

if uploaded_file is None:
    st.info("Upload `test_data.csv` from the GitHub repository to evaluate a model.")
    st.subheader("Dataset and model information")
    c1, c2, c3 = st.columns(3)
    c1.metric("Input features", len(metadata["features"]))
    c2.metric("Saved classifiers", len(models))
    c3.metric("Test records", metadata["test_rows"])
    st.write("Required target column: `Revenue` (True = purchase, False = no purchase).")
    st.stop()

try:
    test_df = pd.read_csv(uploaded_file)
    required_columns = metadata["features"] + [TARGET]
    missing_columns = [col for col in required_columns if col not in test_df.columns]
    if missing_columns:
        st.error(f"CSV is missing required columns: {', '.join(missing_columns)}")
        st.stop()
    if test_df.empty:
        st.error("The uploaded CSV contains no records.")
        st.stop()

    X_test = test_df[metadata["features"]].copy()
    y_test = normalize_target(test_df[TARGET])

    if y_test.nunique() < 2:
        st.error("ROC-AUC requires both target classes. Upload a test CSV containing both Revenue classes.")
        st.stop()

    model = models[selected_model]
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_score),
        "Precision (macro)": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "Recall (macro)": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "F1-score (macro)": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }

    st.subheader(f"Evaluation results — {selected_model}")
    st.caption(f"Evaluated {len(test_df):,} records. No model training was performed.")
    metric_cols = st.columns(3)
    for i, (name, value) in enumerate(metrics.items()):
        metric_cols[i % 3].metric(name, f"{value:.4f}")

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        cm = confusion_matrix(y_test, y_pred, labels=[False, True])
        fig, ax = plt.subplots(figsize=(5, 4))
        image = ax.imshow(cm, interpolation="nearest")
        ax.set(
            xticks=[0, 1], yticks=[0, 1],
            xticklabels=["No purchase", "Purchase"],
            yticklabels=["No purchase", "Purchase"],
            xlabel="Predicted label", ylabel="Actual label",
            title="Confusion Matrix"
        )
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
        for row in range(2):
            for col in range(2):
                ax.text(col, row, str(cm[row, col]), ha="center", va="center")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("Classification report")
        report = classification_report(
            y_test, y_pred, labels=[False, True],
            target_names=["No purchase", "Purchase"],
            output_dict=True, zero_division=0
        )
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(4), use_container_width=True)

    st.subheader("Metric summary")
    st.dataframe(
        pd.DataFrame([metrics]).T.rename(columns={0: "Score"}).style.format("{:.4f}"),
        use_container_width=True
    )

    output_df = test_df.copy()
    output_df["Predicted_Revenue"] = y_pred
    output_df["Purchase_Probability"] = y_score
    st.download_button(
        "Download predictions CSV",
        data=output_df.to_csv(index=False).encode("utf-8"),
        file_name="online_shoppers_predictions.csv",
        mime="text/csv",
    )

except Exception as exc:
    st.error(f"Could not evaluate the uploaded CSV: {exc}")
    st.caption("Check that the CSV contains the expected feature columns and a valid Revenue target.")
