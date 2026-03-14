import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import argparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)

# ─────────────────────────────────────────────
# ARGUMENT PARSER
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--max_depth", type=int, default=None)
parser.add_argument("--min_samples_split", type=int, default=2)
args = parser.parse_args()

# ─────────────────────────────────────────────
# MLFLOW SETUP
# ─────────────────────────────────────────────
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "mlruns"))
mlflow.set_experiment("credit_risk_ci")

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
train = pd.read_csv("credit_risk_dataset_preprocessing/train.csv")
test  = pd.read_csv("credit_risk_dataset_preprocessing/test.csv")

TARGET = "loan_status"
X_train = train.drop(columns=[TARGET])
y_train = train[TARGET]
X_test  = test.drop(columns=[TARGET])
y_test  = test[TARGET]

# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
with mlflow.start_run():

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth if args.max_depth != 0 else None,
        min_samples_split=args.min_samples_split,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    acc       = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall    = recall_score(y_test, y_pred, average="weighted")
    f1        = f1_score(y_test, y_pred, average="weighted")
    roc_auc   = roc_auc_score(y_test, y_pred_prob)

    # Log params
    mlflow.log_param("n_estimators",      args.n_estimators)
    mlflow.log_param("max_depth",         args.max_depth)
    mlflow.log_param("min_samples_split", args.min_samples_split)

    # Log metrics
    mlflow.log_metric("accuracy",  acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall",    recall)
    mlflow.log_metric("f1_score",  f1)
    mlflow.log_metric("roc_auc",   roc_auc)

    # Log model
    mlflow.sklearn.log_model(model, artifact_path="model")

    # Artifacts
    os.makedirs("outputs", exist_ok=True)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    cm_path = "outputs/confusion_matrix.png"
    fig.savefig(cm_path, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(cm_path, artifact_path="plots")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}", color="darkorange")
    ax2.plot([0, 1], [0, 1], linestyle="--", color="navy")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")
    ax2.legend(loc="lower right")
    roc_path = "outputs/roc_curve.png"
    fig2.savefig(roc_path, bbox_inches="tight")
    plt.close(fig2)
    mlflow.log_artifact(roc_path, artifact_path="plots")

    # Classification Report
    report = classification_report(y_test, y_pred, output_dict=True)
    report_path = "outputs/classification_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    mlflow.log_artifact(report_path, artifact_path="reports")

    print(f"Accuracy : {acc:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print("✅ Training selesai!")