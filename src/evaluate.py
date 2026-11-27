# AINT602 ML Pipeline Evaluation Script
# Matt Chamley 26 November 2026

# Evaluation choices
# Accuracy is reported but not used to choose a model, because about 86% of patients are
# not readmitted and a model that never flags anyone scores that without helping.

# Models are compared on ROC-AUC, with recall, precision and the confusion matrix
# showing the false negatives (missed readmissions) and false positives (false alarms).
# Results are repeated by sex and age band to check the model works across groups.


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    precision_score, recall_score, roc_auc_score, roc_curve,
)

from src.features import add_features


def score(model, X, y):
    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)
    return {
        "accuracy": accuracy_score(y, preds),
        "roc_auc": roc_auc_score(y, probs),
        "pr_auc": average_precision_score(y, probs),
        "recall": recall_score(y, preds, zero_division=0),
        "precision": precision_score(y, preds, zero_division=0),
        "brier": brier_score_loss(y, probs),
    }


def confusion(model, X, y):
    tn, fp, fn, tp = confusion_matrix(y, model.predict(X)).ravel()
    return {"true_negative": int(tn), "false_positive": int(fp),
            "false_negative": int(fn), "true_positive": int(tp)}


def group_report(model, X, y, column):
    data = add_features(X)
    data["actual"] = y.values
    data["predicted"] = model.predict(X)
    rows = []
    for group, part in data.groupby(column):
        rows.append({
            "group": group,
            "patients": len(part),
            "readmission_rate": part["actual"].mean(),
            "recall": recall_score(part["actual"], part["predicted"], zero_division=0),
            "predicted_high_risk_rate": part["predicted"].mean(),
        })
    return pd.DataFrame(rows)


def save_roc_plot(models, X, y, path):
    plt.figure(figsize=(5, 5))
    for name, model in models.items():
        fpr, tpr, _ = roc_curve(y, model.predict_proba(X)[:, 1])
        plt.plot(fpr, tpr, label=name)
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curves on the validation set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
