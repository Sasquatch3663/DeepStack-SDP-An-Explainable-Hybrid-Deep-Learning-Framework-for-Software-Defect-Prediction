"""
visualization.py

Final research-grade visualization module.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

FIG_DIR = "reports/figures"
os.makedirs(FIG_DIR, exist_ok=True)


# ----------------------------------
# LOAD RESULTS
# ----------------------------------
def load_all_results():
    files = [
        "results/model_results.csv",
        "results/dl_results.csv",
        "results/hybrid_results.csv",
        "results/tuned_models.csv"
    ]

    dfs = []
    for file in files:
        if os.path.exists(file):
            dfs.append(pd.read_csv(file))

    if not dfs:
        raise ValueError("No result files found")

    return pd.concat(dfs, ignore_index=True)


# ----------------------------------
# MODEL TYPE CLASSIFICATION
# ----------------------------------
def classify_model(model):
    if "Hybrid" in model:
        return "Hybrid"
    elif model in ["CNN", "LSTM"]:
        return "Deep Learning"
    else:
        return "Machine Learning"


# ----------------------------------
# F1 SCORE (SORTED)
# ----------------------------------
def plot_f1(df):
    df = df.sort_values(by="f1_score", ascending=False)

    plt.figure(figsize=(12, 6))
    plt.barh(df["model"], df["f1_score"])

    plt.xlabel("F1 Score")
    plt.title("F1 Score Comparison")
    plt.gca().invert_yaxis()

    plt.savefig(os.path.join(FIG_DIR, "f1_score.png"), bbox_inches="tight")
    plt.close()


# ----------------------------------
# ACCURACY
# ----------------------------------
def plot_accuracy(df):
    df = df.sort_values(by="accuracy", ascending=False)

    plt.figure(figsize=(12, 6))
    plt.barh(df["model"], df["accuracy"])

    plt.xlabel("Accuracy")
    plt.title("Accuracy Comparison")
    plt.gca().invert_yaxis()

    plt.savefig(os.path.join(FIG_DIR, "accuracy.png"), bbox_inches="tight")
    plt.close()


# ----------------------------------
# PRECISION vs RECALL (GROUPED)
# ----------------------------------
def plot_precision_recall(df):
    df["type"] = df["model"].apply(classify_model)

    plt.figure(figsize=(8, 6))

    for t in df["type"].unique():
        subset = df[df["type"] == t]
        plt.scatter(subset["precision"], subset["recall"], label=t, s=80)

    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.title("Precision vs Recall (Grouped)")
    plt.legend()

    plt.savefig(os.path.join(FIG_DIR, "precision_recall.png"), bbox_inches="tight")
    plt.close()


# ----------------------------------
# MULTI METRIC PLOT
# ----------------------------------
def plot_all_metrics(df):
    df = df.sort_values(by="f1_score", ascending=False).head(10)

    x = range(len(df))

    plt.figure(figsize=(12, 6))

    for metric in ["accuracy", "precision", "recall", "f1_score"]:
        plt.plot(x, df[metric], marker="o", label=metric)

    plt.xticks(x, df["model"], rotation=45)
    plt.title("Top Models - Metric Comparison")
    plt.legend()

    plt.savefig(os.path.join(FIG_DIR, "all_metrics.png"), bbox_inches="tight")
    plt.close()


# ----------------------------------
# ROC CURVE
# ----------------------------------
def plot_roc(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")

    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()

    plt.savefig(os.path.join(FIG_DIR, "roc_curve.png"), bbox_inches="tight")
    plt.close()


# ----------------------------------
# CONFUSION MATRIX
# ----------------------------------
def plot_confusion_matrix(cm):
    plt.figure()
    plt.imshow(cm)

    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.text(j, i, cm[i][j], ha="center", va="center")

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig(os.path.join(FIG_DIR, "confusion_matrix.png"), bbox_inches="tight")
    plt.close()