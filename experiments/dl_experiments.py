"""
dl_experiments.py

Deep learning experiments integrated with Phase 6 pipeline.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import pandas as pd
import numpy as np

from src.utils.helper_functions import load_object
from src.utils.logger import get_logger

from src.models.baseline_models.logistic_regression import prepare_data
from src.models.deep_learning_models.cnn_model import train_cnn
from src.models.deep_learning_models.lstm_model import train_lstm

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from imblearn.over_sampling import SMOTE, ADASYN

logger = get_logger("DL_Experiment")


# -------------------------------
# Evaluation
# -------------------------------
def evaluate_dl(model, X, y):
    preds = model.predict(X)

    y_pred = (preds > 0.5).astype(int)

    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1_score": f1_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, preds)
    }


# -------------------------------
# Feature Selection
# -------------------------------
def fit_feature_selector(X, y):
    k = min(300, X.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    return selector, X_new



def apply_sampling(X, y, method=None):
    if method == "smote":
        return SMOTE(random_state=42).fit_resample(X, y)

    if method == "adasyn":
        return ADASYN(random_state=42).fit_resample(X, y)

    return X, y


# -------------------------------
# Main DL Experiment
# -------------------------------
def run_dl_experiments():
    df = load_object("data/processed/encoded.pkl")

    X, y = prepare_data(df)

    # Feature selection
    selector, X_selected = fit_feature_selector(X, y)

    datasets = {
        "original": (X_selected, y),
        "smote": apply_sampling(X_selected, y, "smote"),
        "adasyn": apply_sampling(X_selected, y, "adasyn"),
    }

    results = []

    for d_name, (X_data, y_data) in datasets.items():
        logger.info(f"\nRunning DL models on {d_name}")


        X_cnn = X_data.reshape(X_data.shape[0], X_data.shape[1], 1)

        cnn_model = train_cnn(X_cnn, y_data)
        cnn_metrics = evaluate_dl(cnn_model, X_cnn, y_data)

        cnn_metrics.update({"model": "CNN", "dataset": d_name})
        results.append(cnn_metrics)


        X_lstm = X_data.reshape(X_data.shape[0], 1, X_data.shape[1])

        lstm_model = train_lstm(X_lstm, y_data)
        lstm_metrics = evaluate_dl(lstm_model, X_lstm, y_data)

        lstm_metrics.update({"model": "LSTM", "dataset": d_name})
        results.append(lstm_metrics)

    return pd.DataFrame(results)


# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    logger.info("Starting DL experiments...")

    results_df = run_dl_experiments()

    os.makedirs("results", exist_ok=True)
    results_df.to_csv("results/dl_results.csv", index=False)

    print("\nDL Results:\n", results_df)