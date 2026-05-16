"""
all_models_experiment.py

Final research-grade experiment pipeline:
- Feature selection BEFORE sampling
- Sampling applied dynamically
- No memory issues
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import pandas as pd

from src.utils.helper_functions import load_object
from src.utils.logger import get_logger

from src.models.baseline_models.logistic_regression import prepare_data, train_logistic_regression
from src.models.baseline_models.decision_tree import train_decision_tree
from src.models.baseline_models.random_forest import train_random_forest
from src.models.baseline_models.svm import train_svm
from src.models.baseline_models.xgboost_model import train_xgboost

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from imblearn.over_sampling import SMOTE, ADASYN

logger = get_logger("ExperimentRunner")


# -------------------------------
# Evaluation
# -------------------------------
def evaluate(model, X, y):
    y_pred = model.predict(X)

    try:
        y_prob = model.predict_proba(X)[:, 1]
    except:
        try:
            y_prob = model.decision_function(X)
        except:
            y_prob = y_pred

    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1_score": f1_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, y_prob)
    }


# -------------------------------
# Feature Selection
# -------------------------------
def fit_feature_selector(X, y):
    k = min(300, X.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    return selector, X_new


def transform_features(selector, X):
    return selector.transform(X)


# -------------------------------
# Sampling
# -------------------------------
def apply_sampling(X, y, method=None):
    if method == "smote":
        sampler = SMOTE(random_state=42)
        return sampler.fit_resample(X, y)

    if method == "adasyn":
        sampler = ADASYN(random_state=42)
        return sampler.fit_resample(X, y)

    return X, y


# -------------------------------
# Models
# -------------------------------
def get_models():
    return {
        "LogisticRegression": train_logistic_regression,
        "DecisionTree": train_decision_tree,
        "RandomForest": train_random_forest,
        "SVM_SGD": train_svm,
        "XGBoost": train_xgboost,
    }


# -------------------------------
# Main Experiment
# -------------------------------
def run_experiments():
    df = load_object("data/processed/encoded.pkl")

    X, y = prepare_data(df)

    # Feature selection (ONLY ONCE)
    selector, X_selected = fit_feature_selector(X, y)

    datasets = {
        "original": (X_selected, y),
        "smote": apply_sampling(X_selected, y, "smote"),
        "adasyn": apply_sampling(X_selected, y, "adasyn"),
    }

    models = get_models()
    results = []

    for d_name, (X_data, y_data) in datasets.items():
        logger.info(f"\nRunning on dataset: {d_name}")

        for m_name, train_fn in models.items():
            logger.info(f"Training {m_name} on {d_name}")

            model = train_fn(X_data, y_data)

            metrics = evaluate(model, X_data, y_data)

            metrics.update({
                "model": m_name,
                "dataset": d_name
            })

            results.append(metrics)

    return pd.DataFrame(results)


# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    logger.info("Starting experiment pipeline...")

    results_df = run_experiments()

    os.makedirs("results", exist_ok=True)
    results_df.to_csv("results/model_results.csv", index=False)

    print("\nResults:\n", results_df)
    print("\nSaved to results/model_results.csv")