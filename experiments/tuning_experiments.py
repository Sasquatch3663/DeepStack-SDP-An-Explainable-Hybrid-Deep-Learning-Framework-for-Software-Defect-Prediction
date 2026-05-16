"""
tuning_experiments.py

Runs tuning on multiple models.
"""

import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.helper_functions import load_object
from src.utils.logger import get_logger

from src.models.baseline_models.logistic_regression import prepare_data
from src.training.hyperparameter_tuning import tune_model
from src.training.cross_validation import perform_cross_validation

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE

logger = get_logger("TuningExperiment")


def run_tuning():
    df = load_object("data/processed/encoded.pkl")

    # Step 1: Prepare data
    X, y = prepare_data(df)

    # Step 2: Feature selection
    selector = SelectKBest(f_classif, k=min(300, X.shape[1]))
    X = selector.fit_transform(X, y)

    # Step 3: SMOTE
    X, y = SMOTE(random_state=42).fit_resample(X, y)

    results = []

    models = {
        "RandomForest": (
            RandomForestClassifier(),
            {"n_estimators": [50, 100], "max_depth": [None, 10]}
        ),
        "XGBoost": (
            XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
            {"n_estimators": [50, 100], "max_depth": [3, 6]}
        ),
        "SVM_SGD": (
            SGDClassifier(),
            {"alpha": [0.0001, 0.001]}
        )
    }

    for name, (model, params) in models.items():
        logger.info(f"Tuning {name}")

        best_model, best_params = tune_model(model, params, X, y)

        # Cross-validation
        cv_results = perform_cross_validation(best_model, X, y)

        f1 = f1_score(y, best_model.predict(X))

        results.append({
            "model": name,
            "f1_score": f1,
            "best_params": str(best_params),
            **cv_results
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    df = run_tuning()

    os.makedirs("results", exist_ok=True)
    df.to_csv("results/tuned_models.csv", index=False)

    print(df)