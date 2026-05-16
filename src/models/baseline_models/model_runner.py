"""
model_runner.py

Train all baseline models together
"""

if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.utils.helper_functions import load_object, save_object
from src.models.baseline_models.logistic_regression import train_logistic_regression
from src.models.baseline_models.decision_tree import train_decision_tree
from src.models.baseline_models.random_forest import train_random_forest
from src.models.baseline_models.svm import train_svm
from src.models.baseline_models.xgboost_model import train_xgboost

import os

print("Loading dataset...")
df = load_object("data/processed/smote_data.pkl")

os.makedirs("models", exist_ok=True)

print("Training models...")

save_object("models/lr.pkl", train_logistic_regression(df))
save_object("models/dt.pkl", train_decision_tree(df))
save_object("models/rf.pkl", train_random_forest(df))
save_object("models/svm.pkl", train_svm(df))
save_object("models/xgb.pkl", train_xgboost(df))

print("All models trained and saved!")