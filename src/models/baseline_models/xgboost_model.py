"""
xgboost_model.py
"""

from xgboost import XGBClassifier


def train_xgboost(X, y):
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X, y)
    return model