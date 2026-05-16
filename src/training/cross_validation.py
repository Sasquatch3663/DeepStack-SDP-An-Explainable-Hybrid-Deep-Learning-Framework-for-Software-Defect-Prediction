"""
cross_validation.py

Performs k-fold validation.
"""

from sklearn.model_selection import cross_val_score
import numpy as np


def perform_cross_validation(model, X, y, cv=5):
    """
    Evaluate model using cross-validation.
    """

    scores = cross_val_score(model, X, y, cv=cv, scoring="f1")

    return {
        "cv_mean_f1": np.mean(scores),
        "cv_std_f1": np.std(scores)
    }
