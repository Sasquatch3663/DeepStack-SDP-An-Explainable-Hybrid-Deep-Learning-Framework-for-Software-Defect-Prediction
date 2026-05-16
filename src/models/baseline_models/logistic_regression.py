"""
logistic_regression.py

Handles:
- Data preparation
- Logistic Regression training
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression


def prepare_data(df, target_column="defects"):
    """
    Lightweight preprocessing for large tabular data.
    """

    # Clean target
    df[target_column] = pd.to_numeric(df[target_column], errors="coerce")
    df = df.dropna(subset=[target_column])
    df[target_column] = df[target_column].astype(int)

    # Split
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Convert to numeric
    X = X.apply(pd.to_numeric, errors="coerce")

    # Drop highly sparse columns
    missing_ratio = X.isna().mean()
    X = X.loc[:, missing_ratio < 0.9]

    # Fill remaining missing values
    X = X.fillna(0)

    return X, y


def train_logistic_regression(X, y):
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model