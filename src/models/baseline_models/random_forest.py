"""
random_forest.py
"""

from sklearn.ensemble import RandomForestClassifier


def train_random_forest(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model