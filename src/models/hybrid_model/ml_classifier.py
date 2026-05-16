"""
ml_classifier.py

ML classifier on extracted features
"""

from sklearn.ensemble import RandomForestClassifier


def train_ml_classifier(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model