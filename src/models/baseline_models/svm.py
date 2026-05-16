"""
svm.py (SGD-based SVM)
"""

from sklearn.linear_model import SGDClassifier


def train_svm(X, y):
    model = SGDClassifier(loss="hinge", max_iter=1000, random_state=42)
    model.fit(X, y)
    return model