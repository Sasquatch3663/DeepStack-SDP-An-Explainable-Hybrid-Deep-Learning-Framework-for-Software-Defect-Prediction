"""
decision_tree.py
"""

from sklearn.tree import DecisionTreeClassifier


def train_decision_tree(X, y):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)
    return model