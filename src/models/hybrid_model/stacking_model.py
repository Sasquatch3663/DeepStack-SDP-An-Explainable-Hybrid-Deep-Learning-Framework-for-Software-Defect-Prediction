import numpy as np
from sklearn.model_selection import KFold
from sklearn.base import clone

class StackingModel:
    def __init__(self, base_models, n_folds=5):
        self.base_models = base_models
        self.n_folds = n_folds
        self.trained_models = []

    def fit(self, X, y):
        self.trained_models = []
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        for model in self.base_models:
            models_fold = []

            for train_idx, val_idx in kf.split(X):
                X_train, y_train = X[train_idx], y[train_idx]

                m = clone(model)
                m.fit(X_train, y_train)

                models_fold.append(m)

            self.trained_models.append(models_fold)

    def predict(self, X):
        S_test = np.zeros((X.shape[0], len(self.base_models)))

        for i, models in enumerate(self.trained_models):
            preds = np.column_stack([m.predict(X) for m in models])
            S_test[:, i] = preds.mean(axis=1)

        return S_test