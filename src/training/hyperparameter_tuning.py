"""
hyperparameter_tuning.py

Handles randomized hyperparameter search.
"""

from sklearn.model_selection import RandomizedSearchCV


def tune_model(model, param_grid, X, y, n_iter=10):
    """
    Perform randomized search to find best parameters.
    """

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring="f1",
        cv=3,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    search.fit(X, y)

    return search.best_estimator_, search.best_params_