"""
lime_explainer.py

LIME explainability with safe NumPy handling and image output.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer


class LimeExplainer:
    def __init__(self, X_train, feature_names=None):
        """
        Initialize LIME with safe NumPy conversion.
        """

        # 🔥 FIX: Convert to NumPy (CRITICAL)
        X_train = np.asarray(X_train, dtype=np.float32)

        if feature_names is None:
            feature_names = [f"f{i}" for i in range(X_train.shape[1])]

        self.explainer = LimeTabularExplainer(
            training_data=X_train,
            feature_names=feature_names,
            mode="classification"
        )

    def explain(self, model, X_instance, num_features=10):
        """
        Generate explanation for one instance.
        """
        X_instance = np.asarray(X_instance, dtype=np.float32)

        explanation = self.explainer.explain_instance(
            X_instance,
            model.predict_proba,
            num_features=num_features
        )

        return explanation

    def save_plot(self, explanation, path="results/lime_explanation.png"):
        """
        Save LIME explanation as image.
        """
        os.makedirs("results", exist_ok=True)

        fig = explanation.as_pyplot_figure()
        fig.tight_layout()
        fig.savefig(path, dpi=300)
        plt.close(fig)