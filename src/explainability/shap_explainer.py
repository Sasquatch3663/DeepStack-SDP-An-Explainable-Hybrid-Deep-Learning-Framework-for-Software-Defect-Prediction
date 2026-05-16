"""
shap_explainer.py

SHAP explainability saving ONLY images (paper-ready).
"""

import numpy as np
import shap
import matplotlib.pyplot as plt
import os


class ShapExplainer:
    def __init__(self, model, mode="auto"):
        self.model = model
        self.mode = mode
        self.explainer = None

    def fit(self, X_background):
        X_background = np.asarray(X_background, dtype=np.float32)

        if self.mode == "tree":
            self.explainer = shap.TreeExplainer(self.model)
        else:
            self.explainer = shap.Explainer(self.model.predict_proba, X_background)

    def compute_shap(self, X, max_samples=500):
        X = np.asarray(X, dtype=np.float32)
        X_sample = X[:max_samples]
        shap_values = self.explainer(X_sample)
        return shap_values, X_sample

    def save_summary_plot(self, shap_values):
        os.makedirs("results", exist_ok=True)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values[:, :, 1], show=False)
        plt.tight_layout()
        plt.savefig("results/shap_summary.png", dpi=300)
        plt.close()

    def save_bar_plot(self, shap_values):
        os.makedirs("results", exist_ok=True)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values[:, :, 1],
            plot_type="bar",
            show=False
        )
        plt.tight_layout()
        plt.savefig("results/shap_bar.png", dpi=300)
        plt.close()

    def save_waterfall_plot(self, shap_values, index=0):
        os.makedirs("results", exist_ok=True)

        plt.figure(figsize=(8, 6))
        shap.plots.waterfall(shap_values[index, :, 1], show=False)
        plt.tight_layout()
        plt.savefig("results/shap_waterfall.png", dpi=300)
        plt.close()