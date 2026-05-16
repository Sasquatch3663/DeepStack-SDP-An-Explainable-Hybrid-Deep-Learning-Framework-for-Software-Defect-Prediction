"""
explainability_pipeline.py

Generates SHAP + LIME explainability figures (paper-ready).
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import joblib
import numpy as np

from src.explainability.shap_explainer import ShapExplainer
from src.explainability.lime_explainer import LimeExplainer


def run_explainability():

    os.makedirs("results", exist_ok=True)

    print("🔹 Loading data...")

    X_test = joblib.load("data/processed/X_test.pkl")

    # 🔥 FIX: Ensure NumPy format
    X_test = np.asarray(X_test, dtype=np.float32)

    print("🔹 Loading model...")
    model = joblib.load("models/baseline_smote.pkl")

    # =========================
    # 🔹 SHAP
    # =========================
    print("🔹 Running SHAP...")

    shap_exp = ShapExplainer(model, mode="tree")
    shap_exp.fit(X_test[:200])

    shap_values, X_sample = shap_exp.compute_shap(X_test)

    shap_exp.save_summary_plot(shap_values)
    shap_exp.save_bar_plot(shap_values)
    shap_exp.save_waterfall_plot(shap_values)

    # =========================
    # 🔹 LIME
    # =========================
    print("🔹 Running LIME...")

    lime_exp = LimeExplainer(X_test)

    explanation = lime_exp.explain(model, X_test[0])
    lime_exp.save_plot(explanation)

    print("\n✅ All explainability images saved in /results/")


if __name__ == "__main__":
    run_explainability()