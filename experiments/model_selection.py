"""
model_selection.py

Selects best model based on tuning results.
"""

import pandas as pd


def select_best_model(path="results/tuned_models.csv"):
    df = pd.read_csv(path)

    best = df.sort_values(by="f1_score", ascending=False).iloc[0]

    print("\nBest Model Selected:\n")
    print(best)

    return best