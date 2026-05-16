"""
model_comparison.py

Creates structured comparison tables.
"""

import pandas as pd


def generate_full_comparison(df):
    """
    Creates pivot table: Model vs Dataset (F1 Score)
    """

    if "dataset" in df.columns:
        return df.pivot_table(
            index="model",
            columns="dataset",
            values="f1_score"
        )

    return df


def save_comparison_table(df, path="results/final_comparison.csv"):
    df.to_csv(path)
    print(f"Saved comparison table → {path}")