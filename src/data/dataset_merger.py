"""
dataset_merger.py

Merges datasets from multiple sources with schema handling.
"""

# ✅ SAFE EXECUTION FIX
if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pandas as pd
from typing import Dict
from src.utils.helper_functions import save_object
import os

from src.utils.logger import get_logger

logger = get_logger("DatasetMerger")


def align_columns(datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Align columns across datasets (union of all columns).
    Missing columns are filled with NaN.
    """
    all_columns = set()

    for df in datasets.values():
        all_columns.update(df.columns)

    aligned_datasets = {}

    for name, df in datasets.items():
        aligned_df = df.copy()

        for col in all_columns:
            if col not in aligned_df.columns:
                aligned_df[col] = pd.NA

        aligned_datasets[name] = aligned_df[list(all_columns)]

    return aligned_datasets


def merge_datasets(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge datasets into one with source tracking.
    """
    if not datasets:
        raise ValueError("No datasets provided.")

    # Align schemas
    datasets = align_columns(datasets)

    merged_list = []

    for name, df in datasets.items():
        temp_df = df.copy()
        temp_df["source_dataset"] = name  # track origin
        merged_list.append(temp_df)

    merged_df = pd.concat(merged_list, ignore_index=True)

    logger.info(f"Merged dataset shape: {merged_df.shape}")
    return merged_df



# (only __main__ updated)

if __name__ == "__main__":
    from src.data.load_dataset import load_all_datasets
    from src.preprocessing.label_standardizer import standardize_all_datasets
    from src.preprocessing.label_standardizer import standardize_all_datasets

    logger.info("Running dataset_merger standalone...")

    datasets = load_all_datasets("data/raw")
    datasets = standardize_all_datasets(datasets)

    merged = merge_datasets(datasets)

    import os
    from src.utils.helper_functions import save_object

    os.makedirs("data/processed", exist_ok=True)
    save_object("data/processed/merged_dataset.pkl", merged)

    print("Merged dataset saved!")