"""
missing_value_handler.py

Handles advanced missing value strategies.
"""

# ✅ SAFE EXECUTION FIX
if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pandas as pd
import os

from src.utils.logger import get_logger
from src.utils.helper_functions import save_object

logger = get_logger("MissingValueHandler")


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply advanced missing value handling.
    """
    df = df.copy()

    # Drop columns with >50% missing values
    threshold = 0.5
    missing_ratio = df.isnull().mean()
    cols_to_drop = missing_ratio[missing_ratio > threshold].index

    df = df.drop(columns=cols_to_drop)
    logger.info(f"Dropped {len(cols_to_drop)} columns with >50% missing values")

    # Fill numeric columns with median
    numeric_cols = df.select_dtypes(include=["number"]).columns

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        median_val = df[col].median()
        if pd.notna(median_val):
            df[col] = df[col].fillna(median_val)

    # Fill categorical with mode
    categorical_cols = df.select_dtypes(exclude=["number"]).columns

    for col in categorical_cols:
        mode_val = df[col].mode()
        fill_val = mode_val[0] if not mode_val.empty else "unknown"
        df[col] = df[col].fillna(fill_val)

    logger.info("Missing value handling completed")

    return df


# ✅ Standalone Execution
if __name__ == "__main__":
    from src.data.data_cleaning import clean_dataset
    from src.data.dataset_merger import merge_datasets
    from src.data.load_dataset import load_all_datasets
    from src.preprocessing.label_standardizer import standardize_all_datasets

    logger.info("Running missing_value_handler standalone...")

    datasets = load_all_datasets("data/raw")
    datasets = standardize_all_datasets(datasets)
    merged = merge_datasets(datasets)
    cleaned = clean_dataset(merged)

    processed = handle_missing_values(cleaned)

    # Save
    os.makedirs("data/processed", exist_ok=True)
    save_object("data/processed/missing_handled.pkl", processed)

    print("Missing value handling complete!")