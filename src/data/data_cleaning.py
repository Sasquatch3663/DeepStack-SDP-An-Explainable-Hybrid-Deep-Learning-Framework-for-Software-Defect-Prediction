"""
data_cleaning.py

Handles dataset cleaning and normalization.
"""


if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pandas as pd
from src.utils.helper_functions import save_object
import os

from src.utils.logger import get_logger

logger = get_logger("DataCleaning")


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    original_shape = df.shape

    df = df.drop_duplicates()

    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    # Target must already be standardized → just ensure clean
    if "defects" in df.columns:
        df["defects"] = pd.to_numeric(df["defects"], errors="coerce")
        df = df.dropna(subset=["defects"])
        df["defects"] = df["defects"].astype(int)

    # Numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns

    for col in numeric_cols:
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            median_value = df[col].median()
            if pd.notna(median_value):
                df[col] = df[col].fillna(median_value)
        except:
            continue

    # Categorical columns
    categorical_cols = df.select_dtypes(exclude=["number"]).columns

    for col in categorical_cols:
        try:
            df[col] = df[col].fillna("unknown")
        except:
            continue

    logger.info(f"Cleaned dataset: {original_shape} -> {df.shape}")

    return df



if __name__ == "__main__":
    from src.data.load_dataset import load_all_datasets
    from src.data.dataset_merger import merge_datasets
    from src.preprocessing.label_standardizer import standardize_all_datasets

    logger.info("Running data_cleaning standalone...")

    datasets = load_all_datasets("data/raw")
    datasets = standardize_all_datasets(datasets)
    merged = merge_datasets(datasets)
    cleaned = clean_dataset(merged)

    
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)

    save_object(os.path.join(processed_dir, "cleaned_dataset.pkl"), cleaned)

    print("Cleaned dataset saved!")
