"""
label_standardizer.py

Standardizes target labels across heterogeneous datasets.
"""

# ✅ SAFE EXECUTION FIX
if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pandas as pd
import os

from src.utils.logger import get_logger
from src.utils.helper_functions import save_object

logger = get_logger("LabelStandardizer")


# 🔥 POSSIBLE TARGET COLUMN NAMES
POSSIBLE_TARGETS = [
    "defects", "defect", "bug", "bugs",
    "is_defective", "is_buggy", "label", "target"
]


def detect_target_column(df: pd.DataFrame) -> str:
    """
    Detect target column from dataset.
    """
    for col in df.columns:
        if col.lower() in POSSIBLE_TARGETS:
            return col
    return None


def standardize_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert target column to unified binary 'defects'.
    """
    df = df.copy()

    target_col = detect_target_column(df)

    if target_col is None:
        logger.warning("No target column found. Dropping dataset.")
        return None

    logger.info(f"Detected target column: {target_col}")

    # Rename to 'defects'
    df.rename(columns={target_col: "defects"}, inplace=True)

    # Convert to numeric
    df["defects"] = pd.to_numeric(df["defects"], errors="coerce")

    # Drop invalid rows
    before = df.shape[0]
    df = df.dropna(subset=["defects"])
    after = df.shape[0]

    logger.info(f"Dropped {before - after} rows due to invalid labels")

    # Convert to binary
    df["defects"] = df["defects"].apply(lambda x: 1 if x > 0 else 0)

    # Ensure integer type
    df["defects"] = df["defects"].astype(int)

    return df


def standardize_all_datasets(datasets: dict) -> dict:
    """
    Apply label standardization to all datasets.
    """
    standardized = {}

    for name, df in datasets.items():
        result = standardize_target(df)

        if result is not None and not result.empty:
            standardized[name] = result
        else:
            logger.warning(f"Dropping dataset: {name} (no valid labels)")

    logger.info(f"Standardized {len(standardized)} datasets")

    return standardized


# ✅ Standalone Execution
if __name__ == "__main__":
    from src.data.load_dataset import load_all_datasets

    logger.info("Running label_standardizer standalone...")

    datasets = load_all_datasets("data/raw")

    standardized = standardize_all_datasets(datasets)

    # Merge all standardized datasets
    merged_df = pd.concat(standardized.values(), ignore_index=True)

    logger.info(f"Final standardized dataset shape: {merged_df.shape}")

    # Save
    os.makedirs("data/processed", exist_ok=True)
    save_object("data/processed/standardized_dataset.pkl", merged_df)

    print("Label standardization complete!")