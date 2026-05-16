"""
feature_builder.py

Creates new features from existing data.
"""

# ✅ SAFE EXECUTION FIX
if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pandas as pd
import os

from src.utils.logger import get_logger
from src.utils.helper_functions import save_object

logger = get_logger("FeatureBuilder")


def build_features(df: pd.DataFrame, target_column: str = "defects") -> pd.DataFrame:
    df = df.copy()

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")

    numeric_cols = df.select_dtypes(include=["number"]).columns
    numeric_cols = [col for col in numeric_cols if col != target_column]

    # 🔥 Handle empty case
    if len(numeric_cols) < 2:
        logger.warning("Not enough numeric columns for feature engineering")
        return df

    try:
        df["feature_sum"] = df[numeric_cols].sum(axis=1)
        df["feature_mean"] = df[numeric_cols].mean(axis=1)
        df["feature_std"] = df[numeric_cols].std(axis=1)

        logger.info("Feature engineering completed")

    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")

    return df


# ✅ Standalone Execution
if __name__ == "__main__":
    from src.preprocessing.data_scaling import scale_data
    from src.preprocessing.normalization import normalize_data
    from src.preprocessing.missing_value_handler import handle_missing_values
    from src.data.data_cleaning import clean_dataset
    from src.data.dataset_merger import merge_datasets
    from src.data.load_dataset import load_all_datasets
    from src.preprocessing.label_standardizer import standardize_all_datasets

    logger.info("Running feature_builder standalone...")

    datasets = load_all_datasets("data/raw")
    datasets = standardize_all_datasets(datasets)
    merged = merge_datasets(datasets)
    cleaned = clean_dataset(merged)
    processed = handle_missing_values(cleaned)
    normalized = normalize_data(processed)
    scaled = scale_data(normalized)

    featured = build_features(scaled)

    os.makedirs("data/processed", exist_ok=True)
    save_object("data/processed/featured.pkl", featured)

    print("Feature building complete!")