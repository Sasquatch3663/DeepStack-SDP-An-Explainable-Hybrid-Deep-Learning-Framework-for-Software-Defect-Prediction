"""
data_scaling.py

Applies Standard Scaling.
"""

# ✅ SAFE EXECUTION FIX
if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

from src.utils.logger import get_logger
from src.utils.helper_functions import save_object

logger = get_logger("DataScaling")


def scale_data(df: pd.DataFrame, target_column: str = "defects") -> pd.DataFrame:
    """
    Apply StandardScaler.
    """
    df = df.copy()

    if target_column not in df.columns:
        raise ValueError("Target column not found")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    numeric_cols = X.select_dtypes(include=["number"]).columns

    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    df_scaled = pd.concat([X, y], axis=1)

    logger.info("Scaling applied successfully")

    return df_scaled


# ✅ Standalone Execution
if __name__ == "__main__":
    from src.preprocessing.normalization import normalize_data
    from src.preprocessing.missing_value_handler import handle_missing_values
    from src.data.data_cleaning import clean_dataset
    from src.data.dataset_merger import merge_datasets
    from src.data.load_dataset import load_all_datasets
    from src.preprocessing.label_standardizer import standardize_all_datasets

    logger.info("Running data_scaling standalone...")

    datasets = load_all_datasets("data/raw")
    datasets = standardize_all_datasets(datasets)
    merged = merge_datasets(datasets)
    cleaned = clean_dataset(merged)
    processed = handle_missing_values(cleaned)
    normalized = normalize_data(processed)

    scaled = scale_data(normalized)

    os.makedirs("data/processed", exist_ok=True)
    save_object("data/processed/scaled.pkl", scaled)

    print("Scaling complete!")