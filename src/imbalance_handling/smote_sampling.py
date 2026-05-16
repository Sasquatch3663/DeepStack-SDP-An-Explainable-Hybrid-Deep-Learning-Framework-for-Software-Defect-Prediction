"""
smote_sampling.py

Applies SMOTE to balance dataset.
"""

# ✅ SAFE EXECUTION FIX
if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pandas as pd
import os
from imblearn.over_sampling import SMOTE

from src.utils.logger import get_logger
from src.utils.helper_functions import save_object

logger = get_logger("SMOTE")


def apply_smote(df: pd.DataFrame, target_column: str = "defects") -> pd.DataFrame:
    if target_column not in df.columns:
        raise ValueError("Target column not found")

    df = df.copy()

    # Target cleaning
    df[target_column] = pd.to_numeric(df[target_column], errors="coerce")
    df = df.dropna(subset=[target_column])

    if df.shape[0] == 0:
        raise ValueError("All rows dropped due to invalid target column.")

    df[target_column] = df[target_column].astype(int)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # 🔥 CRITICAL FIX
    X = X.fillna(0)
    X = X.replace([float("inf"), float("-inf")], 0)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled[target_column] = y_resampled

    logger.info(f"SMOTE applied: {df.shape} -> {df_resampled.shape}")

    return df_resampled


# ✅ Standalone Execution
if __name__ == "__main__":
    from src.preprocessing.encoding import encode_categorical
    from src.feature_engineering.feature_builder import build_features
    from src.preprocessing.data_scaling import scale_data
    from src.preprocessing.normalization import normalize_data
    from src.preprocessing.missing_value_handler import handle_missing_values
    from src.data.data_cleaning import clean_dataset
    from src.data.dataset_merger import merge_datasets
    from src.data.load_dataset import load_all_datasets
    from src.preprocessing.label_standardizer import standardize_all_datasets


    logger.info("Running SMOTE standalone...")

    datasets = load_all_datasets("data/raw")
    datasets = standardize_all_datasets(datasets)
    merged = merge_datasets(datasets)
    cleaned = clean_dataset(merged)
    processed = handle_missing_values(cleaned)
    normalized = normalize_data(processed)
    scaled = scale_data(normalized)
    featured = build_features(scaled)
    encoded = encode_categorical(featured)

    smote_df = apply_smote(encoded)

    os.makedirs("data/processed", exist_ok=True)
    save_object("data/processed/smote_data.pkl", smote_df)

    print("SMOTE applied successfully!")