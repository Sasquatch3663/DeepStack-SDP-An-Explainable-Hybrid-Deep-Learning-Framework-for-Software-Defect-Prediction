"""
class_weighting.py

Computes class weights for imbalanced datasets.
"""

# ✅ SAFE EXECUTION FIX
if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pandas as pd
import numpy as np
import os
from sklearn.utils.class_weight import compute_class_weight

from src.utils.logger import get_logger
from src.utils.helper_functions import save_object

logger = get_logger("ClassWeighting")


def compute_weights(df: pd.DataFrame, target_column: str = "defects") -> dict:
    if target_column not in df.columns:
        raise ValueError("Target column not found")

    y = df[target_column].copy()

    y = pd.to_numeric(y, errors="coerce")
    y = y.dropna()
    y = y.astype(int)

    unique_vals = np.unique(y)
    if len(unique_vals) < 2:
        raise ValueError(f"Only one class found: {unique_vals}")

    weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_vals,
        y=y
    )

    class_weights = dict(zip(unique_vals, weights))

    logger.info(f"Computed class weights: {class_weights}")

    return class_weights


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



    logger.info("Running class_weighting standalone...")

    datasets = load_all_datasets("data/raw")
    datasets = standardize_all_datasets(datasets)
    merged = merge_datasets(datasets)
    cleaned = clean_dataset(merged)
    processed = handle_missing_values(cleaned)
    normalized = normalize_data(processed)
    scaled = scale_data(normalized)
    featured = build_features(scaled)
    encoded = encode_categorical(featured)

    weights = compute_weights(encoded)

    os.makedirs("data/processed", exist_ok=True)
    save_object("data/processed/class_weights.pkl", weights)

    print("Class weights saved!")