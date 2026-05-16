"""
encoding.py

Handles categorical feature encoding with strict numeric enforcement
and removal of non-feature columns (IDs, filenames, high-cardinality).
"""

# ✅ SAFE EXECUTION FIX
if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pandas as pd
import numpy as np
import os

from src.utils.logger import get_logger
from src.utils.helper_functions import save_object

logger = get_logger("Encoding")


def _drop_non_feature_columns(X: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that are likely identifiers / filenames / useless high-cardinality strings.
    """
    cols_to_drop = []

    for col in X.columns:
        if X[col].dtype == "object":
            col_as_str = X[col].astype(str)

            # Heuristic 1: file-like values (e.g., *.csv)
            has_file_pattern = col_as_str.str.contains(r"\.csv", case=False, na=False).any()

            # Heuristic 2: very high cardinality (almost unique per row)
            nunique = col_as_str.nunique(dropna=True)
            high_cardinality = nunique > 0.9 * len(X)

            if has_file_pattern or high_cardinality:
                cols_to_drop.append(col)

    if cols_to_drop:
        logger.info(f"Dropping non-feature columns: {cols_to_drop}")
        X = X.drop(columns=cols_to_drop)

    return X


def encode_categorical(df: pd.DataFrame, target_column: str = "defects") -> pd.DataFrame:
    df = df.copy()

    if target_column not in df.columns:
        raise ValueError("Target column not found")

    # =============================
    # 🔥 STEP 1: CLEAN STRING VALUES
    # =============================
    df.replace(["unknown", "UNK", "?", "None", None], np.nan, inplace=True)

    # =============================
    # 🔥 STEP 2: TARGET CLEANING
    # =============================
    df[target_column] = pd.to_numeric(df[target_column], errors="coerce")
    df = df.dropna(subset=[target_column])
    df[target_column] = df[target_column].astype(int)

    # =============================
    # 🔥 STEP 3: SPLIT FEATURES
    # =============================
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # =============================
    # 🔥 STEP 3.5: DROP NON-FEATURE COLS (CRITICAL)
    # =============================
    X = _drop_non_feature_columns(X)

    # =============================
    # 🔥 STEP 4: COERCE NUMERIC WHERE POSSIBLE
    # =============================
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = pd.to_numeric(X[col], errors="ignore")

    # =============================
    # 🔥 STEP 5: ENCODE CATEGORICALS
    # =============================
    categorical_cols = X.select_dtypes(exclude=["number"]).columns
    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        logger.info(f"Encoded {len(categorical_cols)} categorical columns")

    # =============================
    # 🔥 STEP 6: FINAL NUMERIC SAFETY
    # =============================
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(0)

    # Keep only numeric columns (hard filter)
    X = X.select_dtypes(include=[np.number])

    # Force dtype for DL compatibility
    X = X.astype(np.float32)

    # =============================
    # 🔥 STEP 7: RECOMBINE
    # =============================
    df_encoded = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)

    return df_encoded


# =============================
# ✅ STANDALONE EXECUTION
# =============================
if __name__ == "__main__":
    from src.feature_engineering.feature_builder import build_features
    from src.preprocessing.data_scaling import scale_data
    from src.preprocessing.normalization import normalize_data
    from src.preprocessing.missing_value_handler import handle_missing_values
    from src.data.data_cleaning import clean_dataset
    from src.data.dataset_merger import merge_datasets
    from src.data.load_dataset import load_all_datasets
    from src.preprocessing.label_standardizer import standardize_all_datasets

    logger.info("Running encoding standalone...")

    datasets = load_all_datasets("data/raw")
    datasets = standardize_all_datasets(datasets)
    merged = merge_datasets(datasets)
    cleaned = clean_dataset(merged)
    processed = handle_missing_values(cleaned)
    normalized = normalize_data(processed)
    scaled = scale_data(normalized)
    featured = build_features(scaled)

    encoded = encode_categorical(featured)

    os.makedirs("data/processed", exist_ok=True)
    save_object("data/processed/encoded.pkl", encoded)

    print("✅ Encoding complete! Clean numeric dataset saved.")