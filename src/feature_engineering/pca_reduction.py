"""
pca_reduction.py

Applies PCA for dimensionality reduction.
"""

# ✅ SAFE EXECUTION FIX
if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pandas as pd
import os
from sklearn.decomposition import PCA

from src.utils.logger import get_logger
from src.utils.helper_functions import save_object

logger = get_logger("PCA")


def apply_pca(df: pd.DataFrame, target_column: str = "defects", n_components: int = 10) -> pd.DataFrame:
    """
    Apply PCA.
    """
    df = df.copy()

    X = df.drop(columns=[target_column])
    y = df[target_column]

    numeric_cols = X.select_dtypes(include=["number"]).columns

    pca = PCA(n_components=min(n_components, len(numeric_cols)))
    X_pca = pca.fit_transform(X[numeric_cols])

    pca_cols = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    X_pca_df = pd.DataFrame(X_pca, columns=pca_cols)

    df_final = pd.concat([X_pca_df, y.reset_index(drop=True)], axis=1)

    logger.info(f"PCA applied with {X_pca.shape[1]} components")

    return df_final


# ✅ Standalone Execution
if __name__ == "__main__":
    from src.feature_engineering.feature_selection import select_features
    from src.feature_engineering.feature_builder import build_features
    from src.preprocessing.data_scaling import scale_data
    from src.preprocessing.normalization import normalize_data
    from src.preprocessing.missing_value_handler import handle_missing_values
    from src.data.data_cleaning import clean_dataset
    from src.data.dataset_merger import merge_datasets
    from src.data.load_dataset import load_all_datasets
    from src.preprocessing.label_standardizer import standardize_all_datasets

    logger.info("Running PCA standalone...")

    datasets = load_all_datasets("data/raw")
    datasets = standardize_all_datasets(datasets)
    merged = merge_datasets(datasets)
    cleaned = clean_dataset(merged)
    processed = handle_missing_values(cleaned)
    normalized = normalize_data(processed)
    scaled = scale_data(normalized)
    featured = build_features(scaled)
    selected = select_features(featured)

    pca_df = apply_pca(selected)

    os.makedirs("data/processed", exist_ok=True)
    save_object("data/processed/pca_data.pkl", pca_df)

    print("PCA reduction complete!")