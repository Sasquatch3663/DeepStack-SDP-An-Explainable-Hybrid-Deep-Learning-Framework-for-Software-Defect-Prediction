"""
feature_selection.py

Selects important features using statistical methods.
"""

# ✅ SAFE EXECUTION FIX
if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pandas as pd
import os
from sklearn.feature_selection import SelectKBest, f_classif

from src.utils.logger import get_logger
from src.utils.helper_functions import save_object

logger = get_logger("FeatureSelection")


def select_features(df: pd.DataFrame, target_column: str = "defects", k: int = 20) -> pd.DataFrame:
    """
    Select top k features.
    """
    df = df.copy()

    X = df.drop(columns=[target_column])
    y = df[target_column]

    numeric_cols = X.select_dtypes(include=["number"]).columns

    selector = SelectKBest(score_func=f_classif, k=min(k, len(numeric_cols)))
    X_selected = selector.fit_transform(X[numeric_cols], y)

    selected_cols = numeric_cols[selector.get_support()]
    X_selected_df = pd.DataFrame(X_selected, columns=selected_cols)

    df_final = pd.concat([X_selected_df, y.reset_index(drop=True)], axis=1)

    logger.info(f"Selected {len(selected_cols)} features")

    return df_final


# ✅ Standalone Execution
if __name__ == "__main__":
    from src.feature_engineering.feature_builder import build_features
    from src.preprocessing.data_scaling import scale_data
    from src.preprocessing.normalization import normalize_data
    from src.preprocessing.missing_value_handler import handle_missing_values
    from src.data.data_cleaning import clean_dataset
    from src.data.dataset_merger import merge_datasets
    from src.data.load_dataset import load_all_datasets
    from src.preprocessing.label_standardizer import standardize_all_datasets

    logger.info("Running feature_selection standalone...")

    datasets = load_all_datasets("data/raw")
    datasets = standardize_all_datasets(datasets)
    merged = merge_datasets(datasets)
    cleaned = clean_dataset(merged)
    processed = handle_missing_values(cleaned)
    normalized = normalize_data(processed)
    scaled = scale_data(normalized)
    featured = build_features(scaled)

    selected = select_features(featured)

    os.makedirs("data/processed", exist_ok=True)
    save_object("data/processed/selected_features.pkl", selected)

    print("Feature selection complete!")