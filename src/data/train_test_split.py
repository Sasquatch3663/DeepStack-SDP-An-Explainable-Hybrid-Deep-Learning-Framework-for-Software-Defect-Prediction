"""
train_test_split.py

Creates train/test splits from encoded data with strict numeric enforcement.
"""

# ✅ SAFE EXECUTION FIX
if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger
from src.utils.helper_functions import load_object, save_object

logger = get_logger("TrainTestSplit")


def create_splits(
    input_path="data/processed/encoded.pkl",
    target_column="defects",
    test_size=0.2,
    random_state=42,
):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} not found. Run encoding first.")

    df = load_object(input_path)

    if target_column not in df.columns:
        raise ValueError("Target column missing in encoded data")

    # Ensure numeric only (hard safety)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X = X.select_dtypes(include=[np.number]).astype(np.float32)
    X = X.fillna(0)

    y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    os.makedirs("data/processed", exist_ok=True)
    save_object("data/processed/X_train.pkl", X_train)
    save_object("data/processed/X_test.pkl", X_test)
    save_object("data/processed/y_train.pkl", y_train)
    save_object("data/processed/y_test.pkl", y_test)

    logger.info("Train-test split created and saved.")
    print("✅ Train-test data saved.")


if __name__ == "__main__":
    create_splits()