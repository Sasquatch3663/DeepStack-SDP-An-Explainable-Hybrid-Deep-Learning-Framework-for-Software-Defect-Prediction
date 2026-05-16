"""
load_dataset.py

Flexible dataset loader for:
- NASA PROMISE datasets
- Kaggle defect dataset
- Any CSV-based dataset
"""

# ✅ SAFE EXECUTION FIX
if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import os
import pandas as pd
from typing import Dict

from src.utils.logger import get_logger

logger = get_logger("DataLoader")


def load_csv(file_path: str) -> pd.DataFrame:
    """Load a single CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)
    logger.info(f"Loaded: {file_path} | Shape: {df.shape}")
    return df


def load_folder(folder_path: str) -> Dict[str, pd.DataFrame]:
    """
    Load all CSV files from a folder.
    """
    datasets = {}

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            full_path = os.path.join(folder_path, file)
            datasets[file] = load_csv(full_path)

    logger.info(f"{folder_path} -> {len(datasets)} datasets loaded")
    return datasets


def load_all_datasets(base_path: str = "data/raw") -> Dict[str, pd.DataFrame]:
    """
    Load datasets from all subfolders inside base_path.

    Example:
    data/raw/
        nasa_promise/
        defect_dataset/
    """
    all_datasets = {}

    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)

        if os.path.isdir(folder_path):
            folder_datasets = load_folder(folder_path)

            # Prefix dataset names with folder name
            for name, df in folder_datasets.items():
                key = f"{folder}_{name}"
                all_datasets[key] = df

    logger.info(f"Total datasets loaded from all sources: {len(all_datasets)}")
    return all_datasets


# ✅ Standalone Execution
if __name__ == "__main__":
    logger.info("Running load_dataset standalone...")

    print("\n--- Loading NASA PROMISE ---")
    nasa = load_folder("data/raw/nasa_promise")
    print(f"NASA datasets: {list(nasa.keys())}")

    print("\n--- Loading DEFECT DATASET ---")
    defect = load_folder("data/raw/defect_dataset")
    print(f"Defect datasets: {list(defect.keys())}")

    print("\n--- Loading ALL DATASETS ---")
    all_data = load_all_datasets("data/raw")
    print(f"Total datasets loaded: {len(all_data)}")