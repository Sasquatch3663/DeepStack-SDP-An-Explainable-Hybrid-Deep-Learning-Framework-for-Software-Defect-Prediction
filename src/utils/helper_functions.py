"""
helper_functions.py

Contains reusable utility functions used across the project.
"""

import os
import joblib
import numpy as np
from typing import Any


def save_object(file_path: str, obj: Any) -> None:
    """
    Save Python object using joblib.

    Args:
        file_path (str): Path to save object
        obj (Any): Object to save
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(obj, file_path)


def load_object(file_path: str) -> Any:
    """
    Load Python object using joblib.

    Args:
        file_path (str): Path to object

    Returns:
        Any: Loaded object
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    return joblib.load(file_path)


def create_directory(path: str) -> None:
    """
    Create directory if it doesn't exist.

    Args:
        path (str): Directory path
    """
    os.makedirs(path, exist_ok=True)


def get_file_extension(file_path: str) -> str:
    """
    Get file extension.

    Args:
        file_path (str)

    Returns:
        str
    """
    return os.path.splitext(file_path)[-1]


def set_random_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed (int)
    """
    np.random.seed(seed)


# Standalone test
if __name__ == "__main__":
    print("🔧 Testing helper functions...")

    sample_data = {"test": 123}
    save_path = "temp/test.pkl"

    save_object(save_path, sample_data)
    loaded = load_object(save_path)

    print("Saved and Loaded Object:", loaded)