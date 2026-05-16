"""
config_loader.py

Handles loading configuration files (YAML/JSON).
Provides a centralized way to manage project settings.
"""

import os
import yaml
import json
from typing import Any, Dict


def load_config(file_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML or JSON file.

    Args:
        file_path (str): Path to config file

    Returns:
        dict: Configuration dictionary
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Config file not found: {file_path}")

    try:
        if file_path.endswith(".yaml") or file_path.endswith(".yml"):
            with open(file_path, "r") as file:
                config = yaml.safe_load(file)
        elif file_path.endswith(".json"):
            with open(file_path, "r") as file:
                config = json.load(file)
        else:
            raise ValueError("Unsupported config format. Use YAML or JSON.")

        return config

    except Exception as e:
        raise RuntimeError(f"Error loading config: {e}")


def get_project_root() -> str:
    """
    Get absolute path of project root.

    Returns:
        str: Project root directory
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))


# Standalone test
if __name__ == "__main__":
    print("🔍 Testing config loader...")

    # Example test (create a sample config manually if needed)
    root = get_project_root()
    print(f"Project Root: {root}")
