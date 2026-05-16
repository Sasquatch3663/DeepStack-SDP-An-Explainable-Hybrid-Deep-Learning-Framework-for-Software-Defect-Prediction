"""
neural_network_utils.py

Shared utilities for DL models.
"""

import numpy as np
import pandas as pd


def prepare_dl_data(X, y):
    """
    Converts tabular data into DL-compatible format.
    """

    # Ensure numpy arrays
    X = np.array(X)
    y = np.array(y)

    return X, y