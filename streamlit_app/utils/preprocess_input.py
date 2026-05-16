import numpy as np
import pandas as pd


def clean_input(df):
    df = df.copy()

    # Handle unknown values
    df = df.replace(["unknown", "UNK", "?"], 0)

    # Convert numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    df = df.fillna(0)

    return df


def prepare_array(df):
    return df.values.astype(np.float32)