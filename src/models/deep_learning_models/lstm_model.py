"""
lstm_model.py

LSTM model for sequential learning.
"""

import numpy as np
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore


def build_lstm(input_shape):
    model = Sequential()

    model.add(LSTM(64, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(0.3))

    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train_lstm(X, y):
    """
    Train LSTM model.

    Expected input shape:
    (samples, timesteps, features)
    """

    if len(X.shape) != 3:
        raise ValueError("LSTM expects input shape (samples, timesteps, features)")

    model = build_lstm((X.shape[1], X.shape[2]))

    history = model.fit(
        X,
        y,
        epochs=150,
        batch_size=32,
        verbose=1  
    )

    return model