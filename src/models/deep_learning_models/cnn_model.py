"""
cnn_model.py

CNN model for feature extraction + classification.
"""

import numpy as np
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore


def build_cnn(input_shape):
    model = Sequential()

    model.add(Conv1D(
        filters=32,
        kernel_size=3,
        activation="relu",
        input_shape=input_shape
    ))

    model.add(Flatten())

    model.add(Dense(64, activation="relu"))

    # ✅ IMPORTANT: Named feature layer
    model.add(Dense(32, activation="relu", name="feature_layer"))

    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train_cnn(X, y):
    """
    Train CNN model.

    Expected input shape:
    (samples, features, 1)
    """

    if len(X.shape) != 3:
        raise ValueError("CNN expects input shape (samples, features, 1)")

    model = build_cnn((X.shape[1], X.shape[2]))

    history = model.fit(
        X,
        y,
        epochs=150,
        batch_size=32,
        verbose=1  
    )

    return model