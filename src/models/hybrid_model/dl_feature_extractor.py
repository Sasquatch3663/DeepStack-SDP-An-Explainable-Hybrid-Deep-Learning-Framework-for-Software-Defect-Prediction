"""
dl_feature_extractor.py

Extract features using CNN
"""

import numpy as np
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Input # type: ignore


def build_feature_extractor(input_shape):
    inputs = Input(shape=input_shape)

    x = Conv1D(32, 3, activation='relu')(inputs)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)

    # Output features (not classification)
    model = Model(inputs, x)

    return model


def extract_features(X):
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = build_feature_extractor((X.shape[1], 1))

    features = model.predict(X, verbose=1)

    return features