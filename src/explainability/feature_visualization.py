"""
feature_visualization.py

Visualizes CNN extracted features for hybrid model.
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model # type: ignore
import os


class FeatureVisualizer:
    def __init__(self, cnn_model):
        self.cnn_model = cnn_model

    def extract_features(self, X):
        X = np.asarray(X, dtype=np.float32)

        feature_model = Model(
            inputs=self.cnn_model.input,
            outputs=self.cnn_model.get_layer("feature_layer").output
        )

        X_cnn = X.reshape(X.shape[0], X.shape[1], 1)
        features = feature_model.predict(X_cnn, verbose=0)

        return features

    def plot_distribution(self, features):
        os.makedirs("results", exist_ok=True)

        plt.figure(figsize=(8, 4))
        plt.hist(features.flatten(), bins=50)
        plt.title("Feature Distribution")
        plt.savefig("results/feature_distribution.png", dpi=300)
        plt.close()

    def compare_features(self, X, features):
        os.makedirs("results", exist_ok=True)

        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.plot(X[0])
        plt.title("Original Features")

        plt.subplot(1, 2, 2)
        plt.plot(features[0])
        plt.title("CNN Features")

        plt.savefig("results/feature_comparison.png", dpi=300)
        plt.close()