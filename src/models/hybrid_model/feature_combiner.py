import numpy as np

class FeatureCombiner:
    def __init__(self):
        pass

    def combine(self, X_original, X_dl_features):
        """
        Combine original selected features with DL extracted features
        """
        if len(X_original) != len(X_dl_features):
            raise ValueError("Mismatch in sample sizes")

        return np.hstack((X_original, X_dl_features))