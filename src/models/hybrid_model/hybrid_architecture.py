import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Model # type: ignore

from src.models.hybrid_model.feature_combiner import FeatureCombiner
from src.models.hybrid_model.stacking_model import StackingModel
from src.models.hybrid_model.meta_model import MetaModel


class HybridModel:
    def __init__(self, dl_model):
        self.dl_model = dl_model
        self.combiner = FeatureCombiner()
        self.meta_model = MetaModel()

        # Extract intermediate layer for features
        self.feature_extractor = Model(
            inputs=dl_model.input,
            outputs=dl_model.get_layer("feature_layer").output
        )

        self.base_models = [
            RandomForestClassifier(n_estimators=150, random_state=42),
            XGBClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        ]

        self.stacking = StackingModel(self.base_models)

    # -----------------------------------
    # DL FEATURE EXTRACTION (FIXED)
    # -----------------------------------
    def extract_dl_features(self, X):
        X = X.reshape(X.shape[0], X.shape[1], 1)
        return self.feature_extractor.predict(X, verbose=0)

    # -----------------------------------
    # TRAIN
    # -----------------------------------
    def fit(self, X_train, y_train):
        # DL Features
        dl_train = self.extract_dl_features(X_train)

        # Combine features
        X_train_combined = self.combiner.combine(X_train, dl_train)

        # Stacking (NO TEST DATA HERE)
        self.stacking.fit(X_train_combined, y_train)
        S_train = self.stacking.predict(X_train_combined)

        # Meta model
        self.meta_model.train(S_train, y_train)

        return self

    # -----------------------------------
    # PREDICT
    # -----------------------------------
    def predict(self, X):

        dl_features = self.extract_dl_features(X)

        X_combined = self.combiner.combine(X, dl_features)

        S_test = self.stacking.predict(X_combined)

        return self.meta_model.predict(S_test)