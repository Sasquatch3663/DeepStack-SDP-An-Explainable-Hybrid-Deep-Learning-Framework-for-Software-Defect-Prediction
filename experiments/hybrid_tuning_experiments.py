import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.models.hybrid_model.hybrid_architecture import HybridModel
from src.models.deep_learning_models.cnn_model import build_cnn
from src.evaluation.metrics import evaluate_model

from tensorflow.keras.models import Model # type: ignore


# =========================
# 🔹 PARAMETER SPACES
# =========================

rf_param_dist = {
    "n_estimators": [100, 150, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}

xgb_param_dist = {
    "n_estimators": [100, 150, 200],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}


# =========================
# 🔹 TUNING FUNCTION
# =========================

def tune_model(model, param_dist, X, y):
    search = RandomizedSearchCV(
        model,
        param_dist,
        n_iter=5,
        scoring="f1",
        cv=3,
        verbose=1,
        n_jobs=-1,
        random_state=42,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_


# =========================
# 🔹 MAIN FUNCTION
# =========================

def run_hybrid_tuning():
    print("🔹 Loading data...")

    X_train = joblib.load("data/processed/X_train.pkl")
    X_test = joblib.load("data/processed/X_test.pkl")
    y_train = joblib.load("data/processed/y_train.pkl")
    y_test = joblib.load("data/processed/y_test.pkl")

    # =========================
    # 🔥 CLEAN DATA (CRITICAL FIX)
    # =========================
    if hasattr(X_train, "replace"):
        X_train = X_train.replace(["unknown", "UNK", "?", None], 0)
        X_test  = X_test.replace(["unknown", "UNK", "?", None], 0)

    # Convert to numpy
    X_train = X_train.values if hasattr(X_train, "values") else X_train
    X_test  = X_test.values if hasattr(X_test, "values") else X_test

    # Force numeric
    X_train = X_train.astype(np.float32)
    X_test  = X_test.astype(np.float32)

    # Handle NaNs
    X_train = np.nan_to_num(X_train)
    X_test  = np.nan_to_num(X_test)

    y_train = np.array(y_train).astype(np.int32)
    y_test  = np.array(y_test).astype(np.int32)

    # =========================
    # 🔹 CNN TRAINING
    # =========================
    print("🔹 Training CNN...")

    input_shape = (X_train.shape[1], 1)
    cnn = build_cnn(input_shape=input_shape)

    X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_cnn  = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    cnn.fit(X_train_cnn, y_train, epochs=150, batch_size=32, verbose=1)

    # =========================
    # 🔹 FEATURE EXTRACTION
    # =========================
    feature_extractor = Model(
        inputs=cnn.input,
        outputs=cnn.get_layer("feature_layer").output
    )

    dl_train = feature_extractor.predict(X_train_cnn, verbose=1)
    dl_test  = feature_extractor.predict(X_test_cnn, verbose=1)

    # Combine features (NOW SAFE)
    X_train_combined = np.hstack((X_train, dl_train))
    X_test_combined  = np.hstack((X_test, dl_test))

    # =========================
    # 🔹 TUNING BASE MODELS
    # =========================
    print("🔹 Tuning Random Forest...")
    best_rf, rf_params = tune_model(
        RandomForestClassifier(random_state=42),
        rf_param_dist,
        X_train_combined,
        y_train
    )

    print("🔹 Tuning XGBoost...")
    best_xgb, xgb_params = tune_model(
        XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42
        ),
        xgb_param_dist,
        X_train_combined,
        y_train
    )

    # =========================
    # 🔹 META MODEL
    # =========================
    print("🔹 Training Meta Model...")

    S_train = np.column_stack([
        best_rf.predict(X_train_combined),
        best_xgb.predict(X_train_combined)
    ])

    S_test = np.column_stack([
        best_rf.predict(X_test_combined),
        best_xgb.predict(X_test_combined)
    ])

    best_meta, meta_params = tune_model(
        XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42
        ),
        xgb_param_dist,
        S_train,
        y_train
    )

    # =========================
    # 🔹 FINAL HYBRID MODEL
    # =========================
    print("🔹 Building final Hybrid Model...")

    hybrid = HybridModel(cnn)
    hybrid.base_models = [best_rf, best_xgb]
    hybrid.meta_model.model = best_meta

    hybrid.fit(X_train, y_train)

    y_pred = hybrid.predict(X_test)

    # =========================
    # 🔹 EVALUATION
    # =========================
    results = evaluate_model(y_test, y_pred)

    output = {
        **results,
        "rf_params": rf_params,
        "xgb_params": xgb_params,
        "meta_params": meta_params
    }

    os.makedirs("results", exist_ok=True)
    pd.DataFrame([output]).to_csv("results/hybrid_tuned.csv", index=False)

    print("\n✅ FINAL TUNED RESULTS:")
    print(output)


if __name__ == "__main__":
    run_hybrid_tuning()