"""
hybrid_model_experiments.py

Runs Hybrid Model (DL + Stacking + Meta Model) experiment
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

from src.utils.helper_functions import load_object
from src.models.baseline_models.logistic_regression import prepare_data
from src.models.deep_learning_models.cnn_model import build_cnn
from src.models.hybrid_model.hybrid_architecture import HybridModel

from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE


def run_hybrid_experiment():

    print("🚀 Starting Advanced Hybrid Model Experiment...")

    # ===============================
    # LOAD DATA
    # ===============================
    data_path = "data/processed/encoded.pkl"

    if not os.path.exists(data_path):
        raise FileNotFoundError("❌ encoded.pkl not found. Run preprocessing first.")

    df = load_object(data_path)

    # ===============================
    # PREPARE DATA
    # ===============================
    X, y = prepare_data(df)

    # ===============================
    # FEATURE SELECTION
    # ===============================
    selector = SelectKBest(f_classif, k=min(300, X.shape[1]))
    X = selector.fit_transform(X, y)

    print("Feature shape after selection:", X.shape)

    # ===============================
    # TRAIN-TEST SPLIT
    # ===============================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    # ===============================
    # HANDLE IMBALANCE (TRAIN ONLY)
    # ===============================
    X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)

    print("After SMOTE:", X_train.shape)

    # ===============================
    # BUILD CNN MODEL
    # ===============================
    input_shape = (X_train.shape[1], 1)
    dl_model = build_cnn(input_shape=input_shape)

    # Reshape for CNN
    X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    print("Training CNN feature extractor...")

    dl_model.fit(
        X_train_cnn,
        y_train,
        epochs=150,
        batch_size=32,
        verbose=1
    )

    # ===============================
    # HYBRID MODEL
    # ===============================
    hybrid = HybridModel(dl_model)

    print("Training Hybrid Model...")

    hybrid.fit(X_train, y_train)

    # ===============================
    # PREDICTION
    # ===============================
    print("Evaluating on test set...")

    preds = hybrid.predict(X_test)

    # ===============================
    # EVALUATION
    # ===============================
    print("\n📊 Classification Report:\n")
    print(classification_report(y_test, preds))

    f1 = f1_score(y_test, preds)

    print(f"\n✅ Hybrid Model F1 Score: {f1:.4f}")

    # ===============================
    # SAVE RESULTS
    # ===============================
    os.makedirs("results", exist_ok=True)

    result_df = pd.DataFrame([{
        "model": "Hybrid_Stacking_DL",
        "f1_score": f1
    }])

    result_path = "results/hybrid_results.csv"

    if os.path.exists(result_path):
        result_df.to_csv(result_path, mode='a', header=False, index=False)
    else:
        result_df.to_csv(result_path, index=False)

    print(f"\n💾 Results saved to {result_path}")


if __name__ == "__main__":
    run_hybrid_experiment()