"""
save_models.py

Trains and saves models for ALL strategies (Option B):
- baseline_original
- baseline_smote
- baseline_adasyn
- baseline_class_weight
- hybrid_smote (CNN + stacking + meta)

Run AFTER preprocessing pipeline is complete.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from imblearn.over_sampling import SMOTE, ADASYN

# 🔹 Your modules
from src.models.hybrid_model.hybrid_architecture import HybridModel
from src.models.deep_learning_models.cnn_model import build_cnn


# =========================
# 🔹 CONFIG
# =========================
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


# =========================
# 🔹 DATA LOADING (SAFE)
# =========================
def load_data():
    print("🔹 Loading encoded dataset...")

    possible_paths = [
        "data/processed/encoded.pkl",
        "../data/processed/encoded.pkl"
    ]

    df = None
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ Found dataset: {path}")
            try:
                df = pd.read_pickle(path)
            except:
                df = joblib.load(path)
            break

    if df is None:
        raise FileNotFoundError("❌ encoded.pkl not found. Run preprocessing first.")

    # 🔥 CLEAN DATA (VERY IMPORTANT)
    df = df.replace(["unknown", "UNK", "?", None], 0)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.fillna(0)

    X = df.drop(columns=["defects"]).values.astype(np.float32)
    y = df["defects"].values.astype(int)

    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


# =========================
# 🔹 DATA STRATEGIES
# =========================
def apply_smote(X, y):
    print("🔹 Applying SMOTE...")
    return SMOTE(random_state=42).fit_resample(X, y)


def apply_adasyn(X, y):
    print("🔹 Applying ADASYN...")
    return ADASYN(random_state=42).fit_resample(X, y)


def get_class_weights(y):
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y),
        y=y
    )
    return {i: weights[i] for i in range(len(weights))}


# =========================
# 🔹 BASELINE MODEL
# =========================
def train_baseline(X, y, class_weight=None):
    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight=class_weight
    )
    model.fit(X, y)
    return model


def save_baseline(model, name):
    path = os.path.join(MODEL_DIR, f"baseline_{name}.pkl")
    joblib.dump(model, path)
    print(f"✅ Saved {path}")


# =========================
# 🔹 CNN MODEL
# =========================
def train_cnn(X, y):
    print("🔹 Training CNN...")

    X_cnn = X.reshape(X.shape[0], X.shape[1], 1)

    model = build_cnn((X.shape[1], 1))

    model.fit(
        X_cnn,
        y,
        epochs=25,   # keep small for saving
        batch_size=64,
        verbose=1
    )

    return model


# =========================
# 🔹 HYBRID MODEL
# =========================
def train_hybrid(X, y):
    print("🔹 Training Hybrid Model...")

    cnn = train_cnn(X, y)

    hybrid = HybridModel(cnn)
    hybrid.fit(X, y)

    return hybrid


def save_hybrid(hybrid, name):
    print(f"🔹 Saving Hybrid ({name})...")

    save_dir = os.path.join(MODEL_DIR, f"hybrid_{name}")
    os.makedirs(save_dir, exist_ok=True)

    # CNN
    hybrid.dl_model.save(os.path.join(save_dir, "cnn.h5"))

    # Stacking
    joblib.dump(hybrid.stacking, os.path.join(save_dir, "stacking.pkl"))

    # Meta model
    joblib.dump(hybrid.meta_model.model, os.path.join(save_dir, "meta.pkl"))

    print(f"✅ Hybrid saved in {save_dir}")


# =========================
# 🔹 MAIN PIPELINE
# =========================
def main():
    X_train, X_test, y_train, y_test = load_data()

    # -------------------------
    # 🔹 ORIGINAL
    # -------------------------
    print("\n🚀 Training ORIGINAL")
    model_orig = train_baseline(X_train, y_train)
    save_baseline(model_orig, "original")

    # -------------------------
    # 🔹 SMOTE
    # -------------------------
    print("\n🚀 Training SMOTE")
    X_sm, y_sm = apply_smote(X_train, y_train)
    model_sm = train_baseline(X_sm, y_sm)
    save_baseline(model_sm, "smote")

    # -------------------------
    # 🔹 ADASYN
    # -------------------------
    print("\n🚀 Training ADASYN")
    X_ad, y_ad = apply_adasyn(X_train, y_train)
    model_ad = train_baseline(X_ad, y_ad)
    save_baseline(model_ad, "adasyn")

    # -------------------------
    # 🔹 CLASS WEIGHT
    # -------------------------
    print("\n🚀 Training CLASS WEIGHT")
    cw = get_class_weights(y_train)
    model_cw = train_baseline(X_train, y_train, class_weight=cw)
    save_baseline(model_cw, "class_weight")

    # -------------------------
    # 🔹 HYBRID (BEST → SMOTE)
    # -------------------------
    print("\n🚀 Training HYBRID (SMOTE)")
    hybrid_sm = train_hybrid(X_sm, y_sm)
    save_hybrid(hybrid_sm, "smote")

    print("\n🎉 ALL MODELS SAVED SUCCESSFULLY!")


if __name__ == "__main__":
    main()