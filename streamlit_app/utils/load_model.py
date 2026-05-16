import os
import joblib
from tensorflow.keras.models import load_model # type: ignore

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../models")
)


# =========================
# 🔹 BASELINE
# =========================
def load_baseline(strategy):
    path = os.path.join(BASE_DIR, f"baseline_{strategy}.pkl")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Baseline model not found: {path}")

    return joblib.load(path)


# =========================
# 🔹 HYBRID
# =========================
def load_hybrid(strategy):
    path = os.path.join(BASE_DIR, f"hybrid_{strategy}")

    cnn = load_model(os.path.join(path, "cnn.h5"))
    stacking = joblib.load(os.path.join(path, "stacking.pkl"))
    meta = joblib.load(os.path.join(path, "meta.pkl"))

    return {
        "cnn": cnn,
        "stacking": stacking,
        "meta": meta
    }


# =========================
# 🔹 MAIN LOADER
# =========================
def load_model(model_type, strategy):

    if model_type == "baseline":
        return load_baseline(strategy)

    elif model_type == "hybrid":
        return load_hybrid(strategy)

    else:
        raise ValueError("Invalid model type")
