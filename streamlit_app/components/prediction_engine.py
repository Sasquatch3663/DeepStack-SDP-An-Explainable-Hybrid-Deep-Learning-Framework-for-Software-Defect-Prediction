import numpy as np


# =========================
# 🔹 BASELINE
# =========================
def predict_baseline(model, X):
    preds = model.predict(X)

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[:, 1]
    else:
        prob = None

    return preds, prob


# =========================
# 🔹 HYBRID (FULL PIPELINE)
# =========================
def predict_hybrid(hybrid_model, X):

    cnn = hybrid_model["cnn"]
    stacking = hybrid_model["stacking"]
    meta = hybrid_model["meta"]

    # CNN feature extraction
    X_cnn = X.reshape(X.shape[0], X.shape[1], 1)
    dl_features = cnn.predict(X_cnn, verbose=0)

    # Combine features
    X_combined = np.hstack((X, dl_features))

    # Stacking → meta features
    S_test = stacking.predict(X_combined)

    # Final prediction
    preds = meta.predict(S_test)

    if hasattr(meta, "predict_proba"):
        prob = meta.predict_proba(S_test)[:, 1]
    else:
        prob = None

    return preds, prob, X_combined, S_test