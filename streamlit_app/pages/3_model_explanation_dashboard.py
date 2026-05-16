import streamlit as st
import shap

st.title("🧠 Model Explanation")

if "X" not in st.session_state:
    st.warning("Run prediction first")
    st.stop()

X = st.session_state["X"]
model = st.session_state["model"]
model_type = st.session_state["model_type"]

index = st.slider("Select Sample", 0, min(50, len(X)-1), 0)

# =========================
# 🔹 BASELINE
# =========================
if model_type == "baseline":

    explainer = shap.Explainer(model.predict_proba, X[:300])
    shap_values = explainer(X[:300])

    force_plot = shap.force_plot(
        shap_values.base_values[index, 1],
        shap_values.values[index, :, 1],
        X[index]
    )

# =========================
# 🔹 HYBRID
# =========================
else:
    S_test = st.session_state["S_test"]
    meta = model["meta"]

    explainer = shap.Explainer(meta.predict_proba, S_test[:300])
    shap_values = explainer(S_test[:300])

    force_plot = shap.force_plot(
        shap_values.base_values[index, 1],
        shap_values.values[index, :, 1],
        S_test[index]
    )

# =========================
# 🔹 RENDER
# =========================
st.components.v1.html(shap.getjs() + force_plot.html(), height=400)