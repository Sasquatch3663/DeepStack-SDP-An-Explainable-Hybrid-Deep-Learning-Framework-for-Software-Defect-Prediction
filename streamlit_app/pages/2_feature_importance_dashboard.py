import streamlit as st
import shap
import matplotlib.pyplot as plt

st.title("📊 Feature Importance (SHAP)")

if "X" not in st.session_state:
    st.warning("Run prediction first")
    st.stop()

X = st.session_state["X"]
model = st.session_state["model"]
model_type = st.session_state["model_type"]

if model_type != "baseline":
    st.warning("SHAP feature importance only available for baseline model")
    st.stop()

X_sample = X[:300]

explainer = shap.Explainer(model.predict_proba, X_sample)
shap_values = explainer(X_sample)

st.subheader("Global Feature Importance")

fig = plt.figure()
shap.plots.bar(shap_values[:, :, 1], show=False)
st.pyplot(fig)