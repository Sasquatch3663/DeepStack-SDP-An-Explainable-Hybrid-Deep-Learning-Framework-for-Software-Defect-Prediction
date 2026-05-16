import streamlit as st
import pandas as pd

from components.file_uploader import upload_file
from components.prediction_engine import predict_baseline, predict_hybrid
from utils.preprocess_input import clean_input, prepare_array
from utils.load_model import load_model

st.set_page_config(layout="wide")

st.title("🚀 Software Defect Prediction Dashboard")

# =========================
# 🔹 SIDEBAR
# =========================
st.sidebar.header("⚙️ Settings")

model_type = st.sidebar.selectbox("Model", ["baseline", "hybrid"])

strategy = st.sidebar.selectbox(
    "Data Strategy",
    ["original", "smote", "adasyn", "class_weight"]
)

# =========================
# 🔹 FILE UPLOAD
# =========================
df = upload_file()

if df is not None:

    df_clean = clean_input(df)
    X = prepare_array(df_clean)

    model = load_model(model_type, strategy)

    # =========================
    # 🔹 PREDICTION
    # =========================
    if model_type == "baseline":
        preds, prob = predict_baseline(model, X)
    else:
        preds, prob, X_combined, S_test = predict_hybrid(model, X)

    df["Prediction"] = preds

    if prob is not None:
        df["Confidence"] = prob

    # =========================
    # 🔹 DASHBOARD METRICS
    # =========================
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Samples", len(df))
    col2.metric("Defective", int((preds == 1).sum()))
    col3.metric("Non-Defective", int((preds == 0).sum()))

    st.divider()

    # =========================
    # 🔹 TABLE
    # =========================
    st.subheader("📄 Prediction Results")
    st.dataframe(df.head(50))

    # Save for SHAP page
    st.session_state["X"] = X
    st.session_state["model"] = model
    st.session_state["model_type"] = model_type

    if model_type == "hybrid":
        st.session_state["S_test"] = S_test