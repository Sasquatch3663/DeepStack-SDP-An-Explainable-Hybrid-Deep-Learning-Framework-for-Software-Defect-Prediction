import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from components.prediction_engine import predict_baseline, predict_hybrid
from utils.load_model import load_model
from utils.report_generator import compute_metrics, generate_csv, generate_pdf

st.title("🏆 Model Leaderboard & Comparison")

# =========================
# 🔹 CHECK DATA
# =========================
if "X" not in st.session_state:
    st.warning("Run prediction first")
    st.stop()

X = st.session_state["X"]

# Try to get real labels
y_true = st.session_state.get("y_true", None)

if y_true is None:
    st.warning("⚠️ Using dummy labels (upload dataset with labels for real evaluation)")
    import numpy as np
    y_true = np.random.randint(0, 2, len(X))


# =========================
# 🔹 SELECT METRIC
# =========================
metric_choice = st.selectbox(
    "Select Ranking Metric",
    ["F1 Score", "Accuracy", "ROC AUC"]
)


# =========================
# 🔹 MODEL EVALUATION
# =========================
strategies = ["original", "smote", "adasyn", "class_weight"]

results = []
roc_data = {}

for strategy in strategies:

    model_b = load_model("baseline", strategy)
    preds, prob = predict_baseline(model_b, X)

    metrics = compute_metrics(y_true, preds, prob)
    metrics["Model"] = f"Baseline-{strategy}"

    results.append(metrics)

    if prob is not None:
        fpr, tpr, _ = roc_curve(y_true, prob)
        roc_data[f"Baseline-{strategy}"] = (fpr, tpr)


# 🔹 HYBRID (SMOTE)
model_h = load_model("hybrid", "smote")
preds_h, prob_h, _, _ = predict_hybrid(model_h, X)

metrics_h = compute_metrics(y_true, preds_h, prob_h)
metrics_h["Model"] = "Hybrid-SMOTE"

results.append(metrics_h)

if prob_h is not None:
    fpr, tpr, _ = roc_curve(y_true, prob_h)
    roc_data["Hybrid-SMOTE"] = (fpr, tpr)


# =========================
# 🔹 CREATE DATAFRAME
# =========================
df = pd.DataFrame(results)

df = df.set_index("Model")

# Sort leaderboard
df_sorted = df.sort_values(by=metric_choice, ascending=False)


# =========================
# 🔹 DISPLAY LEADERBOARD
# =========================
st.subheader("🏆 Leaderboard")

st.dataframe(
    df_sorted.style.highlight_max(axis=0, color="lightgreen"),
    use_container_width=True
)


# =========================
# 🔹 BEST MODEL
# =========================
best_model_name = df_sorted.index[0]
best_score = df_sorted.iloc[0][metric_choice]

st.success(f"🥇 Best Model: **{best_model_name}** ({metric_choice}: {best_score:.4f})")


# =========================
# 🔹 AUTO-SELECTION OPTION
# =========================
use_best = st.checkbox("Use Best Model for Prediction")

if use_best:
    st.session_state["best_model"] = best_model_name
    st.success("Best model saved for prediction dashboard")


# =========================
# 🔹 ROC CURVE
# =========================
st.subheader("📈 ROC Curve")

fig, ax = plt.subplots()

for name, (fpr, tpr) in roc_data.items():
    ax.plot(fpr, tpr, label=name)

ax.plot([0, 1], [0, 1], linestyle="--")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend()

st.pyplot(fig)


# =========================
# 🔹 DOWNLOAD REPORTS
# =========================
st.subheader("📥 Download Reports")

csv_data = generate_csv(df.to_dict("index"))

st.download_button(
    "Download CSV",
    csv_data,
    file_name="leaderboard.csv",
    mime="text/csv"
)

pdf_file = generate_pdf(df.to_dict("index"))

with open(pdf_file, "rb") as f:
    st.download_button(
        "Download PDF",
        f,
        file_name="leaderboard_report.pdf"
    )