import streamlit as st

st.set_page_config(
    page_title="SDP AI Dashboard",
    layout="wide",
    page_icon="🚀"
)

st.title("🚀 Software Defect Prediction System")

st.markdown("""
### 💡 Features
- Baseline vs Hybrid Model
- Multiple Data Balancing Strategies
- SHAP Explainability
- Interactive Dashboard
""")

st.sidebar.success("Select a page")