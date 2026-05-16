import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def show_predictions(preds):
    st.subheader("Predictions")
    st.write(preds[:20])


def show_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", ax=ax)

    st.pyplot(fig)