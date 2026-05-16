import streamlit as st
import pandas as pd


def upload_file():
    file = st.file_uploader("Upload CSV File", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.success("File uploaded successfully")
        st.write(df.head())
        return df

    return None