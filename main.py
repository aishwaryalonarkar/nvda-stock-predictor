"""
NVDA Stock Predictor — Multi-Page Streamlit App
================================================
Entry point. Streamlit will auto-discover pages/ directory.
Run: streamlit run main.py
"""
import streamlit as st

st.set_page_config(
    page_title="NVDA Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Immediately redirect to the Home page
st.switch_page("pages/Home.py")
