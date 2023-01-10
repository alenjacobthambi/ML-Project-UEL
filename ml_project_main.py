import streamlit as st

st.sidebar.title('Welcome to ML Project')
page = st.sidebar.radio(
    "What would you like to know about this project ?",
    ("Data Exploration", 'Machine Learning Models')
)