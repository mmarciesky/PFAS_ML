import streamlit as st

st.set_page_config(page_title="PFAS Predict", layout="wide")

home_page = st.Page("pages/Home_Landing.py", title="Home", default=True)
predict_page = st.Page("pages/0_Predict_BDE.py", title="Predict")
results_page = st.Page("pages/3_Results.py", title="Results")
how_to_page = st.Page("pages/1_How_to_Use.py", title="How to Use")
about_page = st.Page("pages/2_About.py", title="About")
explorer_page = st.Page("pages/4_Data_Explorer.py", title="Data Explorer")

pg = st.navigation({
    "": [home_page],
    "Database": [explorer_page],
    "BDE Predictor": [predict_page, results_page, how_to_page, about_page],
})

pg.run()