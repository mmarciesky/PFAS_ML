import streamlit as st

st.set_page_config(page_title="PFAS Foundry", layout="wide")

home_page = st.Page("pages/Home_Landing.py", title="Home", default=True)
predict_page = st.Page("pages/0_Predict_BDE.py", title="Predict")
results_page = st.Page("pages/3_Results.py", title="Results")
how_to_page = st.Page("pages/1_How_to_Use.py", title="How to Use")
about_page = st.Page("pages/2_About.py", title="About")
explorer_page = st.Page("pages/4_Data_Explorer.py", title="Data Explorer")
redox_page = st.Page("pages/5_Redox_Predictor.py", title="Redox Predictor")
partition_page = st.Page("pages/6_Partition_Predictor.py", title="Partition Predictor")
dipole_page = st.Page("pages/7_Dipole_Predictor.py", title="Dipole Predictor")
homo_page = st.Page("pages/8_HOMO_LUMO_Predictor.py", title="HOMO-LUMO Predictor")

pg = st.navigation({
    "": [home_page],
    "Database": [explorer_page],
    "BDE Predictor": [predict_page, how_to_page, about_page],
    "Redox Potential Predictor": [redox_page],
    "Partition Coefficient Predictor": [partition_page],
    "Dipole Moment Predictor": [dipole_page],
    "HOMO-LUMO Gap Predictor": [homo_page],
})

pg.run()
