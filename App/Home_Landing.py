import streamlit as st
import pandas as pd
from pathlib import Path

st.title("PFAS Predict")
st.subheader("Machine learning predictions and a curated quantum chemistry database for PFAS")

st.markdown(
    "This platform brings together machine-learned property predictors and a "
    "curated quantum chemistry database for per- and polyfluoroalkyl substances (PFAS)."
)

st.divider()

MAIN_TABLE_PATH = Path(__file__).parent.parent / "main_table.csv"
BDE_TABLE_PATH = Path(__file__).parent.parent / "BDE_table.csv"

stat_cols = st.columns(3)

if MAIN_TABLE_PATH.exists():
    main_df = pd.read_csv(MAIN_TABLE_PATH, usecols=['Inchikey'])
    stat_cols[0].metric("Molecules in database", main_df['Inchikey'].nunique())
else:
    stat_cols[0].metric("Molecules in database", "—")

if BDE_TABLE_PATH.exists():
    bde_df = pd.read_csv(BDE_TABLE_PATH, usecols=['Parent_SMILES'])
    stat_cols[1].metric("BDE values computed", len(bde_df))
else:
    stat_cols[1].metric("BDE values computed", "—")

stat_cols[2].metric("ML models available", 1)  # bump this as you add more

st.divider()
st.subheader("Explore the platform")

card_cols = st.columns(2)
with card_cols[0]:
    with st.container(border=True):
        st.markdown("**Data Explorer**")
        st.write("Browse the curated PFAS database — properties, structures, and bond dissociation energies.")
        st.page_link("pages/4_Data_Explorer.py", label="Open Data Explorer")

with card_cols[1]:
    with st.container(border=True):
        st.markdown("**BDE Predictor**")
        st.write("Predict bond dissociation energies for a PFAS molecule from a SMILES string or CSV upload.")
        st.page_link("pages/0_Predict_BDE.py", label="Open BDE Predictor")

st.caption("More predictive models are planned for this platform.")
st.divider()
st.subheader("Model roadmap")

STATUS_COLORS = {
    "Validated": "#4a8f6b",
    "Preliminary": "#8fbf9f",
    "Functional but weak": "#c98a4b",
    "Benchmarked / non-automated": "#e0b84b",
    "Not started": "#9a9a9a",
}

def status_badge(label):
    color = STATUS_COLORS.get(label, "#9a9a9a")
    return (
        f'<span style="background-color:{color}; color:#1a1a1a; padding:2px 10px; '
        f'border-radius:12px; font-size:0.8em; font-weight:600;">{label}</span>'
    )

# (name, status, page path or None if not built yet)
roadmap = [
    ("BDE Model", "Preliminary", "pages/0_Predict_BDE.py"),
    ("Redox Model", "Benchmarked / non-automated", None),
    ("Partition Coefficient Model", "Not started", None),
    ("Toxicology Model", "Not started", None),
]

for name, status, link in roadmap:
    cols = st.columns([3, 3, 2])
    cols[0].markdown(f"**{name}**")
    cols[1].markdown(status_badge(status), unsafe_allow_html=True)
    if link:
        cols[2].page_link(link, label="Open")
    else:
        cols[2].caption("Coming soon")
##############################
# SIDE BAR #
#######################
st.sidebar.markdown(
"""
<small>
<strong>Developed by</strong><br>
<a href="https://www.linkedin.com/in/mmarciesky" target="_blank"> Dr. Mel Marciesky </a><br>

<strong>Affiliations</strong><br>
<a href="https://www.modelnglab.com/" target="_blank">Ng Lab</a><br>
<a href="https://keithlab.pitt.edu/" target="_blank">Keith Lab</a>  

<br>
<strong>Model</strong><br>
Version v0.1 (Preliminary)<br>
<em>v1.0 planned — Summer 2026</em>
</small>
""",
unsafe_allow_html=True
)