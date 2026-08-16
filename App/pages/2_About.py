import streamlit as st

st.set_page_config(page_title="About", layout="wide")
st.caption("Model version: v0.1 (Preliminary)")
st.title("About This App")

# ── Overview ────────────────────────────────────────────────────────────────
st.markdown("""
## PFAS BDE Predictor

This application predicts **bond dissociation energies (BDEs)** for PFAS molecules using a machine learning model trained on 701 bonds from DFT (ωB97X-V/def2-TZVPD) calculations. 

The model is designed to support **rapid screening and mechanistic insight**, helping identify bonds that may be more susceptible to cleavage under various conditions.

<br>

<strong>Model Version</strong><br>
v0.1 (Preliminary)<br>
<em>v1.0 planned — Summer 2026</em>

<br>

<strong>Underlying Data and QM methodolgy</strong><br>
<a href="https://github.com/mmarciesky/PFAS_Database" target="_blank">
PFAS Quantum Chemistry Database
</a>

""", unsafe_allow_html=True)

st.divider()

# ── Model Details ────────────────────────────────────────────────────────────
st.markdown("## Model Details")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
### Algorithm
- **Model:** XGBoost (eXtreme Gradient Boosting), Optuna-tuned
""")

with col2:
    st.markdown("""
### Performance
 Holdout MAE:  1.357 kcal/mol \n
 Holdout RMSE: 3.415 kcal/mol \n
 Holdout R²:   0.960

Performance reflects agreement with reference DFT calculations on a held-out set of parent PFAS molecules never seen during training.
""")

st.divider()

# ── Scope ─────────────────────────────────────────────────────────────────────
st.markdown("""
## Molecular Scope & Domain

The model is trained on PFAS and PFAS-like structures and performs best within this chemical space.

**Supported environments:**
- Gas phase  
- Implicit water  
- Implicit DMSO  

Predictions for structures outside the training domain should be interpreted with caution.
""", unsafe_allow_html=True)

st.divider()


# ── Footer ────────────────────────────────────────────────────────────────────
st.caption("Version v0.1 (Preliminary) · Last updated 2026")
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