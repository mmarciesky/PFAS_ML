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
- **Model:** XGBoost (eXtreme Gradient Boosting)  
""")

with col2:
    st.markdown("""
### Performance
 MAE:  1.442 ± 0.122 kcal/mol \n
 RMSE: 3.248 ± 0.318 kcal/mol \n
 R²:   0.953 ± 0.008

Performance reflects agreement with reference DFT calculations within the training domain.
""")

st.divider()

# ── Scope ─────────────────────────────────────────────────────────────────────
st.markdown("""
## Molecular Scope & Domain

The model is trained on PFAS and PFAS-like structures and performs best within this chemical space.

**Supported environments:**
- Gas phase  
- Implicit water  

Predictions for structures outside the training domain should be interpreted with caution.
""", unsafe_allow_html=True)

st.divider()
st.markdown("## Next Version (v1.0)")

st.markdown("""

In development — expanding the model to support:
- Multi-solvent predictions (water, gas, DMSO)
- Multi-property outputs (e.g., BDE, partition coefficents)
- A significantly expanded PFAS quantum chemistry dataset   

These improvements aim to enable broader chemical coverage and more comprehensive reactivity insights.
""", unsafe_allow_html=True)
# ── Citation ──────────────────────────────────────────────────────────────────
st.markdown("""
## Citation

If you use this tool in your research, please cite: \n
App: Marciesky, M. PFAS BDE Predictor (v0.1-preliminary). 2026. [GitHub](https://github.com/mmarciesky/PFAS_ML) link — DOI forthcoming \n
Database: PFAS Quantum Chemistry Database. [GitHub](https://github.com/mmarciesky/PFAS_Database) — DOI forthcoming via Zenodo
""")

st.markdown("""
## Acknowledgments
This tool was developed with the support of the Ng Lab and Keith Lab
at University of Pittsburgh. 
""")
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
