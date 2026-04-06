import streamlit as st
st.caption("Model version: v0.1 (Preliminary)")
st.title("How to Use")

st.markdown("""
## Overview
This tool predicts bond dissociation energies (BDEs) for PFAS molecules using a QM-trained machine learning model.

---

## Input

### Option 1: SMILES Input
- Enter a valid SMILES string
- The model will identify bonds and generate predictions

### Option 2: File Upload
The uploaded file must be a CSV file with a "SMILES" column. The "solvent" column can be included as well with either "gas" or "water" in each cell.

---

## Output Explanation

The results table includes:

- **BDE Prediction**  
  Predicted bond dissociation energy (kcal/mol)

- **BDE_Lower_90 / BDE_Upper_90**  
  90% confidence interval for the prediction  

- **In_Domain**  
  Indicates whether the input is similar to the training data  
  - `True` → prediction is more reliable  
  - `False` → extrapolation (use caution)

- **AD_Score**  
  Applicability domain score  
  - Higher values → closer to training data  
  - Lower values → less reliable prediction  

---

## Interpreting Results

- Predictions within the domain (`In_Domain = True`) are generally more reliable  
- Wider confidence intervals indicate higher uncertainty  
- Use caution when evaluating molecules outside the training domain  

---

## Limitations

- Model is trained on PFAS and PFAS-like structures  
- Predictions may not generalize to unrelated chemistries  
- Uncertainty reflects model confidence relative to training DFT data, not absolute experimental error (see About) for more information.

---
""")

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
