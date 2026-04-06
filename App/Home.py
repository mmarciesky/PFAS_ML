# ============================================================
# PART 1: Imports
# ============================================================
import streamlit as st
import sys
import json
import pickle
import argparse
import warnings
from pathlib import Path
import subprocess
import tempfile
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw
from rdkit.DataStructs import CreateFromBitString
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')

BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(BASE_DIR / "src")) 
sys.path.insert(0, str(BASE_DIR))
from bde_utils import (
    smiles_to_fp,
    predict_bde,
    canonicalize_smiles,
    make_sorted_pair,
    remove_dummy_atoms_and_add_radicals,
    dedupe_bonds,
    fragment_molecules,
    convert_parent_pfas_to_dataframe,
    show_image_grid,
    show_image_grid_save
)

from predict import ( load_model,fragment_and_prepare_bonds,check_applicability_domain,generate_predictions)

warnings.filterwarnings('ignore')

input_path = None
run_button = False
sns.set_style("whitegrid")
st.caption("Model version: v0.1 (Preliminary)")
st.title("PFAS BDE Predictor")

### Handle inputs for single smiles or CSV file
st.header("Input a SMILES string or CSV file with a SMILES column:" )
input_type = st.selectbox(
    "Choose input type:",
    ["Single SMILES", "Upload CSV"]
)
validate = st.checkbox("Include training data validation")
if input_type == "Single SMILES":
    smiles = st.text_input("Enter a SMILES string:")
    solvent = st.selectbox(
        "Choose solvent:",
        ["gas", "water"]
    )

    if smiles:
        mol = Chem.MolFromSmiles(smiles)

        if mol is not None:
            st.success("Valid SMILES")
        else:
            st.error("Invalid SMILES")
        run_button = st.button("Run Prediction")
        if run_button:

            df = pd.DataFrame({
                "SMILES": [smiles],
                "solvent": [solvent]
            })

            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                df.to_csv(tmp.name, index=False)
                input_path = tmp.name

            st.success(f"Temporary file created: {input_path}")
            
elif input_type == "Upload CSV":
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.write("Preview of uploaded data:")
        st.dataframe(df.head())

        if "SMILES" in df.columns:
            st.success("SMILES column found!")
            run_button = st.button("Run Prediction")

            if run_button:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                    df.to_csv(tmp.name, index=False)
                    input_path = tmp.name

                st.success(f"Temporary file created: {input_path}")

        else:
            st.error("CSV must contain a column named 'SMILES'")
################################
#### Run the BACKEND #####
################################
if run_button and input_path is not None:
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name
    image_dir = tempfile.mkdtemp()

    cmd = [
        sys.executable,
        "predict.py",
        "--input", input_path,
        "--output", output_path,
        "--visualize", image_dir
    ]
    if validate:
        cmd.append("--validate")

    with st.spinner("Running prediction..."):
        result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        if os.path.exists(output_path):
            results_df = pd.read_csv(output_path)

            image_dir_path = Path(image_dir)
            bde_images = sorted(image_dir_path.glob("molecules_bde_*.png"))

            st.session_state["results_df"] = results_df
            st.session_state["bde_images"] = [str(img) for img in bde_images]
            st.session_state["has_results"] = True

            st.switch_page("pages/3_Results.py")
        else:
            st.error("Output file was not created.")
    else:
        st.error("Prediction failed.")
        st.text(result.stderr)
        
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
