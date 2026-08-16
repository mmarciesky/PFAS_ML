import streamlit as st
import sys
import subprocess
import tempfile
import os
import warnings
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR))
from bde_utils import canonicalize_smiles

warnings.filterwarnings('ignore')

banner_imag = Path(__file__).parent.parent / "assets" / "PFAS_predict.png"
st.set_page_config(page_title="PFAS-Predict", page_icon=str(banner_imag), layout="wide")

st.caption("Model version: v0.1 (Preliminary)")
col1, col2 = st.columns([3, 1])
with col1:
    st.title("PFAS BDE Predictor")
    st.header("Input a SMILES string or CSV file with a SMILES column:")
with col2:
    img_path = Path(__file__).parent.parent / "assets" / "PFAS.png"
    st.image(str(img_path), use_container_width=True)

### Handle inputs for single smiles or CSV file
input_type = st.selectbox("Choose input type:", ["Single SMILES", "Upload CSV"])
validate = st.checkbox("Include training data validation")

input_path = None
run_button = False

if input_type == "Single SMILES":
    smiles = st.text_input("Enter a SMILES string:", value="FC(F)(F)C(F)(F)C(F)(F)C(=O)[O-]")
    solvent = st.selectbox("Choose solvent:", ["gas", "water", "DMSO"])  

    if smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            st.success("Valid SMILES")
        else:
            st.error("Invalid SMILES")
        run_button = st.button("Run Prediction")
        if run_button:
            df = pd.DataFrame({"SMILES": [smiles], "solvent": [solvent]})
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                df.to_csv(tmp.name, index=False)
                input_path = tmp.name

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
        else:
            st.error("CSV must contain a column named 'SMILES'")

################################
#### Run the BACKEND #####
################################
if run_button and input_path is not None:
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name
    image_dir = tempfile.mkdtemp()
    cmd = [
        sys.executable, str(BASE_DIR / "predict.py"),
        "--input", input_path,
        "--output", output_path,
        "--visualize", image_dir,
        "--verbose",
    ]
    if validate:
        cmd.append("--validate")
    with st.spinner("Running prediction..."):
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(BASE_DIR))
    if result.returncode == 0:
        if os.path.exists(output_path):
            results_df = pd.read_csv(output_path)
            image_dir_path = Path(image_dir)
            bde_images = sorted(image_dir_path.glob("molecules_bde_*.png"))
            st.session_state["results_df"] = results_df
            st.session_state["bde_images"] = [str(img) for img in bde_images]
            st.session_state["has_results"] = True
        else:
            st.error("Output file was not created.")
    else:
        st.error("Prediction failed.")
        st.text(result.stderr)

################################
#### RESULTS (inline, no page switch) #####
################################
if st.session_state.get("has_results", False):
    st.divider()
    st.subheader("Prediction Results")

    bde_images = st.session_state.get("bde_images", [])
    results_df = st.session_state.get("results_df", None)

    if bde_images:
        st.subheader("BDE Visualizations")
        for img in bde_images:
            st.image(img, use_container_width=True)
            with open(img, "rb") as file:
                st.download_button(
                    label=f"Download {Path(img).name}",
                    data=file,
                    file_name=Path(img).name,
                    mime="image/png",
                    key=f"dl_{Path(img).name}",
                )

    if results_df is not None:
        st.subheader("Prediction Table")
        st.dataframe(results_df, use_container_width=True)
        st.download_button(
            label="Download Results CSV",
            data=results_df.to_csv(index=False),
            file_name="bde_results.csv",
            mime="text/csv",
        )

    if st.button("Clear Results"):
        st.session_state.pop("results_df", None)
        st.session_state.pop("bde_images", None)
        st.session_state["has_results"] = False
        st.rerun()

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