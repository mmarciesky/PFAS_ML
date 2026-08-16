import streamlit as st
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR))
import property_utils as pu

banner_imag = Path(__file__).parent.parent / "assets" / "PFAS_predict.png"
st.set_page_config(page_title="PFAS-Predict", page_icon=str(banner_imag), layout="wide")

st.caption("Model version: v0.1 (Preliminary) -- small dataset, ~30 molecules, treat as provisional")

col1, col2 = st.columns([3, 1])
with col1:
    st.title("PFAS Redox Potential Predictor")
    st.header("Input a SMILES string or CSV file with a SMILES column:")
with col2:
    img_path = Path(__file__).parent.parent / "assets" / "PFAS.png"
    st.image(str(img_path), use_container_width=True)

MODEL_DIR = BASE_DIR / "ML_Models"

@st.cache_resource
def load_models():
    ox = pu.load_property_model("oxidation_potential", MODEL_DIR)
    red = pu.load_property_model("reduction_potential", MODEL_DIR)
    return ox, red

try:
    (ox_model, ox_encoders, ox_fps, ox_meta), (red_model, red_encoders, red_fps, red_meta) = load_models()
    models_loaded = True
except FileNotFoundError:
    models_loaded = False
    st.error(f"Model files not found in {MODEL_DIR}. Make sure oxidation_potential_*.pkl/json "
             f"and reduction_potential_*.pkl/json exist there.")

if models_loaded:
    input_type = st.selectbox("Choose input type:", ["Single SMILES", "Upload CSV"])

    if input_type == "Single SMILES":
        smiles = st.text_input("Enter a SMILES string:", value="FC(F)(F)C(F)(F)C(F)(F)C(=O)[O-]")
        solvent = st.selectbox("Choose solvent:", ["water"])  # water-only model -- no other option offered

        if smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                st.success("Valid SMILES")
            else:
                st.error("Invalid SMILES")

            run_button = st.button("Run Prediction")
            if run_button and mol is not None:
                try:
                    ox_result = pu.predict_with_confidence(smiles, ox_model, ox_encoders, ox_fps, ox_meta)
                    red_result = pu.predict_with_confidence(smiles, red_model, red_encoders, red_fps, red_meta)
                    st.session_state["redox_single_result"] = {
                        "smiles": smiles, "mol": mol, "ox": ox_result, "red": red_result,
                    }
                except Exception as e:
                    st.session_state["redox_single_result"] = {"error": str(e)}

        if "redox_single_result" in st.session_state:
            result = st.session_state["redox_single_result"]
            if "error" in result:
                st.error(f"Prediction failed: {result['error']}")
            else:
                col_img, col_results = st.columns([1, 2])
                with col_img:
                    img = Draw.MolToImage(result["mol"], size=(300, 300))
                    st.image(img, caption="Input structure")

                with col_results:
                    ox_result, red_result = result["ox"], result["red"]
                    st.write(f"Detected protonation state: **{ox_result['protonation_state']}**")

                    res_col1, res_col2 = st.columns(2)
                    with res_col1:
                        st.metric("Oxidation potential (V vs SHE)", f"{ox_result['prediction']:.3f}")
                        st.caption(f"90% interval: [{ox_result['lower_90']:.3f}, {ox_result['upper_90']:.3f}]")
                        if ox_result['in_domain']:
                            st.success(f"In domain (AD score: {ox_result['ad_score']:.3f})")
                        else:
                            st.warning(f"Outside training domain (AD score: {ox_result['ad_score']:.3f})")

                    with res_col2:
                        st.metric("Reduction potential (V vs SHE)", f"{red_result['prediction']:.3f}")
                        st.caption(f"90% interval: [{red_result['lower_90']:.3f}, {red_result['upper_90']:.3f}]")
                        if red_result['in_domain']:
                            st.success(f"In domain (AD score: {red_result['ad_score']:.3f})")
                        else:
                            st.warning(f"Outside training domain (AD score: {red_result['ad_score']:.3f})")

    elif input_type == "Upload CSV":
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(batch_df.head())

            if "SMILES" not in batch_df.columns:
                st.error("CSV must contain a column named 'SMILES'")
            else:
                st.success("SMILES column found!")

                has_solvent_col = "solvent" in batch_df.columns
                if has_solvent_col:
                    n_non_water = (batch_df["solvent"].astype(str).str.lower() != "water").sum()
                    if n_non_water > 0:
                        st.warning(f"{n_non_water} row(s) have a solvent other than water -- "
                                   f"those rows will be skipped (this model is water-only).")
                else:
                    st.info("No 'solvent' column found -- assuming water for all rows.")

                run_button = st.button("Run Prediction")

                if run_button:
                    rows = []
                    progress = st.progress(0)
                    for i, row_data in batch_df.iterrows():
                        smi = row_data["SMILES"]
                        row = {"SMILES": smi}

                        if has_solvent_col and str(row_data["solvent"]).lower() != "water":
                            row["status"] = "skipped -- solvent is not water"
                            rows.append(row)
                            progress.progress((i + 1) / len(batch_df))
                            continue

                        try:
                            ox_r = pu.predict_with_confidence(smi, ox_model, ox_encoders, ox_fps, ox_meta)
                            red_r = pu.predict_with_confidence(smi, red_model, red_encoders, red_fps, red_meta)
                            row.update({
                                "status": "ok",
                                "protonation_state": ox_r["protonation_state"],
                                "oxidation_potential_pred": ox_r["prediction"],
                                "oxidation_lower_90": ox_r["lower_90"],
                                "oxidation_upper_90": ox_r["upper_90"],
                                "oxidation_in_domain": ox_r["in_domain"],
                                "oxidation_ad_score": ox_r["ad_score"],
                                "reduction_potential_pred": red_r["prediction"],
                                "reduction_lower_90": red_r["lower_90"],
                                "reduction_upper_90": red_r["upper_90"],
                                "reduction_in_domain": red_r["in_domain"],
                                "reduction_ad_score": red_r["ad_score"],
                            })
                        except Exception as e:
                            row["status"] = f"failed: {e}"
                        rows.append(row)
                        progress.progress((i + 1) / len(batch_df))

                    st.session_state["redox_results_df"] = pd.DataFrame(rows)

        if "redox_results_df" in st.session_state:
            st.subheader("Batch Prediction Results")
            st.dataframe(st.session_state["redox_results_df"], use_container_width=True)
            st.download_button(
                "Download Results CSV",
                data=st.session_state["redox_results_df"].to_csv(index=False),
                file_name="redox_batch_results.csv",
                mime="text/csv",
            )

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