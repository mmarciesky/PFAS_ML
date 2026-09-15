import streamlit as st
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR))
import partition_utils as ptu

banner_imag = Path(__file__).parent.parent / "assets" / "PFAS_predict.png"
st.set_page_config(page_title="PFAS-Predict", page_icon=str(banner_imag), layout="wide")

st.caption("Model version: v0.1 (Preliminary) -- trained on neutral species only; "
           "charged molecules are outside the training set")

col1, col2 = st.columns([3, 1])
with col1:
    st.title("PFAS Partition Coefficient Predictor")
    st.header("Input a SMILES string or CSV file with a SMILES column:")
with col2:
    img_path = Path(__file__).parent.parent / "assets" / "PFAS.png"
    st.image(str(img_path), use_container_width=True)

MODEL_DIR = BASE_DIR / "ML_Models"


@st.cache_resource
def load_partition_models_cached(model_dir_str):
    """Named uniquely and given an argument -- st.cache_resource keys on the
    function name, so a no-arg load_models() would collide with other pages."""
    kow = ptu.load_partition_model("logKow", Path(model_dir_str))
    kaw = ptu.load_partition_model("logKaw", Path(model_dir_str))
    return kow, kaw


try:
    (kow_model, kow_encoders, kow_fps, kow_meta), \
        (kaw_model, kaw_encoders, kaw_fps, kaw_meta) = load_partition_models_cached(str(MODEL_DIR))
    models_loaded = True
except FileNotFoundError:
    models_loaded = False
    st.error(f"Model files not found in {MODEL_DIR}. Make sure logKow_*.pkl/json "
             f"and logKaw_*.pkl/json exist there.")


def render_result(label, result, unit_note):
    st.metric(label, f"{result['prediction']:.3f}")
    st.caption(f"90% interval: [{result['lower_90']:.3f}, {result['upper_90']:.3f}]  ·  {unit_note}")
    if result["in_domain"]:
        st.success(f"In domain (AD score: {result['ad_score']:.3f})")
    else:
        st.warning(f"Outside training domain (AD score: {result['ad_score']:.3f} "
                   f"< {result['domain_threshold']:.3f})")


if models_loaded:
    input_type = st.selectbox("Choose input type:", ["Single SMILES", "Upload CSV"])

    if input_type == "Single SMILES":
        # Neutral default -- the models were trained without charged species.
        smiles = st.text_input("Enter a SMILES string:",
                               value="FC(F)(F)C(F)(F)C(F)(F)C(=O)O")

        if smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                st.success("Valid SMILES")
                if ptu.is_charged(smiles):
                    st.warning(
                        "This molecule carries a net formal charge. Charged species were "
                        "excluded from training, so both predictions are extrapolations "
                        "regardless of the AD score. Consider the neutral (protonated) form."
                    )
            else:
                st.error("Invalid SMILES")

            run_button = st.button("Run Prediction")
            if run_button and mol is not None:
                try:
                    kow_result = ptu.predict_with_confidence(
                        smiles, kow_model, kow_encoders, kow_fps, kow_meta)
                    kaw_result = ptu.predict_with_confidence(
                        smiles, kaw_model, kaw_encoders, kaw_fps, kaw_meta)
                    st.session_state["partition_single_result"] = {
                        "smiles": smiles, "mol": mol,
                        "kow": kow_result, "kaw": kaw_result,
                    }
                except Exception as e:
                    st.session_state["partition_single_result"] = {"error": str(e)}

        if "partition_single_result" in st.session_state:
            result = st.session_state["partition_single_result"]
            if "error" in result:
                st.error(f"Prediction failed: {result['error']}")
            else:
                col_img, col_results = st.columns([1, 2])
                with col_img:
                    img = Draw.MolToImage(result["mol"], size=(300, 300))
                    st.image(img, caption="Input structure")

                with col_results:
                    kow_result, kaw_result = result["kow"], result["kaw"]
                    st.write(f"Detected headgroup: **{kow_result['headgroup']}**")

                    res_col1, res_col2 = st.columns(2)
                    with res_col1:
                        render_result("logKow (octanol-water)", kow_result,
                                      "higher = more lipophilic")
                    with res_col2:
                        render_result("logKaw (air-water)", kaw_result,
                                      "lower = less volatile")


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
                st.info("These models are molecule-level -- logKow and logKaw do not depend "
                        "on solvent, so any 'solvent' column is ignored.")

                run_button = st.button("Run Prediction")

                if run_button:
                    rows = []
                    progress = st.progress(0)
                    for i, row_data in batch_df.iterrows():
                        smi = row_data["SMILES"]
                        row = {"SMILES": smi}
                        try:
                            kow_r = ptu.predict_with_confidence(
                                smi, kow_model, kow_encoders, kow_fps, kow_meta)
                            kaw_r = ptu.predict_with_confidence(
                                smi, kaw_model, kaw_encoders, kaw_fps, kaw_meta)
                            row.update({
                                "status": "ok",
                                "headgroup": kow_r["headgroup"],
                                "charged": kow_r["charged"],
                                "logKow_pred": kow_r["prediction"],
                                "logKow_lower_90": kow_r["lower_90"],
                                "logKow_upper_90": kow_r["upper_90"],
                                "logKow_in_domain": kow_r["in_domain"],
                                "logKow_ad_score": kow_r["ad_score"],
                                "logKaw_pred": kaw_r["prediction"],
                                "logKaw_lower_90": kaw_r["lower_90"],
                                "logKaw_upper_90": kaw_r["upper_90"],
                                "logKaw_in_domain": kaw_r["in_domain"],
                                "logKaw_ad_score": kaw_r["ad_score"],
                            })
                        except Exception as e:
                            row["status"] = f"failed: {e}"
                        rows.append(row)
                        progress.progress((i + 1) / len(batch_df))

                    st.session_state["partition_results_df"] = pd.DataFrame(rows)

        if "partition_results_df" in st.session_state:
            res_df = st.session_state["partition_results_df"]
            st.subheader("Batch Prediction Results")

            n_ok = int((res_df.get("status") == "ok").sum())
            n_charged = int(res_df.get("charged", pd.Series(dtype=bool)).fillna(False).sum())
            n_out = int((~res_df.get("logKow_in_domain", pd.Series(dtype=bool)).fillna(True)).sum())
            summary_cols = st.columns(3)
            summary_cols[0].metric("Predicted", n_ok)
            summary_cols[1].metric("Charged (extrapolation)", n_charged)
            summary_cols[2].metric("Outside domain (logKow)", n_out)

            st.dataframe(res_df, use_container_width=True)
            st.download_button(
                "Download Results CSV",
                data=res_df.to_csv(index=False),
                file_name="partition_batch_results.csv",
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
