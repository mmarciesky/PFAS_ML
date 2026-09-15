import streamlit as st
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR))
import solvent_model_utils as smu

MODEL_PREFIX = "dipole_moment"
SPEC = smu.MODEL_SPECS[MODEL_PREFIX]

banner_imag = Path(__file__).parent.parent / "assets" / "PFAS_predict.png"
st.set_page_config(page_title="PFAS-Predict", page_icon=str(banner_imag), layout="wide")

st.caption("Model version: v0.1 (Preliminary) -- predictions are solvent-specific")

col1, col2 = st.columns([3, 1])
with col1:
    st.title("PFAS Dipole Moment Predictor")
    st.header("Input a SMILES string or CSV file with a SMILES column:")
with col2:
    img_path = Path(__file__).parent.parent / "assets" / "PFAS.png"
    st.image(str(img_path), use_container_width=True)

MODEL_DIR = BASE_DIR / "ML_Models"


@st.cache_resource
def load_solvent_model_cached(prefix, model_dir_str):
    """Prefix is an argument so the cache key differs per model. A no-arg loader
    collides across pages -- two files both defining load_model() means the second
    page silently receives the first page's model, and the feature widths clash."""
    return smu.load_solvent_model(prefix, Path(model_dir_str))


try:
    model, encoders, training_fps, meta = load_solvent_model_cached(
        MODEL_PREFIX, str(MODEL_DIR))
    models_loaded = True
except FileNotFoundError as e:
    models_loaded = False
    st.error(f"Model files not found in {MODEL_DIR}. Make sure {MODEL_PREFIX}_*.pkl/json "
             f"exist there.\n\n{e}")

if models_loaded:
    SOLVENTS = smu.solvent_options(encoders)
    conformal_90, domain_threshold = smu.get_confidence(meta)

    # Guard against loading the wrong model: the input width has to match the
    # feature layout this page builds.
    _expected_scalars = smu.infer_use_scalars(model, encoders)
    if _expected_scalars is not None and _expected_scalars != SPEC["scalars"]:
        st.error(
            f"Loaded model expects {model.n_features_in_} features, which does not "
            f"match the {SPEC['label']} layout. This usually means the wrong model "
            f"was loaded or {MODEL_PREFIX} was trained on a different feature set. "
            f"Clear the Streamlit cache and check {MODEL_PREFIX}_model.pkl."
        )
        st.stop()
    if meta.get("target") not in (None, SPEC["target"]):
        st.warning(f"Metadata target is '{meta.get('target')}' but this page expects "
                   f"'{SPEC['target']}'.")

    if training_fps is None or domain_threshold is None:
        st.warning(
            "This model was saved without applicability-domain artifacts, so no "
            "in-domain check is available. Predictions still carry a 90% interval, "
            "but there is no signal for whether a molecule is unlike the training set. "
            f"Re-save {MODEL_PREFIX} with save_model_with_ad to enable it."
        )

    def render_result(result):
        st.metric(f"{SPEC['label']} ({SPEC['unit']}) in {result['solvent']}",
                  f"{result['prediction']:.3f}")
        st.caption(f"90% interval: [{result['lower_90']:.3f}, {result['upper_90']:.3f}]")
        if result["in_domain"] is None:
            st.info("Domain check unavailable for this model")
        elif result["in_domain"]:
            st.success(f"In domain (AD score: {result['ad_score']:.3f})")
        else:
            st.warning(f"Outside training domain (AD score: {result['ad_score']:.3f} "
                       f"< {result['domain_threshold']:.3f})")

    input_type = st.selectbox("Choose input type:", ["Single SMILES", "Upload CSV"])

    if input_type == "Single SMILES":
        smiles = st.text_input("Enter a SMILES string:",
                               value="FC(F)(F)C(F)(F)C(F)(F)C(=O)[O-]")
        solvent = st.selectbox("Choose solvent:", SOLVENTS)

        if smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                st.success("Valid SMILES")
            else:
                st.error("Invalid SMILES")

            run_button = st.button("Run Prediction")
            if run_button and mol is not None:
                try:
                    result = smu.predict_with_confidence(
                        smiles, solvent, model, encoders, training_fps, meta,
                        SPEC["scalars"])
                    st.session_state[f"{MODEL_PREFIX}_single"] = {
                        "smiles": smiles, "mol": mol, "result": result}
                except Exception as e:
                    st.session_state[f"{MODEL_PREFIX}_single"] = {"error": str(e)}

        if f"{MODEL_PREFIX}_single" in st.session_state:
            state = st.session_state[f"{MODEL_PREFIX}_single"]
            if "error" in state:
                st.error(f"Prediction failed: {state['error']}")
            else:
                col_img, col_results = st.columns([1, 2])
                with col_img:
                    st.image(Draw.MolToImage(state["mol"], size=(300, 300)),
                             caption="Input structure")
                with col_results:
                    res = state["result"]
                    st.write(f"Detected protonation state: **{res['protonation_state']}** "
                             f"(formal charge {res['formal_charge']:+d})")
                    st.caption(
                        "Derived from the structure, not from the database's "
                        "protonation_state column."
                    )
                    render_result(res)

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
                    st.info(f"Using the 'solvent' column. Recognised values: {SOLVENTS}. "
                            f"Rows with anything else are skipped.")
                    batch_solvent = None
                else:
                    st.info("No 'solvent' column found -- choose one for all rows.")
                    batch_solvent = st.selectbox("Choose solvent:", SOLVENTS, key="batch_solv")

                run_button = st.button("Run Prediction")
                if run_button:
                    rows = []
                    progress = st.progress(0)
                    for i, row_data in batch_df.iterrows():
                        smi = row_data["SMILES"]
                        if has_solvent_col:
                            raw = str(row_data["solvent"])
                            match = [s for s in SOLVENTS if s.lower() == raw.lower()]
                            solv_list = match if match else []
                            if not solv_list:
                                rows.append({"SMILES": smi, "solvent": raw,
                                             "status": f"skipped -- unknown solvent"})
                                progress.progress((i + 1) / len(batch_df))
                                continue
                        else:
                            solv_list = [batch_solvent]

                        for solv in solv_list:
                            row = {"SMILES": smi, "solvent": solv}
                            try:
                                r = smu.predict_with_confidence(
                                    smi, solv, model, encoders, training_fps, meta,
                                    SPEC["scalars"])
                                row.update({
                                    "status": "ok",
                                    "protonation_state": r["protonation_state"],
                                    "formal_charge": r["formal_charge"],
                                    f"{SPEC['target']}_pred": r["prediction"],
                                    "lower_90": r["lower_90"],
                                    "upper_90": r["upper_90"],
                                    "in_domain": r["in_domain"],
                                    "ad_score": r["ad_score"],
                                })
                            except Exception as e:
                                row["status"] = f"failed: {e}"
                            rows.append(row)
                        progress.progress((i + 1) / len(batch_df))

                    st.session_state[f"{MODEL_PREFIX}_batch"] = pd.DataFrame(rows)

        if f"{MODEL_PREFIX}_batch" in st.session_state:
            res_df = st.session_state[f"{MODEL_PREFIX}_batch"]
            st.subheader("Batch Prediction Results")

            n_ok = int((res_df.get("status") == "ok").sum())
            n_skip = int((res_df.get("status", pd.Series(dtype=str))
                          .astype(str).str.startswith("skipped")).sum())
            n_out = int((res_df.get("in_domain", pd.Series(dtype=object)) == False).sum())
            c = st.columns(3)
            c[0].metric("Predicted", n_ok)
            c[1].metric("Skipped", n_skip)
            c[2].metric("Outside domain", n_out)

            st.dataframe(res_df, use_container_width=True)
            st.download_button(
                "Download Results CSV",
                data=res_df.to_csv(index=False),
                file_name=f"{MODEL_PREFIX}_batch_results.csv",
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
