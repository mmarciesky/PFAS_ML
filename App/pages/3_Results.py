import streamlit as st
from pathlib import Path
st.set_page_config(page_title="Results", layout="wide")

st.title("Prediction Results")
st.info("Model Version: v0.1 (Preliminary) — Results are for exploratory use")
if not st.session_state.get("has_results", False):
    st.warning("No results available yet. Please run a prediction first.")

    if st.button("Go back to Predict"):
        st.switch_page("Home.py")

else:
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
                    mime="image/png"
                )

    if results_df is not None:
        st.subheader("Prediction Table")
        st.dataframe(results_df, use_container_width=True)
        st.download_button(
        label="Download Results CSV",
        data=results_df.to_csv(index=False),
        file_name="bde_results.csv",
        mime="text/csv"
        )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Back to Predict"):
            st.switch_page("Home.py")

    with col2:
        if st.button("Clear Results"):
            st.session_state.pop("results_df", None)
            st.session_state.pop("bde_images", None)
            st.session_state["has_results"] = False
            st.switch_page("Home.py")
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
