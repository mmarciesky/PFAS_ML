import streamlit as st
import pandas as pd
from pathlib import Path

ACCENTS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

# ---------------------------------------------------------------------------
# Styled helpers
# ---------------------------------------------------------------------------

def styled_metric(col, label, value, accent="#4C72B0"):
    col.markdown(f"""
    <div style="
        border-top: 3px solid {accent};
        background-color: rgba(255,255,255,0.04);
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
    ">
        <div style="font-size: 0.75rem; color: #9a9a9a; text-transform: uppercase; letter-spacing: 0.04em;">
            {label}
        </div>
        <div style="font-size: 1.9rem; font-weight: 600; margin-top: 0.2rem;">
            {value}
        </div>
    </div>
    """, unsafe_allow_html=True)


def feature_card(col, title, description, link, accent):
    with col:
        st.markdown(f"""
        <div style="
            border-top: 4px solid {accent};
            background-color: rgba(255,255,255,0.03);
            border-radius: 8px;
            padding: 1rem 1.2rem 0.6rem 1.2rem;
        ">
            <div style="font-size: 1.15rem; font-weight: 600; margin-bottom: 0.3rem;">{title}</div>
            <div style="color: #b0b0b0; margin-bottom: 0.8rem;">{description}</div>
        </div>
        """, unsafe_allow_html=True)
        st.page_link(link, label=f"Open {title}")


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------

HEADER_ACCENT = "#8172B2"

st.markdown(f"""
<div style="padding: 1.5rem 0 0.5rem 0; border-bottom: 3px solid {HEADER_ACCENT}; margin-bottom: 1.5rem; display: flex; align-items: center; gap: 1rem;">
    <div style="
        width: 56px; height: 56px;
        border: 2px dashed #6a6a6a;
        border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.65rem; color: #8a8a8a; text-align: center; line-height: 1.1;
        flex-shrink: 0;
    ">LOGO<br>TBD</div>
    <div>
        <h1 style="font-size: 2.6rem; margin-bottom: 0.2rem;">PFAS Foundry</h1>
        <p style="font-size: 1.15rem; color: #9a9a9a; margin-top: 0;">
            Machine learning predictions and a curated quantum chemistry database for PFAS
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

MAIN_TABLE_PATH = Path(__file__).parent.parent / "main_table.csv"
BDE_TABLE_PATH = Path(__file__).parent.parent / "BDE_table.csv"

solvents_seen = set()

if MAIN_TABLE_PATH.exists():
    main_df = pd.read_csv(
        MAIN_TABLE_PATH,
        usecols=['Inchikey', 'Solvent', 'oxidation_potential_V_vs_SHE',
                 'reduction_potential_V_vs_SHE', 'logKow', 'logKaw'],
    )
    solvents_seen |= set(main_df['Solvent'].dropna().unique())
else:
    main_df = None

if BDE_TABLE_PATH.exists():
    bde_df = pd.read_csv(BDE_TABLE_PATH, usecols=['Parent_SMILES', 'Solvent'])
    solvents_seen |= set(bde_df['Solvent'].dropna().unique())
else:
    bde_df = None

# ---------------------------------------------------------------------------
# Row 1: overview counts
# ---------------------------------------------------------------------------

stat_cols = st.columns(4)

styled_metric(stat_cols[0], "Total unique parent molecules",
              main_df['Inchikey'].nunique() if main_df is not None else "—", ACCENTS[0])
styled_metric(stat_cols[1], "BDE values computed",
              len(bde_df) if bde_df is not None else "—", ACCENTS[1])
styled_metric(stat_cols[2], "Solvents covered",
              len(solvents_seen) if solvents_seen else "—", ACCENTS[2])
stat_cols[2].caption(", ".join(sorted(solvents_seen)) if solvents_seen else "—")
styled_metric(stat_cols[3], "Models available", 3, ACCENTS[3])

# ---------------------------------------------------------------------------
# Row 2: property-specific counts (unique molecules with each value)
# ---------------------------------------------------------------------------

stat_cols2 = st.columns(4)

property_labels = [
    ("Unique Oxidation potentials", "oxidation_potential_V_vs_SHE"),
    ("Unique Reduction potentials", "reduction_potential_V_vs_SHE"),
    ("Unique logKow", "logKow"),
    ("Unique logKaw", "logKaw"),
]

for i, (col, (label, colname)) in enumerate(zip(stat_cols2, property_labels)):
    accent = ACCENTS[i % len(ACCENTS)]
    if main_df is not None and colname in main_df.columns:
        n = main_df.loc[main_df[colname].notna(), 'Inchikey'].nunique()
        styled_metric(col, label, n, accent)
    else:
        styled_metric(col, label, "—", accent)

# ---------------------------------------------------------------------------
# Feature cards
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Explore the platform")

card_cols = st.columns(3)
feature_card(card_cols[0], "Data Explorer",
             "Browse the curated PFAS database — properties, structures, and bond dissociation energies.",
             "pages/4_Data_Explorer.py", ACCENTS[2])
feature_card(card_cols[1], "BDE Predictor",
             "Predict bond dissociation energies for a PFAS molecule from a SMILES string or CSV upload.",
             "pages/0_Predict_BDE.py", ACCENTS[0])
feature_card(card_cols[2], "Redox Predictor",
             "Predict the oxidation and reduction potential for a PFAS molecule from a SMILES string or CSV upload.",
             "pages/5_Redox_Predictor.py", ACCENTS[1])
st.caption("More predictive models are planned for this platform.")

# ---------------------------------------------------------------------------
# Model roadmap
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Model roadmap")
st.caption(
    "The DFT-based predictor models below generate molecular features "
    "(bond strength, redox behavior, partitioning, polarity, electronic structure) "
    "that feed into the Toxicology model, which predicts LD50."
)

roadmap = [
    ("BDE Model", "Preliminary", "pages/0_Predict_BDE.py"),
    ("Redox Model", "Preliminary", "pages/0_Predict_BDE.py"),
    ("Partition Coefficient Model", "Coming soon", None),
    ("Dipole Moment Model", "Coming soon", None),
    ("HOMO-LUMO Gap Model", "Coming soon", None),
]

for name, status, link in roadmap:
    cols = st.columns([3, 3, 2])
    cols[0].markdown(f"**{name}**")
    cols[1].caption(status)
    if link:
        cols[2].page_link(link, label="Open")
    else:
        cols[2].caption("—")

st.divider()
with st.container(border=True):
    st.markdown("**Toxicology Model (LD50)** — *downstream, consumes the features above*")
    st.caption("Coming soon — depends on the predictor models being complete enough to generate reliable feature sets.")
st.markdown("## Platform Expansion")

st.markdown("""

This model now supports gas, water, and DMSO phases. The platform has also expanded 
beyond BDE to include a **Redox Potential Predictor** (oxidation and reduction potential).

In development:
- Dipole moment and HOMO-LUMO gap predictors  
- A downstream toxicology (LD50) model, built on top of these predictors as features  
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