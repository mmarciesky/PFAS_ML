import streamlit as st
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
import mol_tools as mt

st.set_page_config(page_title="Data Explorer", layout="wide")
st.title("PFAS Data Explorer")
st.subheader("Data Behind the Models")
st.caption(
    "For Mulliken/Löwdin charges, XYZ coordinates, fragment molecule data, "
    "and vibrational frequency data, see the full database on "
    "[GitHub](https://github.com/mmarciesky/PFAS_Database)."
)
MAIN_TABLE_PATH = Path(__file__).parent.parent / "main_table.csv"
BDE_TABLE_PATH = Path(__file__).parent.parent / "BDE_table.csv"
    
if not MAIN_TABLE_PATH.exists():
    st.error(f"Couldn't find main_table.csv at {MAIN_TABLE_PATH}")
else:
    main_df = pd.read_csv(MAIN_TABLE_PATH)
    main_df = mt.enrich_with_structural_classes(main_df)
    bde_df = mt.load_bde_table(BDE_TABLE_PATH) if BDE_TABLE_PATH.exists() else pd.DataFrame()
    if bde_df.empty:
        st.warning(f"Couldn't find BDE table at {BDE_TABLE_PATH} — BDE sections will be skipped.")
    st.sidebar.header("Filters")
    solvent_options = sorted(
        set(main_df['Solvent'].dropna().unique())
        | set(bde_df['Solvent'].dropna().unique() if not bde_df.empty else [])
    )
    solvent_filter = st.sidebar.multiselect(
        "Solvent", options=solvent_options, default=solvent_options,
    )

    protonation_options = sorted(main_df['protonation_state'].dropna().unique())
    protonation_filter = st.sidebar.multiselect(
        "Protonation state", options=protonation_options, default=protonation_options,
    )
    main_df['headgroup_display'] = main_df['headgroup'].fillna('Unknown')
    headgroup_options = sorted(main_df['headgroup_display'].unique())
    headgroup_filter = st.sidebar.multiselect(
        "Headgroup", options=headgroup_options, default=headgroup_options,
    )


    structural_class_map = {
        'Ether-containing': 'has_ether',
        'Chlorinated': 'has_chlorine',
        'Fluorotelomer-based': 'is_fluorotelomer',
        'Branched': 'is_branched',
        'Cyclic': 'is_cyclic',
        'Nitrogen-containing': 'has_nitrogen',
        'Alkene-containing': 'has_alkene',
    }
    main_df['is_plain'] = ~main_df[list(structural_class_map.values())].any(axis=1)
    structural_class_map['Plain (no special features)'] = 'is_plain'

    structural_class_options = list(structural_class_map.keys())
    structural_class_filter = st.sidebar.multiselect(
        "Structural classes", options=structural_class_options,
        default=structural_class_options,
    )

    selected_cols = [structural_class_map[label] for label in structural_class_filter]
    structural_mask = main_df[selected_cols].any(axis=1) if selected_cols else pd.Series(False, index=main_df.index)

    filtered_df = main_df[
        main_df['Solvent'].isin(solvent_filter)
        & main_df['protonation_state'].isin(protonation_filter)
        & main_df['headgroup_display'].isin(headgroup_filter)
        & structural_mask
    ]
    if not bde_df.empty:
        molecule_attrs = main_df.drop_duplicates(subset='Inchikey')[
            ['Inchikey', 'protonation_state', 'headgroup_display'] + list(structural_class_map.values())
        ]
        bde_df_annotated = bde_df.merge(molecule_attrs, on='Inchikey', how='left')
        bde_structural_mask = (
            bde_df_annotated[selected_cols].any(axis=1)
            if selected_cols else pd.Series(False, index=bde_df_annotated.index)
        )
        filtered_bde_df = bde_df_annotated[
            bde_df_annotated['Solvent'].isin(solvent_filter)
            & bde_df_annotated['protonation_state'].isin(protonation_filter)
            & bde_df_annotated['headgroup_display'].isin(headgroup_filter)
            & bde_structural_mask
        ]
    else:
        filtered_bde_df = bde_df

# --- Quick counts dashboard ---
    st.subheader("At a glance")
    with st.container(border=True):
        metrics = [
            ("Unique molecules", filtered_df['Inchikey'].nunique()),
        ]
        for solvent in solvent_options:
            n = filtered_df.loc[filtered_df['Solvent'] == solvent, 'Inchikey'].nunique()
            metrics.append((f"In {solvent}", n))

        metrics += [
            ("Oxidation potentials", int(filtered_df.loc[filtered_df['oxidation_potential_V_vs_SHE'].notna(), 'Inchikey'].nunique())),
            ("Reduction potentials", int(filtered_df.loc[filtered_df['reduction_potential_V_vs_SHE'].notna(), 'Inchikey'].nunique())),
            ("logKow", int(filtered_df.loc[filtered_df['logKow'].notna(), 'Inchikey'].nunique())),
            ("logKaw", int(filtered_df.loc[filtered_df['logKaw'].notna(), 'Inchikey'].nunique())),
            ("Dipole moment", int(filtered_df.loc[filtered_df['dipole_moment_debye'].notna(), 'Inchikey'].nunique())),
            ("HOMO-LUMO gap", int(filtered_df.loc[filtered_df['homo_lumo_gap_eV'].notna(), 'Inchikey'].nunique())),
        ]

        if not filtered_bde_df.empty:
            metrics += [
                ("Total BDE values", len(filtered_bde_df)),
                ("Parent molecules with BDE data", filtered_bde_df['Inchikey'].nunique()),
            ]

        n_cols = 4
        for row_start in range(0, len(metrics), n_cols):
            row_metrics = metrics[row_start:row_start + n_cols]
            cols = st.columns(n_cols, gap="small")
            for col, (label, value) in zip(cols, row_metrics):
                col.metric(label, value)
    st.subheader("Structural diversity")
    with st.container(border=True):

        n_unique = filtered_df['Inchikey'].nunique()

        class_row = st.columns(2, gap="medium")
        with class_row[0]:
            st.caption(f"Structural motifs present (of {n_unique} unique molecules)")
            fig = mt.plot_structural_class_counts(filtered_df)
            mt.chart_with_expand(fig, key="struct_classes", base_width=440, expanded_width=900)
        with class_row[1]:
            st.caption("Carbon chain length distribution")
            fig = mt.plot_histogram(filtered_df.drop_duplicates(subset='Inchikey'), 'n_carbon')
            mt.chart_with_expand(fig, key="n_carbon", base_width=440, expanded_width=900)
    st.divider()
    left_col, right_col = st.columns(2, gap="medium")

    with left_col:
        st.subheader("Dataset overview")
        with st.container(border=True):
            row_a = st.columns(2, gap="small")
            with row_a[0]:
                st.caption("By headgroup")
                if 'headgroup' in filtered_df.columns:
                    fig = mt.plot_categorical_counts(filtered_df, 'headgroup')
                    mt.chart_with_expand(fig, key="headgroup", base_width=340, expanded_width=800)
            with row_a[1]:
                st.caption("By protonation state")
                if 'protonation_state' in filtered_df.columns:
                    fig = mt.plot_categorical_counts(filtered_df, 'protonation_state')
                    mt.chart_with_expand(fig, key="protonation", base_width=340, expanded_width=800)

            st.divider()

            row_b = st.columns(2, gap="small")
            with row_b[0]:
                st.caption("Fluorination ratio")
                fig = mt.plot_histogram(filtered_df, 'fluorination_ratio')
                mt.chart_with_expand(fig, key="fluor_hist", base_width=340, expanded_width=800)
            with row_b[1]:
                st.caption("Molecular weight")
                fig = mt.plot_histogram(filtered_df, 'mw')
                mt.chart_with_expand(fig, key="mw_hist", base_width=340, expanded_width=800)

            row_c = st.columns(2, gap="small")
            with row_c[0]:
                st.caption("Dipole moment (Debye)")
                fig = mt.plot_histogram(filtered_df, 'dipole_moment_debye')
                mt.chart_with_expand(fig, key="dipole_hist", base_width=340, expanded_width=800)
            with row_c[1]:
                st.caption("HOMO-LUMO gap (eV)")
                fig = mt.plot_histogram(filtered_df, 'homo_lumo_gap_eV')
                mt.chart_with_expand(fig, key="gap_hist", base_width=340, expanded_width=800)

    with right_col:
        st.subheader("Chemical space (UMAP)")
        with st.container(border=True):
            umap_props = st.multiselect(
                "Properties to use",
                options=mt.ALL_PROPERTY_COLS,
                default=['dipole_moment_debye', 'homo_lumo_gap_eV', 'fluorination_ratio'],
                key="umap_props",
            )
            color_options = ['headgroup', 'fluorination_ratio', 'protonation_state', 'Solvent']
            color_by = st.selectbox("Color by", options=color_options, key="umap_color")

            if len(umap_props) >= 2:
                try:
                    umap_df, reducer, scaler = mt.compute_umap(filtered_df, umap_props)
                    hover_cols = ['Inchikey', 'headgroup', 'Solvent'] + umap_props
                    umap_expanded = st.checkbox("Expand", key="expand_umap", value=False)
                    w, h = (1300, 1100) if umap_expanded else (900, 800)
                    fig = mt.plot_embedding_scatter_interactive(umap_df, 'UMAP1', 'UMAP2', color_by, hover_cols, width=w, height=h)
                    st.plotly_chart(fig, use_container_width=False)
                    st.caption(f"{len(umap_df)} rows with complete data for these properties")
                except ValueError as e:
                    st.warning(str(e))
            else:
                st.info("Pick at least 2 properties to run UMAP.")
    st.divider()
    st.subheader("Redox and partition coefficients vs. fluorination")
    with st.container(border=True):
        dist_row1 = st.columns(2, gap="small")
        with dist_row1[0]:
            fig, n = mt.plot_scatter_by_group(filtered_df, 'fluorination_ratio', 'oxidation_potential_V_vs_SHE')
            st.caption(f"Oxidation potential vs. fluorination ratio — n={n}")
            mt.chart_with_expand(fig, key="ox_fluor", base_width=440, expanded_width=900, empty_message="No overlapping data for this combination.")
        with dist_row1[1]:
            fig, n = mt.plot_scatter_by_group(filtered_df, 'fluorination_ratio', 'reduction_potential_V_vs_SHE')
            st.caption(f"Reduction potential vs. fluorination ratio — n={n}")
            mt.chart_with_expand(fig, key="red_fluor", base_width=440, expanded_width=900, empty_message="No overlapping data for this combination.")

        dist_row2 = st.columns(2, gap="small")
        with dist_row2[0]:
            fig, n = mt.plot_scatter_by_group(filtered_df, 'fluorination_ratio', 'logKow')
            st.caption(f"logKow vs. fluorination ratio — n={n}")
            mt.chart_with_expand(fig, key="kow_fluor", base_width=440, expanded_width=900, empty_message="No overlapping data for this combination.")
        with dist_row2[1]:
            fig, n = mt.plot_scatter_by_group(filtered_df, 'fluorination_ratio', 'logKaw')
            st.caption(f"logKaw vs. fluorination ratio — n={n}")
            mt.chart_with_expand(fig, key="kaw_fluor", base_width=440, expanded_width=900, empty_message="No overlapping data for this combination.")
    st.divider()
    st.subheader("Bond dissociation energies")
    if filtered_bde_df.empty:
        st.info("No BDE data available for the current filters.")
    else:
        bde_enriched = mt.enrich_bde_with_bond_type(filtered_bde_df)
        with st.container(border=True):
            bde_row = st.columns(2, gap="small")
            with bde_row[0]:
                st.caption("BDE distribution by solvent")
                fig = mt.plot_histogram_by_group(bde_enriched, 'BDE_wB97X-V', 'Solvent')
                mt.chart_with_expand(fig, key="bde_solvent", base_width=440, expanded_width=900)
            with bde_row[1]:
                st.caption("BDE by bond type (top 8 most common)")
                fig = mt.plot_boxplot_by_group(bde_enriched, 'BDE_wB97X-V', 'bond_type')
                mt.chart_with_expand(fig, key="bde_bondtype", base_width=440, expanded_width=900)
    st.divider()
    st.subheader("Look up a molecule")
    with st.container(border=True):
        query = st.text_input(
            "SMILES", placeholder="O=C([O-])C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",
        )
        lookup_df = None
        is_exact_match = False

        if query:
            canon_query = mt.canonicalize(query)
            if canon_query is None:
                st.error("Couldn't parse that SMILES.")
            else:
                exact = filtered_df[filtered_df['canonical_smiles'] == canon_query]
                if len(exact):
                    st.success(f"Exact match found ({len(exact)} row(s) — one per solvent).")
                    lookup_df = exact
                    is_exact_match = True
                else:
                    st.warning("No exact match in the current filtered view — showing closest neighbor(s) by structure instead.")
                    lookup_df = mt.find_similar(query, filtered_df, top_n=5)

                col_img, col_table = st.columns([1, 3])
                with col_img:
                    img = mt.draw_molecule(query)
                    if img is not None:
                        st.image(img, caption="Query structure")
                with col_table:
                    st.dataframe(lookup_df, use_container_width=True)
                    st.download_button(
                        "Download this lookup as CSV",
                        data=lookup_df.to_csv(index=False),
                        file_name="molecule_lookup.csv",
                        mime="text/csv",
                        key="lookup_download",
                    )

                if lookup_df is not None and len(lookup_df) and not filtered_bde_df.empty:
                    label = "Include BDE data for this molecule" if is_exact_match else "Include BDE data for the closest match"
                    include_bde = st.checkbox(label, value=False, key="include_bde_checkbox")
                    if include_bde:
                        inchikey = lookup_df.iloc[0]['Inchikey']
                        mol_bde = filtered_bde_df[filtered_bde_df['Inchikey'] == inchikey].sort_values('BDE_wB97X-V')
                        if len(mol_bde):
                            st.write(f"{len(mol_bde)} bond(s) with BDE data for this molecule")
                            st.dataframe(mol_bde, use_container_width=True)

                            if is_exact_match:
                                # Use the BDE table's own Parent_SMILES for drawing --
                                # Bond_Index only lines up correctly against the exact
                                # SMILES string it was originally computed from.
                                drawing_smiles = mol_bde.iloc[0]['Parent_SMILES']
                                png = mt.draw_molecule_with_bde_labels(drawing_smiles, mol_bde)
                                if png is not None:
                                    st.image(png, caption="Bonds labeled with BDE (kcal/mol) — verify against a known molecule first")
                            else:
                                st.caption("Structure diagram not shown for nearest-neighbor matches — table only.")
                        else:
                            st.info("No BDE data found for this molecule.")
    st.divider()
    st.subheader("Raw data")
    display_df = lookup_df if lookup_df is not None and len(lookup_df) else filtered_df
    showing_lookup = lookup_df is not None and len(lookup_df) > 0

    with st.expander("View table", expanded=showing_lookup):
        st.write(f"Showing {'lookup result' if showing_lookup else 'all filtered rows'}: {len(display_df)} rows")
        st.dataframe(display_df, use_container_width=True)
        st.download_button(
            "Download this table as CSV",
            data=display_df.to_csv(index=False),
            file_name="main_table_export.csv",
            mime="text/csv",
        )
    st.divider()
    st.subheader("BDE table")
    if not filtered_bde_df.empty:
        with st.expander("View BDE table (separate from the main table)"):
            st.dataframe(filtered_bde_df, use_container_width=True)
            st.download_button(
                "Download BDE table as CSV",
                data=filtered_bde_df.to_csv(index=False),
                file_name="bde_table_export.csv",
                mime="text/csv",
                key="bde_download",
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