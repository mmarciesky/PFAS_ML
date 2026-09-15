import io

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.preprocessing import StandardScaler
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw
# --- Global chart style: transparent backgrounds + light text so charts
# blend into a dark Streamlit theme instead of showing as white boxes.
# NOTE: this assumes dark theme -- if you ever add a light-mode toggle,
# these text colors would need to flip.
plt.rcParams.update({
    'font.size': 9,
    'font.family': 'sans-serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,
    'figure.dpi': 150,
    'text.color': '#e6e6e6',
    'axes.labelcolor': '#e6e6e6',
    'xtick.color': '#e6e6e6',
    'ytick.color': '#e6e6e6',
    'axes.edgecolor': '#5a5a5a',
})

QUALITATIVE_PALETTE = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2', '#937860']

ALL_PROPERTY_COLS = [
    'logKow', 'logKaw',
    'oxidation_potential_V_vs_SHE', 'reduction_potential_V_vs_SHE',
    'dipole_moment_debye', 'homo_lumo_gap_eV',
    'fluorination_ratio', 'mw',
]


def render_fig(fig, width=380, empty_message="No data available for this view."):
    import streamlit as st
    if fig is None:
        st.info(empty_message)
        return
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    buf.seek(0)
    st.image(buf, width=width)
    plt.close(fig)


def coverage_summary(df, cols=ALL_PROPERTY_COLS):
    rows = []
    for col in cols:
        if col not in df.columns:
            continue
        n = df[col].notna().sum()
        rows.append({'property': col, 'n_rows': n, 'pct': round(100 * n / len(df), 1)})
    return pd.DataFrame(rows).sort_values('n_rows', ascending=False)


def plot_correlation_heatmap(df, property_cols, figsize=(5, 4)):
    sub = df[property_cols].dropna(how='all')
    if len(sub) < 2:
        return None, len(sub)
    corr = sub.corr()
    with plt.rc_context({'axes.grid': False}):
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, ax=ax,
                    cbar_kws={'shrink': 0.8}, annot_kws={'color': '#1a1a1a'})
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()
    return fig, len(sub)


def plot_headgroup_counts(df, figsize=(4, 3)):
    counts = df['headgroup'].value_counts()
    fig, ax = plt.subplots(figsize=figsize)
    counts.plot(kind='bar', ax=ax, color=QUALITATIVE_PALETTE[0])
    ax.set_ylabel('Number of molecules')
    ax.set_xlabel('')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig


def compute_umap(df, property_cols, n_components=2, n_neighbors=15, min_dist=0.1, random_state=42):
    sub = df.dropna(subset=property_cols)
    if len(sub) < n_neighbors + 1:
        raise ValueError(
            f"Only {len(sub)} rows have all of {property_cols} -- "
            f"need at least {n_neighbors + 1} for n_neighbors={n_neighbors}."
        )
    scaler = StandardScaler()
    scaled = scaler.fit_transform(sub[property_cols])

    reducer = umap.UMAP(
        n_components=n_components, n_neighbors=n_neighbors,
        min_dist=min_dist, random_state=random_state,
    )
    embedding = reducer.fit_transform(scaled)

    result = sub.copy()
    for i in range(n_components):
        result[f'UMAP{i+1}'] = embedding[:, i]
    return result, reducer, scaler


def plot_embedding_scatter(embed_df, x_col, y_col, color_by, figsize=(4, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    if embed_df[color_by].dtype == object or str(embed_df[color_by].dtype) == 'category':
        for i, (val, sub) in enumerate(embed_df.groupby(color_by)):
            color = QUALITATIVE_PALETTE[i % len(QUALITATIVE_PALETTE)]
            ax.scatter(sub[x_col], sub[y_col], label=str(val), alpha=0.75, s=25, color=color)
        legend = ax.legend(fontsize=7, title=color_by, title_fontsize=8, frameon=False)
        for text in legend.get_texts():
            text.set_color('#e6e6e6')
        legend.get_title().set_color('#e6e6e6')
    else:
        sc = ax.scatter(embed_df[x_col], embed_df[y_col], c=embed_df[color_by], cmap='viridis', s=25)
        cbar = fig.colorbar(sc, ax=ax, label=color_by, shrink=0.8)
        cbar.ax.yaxis.label.set_color('#e6e6e6')
        cbar.ax.tick_params(colors='#e6e6e6')
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig

import plotly.express as px

def plot_embedding_scatter_interactive(embed_df, x_col, y_col, color_by, hover_cols, width=560, height=480):
    is_categorical = embed_df[color_by].dtype == object or str(embed_df[color_by].dtype) == 'category'
    kwargs = dict(
        x=x_col, y=y_col, color=color_by,
        hover_data={c: True for c in hover_cols},
        width=width, height=height,
    )
    if is_categorical:
        kwargs['color_discrete_sequence'] = QUALITATIVE_PALETTE
    else:
        kwargs['color_continuous_scale'] = 'viridis'

    fig = px.scatter(embed_df, **kwargs)
    fig.update_traces(marker=dict(size=7, opacity=0.8, line=dict(width=0.5, color='#1a1a1a')))
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig

def plot_categorical_counts(df, col, figsize=(3.6, 2.6)):
    counts = df[col].dropna().value_counts()
    if counts.empty:
        return None
    fig, ax = plt.subplots(figsize=figsize)
    counts.plot(kind='bar', ax=ax, color=QUALITATIVE_PALETTE[0])
    ax.set_ylabel('Count')
    ax.set_xlabel('')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig


def plot_histogram(df, col, bins=20, figsize=(3.6, 2.6)):
    data = df[col].dropna()
    if data.empty:
        return None
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(data, bins=bins, color=QUALITATIVE_PALETTE[2])
    ax.set_xlabel(col)
    ax.set_ylabel('Count')
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig

def canonicalize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def get_fingerprint(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def find_similar(query_smiles, df, smiles_col='Smiles', top_n=5, radius=2, n_bits=2048):
    """
    Returns rows from df for the top_n most structurally similar molecules
    (Tanimoto on Morgan fingerprints). Scores one fingerprint per unique
    Inchikey first, then joins back -- so a molecule with both Water and
    DMSO rows doesn't get scored twice.
    """
    query_fp = get_fingerprint(query_smiles, radius, n_bits)
    if query_fp is None:
        return df.iloc[0:0]

    unique_mols = df[['Inchikey', smiles_col]].drop_duplicates(subset='Inchikey')
    fps = unique_mols[smiles_col].apply(lambda s: get_fingerprint(s, radius, n_bits))
    sims = fps.apply(lambda fp: DataStructs.TanimotoSimilarity(query_fp, fp) if fp is not None else None)
    unique_mols = unique_mols.assign(similarity=sims)
    top = unique_mols.sort_values('similarity', ascending=False).head(top_n)

    return df.merge(top[['Inchikey', 'similarity']], on='Inchikey', how='inner') \
             .sort_values('similarity', ascending=False)


def draw_molecule(smiles, size=(280, 280)):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Draw.MolToImage(mol, size=size)
def plot_scatter_by_group(df, x_col, y_col, group_col='headgroup', figsize=(4, 3.2)):
    sub = df[[x_col, y_col, group_col]].dropna()
    if sub.empty:
        return None, 0
    fig, ax = plt.subplots(figsize=figsize)
    for i, (val, g) in enumerate(sub.groupby(group_col)):
        color = QUALITATIVE_PALETTE[i % len(QUALITATIVE_PALETTE)]
        ax.scatter(g[x_col], g[y_col], label=str(val), color=color, alpha=0.75, s=25)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    legend = ax.legend(fontsize=7, title=group_col, title_fontsize=8, frameon=False)
    for text in legend.get_texts():
        text.set_color('#e6e6e6')
    legend.get_title().set_color('#e6e6e6')
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig, len(sub)

_ETHER_SMARTS = Chem.MolFromSmarts('[CX4]-[OX2]-[CX4]')          # backbone ether O, excludes C=O/carboxylate
_TELOMER_SMARTS = Chem.MolFromSmarts('[CX4](F)(F)[CH2][CH2]')    # CF2 directly feeding into an unfluorinated CH2CH2


_ALKENE_SMARTS = Chem.MolFromSmarts('[C;!a]=[C;!a]')  # C=C, explicitly excluding aromatic ring bonds

def classify_pfas_structure(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {'has_ether': None, 'has_chlorine': None, 'is_fluorotelomer': None,
                'is_cyclic': None, 'is_branched': None, 'has_nitrogen': None,
                'has_alkene': None, 'n_carbon': None}

    has_ether = mol.HasSubstructMatch(_ETHER_SMARTS)
    has_chlorine = any(atom.GetSymbol() == 'Cl' for atom in mol.GetAtoms())
    is_fluorotelomer = mol.HasSubstructMatch(_TELOMER_SMARTS)
    is_cyclic = mol.GetRingInfo().NumRings() > 0
    has_nitrogen = any(atom.GetSymbol() == 'N' for atom in mol.GetAtoms())
    has_alkene = mol.HasSubstructMatch(_ALKENE_SMARTS)

    is_branched = any(
        sum(1 for nbr in atom.GetNeighbors() if nbr.GetSymbol() == 'C') >= 3
        for atom in mol.GetAtoms() if atom.GetSymbol() == 'C'
    )

    n_carbon = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')

    return {
        'has_ether': has_ether, 'has_chlorine': has_chlorine,
        'is_fluorotelomer': is_fluorotelomer, 'is_cyclic': is_cyclic,
        'is_branched': is_branched, 'has_nitrogen': has_nitrogen,
        'has_alkene': has_alkene, 'n_carbon': n_carbon,
    }


def enrich_with_structural_classes(df, smiles_col='Smiles'):
    """Adds the classify_pfas_structure() columns, computed once per unique molecule."""
    import streamlit as st
    unique_mols = df[['Inchikey', smiles_col]].drop_duplicates(subset='Inchikey')
    classes = pd.DataFrame(list(unique_mols[smiles_col].apply(classify_pfas_structure)))
    unique_mols = pd.concat([unique_mols.reset_index(drop=True), classes.reset_index(drop=True)], axis=1)
    return df.merge(unique_mols.drop(columns=[smiles_col]), on='Inchikey', how='left')


def plot_structural_class_counts(df, figsize=(4, 3)):
    unique_df = df.drop_duplicates(subset='Inchikey')
    if unique_df.empty:
        return None
    classes = {
        'Ether-containing': unique_df['has_ether'].sum(),
        'Chlorinated': unique_df['has_chlorine'].sum(),
        'Fluorotelomer-based': unique_df['is_fluorotelomer'].sum(),
        'Branched': unique_df['is_branched'].sum(),
        'Cyclic': unique_df['is_cyclic'].sum(),
        'Nitrogen-containing': unique_df['has_nitrogen'].sum(),
        'Alkene-containing': unique_df['has_alkene'].sum(),
    }
    fig, ax = plt.subplots(figsize=figsize)
    pd.Series(classes).plot(kind='barh', ax=ax, color=QUALITATIVE_PALETTE[3])
    ax.set_xlabel('Number of molecules')
    ax.invert_yaxis()
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig

def get_bond_type(parent_smiles, bond_index):
    """
    Element pair at the broken bond, e.g. 'C-F'. Relies on Bond_Index
    matching RDKit's own bond ordering when it parses parent_smiles --
    UNVERIFIED against your real pipeline. Sanity check: bond_index=0 on
    a sulfonic acid parent should come back 'O-S' if this alignment holds.
    """
    mol = Chem.MolFromSmiles(parent_smiles)
    if mol is None:
        return None
    bonds = mol.GetBonds()
    idx = int(bond_index)
    if idx >= len(bonds):
        return None
    a1, a2 = bonds[idx].GetBeginAtom().GetSymbol(), bonds[idx].GetEndAtom().GetSymbol()
    return '-'.join(sorted([a1, a2]))

def enrich_bde_with_bond_type(bde_df):
    import streamlit as st
    bde_df = bde_df.copy()
    bde_df['bond_type'] = bde_df.apply(
        lambda r: get_bond_type(r['Parent_SMILES'], r['Bond_Index']), axis=1
    )
    return bde_df


def plot_histogram_by_group(df, col, group_col, bins=20, figsize=(4.2, 3.2)):
    sub = df[[col, group_col]].dropna()
    if sub.empty:
        return None
    fig, ax = plt.subplots(figsize=figsize)
    for i, (val, g) in enumerate(sub.groupby(group_col)):
        color = QUALITATIVE_PALETTE[i % len(QUALITATIVE_PALETTE)]
        ax.hist(g[col], bins=bins, alpha=0.55, label=str(val), color=color)
    ax.set_xlabel('BDE (kcal/mol)')
    ax.set_ylabel('Count')
    legend = ax.legend(fontsize=7, title=group_col, title_fontsize=8, frameon=False)
    for text in legend.get_texts():
        text.set_color('#e6e6e6')
    legend.get_title().set_color('#e6e6e6')
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig


def plot_boxplot_by_group(df, value_col, group_col, figsize=(5, 3.2), top_n_groups=8):
    sub_check = df[[value_col, group_col]].dropna()
    if sub_check.empty:
        return None
    top_groups = sub_check[group_col].value_counts().head(top_n_groups).index
    sub = sub_check[sub_check[group_col].isin(top_groups)]
    fig, ax = plt.subplots(figsize=figsize)
    order = sub.groupby(group_col)[value_col].median().sort_values().index
    data = [sub.loc[sub[group_col] == g, value_col] for g in order]
    try:
        bp = ax.boxplot(data, tick_labels=order, patch_artist=True)
    except TypeError:
        bp = ax.boxplot(data, labels=order, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor(QUALITATIVE_PALETTE[0])
        patch.set_alpha(0.7)
    ax.set_ylabel('BDE (kcal/mol)')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    fig.tight_layout()
    return fig

from rdkit.Chem.Draw import rdMolDraw2D

def draw_molecule_with_bde_labels(smiles, bde_rows, size=(400, 400)):
    """
    Labels each bond in the 2D depiction with its BDE value. Same
    Bond_Index -> RDKit-bond correspondence caveat as get_bond_type().
    Returns PNG bytes (pass directly to st.image), or None if unparseable.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    bonds = mol.GetBonds()
    for _, row in bde_rows.iterrows():
        idx = int(row['Bond_Index'])
        if idx < len(bonds):
            bonds[idx].SetProp('bondNote', f"{row['BDE_wB97X-V']:.1f}")
    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()
def smiles_to_inchikey(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToInchiKey(mol)



def load_bde_table(path):
    import streamlit as st
    bde_df = pd.read_csv(path)
    bde_df['Inchikey'] = bde_df['Parent_SMILES'].apply(smiles_to_inchikey)
    return bde_df
def chart_with_expand(fig, key, base_width=420, expanded_width=900, empty_message="No data available for this view."):
    """
    Draws an 'Expand' checkbox above a chart; when checked, renders at
    expanded_width instead of base_width. key must be unique per chart
    on the page, or Streamlit will throw a duplicate-widget error.
    """
    import streamlit as st
    if fig is None:
        st.info(empty_message)
        return
    expanded = st.checkbox("Expand", key=f"expand_{key}", value=False)
    width = expanded_width if expanded else base_width
    render_fig(fig, width=width, empty_message=empty_message)