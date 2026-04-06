#!/usr/bin/env python
"""
PFAS BDE Prediction Script

Predicts bond dissociation energies for PFAS molecules using pre-trained XGBoost model.
See README.md for full documentation.
"""

import sys
import json
import pickle
import argparse
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw
from rdkit.DataStructs import CreateFromBitString
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')
# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

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

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(model_dir='ML_Models'):
    """
    Load pre-trained XGBoost model, solvent encoder, and confidence artifacts.

    Returns
    -------
    model : XGBRegressor
    encoder : OneHotEncoder
    training_fps : list of RDKit ExplicitBitVect
    metadata : dict
    """
    model_dir = Path(model_dir)

    try:
        with open(model_dir / 'xgboost_bde_model_optimized.pkl', 'rb') as f:
            model = pickle.load(f)
        with open(model_dir / 'solvent_encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
        with open(model_dir / 'training_fps.pkl', 'rb') as f:
            training_fps = pickle.load(f)
        with open(model_dir / 'model_metadata.json', 'r') as f:
            metadata = json.load(f)

        return model, encoder, training_fps, metadata

    except FileNotFoundError as e:
        print(f" ERROR: Model files not found in '{model_dir}/'")
        print("Required files:")
        print("  • xgboost_bde_model_optimized.pkl")
        print("  • solvent_encoder.pkl")
        print("  • training_fps.pkl")
        print("  • model_metadata.json")
        raise


# ============================================================================
# DATA LOADING
# ============================================================================

def load_and_validate_data(input_file, verbose=False):
    """Load molecular data from CSV/Excel and validate format."""
    input_file = Path(input_file)

    if verbose:
        print(f"Loading data from: {input_file}")

    if input_file.suffix == '.csv':
        df = pd.read_csv(input_file)
    elif input_file.suffix in ['.xlsx', '.xls']:
        df = pd.read_excel(input_file)
    else:
        raise ValueError("File must be .csv, .xlsx, or .xls format")

    if verbose:
        print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

    if 'SMILES' not in df.columns:
        raise ValueError(f"Missing required 'SMILES' column. Found: {list(df.columns)}")

    if 'solvent' not in df.columns:
        df['solvent'] = 'gas'
        if verbose:
            print("  Added default solvent='gas'")

    df['SMILES_Canonical'] = df['SMILES'].apply(canonicalize_smiles)
    valid_mask = df['SMILES_Canonical'].notna()
    n_invalid = (~valid_mask).sum()

    if n_invalid > 0 and verbose:
        print(f"   Warning: {n_invalid} invalid SMILES removed")
        for ex in df[~valid_mask]['SMILES'].head(3).tolist():
            print(f"    - {ex[:60]}...")

    df = df[valid_mask].copy()

    if verbose:
        print(f"   {len(df)} valid molecules ready")

    return df


# ============================================================================
# FRAGMENTATION
# ============================================================================

def fragment_and_prepare_bonds(df, verbose=False):
    """Fragment molecules into bonds and prepare dataframe for prediction."""
    if verbose:
        print("Fragmenting molecules into bonds...")

    smiles_list = df['SMILES_Canonical'].tolist()
    Parent_PFAS = fragment_molecules(smiles_list)
    df_bonds = convert_parent_pfas_to_dataframe(Parent_PFAS, df)
    df_bonds = df_bonds.reset_index(drop=True)

    if verbose:
        print(f"   Generated {len(df_bonds)} bonds from {len(Parent_PFAS)} molecules")

    return df_bonds


# ============================================================================
# CONFIDENCE — APPLICABILITY DOMAIN
# ============================================================================

def check_applicability_domain(parent_smiles, frag1_smiles, frag2_smiles,
                                solvent, training_fps, threshold,
                                radius=2, nBits=2048):
    """
    Build concatenated fingerprint (parent + frag1 + frag2 + solvent) matching
    training feature format (6145 bits), compute mean Tanimoto similarity to
    5 nearest neighbors in training set.

    Returns
    -------
    mean_sim : float
    in_domain : bool
    """
    bits = []
    for smi in [parent_smiles, frag1_smiles, frag2_smiles]:
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            bits.extend([0] * nBits)
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
            bits.extend(fp.ToList())

    bits.append(1 if solvent == 'water' else 0)
    query_fp = CreateFromBitString("".join(str(b) for b in bits))

    sims = DataStructs.BulkTanimotoSimilarity(query_fp, training_fps)
    top5 = sorted(sims, reverse=True)[:5]
    mean_score = float(np.mean(top5))
    return mean_score, bool(mean_score >= threshold)


# ============================================================================
# PREDICTION
# ============================================================================

def generate_predictions(df_bonds, model, encoder, training_fps, metadata, verbose=False):
    """
    Generate BDE predictions with conformal intervals and AD checks.

    Returns
    -------
    df_bonds : pd.DataFrame
        Input dataframe with added prediction and confidence columns
    """
    conformal_q90  = metadata['confidence']['conformal_quantile_90']
    domain_threshold = metadata['confidence']['domain_threshold']

    if verbose:
        print(f"Generating predictions for {len(df_bonds)} bonds...")
        print(f"  90% interval width: ±{conformal_q90:.3f} kcal/mol")
        print(f"  Domain threshold:   {domain_threshold:.3f}")

    predictions, lower_bounds, upper_bounds = [], [], []
    in_domain_list, ad_scores = [], []
    skipped = failed = out_of_domain = 0

    for idx, row in df_bonds.iterrows():
        try:
            if row['Frag2_SMILES'] == '':
                predictions.append(np.nan)
                lower_bounds.append(np.nan)
                upper_bounds.append(np.nan)
                in_domain_list.append(np.nan)
                ad_scores.append(np.nan)
                skipped += 1
                continue

            pred = predict_bde(
                row['Parent_SMILES'], row['Frag1_SMILES'], row['Frag2_SMILES'],
                row['solvent'], model, encoder
            )
            predictions.append(pred)
            lower_bounds.append(pred - conformal_q90)
            upper_bounds.append(pred + conformal_q90)

            ad_score, in_dom = check_applicability_domain(
                row['Parent_SMILES'], row['Frag1_SMILES'], row['Frag2_SMILES'],
                row['solvent'], training_fps, domain_threshold
            )
            in_domain_list.append(in_dom)
            ad_scores.append(ad_score)
            if not in_dom:
                out_of_domain += 1

        except Exception as e:
            predictions.append(np.nan)
            lower_bounds.append(np.nan)
            upper_bounds.append(np.nan)
            in_domain_list.append(np.nan)
            ad_scores.append(np.nan)
            failed += 1
            if verbose:
                print(f"  Warning: bond {idx} failed: {e}")

    df_bonds['Predicted_BDE'] = predictions
    df_bonds['BDE_Lower_90']  = lower_bounds
    df_bonds['BDE_Upper_90']  = upper_bounds
    df_bonds['In_Domain']     = in_domain_list
    df_bonds['AD_Score']      = ad_scores

    if verbose:
        successful = df_bonds['Predicted_BDE'].notna().sum()
        in_dom_count = sum(1 for v in in_domain_list if v is True)
        print(f"   Successful:      {successful}/{len(df_bonds)}")
        if skipped:      print(f"  Skipped:         {skipped}")
        if failed:       print(f"  Failed:          {failed}")
        print(f"  In domain:         {in_dom_count}/{successful}")
        if out_of_domain:
            print(f"   Out of domain:  {out_of_domain} — interpret with caution")

    return df_bonds


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_molecules(df, output_file=None, max_display=20):
    """Create grid visualization of input molecules."""
    mols = [Chem.MolFromSmiles(s) for s in df['SMILES_Canonical'].head(max_display)]
    mols = [m for m in mols if m]
    if not mols:
        print("No valid molecules to visualize")
        return

    legends = df['name'].head(len(mols)).tolist() if 'name' in df.columns else None
    grid = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(300, 300),
                                legends=legends, useSVG=False)
    if output_file:
        grid.save(output_file)
        print(f" Molecule grid saved to {output_file}")
    else:
        return grid


def visualize_predictions(df_bonds, output_dir=None, mols_per_row=3, verbose=False):
    """Create molecular visualizations with BDE labels on bonds."""
    if verbose:
        print("Creating molecular visualizations...")

    image_maps = {}

    for (parent_smiles, solvent), group_df in df_bonds.groupby(['Parent_SMILES', 'solvent']):
        mol = Chem.MolFromSmiles(parent_smiles)
        if mol is None:
            continue

        mol = Chem.AddHs(mol)
        Chem.rdDepictor.Compute2DCoords(mol)

        for _, row in group_df.iterrows():
            if pd.notna(row['Predicted_BDE']):
                bond_idx = int(row['Bond_ID'])
                label = f"{row['Predicted_BDE']:.1f}"
                if row['In_Domain'] is False:
                    label += " Warning"
                mol.GetBondWithIdx(bond_idx).SetProp('bondNote', label)

        img = Draw.MolToImage(mol, size=(500, 500))

        if solvent not in image_maps:
            image_maps[solvent] = {}
        mol_label = parent_smiles[:40] + '...' if len(parent_smiles) > 40 else parent_smiles
        image_maps[solvent][mol_label] = img

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for solvent, mol_images in image_maps.items():
            images = list(mol_images.values())
            labels = list(mol_images.keys())
            output_file = output_dir / f"molecules_bde_{solvent}.png"
            fig=show_image_grid_save(images, labels)
            if verbose:
                print(f"   Saved {solvent}: {output_file}")
            fig.savefig(str(output_file))
            plt.close(fig)

    return image_maps


# ============================================================================
# DFT VALIDATION
# ============================================================================

def compare_to_dft(df_bonds, dft_data_dir='Data', verbose=False):
    """Compare predictions to DFT ground truth data (optional validation)."""
    dft_data_dir = Path(dft_data_dir)

    try:
        df_dft_neutral = pd.read_csv(dft_data_dir / "Enthalpy_Neutral_BDE_table.csv").dropna()
        df_dft_neutral['DFT_Source'] = 'Neutral'
        df_dft_anion = pd.read_csv(dft_data_dir / "Enthalpy_Anion_BDE_table.csv").dropna()
        df_dft_anion['DFT_Source'] = 'Anion'
    except FileNotFoundError:
        if verbose:
            print(" DFT reference data not found — skipping validation")
        return df_bonds, None

    if 'Parent_Canon' not in df_bonds.columns:
        df_bonds['Parent_Canon'] = df_bonds['Parent_SMILES'].apply(canonicalize_smiles)
        df_bonds['Frag1_Canon']  = df_bonds['Frag1_SMILES'].apply(canonicalize_smiles)
        df_bonds['Frag2_Canon']  = df_bonds['Frag2_SMILES'].apply(canonicalize_smiles)

    df_bonds['Frag_Pair'] = df_bonds.apply(
        lambda x: make_sorted_pair(x['Frag1_Canon'], x['Frag2_Canon']), axis=1
    )
    for df_dft in [df_dft_neutral, df_dft_anion]:
        df_dft['Parent_Canon'] = df_dft['Parent_SMILES'].apply(canonicalize_smiles)
        df_dft['Frag1_Canon']  = df_dft['Frag1_SMILES'].apply(canonicalize_smiles)
        df_dft['Frag2_Canon']  = df_dft['Frag2_SMILES'].apply(canonicalize_smiles)
        df_dft['Frag_Pair']    = df_dft.apply(
            lambda x: make_sorted_pair(x['Frag1_Canon'], x['Frag2_Canon']), axis=1
        )

    df_dft_combined = pd.concat([df_dft_neutral, df_dft_anion], ignore_index=True)
   

    df_comparison = df_bonds.merge(
        df_dft_combined[['Parent_Canon', 'Frag_Pair', 'BDE_wB97X-V', 'DFT_Source']],
        on=['Parent_Canon', 'Frag_Pair'], how='left'
    )

    df_comparison['Error']          = df_comparison['BDE_wB97X-V'] - df_comparison['Predicted_BDE']
    df_comparison['Absolute_Error'] = df_comparison['Error'].abs()

    matched_df = df_comparison[df_comparison['BDE_wB97X-V'].notna()].copy()

    if matched_df.empty:
        if verbose:
            print(" No matches found with DFT reference data")
        return df_comparison, None

    metrics = {
        'n_total':    len(matched_df),
        'n_neutral':  (matched_df['DFT_Source'] == 'Neutral').sum(),
        'n_anion':    (matched_df['DFT_Source'] == 'Anion').sum(),
        'mae':        matched_df['Absolute_Error'].mean(),
        'rmse':       np.sqrt((matched_df['Error']**2).mean()),
        'bias':       matched_df['Error'].mean(),
        'max_error':  matched_df['Absolute_Error'].max(),
    }

    if verbose:
        print(f"\n Found {metrics['n_total']} matches with DFT data")
        print(f"  MAE:  {metrics['mae']:.3f} kcal/mol")
        print(f"  RMSE: {metrics['rmse']:.3f} kcal/mol")
        print(f"  Bias: {metrics['bias']:+.3f} kcal/mol")
        print(f"  Max:  {metrics['max_error']:.3f} kcal/mol")
    with open('results/validate.log', 'w') as f:
        f.write(f"Found {metrics['n_total']} matches with DFT data\n")
        f.write(f"  MAE:  {metrics['mae']:.3f} kcal/mol\n")
        f.write(f"  RMSE: {metrics['rmse']:.3f} kcal/mol\n")
        f.write(f"  Bias: {metrics['bias']:+.3f} kcal/mol\n")
        f.write(f"  Max:  {metrics['max_error']:.3f} kcal/mol\n")

    return df_comparison, metrics


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Predict BDEs for PFAS molecules')
    parser.add_argument('--input',     required=True,  help='Input CSV/Excel file')
    parser.add_argument('--output',    required=True,  help='Output CSV file')
    parser.add_argument('--model-dir', default='ML_Models', help='Directory containing model files')
    parser.add_argument('--verbose',   '-v', action='store_true', help='Verbose output')
    parser.add_argument('--visualize', help='Directory to save molecule visualizations')
    parser.add_argument('--validate',  action='store_true',
                        help='Compare predictions to DFT reference data (if available)')
    args = parser.parse_args()

    # Load model + confidence artifacts
    model, encoder, training_fps, metadata = load_model(args.model_dir)
    if args.verbose:
        print(" Model and confidence artifacts loaded")

    # Load and validate data
    df = load_and_validate_data(args.input, verbose=args.verbose)

    # Optional: visualize input molecules
    if args.visualize:
        visualize_molecules(df, output_file=Path(args.visualize) / 'input_molecules.png')

    # Fragment into bonds
    df_bonds = fragment_and_prepare_bonds(df, verbose=args.verbose)

    # Predict with confidence
    df_bonds = generate_predictions(df_bonds, model, encoder,
                                    training_fps, metadata, verbose=args.verbose)

    # Optional: validate against DFT
    if args.validate:
        df_bonds, metrics = compare_to_dft(df_bonds, verbose=args.verbose)

    # Optional: visualize predictions
    if args.visualize:
        visualize_predictions(df_bonds, output_dir=args.visualize, verbose=args.verbose)

    # Save results
    df_bonds.to_csv(args.output, index=False)
    if args.verbose:
        print(f"\n Results saved to {args.output}")


if __name__ == '__main__':
    main()