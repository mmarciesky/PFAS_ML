"""
BDE Prediction Utility Functions

Helper functions for PFAS Bond Dissociation Energy predictions using
pre-trained XGBoost model.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import math
import matplotlib.pyplot as plt

# ============================================================================
# MOLECULAR FINGERPRINTING
# ============================================================================

def smiles_to_fp(smiles, radius=2, nBits=2048):
    """
    Convert SMILES string to Morgan fingerprint.
    
    Args:
        smiles (str): SMILES representation of molecule
        radius (int): Morgan fingerprint radius (default: 2, equivalent to ECFP4)
        nBits (int): Length of bit vector (default: 2048)
    
    Returns:
        np.array: Binary fingerprint vector
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nBits)
    return np.array(
        AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    )


# ============================================================================
# BDE PREDICTION
# ============================================================================

def predict_bde(parent_smiles, frag1_smiles, frag2_smiles, solvent, model, solvent_encoder):
    """
    Predict BDE for a single bond.
    
    Args:
        parent_smiles (str): SMILES of parent molecule
        frag1_smiles (str): SMILES of fragment 1 (radical)
        frag2_smiles (str): SMILES of fragment 2 (radical)
        solvent (str): 'water' or 'gas'
        model: Pre-trained XGBoost model
        solvent_encoder: Pre-trained OneHotEncoder for solvent
    
    Returns:
        float: Predicted BDE in kcal/mol
    """
    # Generate fingerprints
    parent_fp = smiles_to_fp(parent_smiles)
    frag1_fp = smiles_to_fp(frag1_smiles)
    frag2_fp = smiles_to_fp(frag2_smiles)
    
    # Encode solvent
    solvent_encoded = solvent_encoder.transform([[solvent]]).flatten()
    
    # Combine features
    features = np.hstack([parent_fp, frag1_fp, frag2_fp, solvent_encoded])
    
    # Predict
    bde_pred = model.predict(features.reshape(1, -1))[0]
    
    return bde_pred


# ============================================================================
# SMILES UTILITIES
# ============================================================================

def canonicalize_smiles(smiles):
    """
    Convert SMILES to canonical form.
    
    Args:
        smiles (str): SMILES string
    
    Returns:
        str or None: Canonical SMILES, or None if parsing fails
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol)
        else:
            return None
    except:
        return None


def make_sorted_pair(frag1, frag2):
    """
    Return fragments in sorted order as tuple.
    Useful for matching bonds regardless of fragment order.
    
    Args:
        frag1 (str): First fragment SMILES
        frag2 (str): Second fragment SMILES
    
    Returns:
        tuple: Sorted tuple of (frag1, frag2)
    """
    return tuple(sorted([frag1, frag2]))


# ============================================================================
# FRAGMENT PROCESSING
# ============================================================================

def remove_dummy_atoms_and_add_radicals(mol):
    """
    Remove dummy atoms (*) and add radicals at their previous bonding positions.
    
    Args:
        mol: RDKit molecule object
    
    Returns:
        RDKit molecule with dummy atoms removed and radicals added
    """
    mol = Chem.RWMol(mol)  # Convert to editable molecule

    dummy_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    
    for dummy_idx in sorted(dummy_atoms, reverse=True):  
        bonds = mol.GetAtomWithIdx(dummy_idx).GetBonds()
        
        if bonds:  # Ensure the dummy atom was bonded
            neighbor = bonds[0].GetOtherAtomIdx(dummy_idx)
            atom = mol.GetAtomWithIdx(neighbor)
            
            # Set radical on the bonded atom
            atom.SetNumRadicalElectrons(atom.GetNumRadicalElectrons() + 1)

        mol.RemoveAtom(dummy_idx)

    return mol


def dedupe_bonds(data_dict):
    """
    Remove duplicate bonds from Parent_PFAS dictionary.
    Keeps first occurrence of each unique fragment pair.
    
    Args:
        data_dict (dict): Parent_PFAS dictionary
    
    Returns:
        dict: Cleaned dictionary with duplicates removed
    """
    cleaned = {}
    for parent_smiles, bonds in data_dict.items():
        seen = set()
        new_bonds = {}
        
        for bond_idx, info in bonds.items():
            pair = info.get('SMILES', [])
            key = frozenset(pair)
            if key not in seen:
                seen.add(key)
                new_bonds[bond_idx] = info
        
        cleaned[parent_smiles] = new_bonds
    
    return cleaned
def fragment_molecules(smiles_list):
    """
    Fragment all molecules into bonds by breaking single bonds.
    
    Args:
        smiles_list (list): List of canonical SMILES strings
    
    Returns:
        dict: Parent_PFAS dictionary with structure:
              {parent_smiles: {bond_idx: {'molecules': [...], 'SMILES': [...]}}}
    """
    from rdkit import Chem
    
    Parent_PFAS = {}
    
    for parent_smiles in smiles_list:
        parent_mol = Chem.MolFromSmiles(parent_smiles)
        if parent_mol is None:
            continue
        
        parent_mol = Chem.AddHs(parent_mol)
        
        single_bond_indices = [
            bond.GetIdx() for bond in parent_mol.GetBonds() 
            if bond.GetBondType() == Chem.BondType.SINGLE
        ]
        
        Chem.rdDepictor.Compute2DCoords(parent_mol)
        
        Parent_PFAS[parent_smiles] = {}
        
        for bond_idx in single_bond_indices:
            fragmented_mol = Chem.FragmentOnBonds(parent_mol, [bond_idx], addDummies=True)
            fragments = Chem.GetMolFrags(fragmented_mol, asMols=True)
            
            final_fragments = []
            fragment_smiles_list = []
            
            for frag in fragments:
                clean_mol = remove_dummy_atoms_and_add_radicals(frag)
                
                for atom in clean_mol.GetAtoms():
                    if atom.HasProp("molAtomMapNumber"):
                        atom.ClearProp("molAtomMapNumber")
                
                smiles_mol = Chem.MolToSmiles(clean_mol, canonical=True)
                final_fragments.append(clean_mol)
                fragment_smiles_list.append(smiles_mol)
            
            Parent_PFAS[parent_smiles][bond_idx] = {
                "molecules": final_fragments,
                "SMILES": fragment_smiles_list
            }
    
    # Remove duplicates
    Parent_PFAS = dedupe_bonds(Parent_PFAS)
    
    return Parent_PFAS
def convert_parent_pfas_to_dataframe(Parent_PFAS, df_original):
    """
    Convert Parent_PFAS dictionary to dataframe format with bond predictions.
    Handles multiple solvents per molecule and canonicalization.
    
    Args:
        Parent_PFAS (dict): Dictionary of fragmented molecules
        df_original (pd.DataFrame): Original dataframe with SMILES and solvent columns
    
    Returns:
        pd.DataFrame: Dataframe with columns: Parent_SMILES, Frag1_SMILES, 
                      Frag2_SMILES, solvent, Bond_ID
    """
    import pandas as pd
    
    # Canonicalize original dataframe SMILES
    if 'SMILES' in df_original.columns:
        df_original['Parent_SMILES_Canonical'] = df_original['SMILES'].apply(canonicalize_smiles)
    
    # Check if solvent column exists
    has_solvent_column = 'solvent' in df_original.columns
    
    rows = []
    
    for parent_smiles, bonds_dict in Parent_PFAS.items():
        # Get all solvents for this parent molecule
        if has_solvent_column and 'Parent_SMILES_Canonical' in df_original.columns:
            solvent_list = df_original[df_original['Parent_SMILES_Canonical'] == parent_smiles]['solvent'].unique()
            if len(solvent_list) == 0:
                solvent_list = ['gas']
        else:
            solvent_list = ['gas']
        
        # Create bonds for each solvent
        for solvent in solvent_list:
            for bond_id, bond_data in bonds_dict.items():
                if 'SMILES' not in bond_data or len(bond_data['SMILES']) == 0:
                    continue
                
                frag1_smiles = bond_data['SMILES'][0]
                frag2_smiles = bond_data['SMILES'][1] if len(bond_data['SMILES']) > 1 else ''
                
                rows.append({
                    'Parent_SMILES': parent_smiles,
                    'Frag1_SMILES': frag1_smiles,
                    'Frag2_SMILES': frag2_smiles,
                    'solvent': solvent,
                    'Bond_ID': bond_id
                })
    
    return pd.DataFrame(rows)
def show_image_grid(images, labels, mols_per_row=3, sub_img_size=(6, 6)):
    """
    images: list of PIL images
    labels: list of strings (same length as images)
    """
    n = len(images)
    if n == 0:
        print("No images to show.")
        return

    cols = mols_per_row
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(cols * sub_img_size[0], rows * sub_img_size[1])
    )

    # axes can be a single Axes if rows*cols == 1
    ax_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, (img, label) in enumerate(zip(images, labels)):
        ax = ax_list[idx]
        ax.imshow(img)
        ax.axis("off")
        # put SMILES (or whatever label) as title
        ax.set_title(label, fontsize=8)

    # Turn off any unused axes
    for idx in range(n, rows * cols):
        ax_list[idx].axis("off")

    plt.tight_layout()
    plt.show()
def show_image_grid_save(images, labels, mols_per_row=3, sub_img_size=(6, 6)):
    """
    images: list of PIL images
    labels: list of strings (same length as images)
    """
    n = len(images)
    if n == 0:
        print("No images to show.")
        return
    cols = int(mols_per_row)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(cols * sub_img_size[0], rows * sub_img_size[1])
    )

    # axes can be a single Axes if rows*cols == 1
    ax_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, (img, label) in enumerate(zip(images, labels)):
        ax = ax_list[idx]
        ax.imshow(img)
        ax.axis("off")
        # put SMILES (or whatever label) as title
        ax.set_title(label, fontsize=8)

    # Turn off any unused axes
    for idx in range(n, rows * cols):
        ax_list[idx].axis("off")
    return fig