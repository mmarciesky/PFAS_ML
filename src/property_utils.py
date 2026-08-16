"""
Property Prediction Utility Functions

Generalized version of bde_utils.py's prediction + applicability-domain
pattern, adapted for whole-molecule property models (oxidation potential,
reduction potential -- dipole/HOMO-LUMO once their data issues are resolved).
"""

import pickle
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.DataStructs import CreateFromBitString


def smiles_to_fp(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits))


def canonicalize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol) if mol is not None else None
    except Exception:
        return None


def get_protonation_state(smiles):
    """Derived from RDKit formal charge -- same logic used when main_table.csv was built."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 'invalid'
    charge = Chem.GetFormalCharge(mol)
    if charge < 0:
        return 'anionic'
    if charge > 0:
        return 'cationic'
    return 'neutral'


def build_query_bitvector(smiles, extra_flags, radius=2, n_bits=2048):
    """Fingerprint bits + categorical flag bits appended -- must match training order exactly."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        bits = [0] * n_bits
    else:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        bits = fp.ToList()
    bits = bits + [int(round(b)) for b in extra_flags]
    return CreateFromBitString("".join(str(b) for b in bits))


def check_applicability_domain(smiles, extra_flags, training_fps, threshold, radius=2, n_bits=2048):
    query_fp = build_query_bitvector(smiles, extra_flags, radius, n_bits)
    sims = DataStructs.BulkTanimotoSimilarity(query_fp, training_fps)
    top5 = sorted(sims, reverse=True)[:5]
    mean_score = float(np.mean(top5))
    return mean_score, bool(mean_score >= threshold)


def load_property_model(prefix, model_dir='.'):
    """
    Loads {prefix}_model.pkl, {prefix}_encoders.pkl, {prefix}_training_fps.pkl,
    {prefix}_metadata.json -- the same four-file layout used for the BDE model,
    just with encoders stored as a dict (e.g. {'protonation': encoder}) instead
    of a single solvent encoder, since these targets have different flags.
    """
    model_dir = Path(model_dir)
    try:
        with open(model_dir / f"{prefix}_model.pkl", 'rb') as f:
            model = pickle.load(f)
        with open(model_dir / f"{prefix}_encoders.pkl", 'rb') as f:
            encoders = pickle.load(f)
        with open(model_dir / f"{prefix}_training_fps.pkl", 'rb') as f:
            training_fps = pickle.load(f)
        with open(model_dir / f"{prefix}_metadata.json") as f:
            metadata = json.load(f)
        return model, encoders, training_fps, metadata
    except FileNotFoundError as e:
        print(f"ERROR: model files not found for prefix '{prefix}' in '{model_dir}/'")
        raise


def predict_with_confidence(smiles, model, encoders, training_fps, metadata, radius=2, n_bits=2048):
    """
    Full prediction + conformal interval + applicability domain check for one SMILES.
    Protonation state is derived automatically from the input structure -- not
    something the user needs to specify.
    """
    canon = canonicalize_smiles(smiles)
    if canon is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")

    proto = get_protonation_state(canon)
    proto_encoded = encoders['protonation'].transform([[proto]]).flatten()

    fp = smiles_to_fp(canon, radius, n_bits)
    features = np.hstack([fp, proto_encoded]).reshape(1, -1)
    pred = float(model.predict(features)[0])

    q90 = metadata['confidence']['conformal_quantile_90']
    threshold = metadata['confidence']['domain_threshold']
    ad_score, in_domain = check_applicability_domain(canon, proto_encoded, training_fps, threshold, radius, n_bits)

    return {
        'prediction': pred,
        'lower_90': pred - q90,
        'upper_90': pred + q90,
        'protonation_state': proto,
        'canonical_smiles': canon,
        'ad_score': ad_score,
        'in_domain': in_domain,
    }