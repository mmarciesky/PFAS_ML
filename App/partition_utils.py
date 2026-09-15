"""
Partition Coefficient Prediction Utilities

Loading and inference helpers for the logKow / logKaw models trained in
Partition_Coefficient_ML_Models.ipynb.

Mirrors the interface of property_utils (load_* / predict_with_confidence) but the
feature construction differs from the redox models in two ways that matter:

  * Model input  = Morgan fingerprint (2048 bits) + 12 RDKit 2D descriptors.
    There is no solvent or protonation one-hot -- the training table is one row
    per molecule and charged species were excluded before training.
  * AD  vector   = Morgan fingerprint only. Tanimoto is a structural measure and
    continuous descriptors do not belong in a bit vector.

DESCRIPTOR_ORDER below must stay byte-identical to RDKIT_DESCRIPTORS in the
training notebook. Reordering it silently shifts every descriptor column and the
model will return confident nonsense.
"""

import json
import pickle
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Crippen, Descriptors
from rdkit.DataStructs import CreateFromBitString

FP_RADIUS = 2
FP_BITS = 2048

# Order is load-bearing. Must match RDKIT_DESCRIPTORS in the training notebook.
DESCRIPTOR_ORDER = [
    ("crippen_logp",  Crippen.MolLogP),
    ("crippen_mr",    Crippen.MolMR),
    ("tpsa",          Descriptors.TPSA),
    ("mw",            Descriptors.MolWt),
    ("heavy_atoms",   Descriptors.HeavyAtomCount),
    ("rotatable",     Descriptors.NumRotatableBonds),
    ("hba",           Descriptors.NumHAcceptors),
    ("hbd",           Descriptors.NumHDonors),
    ("rings",         Descriptors.RingCount),
    ("fraction_csp3", Descriptors.FractionCSP3),
    ("balaban_j",     Descriptors.BalabanJ),
    ("bertz_ct",      Descriptors.BertzCT),
]

# Same SMARTS set and match order used to label headgroups during training.
HEADGROUP_PATTERNS = [
    ("Carboxylate",  "[CX3](=O)[OX2H1,OX1-]"),
    ("Sulphonate",   "[SX4](=O)(=O)[OX2H1,OX1-]"),
    ("Sulphonamide", "[SX4](=O)(=O)[NX3]"),
    ("Phosphate",    "[PX4](=O)([OX2H1,OX1-])"),
    ("Phosphonate",  "[PX4](=O)([CX4])"),
    ("Alkoxide",     "[OX1-][CX4]"),
    ("Alcohol",      "[OX2H1][CX4]"),
]
_COMPILED_HEADGROUPS = [(n, Chem.MolFromSmarts(p)) for n, p in HEADGROUP_PATTERNS]

OTHER_LABEL = "Other"


# ============================================================================
# LOADING
# ============================================================================

def load_partition_model(prefix, model_dir="ML_Models"):
    """
    Load one partition model and its confidence artifacts.

    Parameters
    ----------
    prefix : 'logKow' or 'logKaw'
    model_dir : directory holding the four saved files

    Returns
    -------
    model, encoders, training_fps, metadata
    """
    model_dir = Path(model_dir)
    try:
        with open(model_dir / f"{prefix}_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open(model_dir / f"{prefix}_encoders.pkl", "rb") as f:
            encoders = pickle.load(f)
        with open(model_dir / f"{prefix}_training_fps.pkl", "rb") as f:
            training_fps = pickle.load(f)
        with open(model_dir / f"{prefix}_metadata.json", "r") as f:
            metadata = json.load(f)
        return model, encoders, training_fps, metadata
    except FileNotFoundError:
        print(f"ERROR: {prefix} model files not found in '{model_dir}/'")
        print("Required files:")
        for suffix in ["_model.pkl", "_encoders.pkl", "_training_fps.pkl", "_metadata.json"]:
            print(f"  - {prefix}{suffix}")
        raise


# ============================================================================
# FEATURES
# ============================================================================

def smiles_to_fp_array(smiles, radius=FP_RADIUS, n_bits=FP_BITS):
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return np.zeros(n_bits)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits))


def compute_descriptors(smiles):
    """The 12 RDKit descriptors, in training order. Returns (array, dict)."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    values, named = [], {}
    for name, fn in DESCRIPTOR_ORDER:
        try:
            v = float(fn(mol))
        except Exception:
            v = 0.0
        values.append(v)
        named[name] = v
    return np.array(values), named


def build_feature_vector(smiles):
    """Model input: fingerprint block then descriptor block."""
    fp = smiles_to_fp_array(smiles)
    desc, _ = compute_descriptors(smiles)
    return np.hstack([fp, desc]).reshape(1, -1)


def build_query_bitvector(smiles, radius=FP_RADIUS, n_bits=FP_BITS):
    """AD vector: fingerprint only, matching how training_fps was built."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        bits = [0] * n_bits
    else:
        bits = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits).ToList()
    return CreateFromBitString("".join(str(b) for b in bits))


# ============================================================================
# CHEMISTRY CHECKS
# ============================================================================

def classify_headgroup(smiles):
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None
    for name, patt in _COMPILED_HEADGROUPS:
        if mol.HasSubstructMatch(patt):
            return name
    return OTHER_LABEL


def is_charged(smiles):
    """Charged species were excluded from training, so flag them explicitly."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None
    return Chem.GetFormalCharge(mol) != 0


# ============================================================================
# APPLICABILITY DOMAIN
# ============================================================================

def check_applicability_domain(smiles, training_fps, threshold, top_k=5):
    """Mean Tanimoto to the k nearest training fingerprints."""
    query_fp = build_query_bitvector(smiles)
    sims = DataStructs.BulkTanimotoSimilarity(query_fp, training_fps)
    top = sorted(sims, reverse=True)[:top_k]
    score = float(np.mean(top))
    return score, bool(score >= threshold)


# ============================================================================
# PREDICTION
# ============================================================================

def predict_with_confidence(smiles, model, encoders, training_fps, metadata):
    """
    Predict one partition coefficient with a 90% interval and a domain flag.

    Returns a dict with: prediction, lower_90, upper_90, conformal_90,
    ad_score, in_domain, domain_threshold, headgroup, charged, descriptors.
    """
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")

    X = build_feature_vector(smiles)
    prediction = float(model.predict(X)[0])

    confidence = metadata.get("confidence", {})
    conformal_90 = confidence.get("conformal_quantile_90")
    domain_threshold = confidence.get("domain_threshold")

    if conformal_90 is None:
        raise KeyError("metadata is missing confidence.conformal_quantile_90")
    if domain_threshold is None:
        raise KeyError("metadata is missing confidence.domain_threshold")

    ad_score, in_domain = check_applicability_domain(smiles, training_fps, domain_threshold)
    _, descriptors = compute_descriptors(smiles)

    return {
        "smiles": smiles,
        "prediction": prediction,
        "conformal_90": float(conformal_90),
        "lower_90": prediction - float(conformal_90),
        "upper_90": prediction + float(conformal_90),
        "ad_score": ad_score,
        "in_domain": in_domain,
        "domain_threshold": float(domain_threshold),
        "headgroup": classify_headgroup(smiles),
        "charged": is_charged(smiles),
        "descriptors": descriptors,
    }
