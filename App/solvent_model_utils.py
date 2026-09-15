"""
Multi-solvent property prediction utilities (dipole moment, HOMO-LUMO gap).

Both models take a Morgan fingerprint plus one-hot solvent and protonation blocks.
HOMO-LUMO additionally takes five scalar descriptors. The feature layout is
declared per model in MODEL_SPECS below and must match the training notebook
exactly -- block order is load-bearing.

Two things differ from the redox and partition models:

  * Predictions are solvent-specific. The same molecule has a different dipole
    and a different gap in each solvent, so the caller must supply one.
  * Protonation state is derived from the SMILES formal charge, NOT from the
    main_table 'protonation_state' column. That column disagrees with the actual
    structure on ~32% of rows (it records the charge the QM job ran at), so it
    cannot be trusted at inference time. See PROTONATION NOTE below.
"""

import json
import pickle
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from rdkit.DataStructs import CreateFromBitString

FP_RADIUS = 2
FP_BITS = 2048

# Scalar descriptors for the HOMO-LUMO model, in training order.
# These must reproduce the main_table columns of the same names.
SCALAR_ORDER = ["mw", "fluorination_ratio", "n_F", "n_CF3", "n_CF2"]

MODEL_SPECS = {
    "dipole_moment": {
        "target": "dipole_moment_debye",
        "label": "Dipole moment",
        "unit": "Debye",
        "scalars": False,
    },
    "homo_lumo_gap": {
        "target": "homo_lumo_gap_eV",
        "label": "HOMO-LUMO gap",
        "unit": "eV",
        "scalars": True,
    },
}


# ============================================================================
# LOADING
# ============================================================================

def load_solvent_model(prefix, model_dir="ML_Models"):
    """
    Load a multi-solvent model and its confidence artifacts.

    Returns (model, encoders, training_fps, metadata). training_fps is None when
    the model was saved without applicability-domain artifacts -- callers should
    handle that rather than assume AD is available.
    """
    model_dir = Path(model_dir)
    with open(model_dir / f"{prefix}_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(model_dir / f"{prefix}_metadata.json", "r") as f:
        metadata = json.load(f)

    enc_path = model_dir / f"{prefix}_encoders.pkl"
    if enc_path.exists():
        with open(enc_path, "rb") as f:
            encoders = pickle.load(f)
    else:
        # Fall back to the separately-pickled encoders the dipole cell writes.
        encoders = {}
        for key, fname in [("solvent", f"{prefix}_solvent_encoder.pkl"),
                           ("protonation", f"{prefix}_protonation_encoder.pkl")]:
            p = model_dir / fname
            if p.exists():
                with open(p, "rb") as f:
                    encoders[key] = pickle.load(f)
    if "solvent" not in encoders or "protonation" not in encoders:
        raise FileNotFoundError(
            f"{prefix}: could not find both solvent and protonation encoders in {model_dir}"
        )

    fps_path = model_dir / f"{prefix}_training_fps.pkl"
    if fps_path.exists():
        with open(fps_path, "rb") as f:
            training_fps = pickle.load(f)
    else:
        training_fps = None

    return model, encoders, training_fps, metadata


def solvent_options(encoders):
    """Solvents the model was actually trained on, straight off the encoder."""
    return list(encoders["solvent"].categories_[0])


def get_confidence(metadata):
    """Returns (conformal_90, domain_threshold). Either may be None."""
    conf = metadata.get("confidence")
    if conf is not None:
        return conf.get("conformal_quantile_90"), conf.get("domain_threshold")
    # Flat schema written by train_final_model_with_conformal
    intervals = metadata.get("conformal_intervals", {})
    return intervals.get("q90"), None


# ============================================================================
# PROTONATION NOTE
# ============================================================================
# The models were trained with a protonation one-hot built from main_table's
# 'protonation_state', which is wrong on ~32% of rows. At inference the only
# honest option is to derive it from the structure, which is what this does.
# Where the training label was wrong, the model learned noise on that feature --
# the fingerprint carries the real signal. Retraining on a corrected column is
# the actual fix.

def protonation_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None
    charge = Chem.GetFormalCharge(mol)
    if charge < 0:
        return "anionic"
    if charge > 0:
        return "cationic"
    return "neutral"


def formal_charge(smiles):
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return None if mol is None else Chem.GetFormalCharge(mol)


# ============================================================================
# FEATURES
# ============================================================================

_CF3_SMARTS = Chem.MolFromSmarts("[CX4](F)(F)F")
_CF2_SMARTS = Chem.MolFromSmarts("[CX4](F)(F)")


def compute_scalars(smiles):
    """mw, fluorination_ratio, n_F, n_CF3, n_CF2 -- matching main_table."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")

    n_f = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "F")
    n_h_on_c = sum(a.GetTotalNumHs(includeNeighbors=True)
                   for a in mol.GetAtoms() if a.GetSymbol() == "C")
    denom = n_f + n_h_on_c

    n_cf3 = n_cf2 = 0
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "C":
            continue
        f_count = sum(1 for nb in atom.GetNeighbors() if nb.GetSymbol() == "F")
        if f_count == 3:
            n_cf3 += 1
        elif f_count == 2:
            n_cf2 += 1

    return {
        "mw": float(Descriptors.MolWt(mol)),
        "fluorination_ratio": float(n_f / denom) if denom else 0.0,
        "n_F": float(n_f),
        "n_CF3": float(n_cf3),
        "n_CF2": float(n_cf2),
    }


def smiles_to_fp_array(smiles, radius=FP_RADIUS, n_bits=FP_BITS):
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return np.zeros(n_bits)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits))


def infer_use_scalars(model, encoders):
    """Work out from the model's own input width whether it expects the scalar
    block. Returns None if the model does not expose n_features_in_."""
    n_flags = (len(encoders["solvent"].categories_[0])
               + len(encoders["protonation"].categories_[0]))
    base = FP_BITS + n_flags
    n_in = getattr(model, "n_features_in_", None)
    if n_in is None:
        return None
    if n_in == base:
        return False
    if n_in == base + len(SCALAR_ORDER):
        return True
    raise ValueError(
        f"Model expects {n_in} features; this layout implies {base} (no scalars) "
        f"or {base + len(SCALAR_ORDER)} (with scalars). Encoders may not match the model."
    )


def build_feature_vector(smiles, solvent, encoders, use_scalars):
    """Block order: fingerprint, solvent one-hot, protonation one-hot, [scalars]."""
    protonation = protonation_from_smiles(smiles)
    if protonation is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")

    fp = smiles_to_fp_array(smiles)
    solv = encoders["solvent"].transform([[solvent]]).flatten()
    proto = encoders["protonation"].transform([[protonation]]).flatten()

    blocks = [fp, solv, proto]
    if use_scalars:
        s = compute_scalars(smiles)
        blocks.append(np.array([s[k] for k in SCALAR_ORDER]))
    return np.hstack(blocks).reshape(1, -1), protonation


def build_query_bitvector(smiles, solvent, encoders, radius=FP_RADIUS, n_bits=FP_BITS):
    """
    AD vector: fingerprint + solvent flags + protonation flags, in that order.
    Scalars are deliberately excluded -- Tanimoto is a structural measure and
    continuous descriptors do not belong in a bit vector.
    """
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        bits = [0] * n_bits
    else:
        bits = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits).ToList()

    protonation = protonation_from_smiles(smiles) or "neutral"
    flags = np.hstack([
        encoders["solvent"].transform([[solvent]]).flatten(),
        encoders["protonation"].transform([[protonation]]).flatten(),
    ])
    bits = bits + [int(round(b)) for b in flags]
    return CreateFromBitString("".join(str(b) for b in bits))


# ============================================================================
# PREDICTION
# ============================================================================

def check_applicability_domain(smiles, solvent, encoders, training_fps, threshold, top_k=5):
    query_fp = build_query_bitvector(smiles, solvent, encoders)
    sims = DataStructs.BulkTanimotoSimilarity(query_fp, training_fps)
    score = float(np.mean(sorted(sims, reverse=True)[:top_k]))
    return score, bool(score >= threshold)


def predict_with_confidence(smiles, solvent, model, encoders, training_fps, metadata,
                            use_scalars):
    """
    Predict one solvent-specific property with a 90% interval and, when the
    artifacts exist, an applicability-domain flag.

    ad_score / in_domain are None when the model was saved without training_fps.
    """
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")

    X, protonation = build_feature_vector(smiles, solvent, encoders, use_scalars)
    prediction = float(model.predict(X)[0])

    conformal_90, domain_threshold = get_confidence(metadata)
    if conformal_90 is None:
        raise KeyError(f"{metadata.get('target')}: metadata has no 90% conformal interval")

    ad_score = in_domain = None
    if training_fps is not None and domain_threshold is not None:
        ad_score, in_domain = check_applicability_domain(
            smiles, solvent, encoders, training_fps, domain_threshold)

    known_solvents = solvent_options(encoders)
    return {
        "smiles": smiles,
        "solvent": solvent,
        "solvent_in_training": solvent in known_solvents,
        "protonation_state": protonation,
        "formal_charge": formal_charge(smiles),
        "prediction": prediction,
        "conformal_90": float(conformal_90),
        "lower_90": prediction - float(conformal_90),
        "upper_90": prediction + float(conformal_90),
        "ad_score": ad_score,
        "in_domain": in_domain,
        "domain_threshold": domain_threshold,
    }
