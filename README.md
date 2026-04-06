# PFAS Bond Dissociation Energy Prediction

Machine learning models for predicting homolytic bond dissociation energies (BDEs) in per- and polyfluoroalkyl substances (PFAS).

# PFAS BDE PREDICTION - PRODUCTION NOTEBOOK

"""
This notebook provides a simple interface for predicting Bond Dissociation 
Energies (BDEs) of PFAS molecules using a pre-trained XGBoost model.

USAGE:
1. Load your molecule data (CSV with SMILES and solvent info)
2. Fragment molecules into bonds (creates Parent_PFAS dictionary)
3. Run predictions to get BDEs for all bonds
4. View visualizations with BDEs labeled on molecular structures


### Performance 
 MAE:  1.442 ± 0.122 kcal/mol \n
 RMSE: 3.248 ± 0.318 kcal/mol \n
 R²:   0.953 ± 0.008
- Training set: ~701 bonds from both gas and water solvent phase PFAS
- DFT level: ωB97X-V/def2-TZVPD


## 🚀 Quick Start

### Installation
##### 1. Clone the repository
git clone https://github.com/your-username/your-repo.git
cd your-repo

##### 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

##### 3. Install dependencies
pip install -r requirements.txt

##### 4. Run a test prediction
python predict.py --input Data/Test_ML.csv --output results/test_output.csv --visualize results/figures --validate --verbose

### Predict BDEs
```python
from src.bde_utils import predict_bde

# Predict for a molecule
bde = predict_bde("C(C(C(F)(F)F)(F)F)(F)F", solvent="water")
print(f"BDE: {bde:.2f} kcal/mol")
```

Or use the notebook: `notebooks/Production_PFAS_BDE_Predictor.ipynb`

## 📊 Model Details

- **Algorithm**: XGBoost (hyperparameter tuned)
- **Features**: Morgan fingerprints (radius=2, 2048 bits) + solvent encoding
- **Training**: 700+ bonds from ωB97X-V/def2-TZVP DFT calculations
- **Cross-validation**: GroupKFold (by parent molecule)

### Model Comparison (Pre-Tuning)

| Model | MAE (kcal/mol) | RMSE (kcal/mol) | R² |
|-------|---------------|-----------------|-----|
| XGBoost | 1.582 ± 0.163 | 3.372 ± 0.236 | 0.950 ± 0.007 |
| Gradient Boosting | 1.609 ± 0.196 | 3.212 ± 0.397 | 0.954 ± 0.011 |
| Random Forest | 1.642 ± 0.249 | 3.455 ± 0.477 | 0.946 ± 0.014 |
| Ridge | 1.855 ± 0.178 | 3.699 ± 0.489 | 0.938 ± 0.019 |
| Neural Network | 4.440 ± 0.558 | 8.618 ± 1.058 | 0.670 ± 0.068 |
| Lasso | 5.867 ± 0.327 | 8.826 ± 0.550 | 0.656 ± 0.038 |
| ElasticNet | 7.328 ± 0.385 | 9.579 ± 0.601 | 0.594 ± 0.048 |
| Baseline (Mean) | 13.569 ± 0.213 | 15.078 ± 0.326 | -0.003 ± 0.003 |

See `notebooks/ML_Model_Training.ipynb` for full analysis.

## 📁 Repository Structure
```
├── data/              # Training data and figures
├── App/               # for webapp
├── ML_Models/         # model data and notebook with training steps/model compairsons
├── src/               # Core utility functions
├── Production_PFAS_BDE_Predictor  # Notebook predictor version
├──  predict.py # python predictor version
```

## 🔬 Reproducibility

**Important**: Results may vary across package versions due to numerical precision differences in underlying libraries (NumPy BLAS, XGBoost internals).

- Current results use package versions in `requirements.txt`

To reproduce exactly:
```bash
pip install -r requirements.txt
```
## 🚀 Usage

### Command Line Interface

The prediction script supports multiple modes and options:

#### Complete Example (all features)
```bash
python scripts/predict.py \
  --input Data/molecules.csv \
  --output results/predictions.csv \
  --visualize results/figures/ \
  --validate \
  --verbose
```
input (path) and output (path) is always necessarry. --visulaize (folder) will give images of predictions, --validate will give DFT references (if applicable), --verbose is for printouts. 

### Input File Format

Your input CSV/Excel file must have a `SMILES` column:
```csv
SMILES,name,solvent
C(C(C(F)(F)F)(F)F)(F)F,PFBA,water
FC(F)(F)C(F)(F)C(F)(F)F,PFHxA,gas
CC(C(=O)O)C(F)(F)F,TFA,water
```

**Required columns:**
- `SMILES`: Molecular structures in SMILES format

**Optional columns:**
- `solvent`: Solvent environment (`gas` or `water`). Defaults to `gas` if not provided.
- `name`: Molecule names for reference

### Output

The script generates:

1. **CSV file** (`--output`): Contains all bonds with predicted BDEs
   - Columns: Parent_SMILES, Frag1_SMILES, Frag2_SMILES, Bond_ID, solvent, Predicted_BDE
   
2. **Visualizations** (`--visualize`, optional): PNG images showing molecules with BDE labels
   - One file per solvent: `molecules_bde_gas.png`, `molecules_bde_water.png`
   
3. **Validation metrics** (`--validate`, optional): Comparison to DFT reference data if available

### Command Line Options
```
usage: predict.py [-h] --input INPUT --output OUTPUT [--verbose] 
                  [--visualize VISUALIZE] [--validate]

Predict BDEs for PFAS molecules

required arguments:
  --input INPUT         Input CSV/Excel file with SMILES column
  --output OUTPUT       Output CSV file for predictions

optional arguments:
  -h, --help            Show this help message and exit
  --verbose, -v         Print detailed progress information
  --visualize VISUALIZE Directory to save molecule visualizations
  --validate            Compare predictions to DFT reference data
```


##  Citation
```
If you use this tool in your research, please cite: \n
App: Marciesky, M. PFAS BDE Predictor (v0.1-preliminary). 2026. [GitHub](https://github.com/mmarciesky/PFAS_ML) link — DOI forthcoming \n
Database: PFAS Quantum Chemistry Database. [GitHub](https://github.com/mmarciesky/PFAS_Database) — DOI forthcoming via Zenodo
```


## Acknowledgments
```
This tool was developed with the support of the Ng Lab and Keith Lab
at University of Pittsburgh. 
```

## 📄 License

MIT License
