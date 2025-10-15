# README for ChemMLP Model Codes

This directory contains the Python source code for the machine learning and Graph Neural Network (GNN) models used in ChemMLP, a GUI tool designed to predict molecular properties (e.g., Collision Cross-Section (CCS) values). These files support the manuscript submitted to JASMS (Ebrahimi, S., et al., 2025).

## Available Files
- `Lasso.py`: Implementation of Lasso regression.
- `SVR.py`: Implementation of Support Vector Regression.
- `ml_GUI_de.py`: Implementation of Support Vector Regression (note: duplicate functionality with `SVR.py`).
- `LR.py`: Implementation of Linear Regression.
- `PLS.py`: Implementation of Partial Least Squares (PLS) regression.
- `GBR.py`: Implementation of Gradient Boosting Regression.
- `RF.py`: Implementation of Random Forest (RF) regression.
- `KNR.py`: Implementation of K-Nearest Neighbors (KNN) regression.
- `VR.py`: Implementation of Variational Regression.
- `ANNew.py`: Implementation of Artificial Neural Networks for SMILES-based data.
- `GNN_chem.py`: Implementation of Graph Neural Networks for SMILES-based data.

## Usage
These scripts represent the core model implementations but require the complete ChemMLP environment to run. To use them:
1. Install dependencies: `pip install scikit-learn torch torch-geometric`.
2. The scripts assume preprocessed data (e.g., from the 'Data' folder in `ChemMLP.zip`).
3. Import and instantiate models in a custom script (e.g., `from code.Lasso import LassoModel`).

## Limitations
- This directory includes only model-specific code. The full software (including the GUI, data pipeline, etc.) is not open-sourced to protect intellectual property and is available as `ChemMLP.exe` in the Releases section.
- For complete reproducibility, contact the author (sa.ebrahimi@gmail.com) post-publication.

## License
These files are released under the MIT License (see the repository root LICENSE file).
