# CFVAE Implementation and Ablation Studies

This repository contains the implementation of the CFVAE (Counterfactual Variational Autoencoder) model for medical counterfactuals, along with several ablation studies to analyze the importance of different components.

## Project Overview

This project reproduces and extends the CFVAE model from the paper "CFVAE: Counterfactual Variational Autoencoder for Medical Counterfactuals". The implementation includes:
- Core CFVAE model with VAE and counterfactual generation components
- Multiple ablation studies to analyze model components
- Evaluation metrics and visualization tools

## Final Report
Link to final report: https://docs.google.com/document/d/1i2T0q8kXUgX8uDe7TldSG61dlqKUxbcNKfQ03TvqdMg/edit?usp=sharing 

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the MIMIC-IV Demo dataset from PhysioNet and place it in the `data/` directory.

## Data Processing

1. Run the data processing script:
```bash
python process_data.py
```

This will create processed datasets in the `processed_data/` directory.

## Model Training

### Base Model
To train the base CFVAE model:
```bash
python CFVAE/main_vaeCF.py
```

### Ablation Studies

1. **Feature Normalization Removal**:
```bash
python CFVAE/main_vaeCF_norm.py
```

2. **VAE Component Removal**:
```bash
python CFVAE/main_vaeCF_noVAE.py
```

3. **Hyperparameter Modification**:
```bash
python CFVAE/main_vaeCF_alt.py
```

4. **Sparsity Constraints**:
```bash
python CFVAE/main_vaeCF_sparse.py
```

## Results

The results of each experiment are saved in the `results/` directory. To generate visualizations of the results:

```bash
python generate_charts.py
```

This will create the following visualizations:
- `performance_comparison.png`: Comparison with original paper results
- `ablation_results.png`: Impact of each ablation study
- `training_curves.png`: Training loss curves
- `feature_importance.png`: Feature importance analysis

## Project Structure

```
.
├── CFVAE/
│   ├── main_vaeCF.py           # Base model implementation
│   ├── main_vaeCF_norm.py      # No normalization ablation
│   ├── main_vaeCF_noVAE.py     # No VAE ablation
│   ├── main_vaeCF_alt.py       # Alternative hyperparameters
│   └── main_vaeCF_sparse.py    # With sparsity constraints
├── data/                       # Raw data directory
├── processed_data/             # Processed datasets
├── results/                    # Experiment results
├── models/                     # Saved model checkpoints
├── logs/                       # Training logs
├── process_data.py            # Data processing script
├── generate_charts.py         # Visualization script
├── requirements.txt           # Project dependencies
└── README.md                  # This file
```

## Key Findings

1. **Feature Normalization Removal**:
   - Improved AUC from 0.63 to 0.73
   - Maintained accuracy at 0.85
   - Increased counterfactual proximity values

2. **VAE Component Removal**:
   - Test accuracy dropped to 0.43 (from 0.85)
   - Validation accuracy decreased to 0.76
   - Demonstrated VAE's crucial role in model performance

3. **Hyperparameter Modifications**:
   - Faster training but unstable performance
   - AUC decreased to 0.31
   - Validated original hyperparameter choices

4. **Sparsity Constraints**:
   - Maintained accuracy while improving interpretability
   - AUC decreased to 0.57
   - Demonstrated trade-off between performance and interpretability

## Requirements

- Python 3.9.7
- PyTorch 1.9.0
- NumPy 1.21.2
- Pandas 1.3.3
- scikit-learn 0.24.2
- Matplotlib 3.4.3

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original CFVAE paper link: https://proceedings.mlr.press/v209/nagesh23a/nagesh23a.pdf 
