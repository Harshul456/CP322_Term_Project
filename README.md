# CP322 Regression Project

Machine Learning project for CP322 course focusing on regression tasks on tabular data.

## Project Overview

This project implements a complete machine learning pipeline for two regression datasets:
1. **Energy Efficiency Dataset** - Predicting Heating Load
2. **Energy Appliances Dataset** - Predicting Appliances energy consumption

## Project Structure

```
project/
│
├── data/
│   ├── energy_efficiency_data.csv
│   ├── energy_data_complete.csv
│
├── notebooks/
│   ├── eda_efficiency.ipynb
│   └── eda_appliances.ipynb
│
├── src/
│   ├── data_loader.py          # CSV data loading
│   ├── preprocessing.py        # Data preprocessing pipeline
│   ├── baselines.py            # Linear Regression & Random Forest
│   ├── xgboost_model.py        # XGBoost implementation
│   ├── tabnet_model.py         # TabNet neural network
│   ├── tuning.py               # Optuna hyperparameter tuning
│   ├── shap_analysis.py        # SHAP feature importance
│   └── utils.py                # Utility functions
│
├── results/
│   ├── models/                 # Saved model artifacts
│   ├── metrics/                # Evaluation metrics
│   ├── shap_plots/             # SHAP visualization plots
│   ├── tuning/                 # Hyperparameter tuning results
│   └── logs/                   # Training logs
│
├── main.py                     # Main pipeline script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

1. Clone or download this repository

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the complete pipeline for both datasets:

```bash
python main.py
```

This will:
- Load and preprocess both datasets
- Train baseline models (Linear Regression, Random Forest)
- Train XGBoost model
- Train TabNet neural network
- Perform hyperparameter tuning with Optuna
- Generate SHAP plots for feature importance
- Save all models, metrics, and visualizations

## Pipeline Components

### 1. Data Preprocessing
- Handles missing values
- Identifies and encodes categorical features
- Scales numeric features
- Splits data into train/test sets (80/20)

### 2. Baseline Models
- **Linear Regression**: Simple linear baseline
- **Random Forest**: Ensemble tree-based baseline

### 3. Advanced Models
- **XGBoost**: Gradient boosting with default hyperparameters
- **TabNet**: Neural tabular model using PyTorch

### 4. Hyperparameter Tuning
- Uses Optuna for Bayesian optimization
- Tunes XGBoost hyperparameters:
  - n_estimators
  - max_depth
  - learning_rate
  - subsample
  - colsample_bytree

### 5. SHAP Analysis
- Generates SHAP summary plots
- Creates feature importance visualizations
- Compares feature importance across models

## Results

All results are saved in the `results/` directory:
- **models/**: Trained model artifacts (.joblib files)
- **metrics/**: JSON files with evaluation metrics (RMSE, MAE, R2)
- **shap_plots/**: PNG files with SHAP visualizations
- **tuning/**: Optuna tuning results and best parameters

## Evaluation Metrics

The pipeline evaluates all models using:
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R²** (Coefficient of Determination)

## Datasets

### Energy Efficiency Dataset
- **Target**: Heating Load
- **Features**: Building characteristics (orientation, glazing area, etc.)

### Energy Appliances Dataset
- **Target**: Appliances energy consumption
- **Features**: Temperature, humidity, and other environmental variables

## Requirements

See `requirements.txt` for complete list of dependencies. Key packages:
- pandas, numpy
- scikit-learn
- xgboost
- pytorch-tabnet
- optuna
- shap
- matplotlib

## Notes

- All models use random seed=42 for reproducibility
- TabNet training may take longer than other models
- SHAP analysis uses a sample of test data for faster computation
- Results are automatically organized by dataset name

## Author

CP322 Course Project

