"""
Main pipeline for CP322 Regression Project.
Runs both datasets through the complete ML pipeline.
"""
import os
import sys
from datetime import datetime

# Add project root and src to path for reliable imports
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now import from src package
from utils import set_seed, save_json
from data_loader import load_dataset
from preprocessing import preprocess_data
from baselines import train_linear_regression, train_random_forest
from xgboost_model import train_xgboost
from tabnet_model import train_tabnet
from tuning import tune_xgboost
from shap_analysis import create_shap_plots


def run_pipeline(dataset_name, data_path, target_col, results_base_dir):
    """
    Run the complete ML pipeline for a dataset.
    
    Args:
        dataset_name (str): Name of the dataset
        data_path (str): Path to the dataset CSV
        target_col (str): Name of the target column
        results_base_dir (str): Base directory for results
    """
    print("\n" + "="*80)
    print(f"Processing Dataset: {dataset_name}")
    print(f"Target: {target_col}")
    print("="*80)
    
    # Create dataset-specific results directory
    results_dir = os.path.join(results_base_dir, dataset_name)
    os.makedirs(results_dir, exist_ok=True)
    
    # Load data
    df = load_dataset(data_path)
    
    # Preprocess
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(
        df, target_col, test_size=0.2, random_state=42
    )
    
    # Initialize metrics storage
    all_metrics = {}
    
    # 1. Baseline Models
    print("\n" + "#"*80)
    print("BASELINE MODELS")
    print("#"*80)
    
    baseline_dir = os.path.join(results_dir, "baselines")
    os.makedirs(baseline_dir, exist_ok=True)
    
    # Linear Regression
    lr_metrics = train_linear_regression(
        X_train, y_train, X_test, y_test, 
        os.path.join(baseline_dir, "models")
    )
    all_metrics['LinearRegression'] = lr_metrics
    
    # Random Forest
    rf_metrics = train_random_forest(
        X_train, y_train, X_test, y_test,
        os.path.join(baseline_dir, "models"),
        n_estimators=100, random_state=42
    )
    all_metrics['RandomForest'] = rf_metrics
    
    # 2. XGBoost
    print("\n" + "#"*80)
    print("XGBOOST MODEL")
    print("#"*80)
    
    xgb_dir = os.path.join(results_dir, "xgboost")
    os.makedirs(xgb_dir, exist_ok=True)
    
    xgb_metrics, xgb_model = train_xgboost(
        X_train, y_train, X_test, y_test,
        os.path.join(xgb_dir, "models"),
        n_estimators=100, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    all_metrics['XGBoost'] = xgb_metrics
    
    # 3. TabNet
    print("\n" + "#"*80)
    print("TABNET MODEL")
    print("#"*80)
    
    tabnet_dir = os.path.join(results_dir, "tabnet")
    os.makedirs(tabnet_dir, exist_ok=True)
    
    tabnet_metrics, tabnet_model = train_tabnet(
        X_train, y_train, X_test, y_test,
        os.path.join(tabnet_dir, "models"),
        max_epochs=50, patience=5, batch_size=1024,  # Reduced for faster training
        virtual_batch_size=128, random_state=42
    )
    all_metrics['TabNet'] = tabnet_metrics
    
    # 4. Hyperparameter Tuning
    print("\n" + "#"*80)
    print("HYPERPARAMETER TUNING (Optuna)")
    print("#"*80)
    
    tuning_dir = os.path.join(results_dir, "tuning")
    os.makedirs(tuning_dir, exist_ok=True)
    
    tuning_results, tuned_model = tune_xgboost(
        X_train, y_train, X_test, y_test,
        os.path.join(tuning_dir, "optuna"),
        n_trials=50, random_state=42
    )
    all_metrics['TunedXGBoost'] = {
        'model': 'Tuned XGBoost',
        'RMSE': tuning_results['test_rmse'],
        'best_params': tuning_results['best_params']
    }
    
    # 5. SHAP Analysis
    print("\n" + "#"*80)
    print("SHAP ANALYSIS")
    print("#"*80)
    
    shap_dir = os.path.join(results_dir, "shap_plots")
    os.makedirs(shap_dir, exist_ok=True)
    
    # SHAP for XGBoost
    try:
        shap_result = create_shap_plots(
            xgb_model, X_test, list(X_test.columns),
            shap_dir, f"{dataset_name}_xgboost"
        )
        if shap_result is None:
            print("⚠ SHAP plots for XGBoost skipped due to compatibility issues")
    except Exception as e:
        print(f"⚠ Error generating SHAP plots for XGBoost: {e}")
    
    # SHAP for Random Forest
    try:
        from joblib import load
        rf_model = load(os.path.join(baseline_dir, "models", "random_forest.joblib"))
        shap_result = create_shap_plots(
            rf_model, X_test, list(X_test.columns),
            shap_dir, f"{dataset_name}_randomforest"
        )
        if shap_result is None:
            print("⚠ SHAP plots for Random Forest skipped due to compatibility issues")
    except Exception as e:
        print(f"⚠ Error generating SHAP plots for Random Forest: {e}")
    
    # Save all metrics
    metrics_path = os.path.join(results_dir, "metrics", "all_metrics.json")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    save_json(all_metrics, metrics_path)
    
    print("\n" + "="*80)
    print(f"Pipeline completed for {dataset_name}")
    print(f"Results saved to: {results_dir}")
    print("="*80)
    
    return all_metrics


def main():
    """Main function to run the complete pipeline."""
    print("="*80)
    print("CP322 Regression Project - Complete Pipeline")
    print("="*80)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Define datasets
    datasets = [
        {
        "name": "EnergyEfficiency",
        "path": "data/energy_efficiency_data.csv",
            "target": "Heating_Load"
        },
        {
        "name": "EnergyAppliances",
            "path": "data/energydata_complete.csv",
        "target": "Appliances"
    }
    ]
    
    # Results base directory
    results_base_dir = "results"
    os.makedirs(results_base_dir, exist_ok=True)
    
    # Run pipeline for each dataset
    all_results = {}
    for dataset in datasets:
        try:
            metrics = run_pipeline(
                dataset["name"],
                dataset["path"],
                dataset["target"],
                results_base_dir
            )
            all_results[dataset["name"]] = metrics
        except Exception as e:
            print(f"\nError processing {dataset['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save summary
    summary_path = os.path.join(results_base_dir, "summary.json")
    save_json(all_results, summary_path)
    
    print("\n" + "="*80)
    print("ALL PIPELINES COMPLETED")
    print("="*80)
    print(f"Summary saved to: {summary_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("METRICS SUMMARY")
    print("="*80)
    for dataset_name, metrics in all_results.items():
        print(f"\n{dataset_name}:")
        for model_name, model_metrics in metrics.items():
            if isinstance(model_metrics, dict) and 'RMSE' in model_metrics:
                print(f"  {model_name}: RMSE={model_metrics['RMSE']:.4f}, "
                      f"R2={model_metrics.get('R2', 'N/A'):.4f}")


if __name__ == "__main__":
    main()
