from src.data_loader import load_dataset
from src.preprocessing import preprocess_data
from src.baselines import run_baselines
from src.xgboost_model import run_xgboost
from src.tabnet_model import run_tabnet
from src.tuning import run_optuna_tuning
from src.shap_analysis import run_shap

def run_pipeline(config):
    print(f"\n=== Running experiment for: {config['name']} ===")

    # 1. Load data
    df = load_dataset(config["path"], config["target"])

    # 2. Preprocess
    X_train, X_test, y_train, y_test = preprocess_data(df, config["target"])

    # 3. Baselines
    run_baselines(X_train, X_test, y_train, y_test, config["name"])

    # 4. XGBoost
    run_xgboost(X_train, X_test, y_train, y_test, config["name"])

    # 5. TabNet / FT-transformer
    run_tabnet(X_train, X_test, y_train, y_test, config["name"])

    # 6. Hyperparameter Tuning (Optuna)
    run_optuna_tuning(X_train, y_train, config["name"])

    # 7. SHAP Analysis
    run_shap(df, config["target"], config["name"])

    print(f"=== Finished: {config['name']} ===\n")


if __name__ == "__main__":

    efficiency_config = {
        "name": "EnergyEfficiency",
        "path": "data/energy_efficiency_data.csv",
        "target": "Heating Load"  # or "Cooling Load"
    }

    appliances_config = {
        "name": "EnergyAppliances",
        "path": "data/energy_data_complete.csv",
        "target": "Appliances"
    }

    run_pipeline(efficiency_config)
    run_pipeline(appliances_config)
