"""
Hyperparameter tuning using Optuna for the CP322 regression project.
"""
import os
import optuna
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np


def objective(trial, X_train, y_train, X_val, y_val):
    """
    Objective function for Optuna optimization.
    
    Args:
        trial: Optuna trial object
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        X_val (pd.DataFrame): Validation features
        y_val (pd.Series): Validation target
        
    Returns:
        float: RMSE score
    """
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
    }
    
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    return rmse


def tune_xgboost(X_train, y_train, X_test, y_test, save_dir, 
                n_trials=50, random_state=42):
    """
    Tune XGBoost hyperparameters using Optuna.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
        save_dir (str): Directory to save results
        n_trials (int): Number of Optuna trials
        random_state (int): Random seed
        
    Returns:
        dict: Best parameters and metrics
    """
    print("\n" + "="*50)
    print("Hyperparameter Tuning with Optuna...")
    print("="*50)
    
    # Split training data for validation
    from sklearn.model_selection import train_test_split
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=random_state
    )
    
    # Create study
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=random_state))
    
    # Optimize
    study.optimize(
        lambda trial: objective(trial, X_train_split, y_train_split, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # Get best parameters
    best_params = study.best_params
    print(f"\nBest parameters: {best_params}")
    print(f"Best RMSE: {study.best_value:.4f}")
    
    # Train final model with best parameters on full training set
    best_model = XGBRegressor(**best_params, random_state=random_state, n_jobs=-1, verbosity=0)
    best_model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"Test RMSE with best parameters: {test_rmse:.4f}")
    
    # Save results
    os.makedirs(save_dir, exist_ok=True)
    results = {
        'best_params': best_params,
        'best_cv_rmse': float(study.best_value),
        'test_rmse': float(test_rmse),
        'n_trials': n_trials
    }
    
    import json
    results_path = os.path.join(save_dir, "optuna_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {results_path}")
    
    return results, best_model

