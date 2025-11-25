"""
Baseline models for the CP322 regression project.
"""
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib


def evaluate_model(y_true, y_pred, model_name):
    """
    Evaluate a model and return metrics.
    
    Args:
        y_true (array-like): True target values
        y_pred (array-like): Predicted target values
        model_name (str): Name of the model
        
    Returns:
        dict: Dictionary of metrics
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'model': model_name,
        'RMSE': float(rmse),
        'MAE': float(mae),
        'R2': float(r2)
    }
    
    print(f"\n{model_name} Metrics:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R2: {r2:.4f}")
    
    return metrics


def train_linear_regression(X_train, y_train, X_test, y_test, save_dir):
    """
    Train and evaluate Linear Regression baseline.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
        save_dir (str): Directory to save model and metrics
        
    Returns:
        dict: Evaluation metrics
    """
    print("\n" + "="*50)
    print("Training Linear Regression...")
    print("="*50)
    print(f"Training data shape: {X_train.shape}")
    print(f"Number of features: {X_train.shape[1]}")
    
    # Check for potential issues
    if X_train.shape[1] > X_train.shape[0]:
        print(f"⚠ CRITICAL: More features ({X_train.shape[1]}) than samples ({X_train.shape[0]})!")
        print("  Linear Regression will fail with singular matrix.")
        print("  Using Ridge Regression with regularization instead...")
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0)  # Regularization to handle singular matrix
    elif X_train.shape[1] > 10000:
        print(f"⚠ WARNING: Very large number of features ({X_train.shape[1]}).")
        print("  This may cause numerical issues. Using Ridge Regression with regularization...")
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=0.1)
    else:
        model = LinearRegression()
    
    # Check for NaN or Inf values
    if X_train.isna().any().any() or X_test.isna().any().any():
        print("⚠ WARNING: NaN values detected in features! Filling with 0...")
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)
    
    # Check for infinite values
    import numpy as np
    if np.isinf(X_train.values).any() or np.isinf(X_test.values).any():
        print("⚠ WARNING: Infinite values detected! Replacing with 0...")
        X_train = X_train.replace([np.inf, -np.inf], 0)
        X_test = X_test.replace([np.inf, -np.inf], 0)
    
    # Convert to numpy for faster computation if DataFrame
    if hasattr(X_train, 'values'):
        X_train_np = X_train.values
        X_test_np = X_test.values
    else:
        X_train_np = X_train
        X_test_np = X_test
    
    print("Fitting model...")
    
    import time
    start_time = time.time()
    
    try:
        model.fit(X_train_np, y_train)
        fit_time = time.time() - start_time
        print(f"✓ Model fitted successfully in {fit_time:.2f} seconds!")
    except Exception as e:
        print(f"✗ Error fitting model: {e}")
        print("  This might be due to:")
        print("  - Singular matrix (perfectly correlated features)")
        print("  - Too many features relative to samples")
        print("  - Numerical instability")
        raise
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Evaluate on test set
    metrics = evaluate_model(y_test, y_pred_test, "Linear Regression")
    
    # Save model
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "linear_regression.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    return metrics


def train_random_forest(X_train, y_train, X_test, y_test, save_dir, n_estimators=100, random_state=42):
    """
    Train and evaluate Random Forest baseline.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
        save_dir (str): Directory to save model and metrics
        n_estimators (int): Number of trees
        random_state (int): Random seed
        
    Returns:
        dict: Evaluation metrics
    """
    print("\n" + "="*50)
    print("Training Random Forest...")
    print("="*50)
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Evaluate on test set
    metrics = evaluate_model(y_test, y_pred_test, "Random Forest")
    
    # Save model
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "random_forest.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    return metrics
