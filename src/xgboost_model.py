"""
XGBoost model for the CP322 regression project.
"""
import os
import numpy as np
from xgboost import XGBRegressor
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


def train_xgboost(X_train, y_train, X_test, y_test, save_dir, 
                  n_estimators=100, max_depth=6, learning_rate=0.1, 
                  subsample=0.8, colsample_bytree=0.8, random_state=42):
    """
    Train and evaluate XGBoost model.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
        save_dir (str): Directory to save model and metrics
        n_estimators (int): Number of boosting rounds
        max_depth (int): Maximum tree depth
        learning_rate (float): Learning rate
        subsample (float): Subsample ratio
        colsample_bytree (float): Column subsample ratio
        random_state (int): Random seed
        
    Returns:
        dict: Evaluation metrics
    """
    print("\n" + "="*50)
    print("Training XGBoost...")
    print("="*50)
    
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        n_jobs=-1,
        verbosity=0
    )
    
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Evaluate on test set
    metrics = evaluate_model(y_test, y_pred_test, "XGBoost")
    
    # Save model
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "xgboost.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    return metrics, model

