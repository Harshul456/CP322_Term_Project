"""
TabNet model for the CP322 regression project.
"""
import os
import numpy as np
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch


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


def train_tabnet(X_train, y_train, X_test, y_test, save_dir, 
                 max_epochs=100, patience=10, batch_size=1024, 
                 virtual_batch_size=128, random_state=42):
    """
    Train and evaluate TabNet model.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
        save_dir (str): Directory to save model and metrics
        max_epochs (int): Maximum number of epochs
        patience (int): Early stopping patience
        batch_size (int): Batch size
        virtual_batch_size (int): Virtual batch size
        random_state (int): Random seed
        
    Returns:
        dict: Evaluation metrics
    """
    print("\n" + "="*50)
    print("Training TabNet...")
    print("="*50)
    
    # Convert to numpy arrays
    X_train_np = X_train.values.astype(np.float32)
    X_test_np = X_test.values.astype(np.float32)
    y_train_np = y_train.values.reshape(-1, 1).astype(np.float32)
    y_test_np = y_test.values.reshape(-1, 1).astype(np.float32)
    
    # Initialize TabNet
    model = TabNetRegressor(
        n_d=64,
        n_a=64,
        n_steps=5,
        gamma=1.5,
        n_independent=2,
        n_shared=2,
        lambda_sparse=1e-3,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        mask_type='entmax',
        seed=random_state,
        verbose=1
    )
    
    # Train model
    model.fit(
        X_train=X_train_np,
        y_train=y_train_np,
        eval_set=[(X_test_np, y_test_np)],
        eval_metric=['rmse'],
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        virtual_batch_size=virtual_batch_size
    )
    
    # Predictions
    y_pred_test = model.predict(X_test_np)
    y_pred_test = y_pred_test.flatten()
    
    # Evaluate on test set
    metrics = evaluate_model(y_test, y_pred_test, "TabNet")
    
    # Save model
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "tabnet")
    model.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    return metrics, model

