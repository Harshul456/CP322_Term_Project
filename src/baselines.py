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
    #calculate RMSE MAE r^2 for given model
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

    print("\n" + "="*50)
    print("Training Linear Regression...")
    print("="*50)
    print(f"Training data shape: {X_train.shape}")
    print(f"Number of features: {X_train.shape[1]}")
    
    #check for issues such as more features than samples
    if X_train.shape[1] > X_train.shape[0]:
        print(f" More features ({X_train.shape[1]}) than samples ({X_train.shape[0]})")
        print(" Using Ridge Regression with regularization instead...")
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0)  #Regularization for single matrix
    elif X_train.shape[1] > 10000:
        print(f"Very large number of features ({X_train.shape[1]}).")
        print("This may cause numerical issues. Using Ridge Regression with regularization")
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=0.1)
    else:
        model = LinearRegression()
    
    #check for INF values and replace with 0
    if X_train.isna().any().any() or X_test.isna().any().any():
        print("Filling empty NAN values with 0")
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)
    
    #check for INF values and replace with 0
    import numpy as np
    if np.isinf(X_train.values).any() or np.isinf(X_test.values).any():
        print("Replacing INF values with 0")
        X_train = X_train.replace([np.inf, -np.inf], 0)
        X_test = X_test.replace([np.inf, -np.inf], 0)
    
    #Convert to numpy for faster computation if DataFrame
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
        print(f"Model fitted successfully in {fit_time:.2f} seconds")
    except Exception as e:
        print(f"Error fitting model: {e}")
        raise
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    #evaluate model on test set
    metrics = evaluate_model(y_test, y_pred_test, "Linear Regression")
    
    #Save model
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "linear_regression.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    return metrics


def train_random_forest(X_train, y_train, X_test, y_test, save_dir, n_estimators=100, random_state=42):
    print("\n" + "="*50)
    print("Training Random Forest...")
    print("="*50)

    #random forest
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    #evaluate model on test set
    metrics = evaluate_model(y_test, y_pred_test, "Random Forest")
    
    #Save model
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "random_forest.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    return metrics
