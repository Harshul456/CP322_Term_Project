"""
baselines.py

Train and evaluate simple baseline models: Linear Regression and RandomForest.
Save basic metrics to disk under results/metrics (the main code can do that).
"""
import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

RESULTS_DIR = "results/metrics"
os.makedirs(RESULTS_DIR, exist_ok=True)

def _save_metrics(name, model_name, metrics):
    out_path = os.path.join(RESULTS_DIR, f"{name}__{model_name}__metrics.txt")
    with open(out_path, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    print(f"[baselines] Saved metrics to {out_path}")

def run_baselines(X_train, X_test, y_train, y_test, experiment_name="exp"):
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    preds_lr = lr.predict(X_test)
    metrics_lr = {
        "RMSE": mean_squared_error(y_test, preds_lr, squared=False),
        "MAE": mean_absolute_error(y_test, preds_lr),
        "R2": r2_score(y_test, preds_lr),
    }
    print(f"[baselines] LinearRegression metrics: {metrics_lr}")
    _save_metrics(experiment_name, "linear_regression", metrics_lr)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    preds_rf = rf.predict(X_test)
    metrics_rf = {
        "RMSE": mean_squared_error(y_test, preds_rf, squared=False),
        "MAE": mean_absolute_error(y_test, preds_rf),
        "R2": r2_score(y_test, preds_rf),
    }
    print(f"[baselines] RandomForest metrics: {metrics_rf}")
    _save_metrics(experiment_name, "random_forest", metrics_rf)

    # Optionally save models
    joblib.dump(lr, f"results/models/{experiment_name}_linear_regression.joblib")
    joblib.dump(rf, f"results/models/{experiment_name}_random_forest.joblib")
    print(f"[baselines] Saved baseline models.")
