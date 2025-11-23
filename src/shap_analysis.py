"""
shap_analysis.py

Compute SHAP values for tree models (XGBoost). Produces and saves summary plot.
Requires shap and matplotlib.
"""
import os
import shap
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("results/shap_plots", exist_ok=True)

def run_shap(df, target, experiment_name="exp", xgb_model=None, feature_names=None):
    """
    If xgb_model is None, function will try to load it from results/models/<experiment_name>_xgboost.model
    df: original dataframe (used to produce feature matrix if feature names are provided)
    """
    # Simple path to model file (main pipeline returns model; but this accepts external call too)
    if xgb_model is None:
        try:
            from xgboost import XGBRegressor
            tmp_path = f"results/models/{experiment_name}_xgboost.model"
            # load via XGBoost booster
            model = XGBRegressor()
            model.load_model(tmp_path)
            xgb_model = model
            print(f"[shap] Loaded model from {tmp_path}")
        except Exception as e:
            print("[shap] Could not load model automatically:", e)
            return

    # Prepare a small sample of background data
    if feature_names is None:
        # We assume the caller provides a numpy matrix or preprocessed arrays for shap if needed.
        print("[shap] No feature_names supplied; please call run_shap with feature_names=np.array([...]) and xgb_model model object for correct plots.")
        return

    # Generate SHAP values
    explainer = shap.TreeExplainer(xgb_model)
    # We expect user to pass a 2D numpy array X for shap evaluation as feature_names
    X_for_shap = feature_names['X'] if isinstance(feature_names, dict) else feature_names
    shap_values = explainer.shap_values(X_for_shap)

    # Summary plot
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_for_shap, feature_names=feature_names['names'], show=False)
    out_png = f"results/shap_plots/{experiment_name}_shap_summary.png"
    plt.savefig(out_png, bbox_inches='tight')
    plt.close()
    print(f"[shap] Saved SHAP summary to {out_png}")
    return out_png
