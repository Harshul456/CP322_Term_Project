"""
SHAP analysis for the CP322 regression project.
"""
import os
import shap
import matplotlib.pyplot as plt
import numpy as np


def create_shap_plots(model, X_test, feature_names, save_dir, dataset_name):
    """
    Create SHAP summary plots for tree-based models.
    
    Args:
        model: Trained model (XGBoost or Random Forest)
        X_test (pd.DataFrame): Test features
        feature_names (list): List of feature names
        save_dir (str): Directory to save plots
        dataset_name (str): Name of the dataset
    """
    print("\n" + "="*50)
    print("Generating SHAP plots...")
    print("="*50)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to numpy if needed
    if hasattr(X_test, 'values'):
        X_test_np = X_test.values
    else:
        X_test_np = X_test
    
    # Use a subset for faster computation
    sample_size = min(100, len(X_test_np))
    X_sample = X_test_np[:sample_size]
    
    # Create SHAP explainer - handle XGBoost compatibility issues
    try:
        # For XGBoost, try using the model's predict method directly
        if hasattr(model, 'get_booster'):
            # XGBoost model - use Explainer which handles version differences
            explainer = shap.Explainer(model, X_sample)
            shap_values = explainer(X_sample).values
        else:
            # Try TreeExplainer first (faster for tree models)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
    except Exception as e:
        print(f"TreeExplainer/Explainer failed: {e}")
        print("Trying PermutationExplainer as fallback...")
        try:
            # Use PermutationExplainer which works with any model
            explainer = shap.PermutationExplainer(model.predict, X_sample)
            shap_values = explainer(X_sample).values
        except Exception as e2:
            print(f"PermutationExplainer failed: {e2}")
            print("⚠ Skipping SHAP plots due to compatibility issues.")
            print("  This is a known issue with some XGBoost/SHAP version combinations.")
            return None
    
    # Summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    summary_path = os.path.join(save_dir, f"shap_summary_{dataset_name}.png")
    plt.tight_layout()
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"SHAP summary plot saved to {summary_path}")
    
    # Bar plot (mean absolute SHAP values)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                     plot_type="bar", show=False)
    bar_path = os.path.join(save_dir, f"shap_bar_{dataset_name}.png")
    plt.tight_layout()
    plt.savefig(bar_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"SHAP bar plot saved to {bar_path}")
    
    return shap_values
