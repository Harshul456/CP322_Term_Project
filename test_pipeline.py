"""
Quick test script to verify the pipeline components work correctly.
Run this before running the full main.py pipeline.
"""
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all imports work."""
    print("Testing imports...")
    try:
        from utils import set_seed, save_json
        from data_loader import load_dataset
        from preprocessing import preprocess_data
        from baselines import train_linear_regression, train_random_forest
        from xgboost_model import train_xgboost
        from tabnet_model import train_tabnet
        from tuning import tune_xgboost
        from shap_analysis import create_shap_plots
        print("✓ All imports successful!")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_data_loading():
    """Test data loading."""
    print("\nTesting data loading...")
    try:
        from data_loader import load_dataset
        
        # Test first dataset
        if os.path.exists("data/energy_efficiency_data.csv"):
            df1 = load_dataset("data/energy_efficiency_data.csv")
            print(f"✓ Loaded energy_efficiency_data.csv: {df1.shape}")
        else:
            print("⚠ energy_efficiency_data.csv not found")
        
        # Test second dataset
        if os.path.exists("data/energydata_complete.csv"):
            df2 = load_dataset("data/energydata_complete.csv")
            print(f"✓ Loaded energydata_complete.csv: {df2.shape}")
        else:
            print("⚠ energydata_complete.csv not found")
        
        return True
    except Exception as e:
        print(f"✗ Data loading error: {e}")
        return False


def test_preprocessing():
    """Test preprocessing on a small sample."""
    print("\nTesting preprocessing...")
    try:
        import pandas as pd
        import numpy as np
        from preprocessing import preprocess_data
        
        # Create dummy data
        np.random.seed(42)
        n_samples = 100
        df = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'feature3': np.random.choice(['A', 'B', 'C'], n_samples),
            'target': np.random.randn(n_samples) * 10 + 50
        })
        
        X_train, X_test, y_train, y_test, preprocessor = preprocess_data(
            df, 'target', test_size=0.2, random_state=42
        )
        
        print(f"✓ Preprocessing successful!")
        print(f"  Training set: {X_train.shape}, Test set: {X_test.shape}")
        return True
    except Exception as e:
        print(f"✗ Preprocessing error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_baseline_models():
    """Test baseline models on dummy data."""
    print("\nTesting baseline models...")
    try:
        import pandas as pd
        import numpy as np
        from preprocessing import preprocess_data
        from baselines import train_linear_regression, train_random_forest
        
        # Create dummy data
        np.random.seed(42)
        n_samples = 100
        df = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'target': np.random.randn(n_samples) * 10 + 50
        })
        
        X_train, X_test, y_train, y_test, _ = preprocess_data(
            df, 'target', test_size=0.2, random_state=42
        )
        
        # Test Linear Regression
        os.makedirs("test_results", exist_ok=True)
        train_linear_regression(
            X_train, y_train, X_test, y_test,
            "test_results/models"
        )
        
        # Test Random Forest
        train_random_forest(
            X_train, y_train, X_test, y_test,
            "test_results/models"
        )
        
        print("✓ Baseline models work!")
        return True
    except Exception as e:
        print(f"✗ Baseline models error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("CP322 Pipeline Test Suite")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Data Loading", test_data_loading()))
    results.append(("Preprocessing", test_preprocessing()))
    results.append(("Baseline Models", test_baseline_models()))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n✓ All tests passed! You can run main.py now.")
    else:
        print("\n✗ Some tests failed. Please fix the issues before running main.py.")
    
    # Cleanup
    import shutil
    if os.path.exists("test_results"):
        shutil.rmtree("test_results")
        print("\nCleaned up test_results directory.")


if __name__ == "__main__":
    main()

