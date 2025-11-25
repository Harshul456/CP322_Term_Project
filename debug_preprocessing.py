"""
Quick debug script to check preprocessing speed and identify issues.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import load_dataset
from preprocessing import preprocess_data
import time

print("="*60)
print("Debugging Preprocessing")
print("="*60)

# Test first dataset
print("\n1. Testing Energy Efficiency dataset...")
try:
    start = time.time()
    df = load_dataset("data/energy_efficiency_data.csv")
    load_time = time.time() - start
    print(f"✓ Loaded in {load_time:.2f} seconds")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    
    start = time.time()
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(
        df, "Heating_Load", test_size=0.2, random_state=42
    )
    prep_time = time.time() - start
    print(f"✓ Preprocessed in {prep_time:.2f} seconds")
    print(f"  Final feature count: {X_train.shape[1]}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test second dataset
print("\n2. Testing Energy Appliances dataset...")
try:
    start = time.time()
    df = load_dataset("data/energydata_complete.csv")
    load_time = time.time() - start
    print(f"✓ Loaded in {load_time:.2f} seconds")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns[:10])}... (showing first 10)")
    
    start = time.time()
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(
        df, "Appliances", test_size=0.2, random_state=42
    )
    prep_time = time.time() - start
    print(f"✓ Preprocessed in {prep_time:.2f} seconds")
    print(f"  Final feature count: {X_train.shape[1]}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Debugging complete!")
print("="*60)

