"""
Quick test to see where linear regression is hanging.
Run this to identify the issue.
"""
import sys
import os
import time
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

print("="*60)
print("Quick Linear Regression Test")
print("="*60)

# Step 1: Load data
print("\n[1/4] Loading data...")
from data_loader import load_dataset
start = time.time()
df = load_dataset("data/energy_efficiency_data.csv")
print(f"✓ Loaded in {time.time() - start:.2f}s, shape: {df.shape}")

# Step 2: Preprocess
print("\n[2/4] Preprocessing...")
from preprocessing import preprocess_data
start = time.time()
X_train, X_test, y_train, y_test, _ = preprocess_data(
    df, "Heating_Load", test_size=0.2, random_state=42
)
print(f"✓ Preprocessed in {time.time() - start:.2f}s")
print(f"  Training shape: {X_train.shape}")
print(f"  Test shape: {X_test.shape}")

# Step 3: Train Linear Regression
print("\n[3/4] Training Linear Regression...")
print(f"  Training shape: {X_train.shape}")
print(f"  Feature count: {X_train.shape[1]}")
print(f"  Sample count: {X_train.shape[0]}")

# Check for issues
import numpy as np
if X_train.shape[1] > X_train.shape[0]:
    print(f"  ⚠ WARNING: More features ({X_train.shape[1]}) than samples ({X_train.shape[0]})!")
    print("  This will cause a singular matrix error.")

if X_train.isna().any().any():
    print("  ⚠ WARNING: NaN values detected!")
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

if np.isinf(X_train.values).any():
    print("  ⚠ WARNING: Infinite values detected!")
    X_train = X_train.replace([np.inf, -np.inf], 0)
    X_test = X_test.replace([np.inf, -np.inf], 0)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
start = time.time()
print("  Calling model.fit()...")
try:
    model.fit(X_train, y_train)
    print(f"✓ Trained in {time.time() - start:.2f}s")
except Exception as e:
    print(f"✗ Error: {e}")
    print(f"  Time elapsed: {time.time() - start:.2f}s")
    raise

# Step 4: Predict
print("\n[4/4] Making predictions...")
start = time.time()
y_pred = model.predict(X_test)
print(f"✓ Predicted in {time.time() - start:.2f}s")

# Step 5: Evaluate
print("\nEvaluating...")
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"✓ RMSE: {rmse:.4f}, R2: {r2:.4f}")

print("\n" + "="*60)
print("Test complete! Linear regression should be very fast.")
print("If this hangs, check which step it's stuck on.")
print("="*60)

