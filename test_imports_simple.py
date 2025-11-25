"""
Simple script to test if imports work correctly.
Run this to verify the import setup is correct.
"""
import os
import sys

print("="*60)
print("Testing Import Setup")
print("="*60)

# Get paths
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')

print(f"\nProject root: {project_root}")
print(f"Source path: {src_path}")
print(f"Current working directory: {os.getcwd()}")

# Check if src directory exists
if not os.path.exists(src_path):
    print(f"\n✗ ERROR: src/ directory not found at {src_path}")
    sys.exit(1)

# Check if __init__.py exists
init_file = os.path.join(src_path, '__init__.py')
if not os.path.exists(init_file):
    print(f"\n⚠ WARNING: src/__init__.py not found. Creating it...")
    with open(init_file, 'w') as f:
        f.write('# Package init file\n')
    print("✓ Created src/__init__.py")

# Add to path
if src_path not in sys.path:
    sys.path.insert(0, src_path)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"\nPython path includes:")
print(f"  - {src_path}")
print(f"  - {project_root}")

# Test imports
print("\n" + "="*60)
print("Testing Imports")
print("="*60)

modules_to_test = [
    'utils',
    'data_loader',
    'preprocessing',
    'baselines',
    'xgboost_model',
    'tabnet_model',
    'tuning',
    'shap_analysis'
]

all_passed = True
for module_name in modules_to_test:
    try:
        module = __import__(module_name)
        print(f"✓ {module_name} - OK")
    except ImportError as e:
        print(f"✗ {module_name} - FAILED: {e}")
        all_passed = False

print("\n" + "="*60)
if all_passed:
    print("✓ ALL IMPORTS SUCCESSFUL!")
    print("You can now run: python main.py")
else:
    print("✗ SOME IMPORTS FAILED")
    print("Check the error messages above")
print("="*60)

