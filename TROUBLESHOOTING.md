# Troubleshooting Guide

## Import Errors: "ModuleNotFoundError" or "No module named 'utils'"

### Problem
When running `python main.py` on a different computer, you get errors like:
```
ModuleNotFoundError: No module named 'utils'
```

### Solution 1: Make sure you're in the project root directory

**Check your current directory:**
```bash
# Windows PowerShell
pwd
# Should show: C:\CP322 Project

# If not, navigate to project root:
cd "C:\CP322 Project"
python main.py
```

### Solution 2: Verify src/__init__.py exists

The `src/` directory needs an `__init__.py` file to be recognized as a Python package.

**Check if it exists:**
```bash
# Windows PowerShell
dir src\__init__.py

# If it doesn't exist, create it (it should be there now)
```

### Solution 3: Run from project root

Always run `main.py` from the project root directory:

```
C:\CP322 Project\          ← Run python main.py from here
├── main.py
├── src/
│   ├── __init__.py        ← Must exist
│   ├── utils.py
│   └── ...
└── data/
```

### Solution 4: Alternative - Use absolute imports

If imports still fail, you can modify `main.py` to use absolute imports:

**Option A: Add to PYTHONPATH (Windows)**
```powershell
$env:PYTHONPATH = "C:\CP322 Project"
python main.py
```

**Option B: Use relative imports (if running as module)**
```python
# In main.py, change imports to:
from src.utils import set_seed, save_json
from src.data_loader import load_dataset
# etc...
```

### Solution 5: Verify Python can find the modules

Test if imports work:
```python
# Create test_imports.py
import sys
import os
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

try:
    from utils import set_seed
    print("✓ Import successful!")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    print(f"Project root: {project_root}")
    print(f"Source path: {src_path}")
    print(f"Files in src/: {os.listdir(src_path)}")
```

Run: `python test_imports.py`

### Common Issues:

1. **Wrong directory**: Running from `src/` instead of project root
2. **Missing __init__.py**: `src/__init__.py` doesn't exist
3. **Python version**: Using Python 2 instead of Python 3
4. **Virtual environment**: Dependencies not installed

### Quick Fix Checklist:

- [ ] Run `python main.py` from project root (`C:\CP322 Project\`)
- [ ] Verify `src/__init__.py` exists
- [ ] Check Python version: `python --version` (should be 3.8+)
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Check file structure matches expected layout

### If Still Not Working:

Try this alternative import method in `main.py`:

```python
# At the top of main.py, replace imports with:
import os
import sys

# Get absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, 'src')

# Add to Python path
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, SRC_DIR)

# Now imports should work
from utils import set_seed, save_json
# ... rest of imports
```

