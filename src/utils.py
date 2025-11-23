"""
utils.py - small utilities for metrics, saving objects, and a small reproducibility helper.
"""
import os
import json
import random
import numpy as np

def set_seed(seed=42):
    import random as rn
    rn.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
