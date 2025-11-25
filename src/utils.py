"""
Utility functions for the CP322 regression project.
"""
import json
import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def save_json(data, filepath):
    """
    Save data to a JSON file.
    
    Args:
        data (dict): Data to save
        filepath (str): Path to save the JSON file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)


def load_json(filepath):
    """
    Load data from a JSON file.
    
    Args:
        filepath (str): Path to the JSON file
        
    Returns:
        dict: Loaded data
    """
    with open(filepath, 'r') as f:
        return json.load(f)
