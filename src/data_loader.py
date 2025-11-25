"""
Data loading utilities for the CP322 regression project.
"""
import pandas as pd
import os


def load_dataset(filepath):
    """
    Load a CSV dataset.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"Loaded dataset: {filepath}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    return df

