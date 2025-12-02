
import json
import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    #sets the seed to 42 so we can reproduce the same results
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def save_json(data, filepath):
    #save data to JSON file
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)


def load_json(filepath):
    #load data to JSON file
    with open(filepath, 'r') as f:
        return json.load(f)
