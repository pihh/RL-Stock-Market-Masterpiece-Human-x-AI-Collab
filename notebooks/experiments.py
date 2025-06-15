import hashlib, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ========== GLOBALS ==========
EXPERIMENT_REGISTRY_PATH = Path("/data/experiments/experiment_registry.csv")
## 1.1 Experiment Deduplication

def check_if_experiment_exists(config_hash):
    if not EXPERIMENT_REGISTRY_PATH.exists():
        return False
    registry = pd.read_csv(EXPERIMENT_REGISTRY_PATH)
    return config_hash in registry['hash'].values

def register_experiment(config_hash, config_dict):
    row = pd.DataFrame([{"hash": config_hash, **config_dict}])
    if EXPERIMENT_REGISTRY_PATH.exists():
        registry = pd.read_csv(EXPERIMENT_REGISTRY_PATH)
        registry = pd.concat([registry, row], ignore_index=True)
    else:
        registry = row
    registry.to_csv(EXPERIMENT_REGISTRY_PATH, index=False)

## 1.2 Predictable Episode Discovery (Meta + Signal Analysis)


def experiment_hash(config={}):
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    return config_hash
    #if not check_if_experiment_exists(config_hash):
    #    score = run_experiment(aapl_df, example_config)
    #    example_config['score'] = score
    #    register_experiment(config_hash, example_config)
    #else:
    #    print("Experiment already exists.")
