import pandas as pd
import numpy as np
import pickle
from common.config import SCALER_PATH, COLUMNS_PATH

def load_local_data(path):
    # Load assets
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(COLUMNS_PATH) as f:
        cols = [line.strip() for line in f.readlines()]

    df = pd.read_csv(path)
    
    # Check for 'age' before calculating 'age_years'
    if "age_years" not in df.columns and "age" in df.columns:
        df["age_years"] = df["age"] / 365
    elif "age_years" not in df.columns:
        # If both are missing, initialize with a neutral value or handle error
        df["age_years"] = 0 

    y = df["cardio"].values
    
    # Ensure all expected columns exist to avoid KeyError
    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns in {path}: {missing_cols}")
        for mc in missing_cols:
            df[mc] = 0 # Default value for missing features

    X = df[cols]
    X_scaled = scaler.transform(X)

    return X_scaled.astype(np.float32), y.astype(np.float32)