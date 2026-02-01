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
    
    # Preprocessing
    if "age_years" not in df.columns:
        df["age_years"] = df["age"] / 365

    y = df["cardio"].values
    X = df.drop(columns=["cardio"])[cols]
    X_scaled = scaler.transform(X)

    return X_scaled.astype(np.float32), y.astype(np.float32)