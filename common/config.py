# Paths
COLUMNS_PATH = "./preprocessing/columns.txt"
SCALER_PATH = "./preprocessing/scaler.pkl"

# Federated Learning Params
NUM_CLIENTS = 3
ROUNDS = 5
LOCAL_EPOCHS = 35
BATCH_SIZE = 64


# --- DP Hyperparameters ---
L2_NORM_CLIP = 1.5      # Limits the influence of any single training example
NOISE_MULTIPLIER = 1.1  # Amount of noise added to gradients
NUM_MICROBATCHES = 1    # Usually set to 1 or your batch size

import os

def get_input_dim():
    """Dynamically determine input dimension from columns.txt."""
    if not os.path.exists(COLUMNS_PATH):
        return 12
    with open(COLUMNS_PATH, "r") as f:
        return len([line.strip() for line in f.readlines() if line.strip()])