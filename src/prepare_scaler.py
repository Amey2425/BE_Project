import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import os

DATASET = "./dataset/final_health_data.csv"

# Load dataset
df = pd.read_csv(DATASET)

# Add age_years temporarily
df["age_years"] = df["age"] / 365

# Drop target for scaling
X = df.drop(columns=["cardio"])
y = df["cardio"]

# Fit scaler
scaler = StandardScaler()
scaler.fit(X)

# Save scaler
os.makedirs("./preprocessing", exist_ok=True)

with open("./preprocessing/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save column order
with open("./preprocessing/columns.txt", "w") as f:
    for col in X.columns:
        f.write(col + "\n")

print("✔ Scaler and column order saved.")
