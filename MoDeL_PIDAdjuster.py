# MoDeL_PIDAdaptive.py

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def teacher_delta_zn(e, de, y_diff, base_Kp=0.3, base_Kd=0.05):
    """
    Teacher based on Ziegler–Nichols style tuning.
    """
    # Ku and Tu 
    Ku = 1.0 + e * 2.0 + 0.1 
    Tu = 1.0 + de * 5.0 + 0.05 

    # ZN PID formulas
    Kp_target = 0.6 * Ku
    Kd_target = (Kp_target * Tu) / 8.0

    # Δ values relative to base
    dKp = Kp_target - base_Kp
    dKd = Kd_target - base_Kd

    # Clip to safe ranges
    dKp = np.clip(dKp, -0.5, 0.5)
    dKd = np.clip(dKd, -0.5, 0.5)

    # Normalize to [-1, 1]
    dKp_norm = dKp / 0.5
    dKd_norm = dKd / 0.5

    return dKp_norm, dKd_norm

# ---------------- Dataset ----------------
# Load dataset prepared with error (e), derivative of error (d_e), and y_diff
df = pd.read_csv("DATA/DataPID.csv")  # columns: e, d_e, y_diff
X = df[["e", "d_e", "y_diff"]].values.astype(np.float32)

# Generate teacher labels using heuristic function
labels = [teacher_delta_zn(e, de, y_diff) for e, de, y_diff in X]
y = np.array(labels, dtype=np.float32)  # shape (N, 2) → [ΔKp, ΔKd]

# ---------------- Model ----------------
# Simple MLP with tanh activations
model = keras.Sequential([
    layers.Input(shape=(3,)),          # inputs: [e, d_e, y_diff]
    layers.Dense(64, activation="tanh"),
    layers.Dense(64, activation="tanh"),
    layers.Dense(2, activation="tanh") # outputs: [ΔKp, ΔKd] in [-1, 1]
])

model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="mse")

# ---------------- Training ----------------
history = model.fit(X, y,
                    epochs=60,
                    batch_size=64,
                    validation_split=0.2,
                    verbose=1)

# ---------------- Save Model ----------------
model.save("models\M_PIDAdjuster_yDiff.keras")
print("Model saved as M_PIDAdjuster_yDiff.keras")

# ---------------- Quick Test ----------------
# Test with a sample input: error=0.4, d_error=-0.1, y_diff=0.2
sample = np.array([[0.4, -0.1, 0.2]], dtype=np.float32)
pred = model.predict(sample)[0]
print("Sample ΔKp, ΔKd prediction:", pred)