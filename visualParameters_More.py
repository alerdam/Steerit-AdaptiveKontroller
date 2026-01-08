# visualParameters_More.py

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from pipe import Pipeline

pipe = Pipeline()

# Plot setup
fig, axs = plt.subplots(2, 2, figsize=(10, 6))
(ax1, ax2), (ax3, ax4) = axs

history_len = 100
raw_vals, pid_vals, kalman_vals = [], [], []
kp_vals, kd_vals = [], []
dkp_vals, dkd_vals = [], []
ydiff_vals = []   # NEW list for y_diff

# Line objects
line_raw, = ax1.plot([], [], label="Raw")
line_pid, = ax1.plot([], [], label="PID")
line_kalman, = ax1.plot([], [], label="Kalman")
ax1.set_title("Steering values")
ax1.set_ylim(-1, 1)
ax1.set_xlim(0, history_len)
ax1.legend()

line_kp, = ax2.plot([], [], label="Kp")
line_dkp, = ax2.plot([], [], label="ΔKp_hat")
ax2.set_title("Kp and ΔKp_hat")
ax2.set_ylim(-1, 1)   # fixed range
ax2.set_xlim(0, history_len)
ax2.legend()

line_kd, = ax3.plot([], [], label="Kd")
line_dkd, = ax3.plot([], [], label="ΔKd_hat")
ax3.set_title("Kd and ΔKd_hat")
ax3.set_ylim(-1, 1)   # fixed range
ax3.set_xlim(0, history_len)
ax3.legend()

# Bottom-right panel → y_diff
line_ydiff, = ax4.plot([], [], label="y_diff", color="purple")
ax4.set_title("y_diff between hands")
ax4.set_ylim(-1, 1)   # adjust range if needed
ax4.set_xlim(0, history_len)
ax4.legend()

def update(_):
    out = pipe.step()
    if out is None:
        return line_raw, line_pid, line_kalman, line_kp, line_dkp, line_kd, line_dkd, line_ydiff

    # Append values (None → np.nan)
    raw_vals.append(out["raw"] if out["raw"] is not None else np.nan)
    pid_vals.append(out["pid"] if out["pid"] is not None else np.nan)
    kalman_vals.append(out["kalman"] if out["kalman"] is not None else np.nan)
    kp_vals.append(out["Kp"] if out["Kp"] is not None else np.nan)
    kd_vals.append(out["Kd"] if out["Kd"] is not None else np.nan)
    dkp_vals.append(out["dKp_hat"] if out["dKp_hat"] is not None else np.nan)
    dkd_vals.append(out["dKd_hat"] if out["dKd_hat"] is not None else np.nan)
    ydiff_vals.append(out["y_diff"] if out["y_diff"] is not None else np.nan)

    # Reset lists if history_len exceeded
    if len(raw_vals) >= history_len:
        raw_vals.clear(); pid_vals.clear(); kalman_vals.clear()
        kp_vals.clear(); kd_vals.clear(); dkp_vals.clear(); dkd_vals.clear()
        ydiff_vals.clear()

    x = np.arange(len(raw_vals))

    # Update lines
    line_raw.set_data(x, raw_vals)
    line_pid.set_data(x, pid_vals)
    line_kalman.set_data(x, kalman_vals)

    line_kp.set_data(np.arange(len(kp_vals)), kp_vals)
    line_dkp.set_data(np.arange(len(dkp_vals)), dkp_vals)

    line_kd.set_data(np.arange(len(kd_vals)), kd_vals)
    line_dkd.set_data(np.arange(len(dkd_vals)), dkd_vals)

    line_ydiff.set_data(np.arange(len(ydiff_vals)), ydiff_vals)

    return line_raw, line_pid, line_kalman, line_kp, line_dkp, line_kd, line_dkd, line_ydiff

ani = animation.FuncAnimation(fig, update, interval=30, blit=False)
plt.tight_layout()
plt.show()

pipe.release()