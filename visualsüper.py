import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from pipe import Pipeline

pipe = Pipeline()
history_len = 100
data = {
    "raw": [], "pid": [], "kalman": [],
    "Kp": [], "Kd": [], "y_diff": []
}

fig = plt.figure(figsize=(12, 7))
gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1])

ax_3d = fig.add_subplot(gs[:, 0], projection='3d')
ax_steer = fig.add_subplot(gs[0, 1])
ax_params = fig.add_subplot(gs[1, 1])

line_raw, = ax_steer.plot([], [], label="Raw", alpha=0.5)
line_pid, = ax_steer.plot([], [], label="PID")
line_kalman, = ax_steer.plot([], [], label="Kalman", linewidth=2)

line_ydiff, = ax_params.plot([], [], label="y_diff", color="purple")
line_kp, = ax_params.plot([], [], label="Kp", color="green")
line_kd, = ax_params.plot([], [], label="Kd", color="orange")

def setup_axes():
    ax_steer.set_title("Steering")
    ax_params.set_title("y_diff / Kp / Kd")
    for a in [ax_steer, ax_params]:
        a.set_ylim(-1, 1)
        a.set_xlim(0, history_len)
        a.legend(loc='upper right', fontsize='x-small')

setup_axes()

def update(_):
    out = pipe.step()
    if out is None: return

    for key in data.keys():
        val = out[key] if out.get(key) is not None else np.nan
        data[key].append(val)
        if len(data[key]) > history_len:
            data[key].pop(0)

    x = np.arange(len(data["raw"]))
    line_raw.set_data(x, data["raw"])
    line_pid.set_data(x, data["pid"])
    line_kalman.set_data(x, data["kalman"])
    line_ydiff.set_data(x, data["y_diff"])
    line_kp.set_data(x, data["Kp"])
    line_kd.set_data(x, data["Kd"])

    ax_3d.clear()
    ax_3d.set_xlim([0, 1]); ax_3d.set_ylim([0, 1]); ax_3d.set_zlim([0, 1])
    ax_3d.view_init(elev=-90, azim=-90)

    if out["hands"] is not None:
        for hand in out["hands"]:
            if hand.shape == (21, 3):
                ax_3d.scatter(hand[:, 0], hand[:, 1], hand[:, 2], c='red', s=10)
        msg = f"ANGLE ESTIMATOR: {out['raw']}\nPID: {out['pid']:.2f}\nKalman: {out['kalman']:.2f}"
        color = 'blue'
    else:
        msg = f"(Prediction Only)\nKalman: {out['kalman']:.2f}"
        color = 'red'

    ax_3d.text2D(0.5, 0.92, msg, transform=ax_3d.transAxes, color='white', 
                 weight='bold', ha='center', linespacing=1.5,
                 bbox=dict(facecolor=color, alpha=0.6, boxstyle='round,pad=0.5'))

ani = animation.FuncAnimation(fig, update, interval=30, cache_frame_data=False)
plt.tight_layout()
plt.show()

pipe.release()