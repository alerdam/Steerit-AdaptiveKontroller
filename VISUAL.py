# VISUAL.py

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.animation as animation
from pipe import Pipeline

pipe = Pipeline()

fig = plt.figure("Model Predictions", figsize=(5, 5))
ax = fig.add_subplot(111, projection='3d')

def update(_):
    ax.clear()
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1]); ax.set_zlim([0, 1])
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.view_init(elev=-90, azim=-90)

    out = pipe.step()
    if out is None:
        return ax

    if out["hands"] is not None:
        for hand in out["hands"]:
            if hand.shape == (21, 3):
                ax.scatter(hand[:, 0], hand[:, 1], hand[:, 2], c='red')

        ax.text2D(
            0.5, 0.95,
            f"Kalman θ={out['kalman']:.2f}\nPID θ={out['pid']:.2f}\nRaw={out['raw']:.2f}",
            transform=ax.transAxes,
            fontsize=12,
            color='white',
            weight='bold',
            ha='center',
            bbox=dict(facecolor='blue', alpha=0.6, boxstyle='round,pad=0.5')
        )
    else:
        ax.text2D(
            0.5, 0.95,
            f"Kalman prediction only θ={out['kalman']:.2f}",
            transform=ax.transAxes,
            fontsize=12,
            color='white',
            weight='bold',
            ha='center',
            bbox=dict(facecolor='red', alpha=0.6, boxstyle='round,pad=0.5')
        )

    return ax

ani = animation.FuncAnimation(fig, update, interval=30, blit=False)
plt.show()

pipe.release()