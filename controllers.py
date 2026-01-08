# controllers.py

import numpy as np

# ---------------- Adaptive PID Controller ----------------
class AdaptivePIDController:
    def __init__(self, Kp=0.3, Ki=0.02, Kd=0.05,
                 i_limit=2.0, out_limit=1.0,
                 Kp_min= 0.1, Kp_max=1.0,
                 Kd_min=0.01, Kd_max=1.0):
        self.base_Kp = Kp
        self.Kp = Kp
        self.Ki = Ki
        self.base_Kd = Kd
        self.Kd = Kd
        self.integral = 0.0
        self.prev_error = 0.0
        self.i_limit = i_limit
        self.out_limit = out_limit
        self.Kp_min = Kp_min
        self.Kp_max = Kp_max
        self.Kd_min = Kd_min
        self.Kd_max = Kd_max

    def update(self, error, dKp_hat=0.0, dKd_hat=0.0, alpha=0.2):
        # Derivative of error
        d_error = error - self.prev_error

        # Integral with clamp
        self.integral += error
        self.integral = float(np.clip(self.integral, -self.i_limit, self.i_limit))

        # Kp and Kd are set from base + alpha * hat, then clamped to min/max ranges
        self.Kp = float(np.clip(self.base_Kp + alpha * float(dKp_hat), self.Kp_min, self.Kp_max))
        self.Kd = float(np.clip(self.base_Kd + alpha * float(dKd_hat), self.Kd_min, self.Kd_max))

        # Control output with clamp
        u = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * d_error)
        u = float(np.clip(u, -self.out_limit, self.out_limit))

        # Store prev error
        self.prev_error = error

        return u, self.Kp, self.Kd, d_error

# ---------------- Kalman Filter ----------------
class KalmanFilter:
    def __init__(self, dt=0.03):
        self.x = np.zeros((3,1))
        self.P = np.eye(3) * 0.1
        self.A = np.array([[1, dt, 0.5*dt*dt],
                           [0, 1, dt],
                           [0, 0, 1]])
        self.H = np.array([[1,0,0]])
        self.Q = np.eye(3) * 0.001
        self.R = np.array([[0.05]])

    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x

    def update(self, z):
        z = np.array([[z]])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(3) - K @ self.H) @ self.P
        return self.x