import cv2
import numpy as np
import tensorflow as tf
import handlib
import time
from controllers import AdaptivePIDController, KalmanFilter
from mediapipe import solutions

class Pipeline:
    def __init__(self,
                 angle_model="models/M_Angle.tflite",
                 pid_model="models/M_PIDAdjuster_yDiff.tflite"):
        # Load TFLite models
        self.angle_interpreter = tf.lite.Interpreter(model_path=angle_model)
        self.angle_interpreter.allocate_tensors()
        self.angle_input = self.angle_interpreter.get_input_details()
        self.angle_output = self.angle_interpreter.get_output_details()

        self.pid_interpreter = tf.lite.Interpreter(model_path=pid_model)
        self.pid_interpreter.allocate_tensors()
        self.pid_input = self.pid_interpreter.get_input_details()
        self.pid_output = self.pid_interpreter.get_output_details()

        # MediaPipe
        mp_hands = solutions.hands
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")

        # Controllers
        self.pid_steering = AdaptivePIDController()
        self.kf = KalmanFilter(dt=0.03)
        self.smoothed_output = 0.0

    def step(self):
        ret, image = self.cap.read()
        if not ret:
            return None

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
            hand_pair_array = []
            for hand_landmarks in results.multi_hand_landmarks:
                hand_points = []
                for landmark in hand_landmarks.landmark:
                    lx = abs(float(landmark.x) - 1)
                    ly = float(landmark.y)
                    lz = float(landmark.z)
                    hand_points.append([lx, ly, lz])
                hand_pair_array.append(np.array(hand_points))

            input_data = handlib.feature_prepper(hand_pair_array).reshape(1, -1).astype(np.float32)

            # Angle model
            self.angle_interpreter.set_tensor(self.angle_input[0]['index'], input_data)
            self.angle_interpreter.invoke()
            prediction = float(self.angle_interpreter.get_tensor(self.angle_output[0]['index'])[0][0])
            prediction = round(prediction, 2)

            # Errors
            error = prediction - self.smoothed_output
            d_error = error - self.pid_steering.prev_error

            # y_diff from landmark0 of both hands
            hand0_landmark0 = hand_pair_array[0][0]  # (x,y,z)
            hand1_landmark0 = hand_pair_array[1][0]
            y_diff = hand0_landmark0[1] - hand1_landmark0[1]
            y_diff = np.round(y_diff, 2)

            # PIDAdjuster input now has 3 features
            PIDAdjuster_input = np.array([[error, d_error, y_diff]], dtype=np.float32)
            self.pid_interpreter.set_tensor(self.pid_input[0]['index'], PIDAdjuster_input)
            self.pid_interpreter.invoke()
            dKp_Adjuster, dKd_Adjuster = self.pid_interpreter.get_tensor(self.pid_output[0]['index'])[0]

            if not np.isfinite(dKp_Adjuster): dKp_Adjuster = 0.0
            if not np.isfinite(dKd_Adjuster): dKd_Adjuster = 0.0

            correction, Kp_val, Kd_val, d_error = self.pid_steering.update(
                error, dKp_hat=dKp_Adjuster, dKd_hat=dKd_Adjuster, alpha=0.2
            )
            self.smoothed_output += correction

            # Kalman
            self.kf.predict()
            self.kf.update(self.smoothed_output)
            theta_est, theta_dot, theta_ddot = self.kf.x.flatten()

            return {
                "raw": prediction,
                "pid": self.smoothed_output,
                "kalman": theta_est,
                "Kp": Kp_val,
                "Kd": Kd_val,
                "dKp_hat": dKp_Adjuster,
                "dKd_hat": dKd_Adjuster,
                "y_diff": y_diff,
                "output": self.smoothed_output,
                "hands": hand_pair_array
            }

        else:
            # No hands → Kalman predict only
            self.kf.predict()
            theta_est, theta_dot, theta_ddot = self.kf.x.flatten()
            return {
                "raw": None,
                "pid": None,
                "kalman": theta_est,
                "Kp": None,
                "Kd": None,
                "dKp_hat": None,
                "dKd_hat": None,
                "y_diff": None,
                "output": theta_est,
                "hands": None
            }

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()


# ----------------- RUN LOOP -----------------
if __name__ == "__main__":
    pipe = Pipeline()
    try:
        frame_count = 0
        while True:
            out = pipe.step()
            frame_count += 1
            if out is None:
                print(f"[Frame {frame_count}] Camera read failed.")
                continue

            if out["hands"] is not None:
                print(f"[Frame {frame_count}] Raw={out['raw']:.2f} | PID={out['pid']:.2f} | Kalman={out['kalman']:.2f} "
                      f"| Kp={out['Kp']:.3f} | Kd={out['Kd']:.3f} | ΔKp_hat={out['dKp_hat']:.4f} | ΔKd_hat={out['dKd_hat']:.4f} "
                      f"| y_diff={out['y_diff']:.2f}")
            else:
                print(f"[Frame {frame_count}] No hands → Kalman θ={out['kalman']:.2f}")

            time.sleep(0.03)  # slow down a bit

    except KeyboardInterrupt:
        print("Exiting...")

    finally:
        pipe.release()
        print("Camera released, resources freed.")