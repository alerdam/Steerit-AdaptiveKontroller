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
        step_start = time.perf_counter()

        ret, image = self.cap.read()
        if not ret:
            return None

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        angle_latency_ms = None
        pid_latency_ms = None

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

            # Angle model latency
            start_angle = time.perf_counter()
            self.angle_interpreter.set_tensor(self.angle_input[0]['index'], input_data)
            self.angle_interpreter.invoke()
            prediction = float(self.angle_interpreter.get_tensor(self.angle_output[0]['index'])[0][0])
            end_angle = time.perf_counter()
            angle_latency_ms = (end_angle - start_angle) * 1000

            prediction = round(prediction, 2)

            # Errors
            error = prediction - self.smoothed_output
            d_error = error - self.pid_steering.prev_error

            # y_diff
            hand0_landmark0 = hand_pair_array[0][0]
            hand1_landmark0 = hand_pair_array[1][0]
            y_diff = hand0_landmark0[1] - hand1_landmark0[1]
            y_diff = np.round(y_diff, 2)

            # PIDAdjuster latency
            PIDAdjuster_input = np.array([[error, d_error, y_diff]], dtype=np.float32)
            start_pid = time.perf_counter()
            self.pid_interpreter.set_tensor(self.pid_input[0]['index'], PIDAdjuster_input)
            self.pid_interpreter.invoke()
            dKp_Adjuster, dKd_Adjuster = self.pid_interpreter.get_tensor(self.pid_output[0]['index'])[0]
            end_pid = time.perf_counter()
            pid_latency_ms = (end_pid - start_pid) * 1000

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

            step_end = time.perf_counter()
            step_latency_ms = (step_end - step_start) * 1000

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
                "hands": hand_pair_array,
                "angle_latency_ms": angle_latency_ms,
                "pid_latency_ms": pid_latency_ms,
                "step_latency_ms": step_latency_ms
            }

        else:
            # No hands → Kalman predict only
            self.kf.predict()
            theta_est, theta_dot, theta_ddot = self.kf.x.flatten()
            step_end = time.perf_counter()
            step_latency_ms = (step_end - step_start) * 1000
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
                "hands": None,
                "angle_latency_ms": None,
                "pid_latency_ms": None,
                "step_latency_ms": step_latency_ms
            }

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()


# ----------------- RUN LOOP WITH REPORT -----------------
if __name__ == "__main__":
    pipe = Pipeline()
    frame_count = 0
    angle_latencies = []
    pid_latencies = []
    step_latencies = []

    try:
        while frame_count < 500:  # run for 500 frames
            out = pipe.step()
            frame_count += 1
            if out is None:
                print(f"[Frame {frame_count}] Camera read failed.")
                continue

            if out["hands"] is not None:
                print(f"[Frame {frame_count}] Raw={out['raw']:.2f} | PID={out['pid']:.2f} | Kalman={out['kalman']:.2f} "
                      f"| Angle latency={out['angle_latency_ms']:.2f} ms | PID latency={out['pid_latency_ms']:.2f} ms | Step latency={out['step_latency_ms']:.2f} ms")
            else:
                print(f"[Frame {frame_count}] No hands → Kalman θ={out['kalman']:.2f} | Step latency={out['step_latency_ms']:.2f} ms")

            if out["angle_latency_ms"] is not None:
                angle_latencies.append(out["angle_latency_ms"])
            if out["pid_latency_ms"] is not None:
                pid_latencies.append(out["pid_latency_ms"])
            step_latencies.append(out["step_latency_ms"])

            time.sleep(0.03)

    except KeyboardInterrupt:
        print("Exiting...")

    finally:
        pipe.release()
        print("Camera released, resources freed.")

        # ---- REPORT ----
        def safe_stats(arr):
            if len(arr) == 0:
                return {"avg": float("nan"), "min": float("nan"), "max": float("nan")}
            return {"avg": np.mean(arr), "min": np.min(arr), "max": np.max(arr)}

        report = {
            "TotalFrames": frame_count,
            "AngleLatency": safe_stats(angle_latencies),
            "PIDLatency": safe_stats(pid_latencies),
            "StepLatency": safe_stats(step_latencies)
        }

        with open("FINAL_REPORT.txt", "w") as f:
            f.write("=== FINAL SYSTEM LATENCY REPORT ===\n\n")
            f.write(f"Total frames processed: {report['TotalFrames']}\n\n")
            f.write("[Angle Model Latency]\n")
            f.write(f"Average: {report['AngleLatency']['avg']:.3f} ms\n")
            f.write(f"Min: {report['AngleLatency']['min']:.3f} ms\n")
            f.write(f"Max: {report['AngleLatency']['max']:.3f} ms\n\n")
            f.write("[PID Model Latency]\n")
            f.write(f"Average: {report['PIDLatency']['avg']:.3f} ms\n")
            f.write(f"Min: {report['PIDLatency']['min']:.3f} ms\n")
            f.write(f"Max: {report['PIDLatency']['max']:.3f} ms\n\n")
            f.write("[Step Function Latency]\n")
            f.write(f"Average: {report['StepLatency']['avg']:.3f} ms\n")
            f