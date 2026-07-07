# Steerit: Adaptive Kontroller

Steerit is a modular rotational command engine that generates real-time steering values by processing camera-based hand gestures. By interpreting hand gestures as they mimic the physical act of steering, the system translates human intent through spatial dynamics, converting hand-position gestures into stable control signals.

Instead of simply following coordinates, the architecture identifies specific spatial states and translates them into reflexive responses, mapping user movement patterns directly to the dynamic requirements of the task.


## 🧠 Abstract & Core Research

The core research focuses on integrating unique temporal and spatial signatures—specifically "motion patterns"—into control stability. Steerit investigates a **Machine Learning-Based PID Tuning Model** that dynamically modulates proportional ($K_p$) and derivative ($K_d$) gains. 

Unlike fixed controllers, this model utilizes an expanded input feature set: in addition to standard error ($e$) and derivative of error ($\Delta e$), the system incorporates **spatial displacement features**, such as vertical hand coordinate differences ($y\_diff$). This multi-dimensional input allows the PID controller to exhibit richer, situation-appropriate adaptive behavior.

## 🛠 Feature Engineering & Perception

The system transforms raw visual data into control intent through a specialized extraction process:

### 1. Spatial Landmark Extraction
Using **MediaPipe**, the system tracks 21 3D landmarks for each hand in real-time. To ensure model efficiency, Steerit converts these into **Hand Geometry Features** rather than using raw coordinate sets.

### 2. Engineered Geometric Features
The prediction engine processes an 8-dimensional feature vector derived from relative hand positions:
* **Palm Connection Slopes**: Computed from lines connecting equivalent points on both palms.
* **Knuckle Alignment**: Slopes of the lines connecting the base of the index and pinky fingers (landmarks 5 and 17).
* **Vertical Displacement ($y\_diff$)**: The vertical difference between palm bases (landmark 0), acting as a primary indicator for steering rotation.



## 🏗 Methodologies

### 1. Machine Learning-Adaptive PID Tuning
The system uses a dedicated **TFLite PID Adjuster** to optimize control parameters in real-time.
* **Teacher-Based Learning**: The model is trained using a **Ziegler–Nichols (ZN)** heuristic function to calculate ideal $\Delta K_p$ and $\Delta K_d$ values based on error dynamics.
* **Dynamic Modulation**: Real-time adjustment of $K_p$ and $K_d$ gains based on a 3-feature input: $[e, \Delta e, y\_diff]$.
* **Clamped Outputs**: Predicted tuning values are normalized and clamped to ensure system stability within safe operational ranges.

### 2. State Estimation via Kalman Filter
To handle sensor noise and detection gaps (e.g., MediaPipe landmark flickering), a **Kalman Filter** is implemented:
* **Triple State Tracking**: Estimates angular position, velocity, and acceleration ($\theta, \dot{\theta}, \ddot{\theta}$).
* **Signal Continuity**: Maintains a stable control output even during brief hand detection losses by using the prediction step of the filter.



### 3. Machine Learning Architecture
The framework employs two primary machine learning models:
* **Angle Predictor (`MoDeL_Angle.py`)**: A multi-layer perceptron (MLP) with mixed `tanh` and `sigmoid` activations designed to map 8 geometric features to a steering angle.
* **PID Adjuster (`MoDeL_PIDAdaptive.py`)**: A high-speed MLP utilizing `tanh` activations for smooth, continuous gain modulation between $[-1, 1]$.

## 📂 Project Architecture

The system follows a **Sense-Think-Act** pipeline:

* **Sensing**: MediaPipe extracts 3D landmarks from camera input.
* **Thinking**: 
    * `AngleModel`: Translates geometric hand features into raw control intent.
    * `PIDAdjuster`: Computes $\Delta K_p$ and $\Delta K_d$ to refine the control signal.
    * `KalmanFilter`: Smooths the output using system dynamics.
* **Actuating**: The final stable signal is piped to the target application.

## 📜 License
This project is licensed under the **MIT License**.
