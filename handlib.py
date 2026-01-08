# handlib.py

import numpy as np
import math


# ---------------- DS PREP Functions ----------------
def Explore_DS(data):
    """Splits a NumPy array into train/test and returns the split data."""
    if data is not None:
        print("Data shape:", data.shape)

        xdata = data[:, :8]
        ydata = data[:, 8]
        splitVal = int(len(xdata) * 0.8)  # 80% train, 20% test
        xtrain = xdata[:splitVal, :]
        xtest = xdata[splitVal:, :]
        ytrain = ydata[:splitVal]
        ytest = ydata[splitVal:]
        return xtrain, ytrain, xtest, ytest
    else:
        return None, None, None, None  

def load_csv_to_numpy(filename=""):
    """Loads a CSV file into a NumPy array."""
    try:
        data = np.loadtxt(filename, delimiter=',', skiprows=1)  # skip header row
        return data
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# ---------------- Utility Functions ----------------
def calculate_3d_distance_manual(point1, point2):
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
    return math.sqrt(dx**2 + dy**2 + dz**2)

def extract_palm_landmarks(hand_pair_array):
    """Extracts palm landmarks (0, 1, 5, 9, 13) from a hand pair NumPy array."""
    palm_landmarks_indices = [0, 1, 5, 9, 13]
    palm_landmarks_array = np.zeros((2, 5, 3))
    if hand_pair_array is not None and len(hand_pair_array) == 2:
        for hand_index in range(2):
            if isinstance(hand_pair_array[hand_index], np.ndarray) and hand_pair_array[hand_index].shape == (21, 3):
                for i, palm_index in enumerate(palm_landmarks_indices):
                    palm_landmarks_array[hand_index, i] = hand_pair_array[hand_index][palm_index]
    return palm_landmarks_array

def calculate_distance_vector(point1, point2):
    return np.round(point2 - point1, 2)

def calculate_slope(distance_vector):
    if distance_vector[0] == 0:
        return float('inf')
    return round(distance_vector[1] / distance_vector[0], 2)

def feature_prepper(hand_pair_array):
    """
    Given two hands (each 21x3 landmarks), compute feature vector:
    - 5 slopes from palm connection lines
    - 2 slopes from knuckle lines
    - y_diff between wrist landmarks
    Returns a 1D numpy array of length 8.
    """
    palm_array = extract_palm_landmarks(hand_pair_array)
    slopes_connection = []
    for i in range(5):
        LeftPalm = palm_array[0, i]
        RightPalm = palm_array[1, i]
        distance_vector = calculate_distance_vector(LeftPalm, RightPalm)
        slopes_connection.append(calculate_slope(distance_vector))

    slopes_knuckle = []
    for hand_index in range(2):
        landmark5 = hand_pair_array[hand_index][5]
        landmark17 = hand_pair_array[hand_index][17]
        distance_vector = calculate_distance_vector(np.array(landmark5), np.array(landmark17))
        slopes_knuckle.append(calculate_slope(distance_vector))

    y_diff = hand_pair_array[0][0][1] - hand_pair_array[1][0][1]

    features = np.hstack((np.array(slopes_connection),
                          np.array(slopes_knuckle),
                          np.array([y_diff])))
    return features
