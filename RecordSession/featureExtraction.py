# featureExtraction.py
import numpy as np

def numpy_to_hand_pairs(npy_filename):
    loaded_data = np.load(npy_filename)
    num_frames, num_hands, num_landmarks, coords = loaded_data.shape
    data = []
    for frame_index in range(num_frames):
        frame_data = [[], []]
        hands = []
        for hand_index in range(num_hands):
            hand_landmarks = []
            for landmark_index in range(num_landmarks):
                x, y, z = loaded_data[frame_index, hand_index, landmark_index]
                hand_landmarks.append([x, y, z])
            hands.append((hand_landmarks[0][0], hand_landmarks))
        hands.sort(key=lambda h: h[0])
        frame_data[0], frame_data[1] = hands[0][1], hands[1][1]
        data.append(frame_data)
    return np.array(data)

def extract_palm_landmarks(hand_pair_array):
    num_frames = hand_pair_array.shape[0]
    palm_landmarks_indices = [0, 1, 5, 9, 13]
    palm_landmarks_array = np.zeros((num_frames, 2, 5, 3))
    for frame_index in range(num_frames):
        for hand_index in range(2):
            for i, palm_index in enumerate(palm_landmarks_indices):
                palm_landmarks_array[frame_index, hand_index, i] = hand_pair_array[frame_index, hand_index, palm_index]
    return palm_landmarks_array

def calculate_distance_vector(point1, point2):
    distance_vector = point2 - point1
    return np.round(distance_vector, 2)

def calculate_slope(distance_vector):
    if distance_vector[0] == 0:
        return float('0')
    return round(distance_vector[1] / distance_vector[0], 2)

# Input
npy_filename = 'DATA/landmark_positions.npy'
hand_pair_array = numpy_to_hand_pairs(npy_filename)

if hand_pair_array.shape[0] > 0:
    num_frames = hand_pair_array.shape[0]
    palm_array = extract_palm_landmarks(hand_pair_array)

    slopes_ConnectionLines = np.zeros((num_frames, 5))
    slopes_KnuckleLines = np.zeros((num_frames, 2))
    y_diff_array = np.zeros(num_frames)

    for frame_index in range(num_frames):
        for palm_index in range(5):
            LeftPalm = palm_array[frame_index, 0, palm_index]
            RightPalm = palm_array[frame_index, 1, palm_index]
            distance_vector = calculate_distance_vector(LeftPalm, RightPalm)
            slope = calculate_slope(distance_vector)
            slopes_ConnectionLines[frame_index, palm_index] = slope

        for hand_index in range(2):
            landmark5 = hand_pair_array[frame_index, hand_index, 5]
            landmark17 = hand_pair_array[frame_index, hand_index, 17]
            distance_vector = calculate_distance_vector(np.array(landmark5), np.array(landmark17))
            slope = calculate_slope(distance_vector)
            slopes_KnuckleLines[frame_index, hand_index] = slope

        hand0_landmark0 = hand_pair_array[frame_index, 0, 0]
        hand1_landmark0 = hand_pair_array[frame_index, 1, 0]
        y_diff = hand0_landmark0[1] - hand1_landmark0[1]
        y_diff_array[frame_index] = np.round(y_diff, 2)

    labels_array = np.load("DATA/label_data.npy")

    # Combine all features into one array
    final_array = np.column_stack((slopes_ConnectionLines,
                                   slopes_KnuckleLines,
                                   y_diff_array,
                                   labels_array))

    # Save single npy file
    np.save("DATA/extracted_features.npy", final_array)

    print("Extracted features saved to DATA/extracted_features.npy")
    print("Shape:", final_array.shape)
else:
    print("hand_pair_array is empty. No further processing.")