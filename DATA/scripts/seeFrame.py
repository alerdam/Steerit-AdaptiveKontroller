import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np

def numpy_to_hand_pairs(npy_filename):
    """
    Reads a .npy file containing hand landmark data and converts it into a NumPy array,
    grouped by hand pairs (hand 0 and hand 1), with each hand's landmarks in order.
    """
    loaded_data = np.load(npy_filename)
    num_frames, num_hands, num_landmarks, coords = loaded_data.shape

    data = []
    for frame_index in range(num_frames):
        frame_data = [[], []]
        # Extract hand data for the frame
        hands = []
        for hand_index in range(num_hands):
            hand_landmarks = []
            for landmark_index in range(num_landmarks):
                x, y, z = loaded_data[frame_index, hand_index, landmark_index]
                hand_landmarks.append([x, y, z])
            hands.append((hand_landmarks[0][0], hand_landmarks))  # Store x value with landmarks
        # Sort hands so the one with the smaller x value is indexed as 0
        hands.sort(key=lambda h: h[0])
        # Assign sorted data to frame_data
        frame_data[0], frame_data[1] = hands[0][1], hands[1][1]
        data.append(frame_data)
    return np.array(data)


npy_filename = 'DATA/landmark_positions.npy'
hand_pair_array = numpy_to_hand_pairs(npy_filename)

selected_frame = 30
current_pos = hand_pair_array[30]
print(current_pos)

for hand_index in range(2):
    plt.scatter(current_pos[hand_index][:, 0], 
                current_pos[hand_index][:, 1], c='red')
plt.show()