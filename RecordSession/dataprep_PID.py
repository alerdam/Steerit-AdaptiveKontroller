import numpy as np
import csv
import os

def save_experimental_csv(
    label_npy_path="DATA/label_data.npy",
    landmark_npy_path="DATA/landmark_positions.npy",
    csv_path="DATA/DataPID.csv"
):
    # Load label data
    labels_array = np.load(label_npy_path, allow_pickle=True)
    print("Loaded labels shape:", labels_array.shape)

    # Compute e and d_e
    e_list, de_list = [], []
    prev_e = 0.0
    for i in range(len(labels_array)):
        if isinstance(labels_array[i], (list, tuple, np.ndarray)) and len(labels_array[i]) >= 2:
            label0, label1 = labels_array[i][0], labels_array[i][1]
        else:
            label0 = labels_array[i-1] if i > 0 else labels_array[i]
            label1 = labels_array[i]
        e = label1 - label0
        d_e = e - prev_e if i > 0 else 0.0
        prev_e = e
        e_list.append(e)
        de_list.append(d_e)

    # Load landmark data
    loaded_data = np.load(landmark_npy_path)
    num_frames, num_hands, num_landmarks, coords = loaded_data.shape
    y_diff_list = []

    for frame_index in range(num_frames):
        hand0_landmark0 = loaded_data[frame_index, 0, 0]  # (x,y,z)
        hand1_landmark0 = loaded_data[frame_index, 1, 0]
        y_diff = hand0_landmark0[1] - hand1_landmark0[1]
        y_diff_list.append(np.round(y_diff, 2))

    # Ensure same length
    min_len = min(len(e_list), len(y_diff_list))
    e_list, de_list, y_diff_list = e_list[:min_len], de_list[:min_len], y_diff_list[:min_len]

    # Save to CSV
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["e", "d_e", "y_diff"])
        for e, de, y_diff in zip(e_list, de_list, y_diff_list):
            writer.writerow([e, de, y_diff])

    print(f"Experimental dataset saved to {csv_path}")
    print("Rows:", min_len)

if __name__ == "__main__":
    save_experimental_csv()