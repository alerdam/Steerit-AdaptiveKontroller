import numpy as np
import os

# Load combined features
features_path = "DATA/extracted_features.npy"

try:
    final_array = np.load(features_path)
    print("Extracted features loaded successfully")
    print("Shape:", final_array.shape)
except FileNotFoundError as e:
    print(f"Error: File not found - {e}")
except Exception as e:
    print(f"An error occurred: {e}")

# Column names
column_names = "slopes_connection_lines_1, slopes_connection_lines_2, slopes_connection_lines_3, slopes_connection_lines_4, slopes_connection_lines_5, slopes_knuckle_lines_1, slopes_knuckle_lines_2, y_diff_array, label_array"

# Save to CSV
np.savetxt("DATA/DS_TEMP.csv", final_array, delimiter=",", fmt="%.2f", header=column_names, comments='')
print("CSV file saved successfully with column headers!")

# Append to DATABASE.csv
database_csv_path = "DATA/DATABASE.csv"
file_exists = os.path.exists(database_csv_path)

if not file_exists or os.stat(database_csv_path).st_size == 0:
    np.savetxt(database_csv_path, final_array, delimiter=",", fmt="%.2f", header=column_names, comments='')
    print(f"CSV file saved successfully as {database_csv_path} with header.")
else:
    with open(database_csv_path, 'ab') as f:
        np.savetxt(f, final_array, delimiter=",", fmt="%.2f", comments='')
    print(f"Data successfully appended to {database_csv_path}.")