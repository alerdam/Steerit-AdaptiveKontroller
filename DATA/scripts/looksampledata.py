import numpy as np

"""
Load data
"""
# Define the paths to your .npy files
slopes_connection_lines_path = "DATA/slopes_connection_lines.npy"
slopes_knuckle_lines_path = "DATA/slopes_knuckle_lines.npy"
y_diff_array_path = "DATA\y_diff_array.npy"

# Load the NumPy arrays
try:
    slopes_connection_lines = np.load(slopes_connection_lines_path)
    slopes_knuckle_lines = np.load(slopes_knuckle_lines_path)
    y_diff_array = np.load(y_diff_array_path)

    # Print the loaded arrays (or do whatever you need with them)
    print("slopes_connection_lines:")
    print("Shape:", slopes_connection_lines.shape)

    print("\nslopes_knuckle_lines:")
    print("Shape:", slopes_knuckle_lines.shape)

    print("\ny_diff_array")
    print("Shape:", y_diff_array.shape)


except FileNotFoundError as e:
    print(f"Error: File not found - {e}")

except Exception as e:
    print(f"An error occurred: {e}")