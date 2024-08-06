import numpy as np

# Define the input and output file paths
input_file = '/home/madhurie/scratch/fishdatasets/VMT/label/imghanlon_octopus_30fps_480p1_short/det/det.txt'
output_file = '/home/madhurie/scratch/fishdatasets/VMT/label/imghanlon_octopus_30fps_480p1_short/det/newdet.txt'

# Initialize list to hold the formatted detections
formatted_detections = []

# Read the input file
with open(input_file, 'r') as infile:
    lines = infile.readlines()

# Process each line in the input file
for frame_id, line in enumerate(lines, start=1):
    # Split the line into components using space as delimiter
    parts = line.strip().split()

    # Ensure the parts are in the expected order and have 4 elements
    if len(parts) != 4:
        print(f"Skipping invalid line: {line.strip()}")  # Debug statement
        continue  # Skip lines that don't have exactly 4 elements

    # Extract the relevant data
    try:
        x = float(parts[0])
        y = float(parts[1])
        width = float(parts[2])
        height = float(parts[3])
    except ValueError as e:
        print(f"Skipping line due to value error: {line.strip()} - {e}")  # Debug statement
        continue

    # Placeholder for confidence, using 1.0 as a default
    confidence = 1.0

    # Format the row according to the desired output
    formatted_row = [frame_id, -1, x, y, width, height, confidence, -1, -1, -1]
    formatted_detections.append(formatted_row)

# Convert the list of formatted detections to a numpy array
if formatted_detections:  # Check if the list is not empty
    formatted_detections = np.array(formatted_detections)

    # Check the shape of the array and print an error if the number of columns doesn't match the fmt string
    expected_columns = 10  # Number of elements in each row
    if formatted_detections.shape[1] != expected_columns:
        raise ValueError(f"Number of columns in formatted_detections ({formatted_detections.shape[1]}) does not match expected {expected_columns}")

    # Save the formatted detections to the output file
    np.savetxt(output_file, formatted_detections, fmt='%d,%d,%.2f,%.2f,%.2f,%.2f,%.5f,%d,%d,%d')
    print(f"Formatted detections saved to {output_file}")
else:
    print("No valid detections found to save.")

