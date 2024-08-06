import cv2
import numpy as np

def read_tracking_results(results_file):
    """Read tracking results from a text file."""
    tracks = []
    with open(results_file, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) >= 7:
                frame_id, track_id, x, y, w, h, confidence = map(float, parts[:7])
                tracks.append((int(frame_id), int(track_id), x, y, w, h, confidence))
    return tracks

def create_video_with_tracking(sequence_dir, results_file, output_video_file):
    frame_files = sorted(Path(sequence_dir).glob('*.jpg'))  # Adjust extension as needed
    if not frame_files:
        raise ValueError("No frames found in the directory.")
    
    tracking_results = read_tracking_results(results_file)
    frame_size = (cv2.imread(str(frame_files[0])).shape[1], cv2.imread(str(frame_files[0])).shape[0])
    
    video_saver = cv2.VideoWriter(output_video_file, cv2.VideoWriter_fourcc(*'XVID'), 30, frame_size)  # Adjust fps if needed

    for frame_file in frame_files:
        frame = cv2.imread(str(frame_file))
        frame_id = int(frame_file.stem.split('_')[-1])  # Extract frame ID from filename
        
        # Draw tracking results for the current frame
        for track in tracking_results:
            if track[0] == frame_id:
                x, y, w, h = map(int, (track[2], track[3], track[4], track[5]))
                track_id = int(track[1])
                color = (0, 255, 0)  # Green color for tracking boxes
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f'ID: {track_id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        video_saver.write(frame)

    video_saver.release()
    print(f"Video with tracking saved to {output_video_file}")

# Example usage
from pathlib import Path

sequence_dir = '/home/madhurie/scratch/fishdatasets/VMT/frame/img1'
results_file = '/home/madhurie/scratch/fishdatasets/VMT/results/tracking_results4.npy'
output_video_file = '/home/madhurie/scratch/fishdatasets/VMT/label/output/tracking_results4.avi'

create_video_with_tracking(sequence_dir, results_file, output_video_file)

