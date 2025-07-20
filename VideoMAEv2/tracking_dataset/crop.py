import collections
import glob  # For listing files in a directory
import os

import cv2
import numpy as np

# Provided bounding box data
bb_data_raw = "./tracking_dataset/volleyball.txt"
with open(bb_data_raw, "r") as file:
    bb_data_raw = file.read()
# --- User Inputs ---
frames_folder = "./tracking_dataset/volleyball/img1"  # Path to your folder containing 00000001.jpg, etc.
output_dir = "./tracking_dataset/volleyball/cropped_object_videos/"  # Directory to save output videos
fps = 30  # Frames per second for the output videos
# --- End User Inputs ---

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Parse bounding box data
# We'll store it as: {frame_id: {object_id: [bb_left, bb_top, bb_width, bb_height]}}
parsed_bb_data = collections.defaultdict(dict)
for line in bb_data_raw.strip().split("\n"):
    parts = [float(p) for p in line.split(",")]
    frame = int(parts[0])
    obj_id = int(parts[1])
    bb_left, bb_top, bb_width, bb_height = parts[2:6]
    parsed_bb_data[frame][obj_id] = [
        int(bb_left),
        int(bb_top),
        int(bb_width),
        int(bb_height),
    ]

# Get a sorted list of all image files in the frames folder
# Assumes filenames are like 00000001.jpg, 00000002.jpg, etc.
frame_files = sorted(glob.glob(os.path.join(frames_folder, "*.jpg")))
if not frame_files:
    print(
        f"Error: No .jpg files found in {frames_folder}. Please check the path and file names."
    )
    exit()

# Determine video resolution from the first frame
first_frame_path = frame_files[0]
first_frame = cv2.imread(first_frame_path)
if first_frame is None:
    print(f"Error: Could not read the first frame at {first_frame_path}.")
    exit()

frame_height, frame_width, _ = first_frame.shape
print(f"Detected frame resolution: {frame_width}x{frame_height}")

# Dictionary to hold video writers for each object
object_video_writers = {}
# Dictionary to store the max dimensions for each object, to ensure consistent output video size
object_max_dims = collections.defaultdict(lambda: [0, 0])  # [max_width, max_height]

print(
    "Scanning all frames to determine maximum object dimensions for consistent output video sizes..."
)
# First pass: Determine maximum dimensions for each object across all frames
# This helps create output videos with consistent dimensions, avoiding resizing issues.
for frame_idx, frame_path in enumerate(frame_files):
    current_frame_number = frame_idx + 1  # Frame numbers usually start from 1

    bbs_in_current_frame = parsed_bb_data.get(current_frame_number, {})

    for obj_id, (x, y, w, h) in bbs_in_current_frame.items():
        # Ensure coordinates are within frame boundaries
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame_width - x)
        h = min(h, frame_height - y)

        object_max_dims[obj_id][0] = max(object_max_dims[obj_id][0], w)
        object_max_dims[obj_id][1] = max(object_max_dims[obj_id][1], h)

print("Starting video cropping process...")
# Second pass: Process frames and crop objects
for frame_idx, frame_path in enumerate(frame_files):
    current_frame_number = frame_idx + 1  # Frame numbers usually start from 1
    print(f"Processing frame: {current_frame_number}/{len(frame_files)}")

    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"Warning: Could not read frame {frame_path}. Skipping.")
        continue

    # Get bounding boxes for the current frame
    bbs_in_current_frame = parsed_bb_data.get(current_frame_number, {})

    # Iterate through each object detected in the current frame
    for obj_id, (x, y, w, h) in bbs_in_current_frame.items():
        # Ensure coordinates are within frame boundaries
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame_width - x)
        h = min(h, frame_height - y)

        cropped_object = frame[y : y + h, x : x + w]

        # Get the target dimensions for this object's video
        target_width, target_height = object_max_dims[obj_id]

        # Initialize video writer for this object if not already done
        if obj_id not in object_video_writers:
            output_filename = os.path.join(output_dir, f"object_{obj_id}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4 files
            object_video_writers[obj_id] = cv2.VideoWriter(
                output_filename, fourcc, fps, (target_width, target_height)
            )
            if not object_video_writers[obj_id].isOpened():
                print(
                    f"Error: Could not create video writer for object {obj_id} at {output_filename}"
                )
                continue

        # Resize and pad the cropped object to match the target dimensions
        if cropped_object.shape[0] > 0 and cropped_object.shape[1] > 0:
            resized_cropped_object = np.zeros(
                (target_height, target_width, 3), dtype=np.uint8
            )

            # Calculate position to paste the cropped object (e.g., center it)
            paste_x = (target_width - cropped_object.shape[1]) // 2
            paste_y = (target_height - cropped_object.shape[0]) // 2

            resized_cropped_object[
                paste_y : paste_y + cropped_object.shape[0],
                paste_x : paste_x + cropped_object.shape[1],
            ] = cropped_object

            object_video_writers[obj_id].write(resized_cropped_object)
        else:
            # If the cropped object is empty, write a black frame of target size
            black_frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            object_video_writers[obj_id].write(black_frame)
            print(
                f"Warning: Cropped object {obj_id} in frame {current_frame_number} had zero dimensions. Writing black frame."
            )


# Release all video writers
for obj_id, writer in object_video_writers.items():
    writer.release()
    print(
        f"Saved video for object {obj_id} to {os.path.join(output_dir, f'object_{obj_id}.mp4')}"
    )

print("Video cropping from frames complete.")
