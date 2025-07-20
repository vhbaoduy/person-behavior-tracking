import glob  # To find image files
import os

import cv2
import numpy as np
import pandas as pd

label_names = {
    0: "dancing ballet",
    1: "breakdancing",
    2: "salsa dancing",
    3: "swing dancing",
    4: "zumba",
    # 5: "playing badminton",
    # 6: "playing basketball",
    # 7: "playing ice hockey",
    # 8: "playing tennis",
    # 9: "playing volleyball",
    5: "badminton",
    6: "basketball",
    7: "ice hockey",
    8: "tennis",
    9: "volleyball",
}


def plot_tracker_output_on_images(
    images_folder,
    tracker_results_path,
    label_file,
    output_video_path,
    frame_rate=30,  # Desired frame rate for the output video
    start_frame_id=1,  # The starting frame ID in your tracker results and image names
    max_frames=None,  # Set to an integer to process only the first N frames
):
    """
    Overlays tracking results (bounding boxes and IDs) onto a sequence of images
    and then compiles them into a video.

    Args:
        images_folder (str): Path to the folder containing frame_id.jpg images.
        tracker_results_path (str): Path to the tracker's output .txt file.
        output_video_path (str): Path to save the new video with overlays.
        frame_rate (int): Frame rate for the output video.
        start_frame_id (int): The starting frame ID (e.g., 1 if frame_000001.jpg).
        max_frames (int, optional): Maximum number of frames to process. Defaults to None (all frames).
    """

    # 1. Load Tracker Data
    try:
        # Assuming MOTChallenge format: frame_id, object_id, x, y, width, height, ...
        tracker_data = pd.read_csv(
            tracker_results_path,
            header=None,
            sep=",",
            index_col=False,
            names=[
                "frame",
                "id",
                "bb_left",
                "bb_top",
                "bb_width",
                "bb_height",
                "conf",
                "x",
                "y",
                "z",
            ],
        )
        print(
            f"Loaded tracker data from: {tracker_results_path} (Shape: {tracker_data.shape})"
        )
    except Exception as e:
        print(f"Error loading tracker data from {tracker_results_path}: {e}")
        return

    # Filter out low-confidence detections if 'conf' column is used
    tracker_data = tracker_data[tracker_data["conf"] >= 0.0]

    # 2. Get list of image files and sort them numerically
    # Assumes image names are like '000001.jpg', '000002.jpg', etc.
    image_files = sorted(glob.glob(os.path.join(images_folder, "*.jpg")))
    if not image_files:
        print(f"Error: No .jpg images found in {images_folder}")
        return

    # Determine dimensions from the first image to set up VideoWriter
    first_frame = cv2.imread(image_files[0])
    if first_frame is None:
        print(f"Error: Could not read the first image {image_files[0]}")
        return
    height, width, _ = first_frame.shape
    print(f"Detected image dimensions: {width}x{height}")

    # Prepare VideoWriter for output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4 files
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))
    if not out.isOpened():
        print(f"Error: Could not create output video {output_video_path}")
        return

    # Generate unique colors for each ID
    id_colors = {}
    rng = np.random.default_rng(seed=42)  # For consistent colors

    def get_color(label):
        if label not in id_colors:
            id_colors[label] = tuple(
                rng.integers(0, 256, size=3).tolist()
            )  # BGR format
        return id_colors[label]

    label_df = pd.read_csv(label_file)

    # Loop through each image file
    processed_frame_count = 0
    for i, image_file in enumerate(image_files):
        # Determine the current frame_id from the filename
        # Assuming filename is like '000001.jpg' where '000001' is the frame ID.
        # Adjust parsing if your filenames are different (e.g., 'frame_000001.jpg')
        try:
            # Example for '000001.jpg' or 'v_id_c_000001.jpg'
            # Extracts '000001'
            frame_id_str = os.path.splitext(os.path.basename(image_file))[0]
            # Handle cases like 'v_-6Os86HzwCs_c009_000001' if your image filenames include video name
            if (
                "_" in frame_id_str and frame_id_str.count("_") > 1
            ):  # e.g., v_-6Os86HzwCs_c009_000001
                current_frame_id_val = int(frame_id_str.split("_")[-1])
            else:  # e.g., 000001
                current_frame_id_val = int(frame_id_str)
            # Adjust to match the frame numbering in your tracker file (usually 1-indexed)
            # If your tracker file's frame IDs directly match the number extracted above,
            # then current_frame_id_for_lookup = current_frame_id_val
            # If your tracker file is 1-indexed and images start from a different index, adjust.
            # Most common: tracker file is 1-indexed, and image names are 1-indexed from '000001.jpg'
            current_frame_id_for_lookup = current_frame_id_val

        except ValueError:
            print(
                f"Warning: Could not parse frame ID from filename: {image_file}. Skipping."
            )
            continue

        if max_frames is not None and processed_frame_count >= max_frames:
            print(f"Reached max_frames limit ({max_frames}). Stopping.")
            break

        frame = cv2.imread(image_file)
        if frame is None:
            print(f"Warning: Could not read image {image_file}. Skipping.")
            continue

        # Get detections for the current frame ID from the tracker data
        current_frame_detections = tracker_data[
            tracker_data["frame"] == current_frame_id_for_lookup
        ].copy()

        for _, row in current_frame_detections.iterrows():
            obj_id = int(row["id"])
            x1 = int(row["bb_left"])
            y1 = int(row["bb_top"])
            w = int(row["bb_width"])
            h = int(row["bb_height"])
            # label = label_df[label_df["video_object"] == obj_id]["predict"].values[0]
            x2 = x1 + w
            y2 = y1 + h

            # Clamp coordinates to frame boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width - 1, x2)
            y2 = min(height - 1, y2)

            # color = get_color(label)
            color = get_color(obj_id)
            text_color = (255, 255, 255)  # White text

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw ID text background
            # text = f"ID: {obj_id}, {label_names[label]}"
            text = f"ID: {obj_id}"
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            text_x = x1
            text_y = y1 - 10 if y1 - 10 > text_height + 5 else y1 + text_height + 5
            text_y = max(text_height, text_y)
            text_y = min(height - 5, text_y)

            cv2.rectangle(
                frame,
                (text_x, text_y - text_height - 5),
                (text_x + text_width, text_y + 5),
                color,
                -1,
            )
            cv2.putText(
                frame,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                text_color,
                2,
            )

        out.write(frame)
        processed_frame_count += 1

        if processed_frame_count % 100 == 0:
            print(f"Processed {processed_frame_count} frames from images.")

    print(f"Finished processing. Output video saved to: {output_video_path}")

    out.release()
    cv2.destroyAllWindows()


# --- Example Usage ---
if __name__ == "__main__":
    # --- Configuration ---
    # Adjust these paths to your specific setup for SportMOT and MOTIP
    SPORTMOT_ROOT = "./datasets/DanceTrack"  # Root directory of your SportMOT dataset
    TRACKER_OUTPUT_ROOT = "outputs/r50_deformable_detr_motip_dancetrack/evaluate/default/DanceTrack/val/r50_deformable_detr_motip_dancetrack/tracker"

    # Example: Plotting a sequence from the validation set
    TARGET_SEQUENCE_NAME = "dancetrack0034"  # This is the video ID. Images might be in a folder named this.
    TARGET_SPLIT = "val"  # Or 'train', 'test' - depends on where this sequence is.

    # Paths to the folder containing images for this specific sequence
    # This path is crucial. It must point to the directory holding your .jpg images.
    # Example: SportMOT's "images" folder structure is usually like:
    # SportsMOT/train_val_test_split/images/SPLIT/SEQUENCE_NAME/*.jpg
    input_images_folder = os.path.join(
        SPORTMOT_ROOT, TARGET_SPLIT, TARGET_SEQUENCE_NAME, "img1"
    )

    # Path to the tracker's output .txt file for this sequence
    tracker_txt_file = os.path.join(TRACKER_OUTPUT_ROOT, f"{TARGET_SEQUENCE_NAME}.txt")

    # Output path for the new video
    output_video_dir = "./output_videos_from_images"
    os.makedirs(output_video_dir, exist_ok=True)
    output_video_file = os.path.join(
        output_video_dir, f"{TARGET_SEQUENCE_NAME}_MOTIP_tracked.mp4"
    )
    input_images_folder = "./tracking_dataset/volleyball/img1"
    tracker_txt_file = "./tracking_dataset/volleyball.txt"
    label_file = "./tracking_dataset/volleyball.csv"
    output_video_file = "./tracking_dataset/volleyball/volleyball_MOTIP_tracked.mp4"

    print(f"Attempting to plot images for sequence: {TARGET_SEQUENCE_NAME}")
    print(f"Input Images Folder: {input_images_folder}")
    print(f"Tracker Output: {tracker_txt_file}")
    print(f"Output Video: {output_video_file}")

    # Run the plotting function
    plot_tracker_output_on_images(
        images_folder=input_images_folder,
        tracker_results_path=tracker_txt_file,
        label_file=label_file,
        output_video_path=output_video_file,
        frame_rate=30,  # Standard for MOT. Adjust if your dataset is different.
        start_frame_id=1,  # Most datasets start frame IDs at 1. Verify your image filenames.
        # max_frames=500 # Uncomment to process only first 500 frames for testing
    )
