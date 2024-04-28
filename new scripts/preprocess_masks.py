import os
import cv2
import numpy as np

# Specify the directory containing your mask files
# mask_directory = "rf_data/rough_masks"
# mask_directory = "bmode_data/rough_masks"
mask_directory = "bk_rf_data/bk_segments"

# Specify the directory to save preprocessed masks
# preprocessed_directory = "rf_data/masks"
# preprocessed_directory = "bmode_data/masks"
preprocessed_directory = "bk_rf_data/bk_segments_processed"
os.makedirs(preprocessed_directory, exist_ok=True)

# Iterate through each file in the directory
for filename in os.listdir(mask_directory):
    # Check if the file is a regular file (not a directory)
    # if os.path.isfile(os.path.join(mask_directory, filename)):

    # preprocessing the bk data in bk_segments
    # Check if the file ends with "mask.png"
    if filename.endswith("mask.png"):

        # Process the file
        print(f"Processing mask file: {filename}")

        # Load the mask using OpenCV
        mask_path = os.path.join(mask_directory, filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # For example, thresholding to create a binary mask
        _, binary_mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

        # Save the preprocessed mask to the new directory
        # preprocessed_mask_path = os.path.join(preprocessed_directory, f"preprocessed_{filename}")
        # saving training masks - save in different directory
        preprocessed_mask_path = os.path.join(preprocessed_directory, f"{filename}")
        cv2.imwrite(preprocessed_mask_path, binary_mask)

        print(f"Preprocessed mask saved to: {preprocessed_mask_path}")
