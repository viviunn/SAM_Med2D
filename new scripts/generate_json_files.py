import os
import json


def generate_image2label2(output_json_path, images_directory):
    # Specify the directories and paths
    # images_directory = "bk_bm_data/train/images"
    # masks_directory = "bk_rf_data/bk_segments_processed"
    # masks_directory = "bk_bm_data/train/masks"
    # output_json_path = "bk_bm_data/train/image2label_train.json"
    # output_json_path = "bk_bm_data/test/label2image_test.json"
    
    # # split by patient
    # output_json_path = "patient_split/rf_data/train/image2label_train.json"
    # images_directory = "patient_split/rf_data/train"
    masks_directory = "patient_split/bm_data/train/masks"

    # Initialize dictionary to store file mappings
    file_mappings = {}

    # Iterate through each file in the images directory
    for image_filename in os.listdir(images_directory):
        # Check if the file is a regular file (not a directory)
        if os.path.isfile(os.path.join(images_directory, image_filename)):
            # Process the file
            print(f"Processing image file: {image_filename}")

            # Create the corresponding mask filename
            mask_filename = f"{os.path.splitext(image_filename)[0]}_mask.png"
            mask_path = os.path.join(masks_directory, mask_filename)
            print("maskpath:",mask_path)

            #****** added
            # Replace backslashes with forward slashes in paths
            # image_path = os.path.join(images_directory, image_filename).replace("\", '/')
            # mask_path = os.path.join(masks_directory, mask_filename).replace("\", '/')
            # Replace backslashes with forward slashes in paths
            image_path = os.path.join(images_directory, image_filename).replace('\\', '/')
            mask_path = os.path.join(masks_directory, mask_filename).replace('\\', '/')
            print("maskpath:", mask_path)
            # Check if the mask file exists
            if os.path.isfile(mask_path):
                # Update file mappings
                # image_path = os.path.join(images_directory, image_filename)
                # mask_path = os.path.join(masks_directory, mask_filename)

                # for json train file
                file_mappings[image_path] = mask_path
                # for json test file
                # file_mappings[mask_path] = image_path

                print(f"Mask found for image: {mask_filename}")
    print("file_mappings:",file_mappings)
    # Save the file mappings to a JSON file
    with open(output_json_path, 'w') as json_file:
        json.dump(file_mappings, json_file, indent=4)

    print("Training set JSON file created.")


def replace_slashes_in_json(file_path):
    # Load the JSON file
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)

    # Function to replace backslashes with forward slashes in a string
    def replace_slashes(s):
        return s.replace('\\', '/')

    # Recursively apply the replacement function to all strings in the JSON data
    def process_item(item):
        if isinstance(item, str):
            return replace_slashes(item)
        elif isinstance(item, list):
            return [process_item(elem) for elem in item]
        elif isinstance(item, dict):
            return {replace_slashes(key): process_item(value) for key, value in item.items()}
        else:
            return item

    # Apply the replacement to the entire JSON data
    data = process_item(data)

    # Save the modified JSON data back to the file
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=2)

# # Specify the path to your JSON file
# json_file_path = "bk_rf_data/test/label2image_test.json"
#
# # Call the function to replace slashes in the JSON file
# replace_slashes_in_json(json_file_path)

# main()

# import json
#
# Load the JSON file
# json_file_path = "bk_bm_data/train/image2label_train.json"

# with open(json_file_path, 'r') as json_file:
#     data = json.load(json_file)

# # Update values to lists
# for key, value in data.items():
#     data[key] = [value]

# # Save the modified data back to the file
# with open(json_file_path, 'w') as json_file:
#     json.dump(data, json_file, indent=4)


# combine bm and rf training json file
import json

def modify_json_keys(original_file, modified_file):
    # Load your JSON data
    with open(original_file, 'r') as file:
        data = json.load(file)

    # Iterate through the keys and modify them
    modified_data = {}
    for key in data.keys():
        # Split the key by '/' and replace the first element
        key_parts = key.split('/')
        key_parts[0] = 'bk_rf_bm_data'
        new_key = '/'.join(key_parts)

        # Assign the modified key to the new dictionary
        modified_data[new_key] = data[key]

    # Save the modified data back to JSON
    with open(modified_file, 'w') as file:
        json.dump(modified_data, file, indent=4)

    print(f"JSON file has been modified and saved as '{modified_file}'")

# # Example usage
# modify_json_keys('bk_rf_bm_data/train/image2label_train_old.json', 'bk_rf_bm_data/train/image2label_train_new.json')


def modify_json_keys_values(original_file, modified_file):
    # Load your JSON data
    with open(original_file, 'r') as file:
        data = json.load(file)

    # Iterate through the keys and values and modify them
    modified_data = {}
    for key, values in data.items():
        # Modify the key
        key_parts = key.split('/')
        key_parts[0] = 'bk_rf_bm_data'
        new_key = '/'.join(key_parts)

        # Modify the values
        new_values = []
        for value in values:
            value_parts = value.split('/')
            value_parts[0] = 'bk_rf_bm_data'
            new_values.append('/'.join(value_parts))

        # Assign the modified key and values to the new dictionary
        modified_data[new_key] = new_values

    # Save the modified data back to JSON
    with open(modified_file, 'w') as file:
        json.dump(modified_data, file, indent=4)

    print(f"JSON file has been modified and saved as '{modified_file}'")

# Example usage
# modify_json_keys_values('bk_rf_bm_data/train/image2label_train_old.json', 'bk_rf_bm_data/train/image2label_train_new.json')


# # divide by anatomical locations

import json
import re
from collections import defaultdict
import os

def split_json_by_anatomical_location(input_json_path, output_directory):
    """
    Creates separate JSON files for each anatomical location, grouping images and masks by location.

    Parameters:
    - input_json_path: Path to the original JSON file.
    - output_directory: Directory where the new JSON files will be saved.
    """
    # Load the original JSON file
    with open(input_json_path, 'r') as file:
        data = json.load(file)

    # Initialize a dictionary to store the data grouped by anatomical location
    grouped_data = defaultdict(dict)

    # Iterate over the data and group by anatomical location
    for mask_path, image_path in data.items():
        # Extract the anatomical location from the filename
        parts = mask_path.split('_')
        if len(parts) > 2:
            # Assuming the anatomical location is right before 'mask.png' and after a number
            location = parts[-2]
            grouped_data[location][mask_path] = image_path

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Write the grouped data into separate JSON files per anatomical location
    for location, mappings in grouped_data.items():
        output_file_path = os.path.join(output_directory, f'{location}_images.json')
        with open(output_file_path, 'w') as output_file:
            json.dump(mappings, output_file, indent=4)

    print(f"Created separate JSON files for each anatomical location in '{output_directory}'.")

# # Example usage:
# input_json = 'bk_rf_data/test/label2image_test.json'  # Make sure to update this path
# output_dir = 'location_json'  # Update this path too
# split_json_by_anatomical_location(input_json, output_dir)


def generate_image_to_mask_json(images_directory, masks_directory, output_json_path):
    # Initialize dictionary to store file mappings
    file_mappings = {}

    images_path = os.path.join(images_directory, "images")
    masks_path = os.path.join(masks_directory, "masks")

    # Iterate through each file in the images directory
    for image_filename in os.listdir(images_path):
        # Create the corresponding mask filename by adding "_mask" before the file extension
        mask_filename = f"{os.path.splitext(image_filename)[0]}_mask.png"

        # Construct full paths for both image and mask
        image_full_path = os.path.join(images_directory, "images", image_filename).replace('\\', '/')
        mask_full_path = os.path.join(masks_directory, "masks", mask_filename).replace('\\', '/')

        # Check if the mask file exists to ensure only valid mappings are included
        # if os.path.isfile(os.path.join(masks_path, mask_filename)):
        file_mappings[image_full_path] = [mask_full_path]

    # Save the file mappings to a JSON file
    with open(output_json_path, 'w') as json_file:
        json.dump(file_mappings, json_file, indent=4)

    print("Image to mask JSON mapping file created at:", output_json_path)

# Usage example
# output_json_path = "patient_split/rf_data/train/image2label_train.json"
# images_directory = "patient_split/rf_data/train/images"
# masks_directory = "patient_split/rf_data/train/masks"  # Assuming masks are within the same base train directory

# generate_image_to_mask_json(images_directory, masks_directory, output_json_path)


# ------------------------------------------------------------------
# update mask path after they have already been moved to new directory

# Load the original JSON file
json_file_path = "patient_split/rf_data/train/image2label_train.json"
with open(json_file_path, 'r') as file:
    data = json.load(file)

# Directory where the masks are now located
new_masks_directory = "patient_split/rf_data/train/masks"

# Update the mask paths in the dictionary
updated_data = {}
for image_path, mask_paths in data.items():
    # Update each mask path in the list (assuming there might be multiple masks per image)
    updated_mask_paths = [
        os.path.join(new_masks_directory, os.path.basename(mask_path)) 
        for mask_path in mask_paths
    ]
    # Update the path to use forward slashes (optional, depending on your OS and preferences)
    updated_mask_paths = [path.replace("\\", "/") for path in updated_mask_paths]
    updated_data[image_path] = updated_mask_paths

# Save the updated mappings to a new JSON file (or overwrite the old one if preferred)
updated_json_file_path = "patient_split/rf_data/train/image2label_train_updated.json"
with open(updated_json_file_path, 'w') as file:
    json.dump(updated_data, file, indent=4)

print(f"Updated JSON file saved to {updated_json_file_path}")


def generate_label2image_json(images_directory, new_masks_base_directory, output_json_path):
    """
    Generate json file with masks path as keys and image paths as values. Images should already be in "test" images directory.
    """
    os.makedirs(new_masks_base_directory, exist_ok=True)  # Ensure the directory exists
    
    # Initialize an empty dictionary for the mask-to-image mappings
    label_to_image_mapping = {}
    
    # Iterate through each image in the images directory to construct the mapping
    for image_filename in os.listdir(images_directory):
        if not image_filename.endswith(".png"):  # Adjust the condition based on your image file extensions
            continue  # Skip non-image files
        
        # Construct the mask filename by replacing the image suffix with "_mask.png"
        mask_filename = image_filename.replace(".png", "_mask.png")
        
        # Construct the full paths for the image and mask
        image_path = os.path.join(images_directory, image_filename)
        mask_path = os.path.join(new_masks_base_directory, mask_filename)
        
        # Update the mapping (mask path as key, image path as value)
        label_to_image_mapping[mask_path] = image_path
    

    # Save the mappings to the JSON file
    with open(output_json_path, 'w') as json_file:
        json.dump(label_to_image_mapping, json_file, indent=4)
    
    print(f"Label to image JSON file created at {output_json_path}")


def generate_image2label_with_predicted_masks(images_directory, masks_directory, output_json_path):
    """
    Generate a JSON file mapping image paths to their corresponding (yet to be created) mask paths.

    Args:
    - images_directory: Path to the directory containing images.
    - masks_directory: Path to the directory where masks will be stored.
    - output_json_path: Path to save the output JSON file.
    """
    file_mappings = {}

    # Ensure mask directory path is correctly formatted
    masks_directory = masks_directory.rstrip("/")

    # Iterate through each file in the images directory
    for image_filename in os.listdir(images_directory):
        image_path = os.path.join(images_directory, image_filename)

        # Check if the file is a regular file (not a directory) and it's an image
        if os.path.isfile(image_path) and (image_path.endswith('.png') or image_path.endswith('.jpg')):
            # Generate the corresponding mask filename and path
            mask_filename = f"{os.path.splitext(image_filename)[0]}_mask.png"
            mask_path = os.path.join(masks_directory, mask_filename)

            # Replace backslashes with forward slashes in paths for compatibility
            image_path_unix = image_path.replace('\\', '/')
            mask_path_unix = mask_path.replace('\\', '/')

            # Update file mappings with the generated mask path
            file_mappings[image_path_unix] = [mask_path_unix]
            print(f"Mapping added for image: {image_path_unix} -> {mask_path_unix}")

    # Save the file mappings to a JSON file
    with open(output_json_path, 'w') as json_file:
        json.dump(file_mappings, json_file, indent=4)

    print(f"JSON file created at {output_json_path}")

# Example usage
images_directory = "patient_split/rf_data/train/images"
masks_directory = "patient_split/rf_data/train/masks"
output_json_path = "patient_split/rf_data/train/image2label.json"

generate_image2label_with_predicted_masks(images_directory, masks_directory, output_json_path)



# generate_image2label(images_directory, masks_directory, output_json_path)


def modify_json_train(input_file, output_file):
    """
    For training json, create image2label_train.json.
    Reads a JSON file, modifies its keys by changing the path and replacing the beginning of the filename
    from 'RF' to 'RFv2', and writes the modified data to a new JSON file.

    Parameters:
    - input_file (str): Path to the input JSON file.
    - output_file (str): Path to the output JSON file where the modified data will be saved.
    """
    # Open and read the input JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    modified_data = {}
    for key, value in data.items():
        # Split the original key to replace specific parts
        parts = key.split('/')
        # Modify the specific part of the key according to the requirements
        # new_key_part = parts[-1].replace("RF_", "RFv2_")  # changing RF to RFv2
        new_key_part = parts[-1].replace("RFv2_", "RFv3_")  # changing RFv2 to RFv3
        # Reconstruct the key with the new path and filename
        # new_key = "RF_v2/" + new_key_part  # new RF_v2 directory
        new_key = "RFv3/" + new_key_part  # new RFv3 directory
        # Assign the modified key to the value in the new dictionary
        modified_data[new_key] = value
    
    # Write the modified data to the output JSON file
    with open(output_file, 'w') as f:
        json.dump(modified_data, f, indent=4)


def modify_json_values_and_path(input_file, output_file):
    """
    Modifies the image paths in the values of a JSON file. The modification involves setting a new base directory "RF_v2/"
    and replacing the beginning of the filename from 'RF' to 'RFv2', then saves the modified data to a new JSON file.

    Parameters:
    - input_file (str): Path to the input JSON file.
    - output_file (str): Path to the output JSON file where the modified data will be saved.
    """
    # Open and read the input JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    modified_data = {}
    # Iterate through the original dictionary
    for key, value in data.items():
        # Assuming value is a path like "patient_split/rf_data/val/images/RF_2015_LBM.png"
        # Split the path to get the filename
        filename = value.split('/')[-1]
        
        # Replace "RF_" with "RFv2_" in the filename
        new_filename = filename.replace("RF_", "RFv2_")
        
        # Construct the new value with the new base directory and modified filename
        new_value = "RF_v2/" + new_filename
        
        # Check if the original file exists before modification (optional as per your use case)
        if not os.path.exists(value):
            # If the original file does not exist, skip this key-value pair
            print(f"File does not exist, skipping: {value}")
            continue

        # Update the dictionary with the modified path
        modified_data[key] = new_value
    
    # Write the modified data to the output JSON file
    with open(output_file, 'w') as f:
        json.dump(modified_data, f, indent=4)



def main():
    # Create label2image json
    # img_dir = "patient_split/bm_data/test/images"
    # new_mask_dir = "patient_split/bm_data/test/masks"
    # output_json_path = "patient_split/bm_data/test/label2image_test.json"
    # generate_label2image_json(img_dir, new_mask_dir, output_json_path)
    # -----------------------------------------------------------
    # Create image2label json
    # images_directory = "patient_split/bm_data/train/images"
    # masks_directory = 'patient_split/bm_data/train/masks'
    # output_json_path = 'patient_split/bm_data/train/image2label_train.json'
    # generate_image2label_with_predicted_masks(images_directory, masks_directory, output_json_path)
    # -----------------------------------------------------------
    # Split by anatomical location
    input_json = 'patient_split/cross_val_rf/val2/test/label2image_test.json'  # Make sure to update this path
    output_dir = 'patient_split/cross_val_location'  # Update this path too
    split_json_by_anatomical_location(input_json, output_dir)
    # ------------------------------------------------------------
    # Modify keys in existing (e.g., RF or RFv2) train file
    # input_file = 'RF_v2/train/image2label_train.json'
    # output_file = 'RFv3/train/image2label_train.json'
    # modify_json_train(input_file, output_file)

    # ------------------------------------------------------------
    # Deleting keys image paths that don't exist, the "modify_json_values_and_path" function automatically checks beforehand, add this to above function
    # # Specify the path to your JSON file
    # json_file_path = 'RF_v2/train/image2label_train.json'

    # # Read the JSON data
    # with open(json_file_path, 'r') as file:
    #     data = json.load(file)

    # # Copy of the keys to iterate over, to avoid RuntimeError for changing dict size during iteration
    # keys_to_check = list(data.keys())

    # # Check each key (image path) to see if the file exists
    # for key in keys_to_check:
    #     if not os.path.exists(key):  # Check if the image file does not exist
    #         del data[key]  # Delete the key-value pair if the file does not exist

    # # Write the modified data back to the JSON file
    # with open(json_file_path, 'w') as file:
    #     json.dump(data, file, indent=4)  # 'indent=4' for pretty printing

    # print("JSON file has been updated.")

    # -------------------------------------------------------
    # # Modifying val and test json files
    # input_file = 'RF_v2/test/label2image_test.json'
    # output_file = 'RFv3/test/label2image_test.json'
    # modify_json_values_and_path(input_file, output_file)




if __name__ == '__main__':
    main()