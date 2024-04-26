import os
import json

def create_json(directory_path, json_path):
    data = {}

    for filename in os.listdir(directory_path):
        if filename.endswith('_mask.png'):
            image_name = filename
            boxes = [[0, 30, 160, 179]]
            point_coords = [[0, 0]]
            point_labels = [1]

            data[image_name] = {
                "boxes": boxes,
                "point_coords": point_coords,
                "point_labels": point_labels
            }

    with open(json_path, 'w') as json_file:
        json.dump(data, json_file, indent=2)


def modify_prompt_file(original_file_path, updated_file_path):
    # Read the original JSON file
    with open(original_file_path, 'r') as file:
        original_dict = json.load(file)

    # Create a new dictionary with updated keys
    updated_dict = {}
    for old_key, value in original_dict.items():
        # Replace 'RF' with 'BM' in the key
        new_key = old_key.replace('RF', 'BM')
        updated_dict[new_key] = value

    # Write the updated dictionary to a new JSON file
    with open(updated_file_path, 'w') as file:
        json.dump(updated_dict, file, indent=2)

    print(f"Dictionary updated and saved to '{updated_file_path}'")


def modify_prompt_file2(original_file_path, updated_file_path):
    # Read the original JSON file
    with open(original_file_path, 'r') as file:
        original_dict = json.load(file)

    # Create a new dictionary with updated values
    updated_dict = {}
    for key, value in original_dict.items():
        # Update the "point_coords" value
        value["point_coords"] = [[250, 127]]

        # Update the "point_labels" value
        value["point_labels"] = [0]

        # Add the updated entry to the new dictionary
        updated_dict[key] = value

    # Write the updated dictionary to a new JSON file
    with open(updated_file_path, 'w') as file:
        json.dump(updated_dict, file, indent=2)



if __name__ == '__main__':
    directory_path = 'patient_split/rf_data/train/masks'  # Update with the path to your masks directory
    json_path = 'train_prompts_rf.json'  # Update with your desired JSON file path
    create_json(directory_path, json_path)
    # modify_prompt_file(original_file_path="output.json", updated_file_path="BM_promptpath.json")
    # modify_prompt_file2('output.json', "RF_nopt_prompt.json")
    