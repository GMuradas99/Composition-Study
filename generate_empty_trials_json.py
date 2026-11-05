import os
import json
import pandas as pd

COMBINATIONS_PATH = 'combinations'
OUTPUT_PATH = 'trials.js'

combinations_df = pd.read_csv(os.path.join(COMBINATIONS_PATH, 'all_combinations.csv'))
art_images = pd.read_csv(os.path.join(COMBINATIONS_PATH, 'ART_image_names.csv'))
nat_images = pd.read_csv(os.path.join(COMBINATIONS_PATH, 'NAT_image_names.csv'))

json_dict = {
    'Art': {'Num_Participants': 0, 'Trial_Participants': {}, 'Image_Participants': {}},
    'Nat': {'Num_Participants': 0, 'Trial_Participants': {}, 'Image_Participants': {}},
}

for _, row in combinations_df.iterrows():
    json_dict['Art']['Trial_Participants'][row['Acronym']] = 0
    json_dict['Nat']['Trial_Participants'][row['Acronym']] = 0

for _, row in art_images.iterrows():
    json_dict['Art']['Image_Participants'][row['image_name']] = {}

    json_dict['Art']['Image_Participants'][row['image_name']]['Num_Participants'] = 0
    for _, row2 in combinations_df.iterrows():
        json_dict['Art']['Image_Participants'][row['image_name']][row2['Acronym']] = 0

for _, row in nat_images.iterrows():
    json_dict['Nat']['Image_Participants'][row['image_name']] = {}

    json_dict['Nat']['Image_Participants'][row['image_name']]['Num_Participants'] = 0
    for _, row2 in combinations_df.iterrows():
        json_dict['Nat']['Image_Participants'][row['image_name']][row2['Acronym']] = 0

def save_dict_to_js(data: dict, file_path: str, var_name: str = "trials") -> None:
    """Saves a nested dictionary to a JavaScript file as a named export."""
    json_str = json.dumps(data, indent=4)
    js_content = f"export const {var_name} = {json_str};\n"

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(js_content)

save_dict_to_js(json_dict, OUTPUT_PATH, var_name="trials")