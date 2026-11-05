import pandas as pd

# Load the main dataset and the two mapping files
df = pd.read_csv("parsed_data_data_3/shapes.csv")
art_df = pd.read_csv("combinations/ART_image_names.csv")
nat_df = pd.read_csv("combinations/NAT_image_names.csv")

# Create a dictionary to map the original filenames to their new prefixed names
filename_mapping = {}

# Process ART names (e.g., 'ART01_AS_0001.tif' -> key: 'AS_0001.tif', value: 'ART01_AS_0001.tif')
for new_name in art_df['image_name']:
    # Split only on the first underscore
    parts = new_name.split('_', 1) 
    if len(parts) == 2:
        original_name = parts[1]
        filename_mapping[original_name] = new_name

# Process NAT names in the exact same way
for new_name in nat_df['image_name']:
    parts = new_name.split('_', 1)
    if len(parts) == 2:
        original_name = parts[1]
        filename_mapping[original_name] = new_name

# Apply the mapping to the 'image_filename' column
# If a filename isn't found in the mapping, it keeps its original name
df['image_filename'] = df['image_filename'].map(filename_mapping).fillna(df['image_filename'])

# Save the altered data to a new CSV file
df.to_csv("parsed_data_data_3/shapes_altered.csv", index=False)

print("File successfully updated with numbered prefixes and saved as shapes_altered_numbered.csv")