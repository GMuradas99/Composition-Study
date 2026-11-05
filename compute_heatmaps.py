import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt

import visualizing_functions as visualizer

from tqdm import tqdm

output_dir = 'heatmaps'
shapes_path = 'parsed_data/shapes.csv'

all_df = pd.read_csv(shapes_path)

all_df.head()

conditions = list(all_df['condition'].unique())

nat_image_filenames = all_df.loc[
    all_df["image_type"] == "Nat", "image_filename"
].unique().tolist()

art_image_filenames = all_df.loc[
    all_df["image_type"] == "Art", "image_filename"
].unique().tolist()


# Create a directory to save the heatmaps if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if not os.path.exists(os.path.join(output_dir, 'Art')):
    os.makedirs(os.path.join(output_dir, 'Art'))

if not os.path.exists(os.path.join(output_dir, 'Nat')):
    os.makedirs(os.path.join(output_dir, 'Nat'))

for nat_filename in nat_image_filenames:
    if not os.path.exists(os.path.join(output_dir, 'Nat', nat_filename)):
        os.makedirs(os.path.join(output_dir, 'Nat', nat_filename))
        os.makedirs(os.path.join(output_dir, 'Nat', nat_filename, 'heatmaps'))
        os.makedirs(os.path.join(output_dir, 'Nat', nat_filename, 'raw_masks'))
        os.makedirs(os.path.join(output_dir, 'Nat', nat_filename, 'labeled_image'))

for art_filename in art_image_filenames:
    if not os.path.exists(os.path.join(output_dir, 'Art', art_filename)):
        os.makedirs(os.path.join(output_dir, 'Art', art_filename))
        os.makedirs(os.path.join(output_dir, 'Art', art_filename, 'heatmaps'))
        os.makedirs(os.path.join(output_dir, 'Art', art_filename, 'raw_masks'))
        os.makedirs(os.path.join(output_dir, 'Art', art_filename, 'labeled_image'))

for nat_filename in tqdm(nat_image_filenames):

    img = cv2.imread(visualizer.get_img_path(nat_filename))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    df_only_image = all_df.loc[all_df["image_filename"] == nat_filename]
    conditions = df_only_image["condition"].unique()

    for condition in conditions:
        df_condition = df_only_image.loc[df_only_image["condition"] == condition]

        labeled_image = visualizer.label_image(img.copy(), df_condition, line_thickness=2)
        raw_mask = visualizer.get_total_mask(img.copy(), df_condition)
        heatmap = visualizer.create_heatmap(img.copy(), raw_mask)

        # Save the labeled image, raw mask, and heatmap
        cv2.imwrite(
            os.path.join(output_dir, 'Nat', nat_filename, 'labeled_image', f"{condition}_labeled.png"),
            cv2.cvtColor(labeled_image, cv2.COLOR_RGB2BGR)
        )
        cv2.imwrite(
            os.path.join(output_dir, 'Nat', nat_filename, 'raw_masks', f"{condition}_mask.png"),
            cv2.cvtColor(raw_mask, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            os.path.join(output_dir, 'Nat', nat_filename, 'heatmaps', f"{condition}_heatmap.png"),
            cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR),
        )

for art_filename in tqdm(art_image_filenames):
    img = cv2.imread(visualizer.get_img_path(art_filename))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    df_only_image = all_df.loc[all_df["image_filename"] == art_filename]
    conditions = df_only_image["condition"].unique()

    for condition in conditions:
        df_condition = df_only_image.loc[df_only_image["condition"] == condition]

        labeled_image = visualizer.label_image(img.copy(), df_condition, line_thickness=2)
        raw_mask = visualizer.get_total_mask(img.copy(), df_condition)
        heatmap = visualizer.create_heatmap(img.copy(), raw_mask)

        # Save the labeled image, raw mask, and heatmap
        cv2.imwrite(
            os.path.join(output_dir, 'Art', art_filename, 'labeled_image', f"{condition}_labeled.png"),
            cv2.cvtColor(labeled_image, cv2.COLOR_RGB2BGR)
        )
        cv2.imwrite(
            os.path.join(output_dir, 'Art', art_filename, 'raw_masks', f"{condition}_mask.png"),
            cv2.cvtColor(raw_mask, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            os.path.join(output_dir, 'Art', art_filename, 'heatmaps', f"{condition}_heatmap.png"),
            cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR),
        )