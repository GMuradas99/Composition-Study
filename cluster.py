import os
import cv2
import pandas as pd

from tqdm import tqdm

import cluster_functions as cluster
import visualizing_functions as visualizer

output_dir = 'clustered_labels'
shapes_path = 'parsed_data/shapes.csv'

all_df = pd.read_csv(shapes_path)

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
        os.makedirs(os.path.join(output_dir, 'Nat', nat_filename, 'cluster_visuals'))
        os.makedirs(os.path.join(output_dir, 'Nat', nat_filename, 'clustered_labels'))

for art_filename in art_image_filenames:
    if not os.path.exists(os.path.join(output_dir, 'Art', art_filename)):
        os.makedirs(os.path.join(output_dir, 'Art', art_filename))
        os.makedirs(os.path.join(output_dir, 'Art', art_filename, 'cluster_visuals'))
        os.makedirs(os.path.join(output_dir, 'Art', art_filename, 'clustered_labels'))


for nat_filename in tqdm(nat_image_filenames):

    type_img = 'Nat'
    img_filename = nat_filename

    img = cv2.imread(visualizer.get_img_path(img_filename))

    # Repetition
    filtered_df = all_df[
        (all_df["image_filename"] == img_filename)
        & (all_df["condition"] == "Repetition")
    ]
    representative_indices, cluster_centers, clustered_df = cluster.cluster_shapes_DBSCAN(filtered_df, eps=75, min_samples=3)
    cluster.visualize_clusters(img.copy(), clustered_df, representative_indices, cluster_centers, 
                               save_path=os.path.join(output_dir, type_img, img_filename, 'cluster_visuals', 'Repetition.png'))

    representative_indices_df = pd.DataFrame(representative_indices, columns=['cluster', 'representative_index'])
    representative_indices_df.to_csv(os.path.join(output_dir, type_img, img_filename, 'clustered_labels', 'Repetition_representative_indices.csv'), index=False)
    clustered_df.reset_index(drop=True, inplace=True)
    clustered_df.to_csv(os.path.join(output_dir, type_img, img_filename, 'clustered_labels', 'Repetition_clustered_labels.csv'), index=True)

    # Emphasis
    filtered_df = all_df[
        (all_df["image_filename"] == img_filename)
        & (all_df["condition"] == "Emphasis")
    ]
    representative_indices, cluster_centers, clustered_df = cluster.cluster_shapes_DBSCAN(filtered_df, eps=75, min_samples=3)
    cluster.visualize_clusters(img.copy(), clustered_df, representative_indices, cluster_centers, 
                               save_path=os.path.join(output_dir, type_img, img_filename, 'cluster_visuals', 'Emphasis.png'))
    representative_indices_df = pd.DataFrame(representative_indices, columns=['cluster', 'representative_index'])
    representative_indices_df.to_csv(os.path.join(output_dir, type_img, img_filename, 'clustered_labels', 'Emphasis_representative_indices.csv'), index=False)
    clustered_df.reset_index(drop=True, inplace=True)
    clustered_df.to_csv(os.path.join(output_dir, type_img, img_filename, 'clustered_labels', 'Emphasis_clustered_labels.csv'), index=True)

    # Clustering
    filtered_df = all_df[
            (all_df["image_filename"] == img_filename)
            & (all_df["condition"] == "Clustering")
    ]
    representative_indices, cluster_centers, clustered_df = cluster.cluster_shapes_DBSCAN(filtered_df, eps=75, min_samples=3)
    cluster.visualize_clusters(img.copy(), clustered_df, representative_indices, cluster_centers, 
                                save_path=os.path.join(output_dir, type_img, img_filename, 'cluster_visuals', 'Clustering.png'))
    representative_indices_df = pd.DataFrame(representative_indices, columns=['cluster', 'representative_index'])
    representative_indices_df.to_csv(os.path.join(output_dir, type_img, img_filename, 'clustered_labels', 'Clustering_representative_indices.csv'), index=False)
    clustered_df.reset_index(drop=True, inplace=True)
    clustered_df.to_csv(os.path.join(output_dir, type_img, img_filename, 'clustered_labels', 'Clustering_clustered_labels.csv'), index=True)

    # Configuring
    filtered_df = all_df[
            (all_df["image_filename"] == img_filename)
            & (all_df["condition"] == "Configuring")
    ]
    representative_indices, cluster_centers, clustered_df = cluster.cluster_shapes_DBSCAN(filtered_df, eps=75, min_samples=3)
    cluster.visualize_clusters(img.copy(), clustered_df, representative_indices, cluster_centers,
                                save_path=os.path.join(output_dir, type_img, img_filename, 'cluster_visuals', 'Configuring.png'))
    representative_indices_df = pd.DataFrame(representative_indices, columns=['cluster', 'representative_index'])
    representative_indices_df.to_csv(os.path.join(output_dir, type_img, img_filename, 'clustered_labels', 'Configuring_representative_indices.csv'), index=False)
    clustered_df.reset_index(drop=True, inplace=True)
    clustered_df.to_csv(os.path.join(output_dir, type_img, img_filename, 'clustered_labels', 'Configuring_clustered_labels.csv'), index=True)

    # Linking
    filtered_df = all_df[
            (all_df["image_filename"] == img_filename)
            & (all_df["condition"] == "Linking")
    ]
    representative_indices, cluster_centers, clustered_df = cluster.cluster_lines_DBSCAN(filtered_df, eps=75, min_samples=3)
    cluster.visualize_lines_clusters(img.copy(), clustered_df, representative_indices, cluster_centers,
                                save_path=os.path.join(output_dir, type_img, img_filename, 'cluster_visuals', 'Linking.png'))
    representative_indices_df = pd.DataFrame(representative_indices, columns=['cluster', 'representative_index'])
    representative_indices_df.to_csv(os.path.join(output_dir, type_img, img_filename, 'clustered_labels', 'Linking_representative_indices.csv'), index=False)
    clustered_df.reset_index(drop=True, inplace=True)
    clustered_df.to_csv(os.path.join(output_dir, type_img, img_filename, 'clustered_labels', 'Linking_clustered_labels.csv'), index=True)

    # Layering
    filtered_df = all_df[
            (all_df["image_filename"] == img_filename)
            & (all_df["condition"] == "Layering")
    ]
    filtered_df_red = filtered_df[filtered_df['fill'] == 'rgb(255, 0, 0)']
    filtered_df_blue = filtered_df[filtered_df['fill'] == 'rgb(41, 134, 204)']
    representative_indices_red, cluster_centers_red, clustered_df_red = cluster.cluster_shapes_DBSCAN(filtered_df_red, eps=75, min_samples=3)
    representative_indices_blue, cluster_centers_blue, clustered_df_blue = cluster.cluster_shapes_DBSCAN(filtered_df_blue, eps=75, min_samples=3)
    cluster.visualize_combined_clusters(img.copy(),
                                        clustered_df_red, representative_indices_red, cluster_centers_red,
                                        clustered_df_blue, representative_indices_blue, cluster_centers_blue,
                                        save_path=os.path.join(output_dir, type_img, img_filename, 'cluster_visuals', 'Layering.png'))
    representative_indices_red_df = pd.DataFrame(representative_indices_red, columns=['cluster', 'representative_index'])
    representative_indices_red_df.to_csv(os.path.join(output_dir, type_img, img_filename, 'clustered_labels', 'Layering_red_representative_indices.csv'), index=False)
    representative_indices_blue_df = pd.DataFrame(representative_indices_blue, columns=['cluster', 'representative_index'])
    representative_indices_blue_df.to_csv(os.path.join(output_dir, type_img, img_filename, 'clustered_labels', 'Layering_blue_representative_indices.csv'), index=False)
    clustered_df_red.reset_index(drop=True, inplace=True)
    clustered_df_red.to_csv(os.path.join(output_dir, type_img, img_filename, 'clustered_labels', 'Layering_red_clustered_labels.csv'), index=True)
    clustered_df_blue.reset_index(drop=True, inplace=True)
    clustered_df_blue.to_csv(os.path.join(output_dir, type_img, img_filename, 'clustered_labels', 'Layering_blue_clustered_labels.csv'), index=True)

    # Balance
    filtered_df = all_df[
            (all_df["image_filename"] == img_filename)
            & (all_df["condition"] == "Balance")
    ]
    rep_indices_pairs, centers_pairs, clustered_pairs_df = cluster.extract_and_cluster_pairs(filtered_df, eps=100, min_samples=2)
    cluster.visualize_all_segment_pairs(img, filtered_df, clustered_pairs_df, rep_indices_pairs,
                                         save_path=os.path.join(output_dir, type_img, img_filename, 'cluster_visuals', 'Balance.png'))
    rep_indices_pairs_df = pd.DataFrame(rep_indices_pairs, columns=['cluster', 'representative_index'])
    rep_indices_pairs_df.to_csv(os.path.join(output_dir, type_img, img_filename, 'clustered_labels', 'Balance_representative_indices.csv'), index=False)
    clustered_pairs_df.reset_index(drop=True, inplace=True)
    clustered_pairs_df.to_csv(os.path.join(output_dir, type_img, img_filename, 'clustered_labels', 'Balance_clustered_labels.csv'), index=True)

for art_filename in tqdm(art_image_filenames):

    type_img = 'Art'
    img_filename = art_filename

    img = cv2.imread(visualizer.get_img_path(img_filename))

    # Repetition
    filtered_df = all_df[
        (all_df["image_filename"] == img_filename)
        & (all_df["condition"] == "Repetition")
    ]
    representative_indices, cluster_centers, clustered_df = cluster.cluster_shapes_DBSCAN(filtered_df, eps=75, min_samples=3)
    cluster.visualize_clusters(img.copy(), clustered_df, representative_indices, cluster_centers, 
                               save_path=os.path.join(output_dir, type_img, img_filename, 'cluster_visuals', 'Repetition.png'))

    representative_indices_df = pd.DataFrame(representative_indices, columns=['cluster', 'representative_index'])
    representative_indices_df.to_csv(os.path.join(output_dir, type_img, img_filename, 'clustered_labels', 'Repetition_representative_indices.csv'), index=False)
    clustered_df.reset_index(drop=True, inplace=True)
    clustered_df.to_csv(os.path.join(output_dir, type_img, img_filename, 'clustered_labels', 'Repetition_clustered_labels.csv'), index=True)

    # Emphasis
    filtered_df = all_df[
        (all_df["image_filename"] == img_filename)
        & (all_df["condition"] == "Emphasis")
    ]
    representative_indices, cluster_centers, clustered_df = cluster.cluster_shapes_DBSCAN(filtered_df, eps=75, min_samples=3)
    cluster.visualize_clusters(img.copy(), clustered_df, representative_indices, cluster_centers, 
                               save_path=os.path.join(output_dir, type_img, img_filename, 'cluster_visuals', 'Emphasis.png'))
    representative_indices_df = pd.DataFrame(representative_indices, columns=['cluster', 'representative_index'])
    representative_indices_df.to_csv(os.path.join(output_dir, type_img, img_filename, 'clustered_labels', 'Emphasis_representative_indices.csv'), index=False)
    clustered_df.reset_index(drop=True, inplace=True)
    clustered_df.to_csv(os.path.join(output_dir, type_img, img_filename, 'clustered_labels', 'Emphasis_clustered_labels.csv'), index=True)

    # Clustering
    filtered_df = all_df[
            (all_df["image_filename"] == img_filename)
            & (all_df["condition"] == "Clustering")
    ]
    representative_indices, cluster_centers, clustered_df = cluster.cluster_shapes_DBSCAN(filtered_df, eps=75, min_samples=3)
    cluster.visualize_clusters(img.copy(), clustered_df, representative_indices, cluster_centers, 
                                save_path=os.path.join(output_dir, type_img, img_filename, 'cluster_visuals', 'Clustering.png'))
    representative_indices_df = pd.DataFrame(representative_indices, columns=['cluster', 'representative_index'])
    representative_indices_df.to_csv(os.path.join(output_dir, type_img, img_filename, 'clustered_labels', 'Clustering_representative_indices.csv'), index=False)
    clustered_df.reset_index(drop=True, inplace=True)
    clustered_df.to_csv(os.path.join(output_dir, type_img, img_filename, 'clustered_labels', 'Clustering_clustered_labels.csv'), index=True)

    # Configuring
    filtered_df = all_df[
            (all_df["image_filename"] == img_filename)
            & (all_df["condition"] == "Configuring")
    ]
    representative_indices, cluster_centers, clustered_df = cluster.cluster_shapes_DBSCAN(filtered_df, eps=75, min_samples=3)
    cluster.visualize_clusters(img.copy(), clustered_df, representative_indices, cluster_centers,
                                save_path=os.path.join(output_dir, type_img, img_filename, 'cluster_visuals', 'Configuring.png'))
    representative_indices_df = pd.DataFrame(representative_indices, columns=['cluster', 'representative_index'])
    representative_indices_df.to_csv(os.path.join(output_dir, type_img, img_filename, 'clustered_labels', 'Configuring_representative_indices.csv'), index=False)
    clustered_df.reset_index(drop=True, inplace=True)
    clustered_df.to_csv(os.path.join(output_dir, type_img, img_filename, 'clustered_labels', 'Configuring_clustered_labels.csv'), index=True)

    # Linking
    filtered_df = all_df[
            (all_df["image_filename"] == img_filename)
            & (all_df["condition"] == "Linking")
    ]
    representative_indices, cluster_centers, clustered_df = cluster.cluster_lines_DBSCAN(filtered_df, eps=75, min_samples=3)
    cluster.visualize_lines_clusters(img.copy(), clustered_df, representative_indices, cluster_centers,
                                save_path=os.path.join(output_dir, type_img, img_filename, 'cluster_visuals', 'Linking.png'))
    representative_indices_df = pd.DataFrame(representative_indices, columns=['cluster', 'representative_index'])
    representative_indices_df.to_csv(os.path.join(output_dir, type_img, img_filename, 'clustered_labels', 'Linking_representative_indices.csv'), index=False)
    clustered_df.reset_index(drop=True, inplace=True)
    clustered_df.to_csv(os.path.join(output_dir, type_img, img_filename, 'clustered_labels', 'Linking_clustered_labels.csv'), index=True)

    # Layering
    filtered_df = all_df[
            (all_df["image_filename"] == img_filename)
            & (all_df["condition"] == "Layering")
    ]
    filtered_df_red = filtered_df[filtered_df['fill'] == 'rgb(255, 0, 0)']
    filtered_df_blue = filtered_df[filtered_df['fill'] == 'rgb(41, 134, 204)']
    representative_indices_red, cluster_centers_red, clustered_df_red = cluster.cluster_shapes_DBSCAN(filtered_df_red, eps=75, min_samples=3)
    representative_indices_blue, cluster_centers_blue, clustered_df_blue = cluster.cluster_shapes_DBSCAN(filtered_df_blue, eps=75, min_samples=3)
    cluster.visualize_combined_clusters(img.copy(),
                                        clustered_df_red, representative_indices_red, cluster_centers_red,
                                        clustered_df_blue, representative_indices_blue, cluster_centers_blue,
                                        save_path=os.path.join(output_dir, type_img, img_filename, 'cluster_visuals', 'Layering.png'))
    representative_indices_red_df = pd.DataFrame(representative_indices_red, columns=['cluster', 'representative_index'])
    representative_indices_red_df.to_csv(os.path.join(output_dir, type_img, img_filename, 'clustered_labels', 'Layering_red_representative_indices.csv'), index=False)
    representative_indices_blue_df = pd.DataFrame(representative_indices_blue, columns=['cluster', 'representative_index'])
    representative_indices_blue_df.to_csv(os.path.join(output_dir, type_img, img_filename, 'clustered_labels', 'Layering_blue_representative_indices.csv'), index=False)
    clustered_df_red.reset_index(drop=True, inplace=True)
    clustered_df_red.to_csv(os.path.join(output_dir, type_img, img_filename, 'clustered_labels', 'Layering_red_clustered_labels.csv'), index=True)
    clustered_df_blue.reset_index(drop=True, inplace=True)
    clustered_df_blue.to_csv(os.path.join(output_dir, type_img, img_filename, 'clustered_labels', 'Layering_blue_clustered_labels.csv'), index=True)

    # Balance
    filtered_df = all_df[
            (all_df["image_filename"] == img_filename)
            & (all_df["condition"] == "Balance")
    ]
    rep_indices_pairs, centers_pairs, clustered_pairs_df = cluster.extract_and_cluster_pairs(filtered_df, eps=100, min_samples=2)
    cluster.visualize_all_segment_pairs(img, filtered_df, clustered_pairs_df, rep_indices_pairs,
                                         save_path=os.path.join(output_dir, type_img, img_filename, 'cluster_visuals', 'Balance.png'))
    rep_indices_pairs_df = pd.DataFrame(rep_indices_pairs, columns=['cluster', 'representative_index'])
    rep_indices_pairs_df.to_csv(os.path.join(output_dir, type_img, img_filename, 'clustered_labels', 'Balance_representative_indices.csv'), index=False)
    clustered_pairs_df.reset_index(drop=True, inplace=True)
    clustered_pairs_df.to_csv(os.path.join(output_dir, type_img, img_filename, 'clustered_labels', 'Balance_clustered_labels.csv'), index=True)