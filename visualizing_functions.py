import os
import cv2
import ast
import numpy as np
import pandas as pd

from matplotlib.colors import to_rgb

def get_img_path(img_file, images_directory = 'images'):
    if os.path.exists(img_file):
        return img_file
    cand = os.path.join(images_directory, img_file)
    if os.path.exists(cand):
        return cand

    filename = os.path.basename(img_file)
    for folder in ['ART_SET', 'NAT_SET', '']:
        p = os.path.join(images_directory, folder, filename)
        if os.path.exists(p):
            return p

    for folder in ['ART_SET', 'NAT_SET']:
        dir_path = os.path.join(images_directory, folder)
        if os.path.exists(dir_path):
            for f in os.listdir(dir_path):
                if filename in f or f.endswith(filename):
                    return os.path.join(dir_path, f)

    if filename[:3] == 'ART':
        return os.path.join(images_directory, 'ART_SET', filename)
    else:
        return os.path.join(images_directory, 'NAT_SET', filename)

def transform_color(color) -> tuple:
    """
    Converts a color specification to an RGB tuple and optionally an alpha value.

    Args:
        color (str): Color specification. Supported formats:
            - Hex string (e.g., '#RRGGBB')
            - 'transparent'
            - 'rgba(r, g, b, a)'
            - 'rgb(r, g, b)'
            - Named color (e.g., 'red', 'green')

    Returns:
        tuple: ((R, G, B), alpha) where R, G, B are integers in [0, 255] and alpha is None or a float.
    """
    # Checks if color is a hex string
    if color.startswith('#'):
        # Convert hex to RGB
        color = color.lstrip('#')
        return (tuple(int(color[i:i + 2], 16) for i in (0, 2, 4)), None)
    elif color == 'transparent':
        return ((0, 0, 0), 0)  # Transparent color with zero opacity
    elif color.startswith('rgba'):
        # Extract RGBA values from rgba string
        rgba_values = color[5:-1].split(',')
        rgb = tuple(int(float(c.strip()) * 255) for c in rgba_values[:3])
        _ = float(rgba_values[3].strip())
        return (rgb, None)
    elif color.startswith('rgb'):
        # Extract RGB values from rgb string
        rgb_values = color[4:-1].split(',')
        rgb = tuple(int(float(c.strip()) * 255) for c in rgb_values)
        return (rgb, None)
    else:
        rgb = to_rgb(color)
        return (tuple(int(c * 255) for c in rgb), None)

def convert_to_numpy_array(points):
    # 1. If the entire input is one giant string, convert it to a list of dicts/tuples
    if isinstance(points, str):
        points = ast.literal_eval(points)
        
    # 2. If it's a list, but the items inside are strings instead of dicts, convert them
    elif isinstance(points, list) and len(points) > 0 and isinstance(points[0], str):
        points = [ast.literal_eval(p) for p in points]
        
    coords = []
    for point in points:
        if isinstance(point, dict) and 'x' in point and 'y' in point:
            coords.append([round(float(point['x'])), round(float(point['y']))])
        elif isinstance(point, (tuple, list)) and len(point) >= 2:
            coords.append([round(float(point[0])), round(float(point[1]))])
    return np.array(coords, dtype=np.int32)

def label_image(img: np.ndarray, df: pd.DataFrame, line_thickness: int = 5, fill: bool = True) -> np.ndarray:
    """
    Draws labeled polygons and polylines on an image according to annotation data in a DataFrame.

    Args:
        img (np.ndarray): The input image as a NumPy array (RGB).
        df (pd.DataFrame): DataFrame containing annotation data with columns:
            - 'type': 'polygon' or 'polyline'
            - 'point_coordinates': list of (x, y) tuples
            - 'color': line color (name, hex, or rgb/rgba string)
            - 'fill': fill color (name, hex, or rgb/rgba string)
            - 'opacity': fill opacity (float 0-1)
        line_thickness (int, optional): Thickness of the polygon/polyline outline. Default is 5.
        fill (bool, optional): Whether to fill polygons. Default is True.

    Returns:
        np.ndarray: The image with drawn labels as a NumPy array (RGB).
    """
    img_with_labels = img.copy()

    for _, row in df.iterrows():

        # Figure out color
        fill_color, fill_alpha = transform_color(row['fill'])
        if fill_alpha is None: fill_alpha = row['opacity']
        line_color, _ = transform_color(row['color'])

        # Polygon
        if row['type'] == 'polygon':
            # points = visualizer.convert_to_numpy_array(row['transformed_points'])
            points_list = [
                tuple(map(float, point))
                for point in __import__('ast').literal_eval(row['point_coordinates'])
            ]
            points = np.array(points_list, dtype=np.int32)
            overlay = img_with_labels.copy()
            if fill: cv2.fillPoly(overlay, [points], color=fill_color)
            cv2.addWeighted(overlay, fill_alpha, img_with_labels, 1 - fill_alpha, 0, img_with_labels)
            cv2.polylines(img_with_labels, [points], isClosed=True, color=line_color, thickness=line_thickness)
        # Polyline
        elif row['type'] == 'polyline':
            # points = convert_to_numpy_array(row['transformed_points'])
            points_list = [
                            tuple(map(float, point))
                            for point in __import__('ast').literal_eval(row['point_coordinates'])
                        ]
            points = np.array(points_list, dtype=np.int32)
            cv2.polylines(img_with_labels, [points], isClosed=False, color=line_color, thickness=line_thickness)

        else:
            raise ValueError(f"Unsupported type: {row['type']}")
        
    return img_with_labels

def get_total_mask(img: np.ndarray, df: pd.DataFrame, line_thickness: int = 5) -> np.ndarray:
    """
    Generates a heat mask for all polygons and polylines in the given DataFrame.

    Args:
        img (np.ndarray): The input image as a NumPy array.
        df (pd.DataFrame): DataFrame containing annotation data with columns:
            - 'type': 'polygon' or 'polyline'
            - 'point_coordinates': list of (x, y) tuples
        line_thickness (int, optional): Thickness of the polyline outline. Default is 5.

    Returns:
        np.ndarray: A 2D heat mask where each pixel value indicates the number of times it was covered by an annotation.
    """
    heat_mask = np.zeros(img.shape[:2], dtype=np.float32)

    for _, row in df.iterrows():
        # Polygon
        if row['type'] == 'polygon':
            points_list = [
                            tuple(map(float, point))
                            for point in __import__('ast').literal_eval(row['point_coordinates'])
                        ]
            points = np.array(points_list, dtype=np.int32)
            element_mask = np.zeros(img.shape[:2], dtype=np.float32)
            cv2.fillPoly(element_mask, [points], color=1)
            cv2.polylines(element_mask, [points], isClosed=True, color=1, thickness=line_thickness)

            heat_mask += element_mask

        # Polyline
        elif row['type'] == 'polyline':
            points_list = [
                                        tuple(map(float, point))
                                        for point in __import__('ast').literal_eval(row['point_coordinates'])
                                    ]
            points = np.array(points_list, dtype=np.int32)
            element_mask = np.zeros(img.shape[:2], dtype=np.float32)
            cv2.polylines(element_mask, [points], isClosed=False, color=1, thickness=line_thickness)

            heat_mask += element_mask

        else:
            raise ValueError(f"Unsupported type: {row['type']}")

    return heat_mask

def create_heatmap(img: np.ndarray, mask: np.ndarray, blur: int = 5) -> np.ndarray:
    """
    Creates a heatmap overlay on an image using a mask.

    Args:
        img (np.ndarray): The original image as a NumPy array (RGB).
        mask (np.ndarray): The mask array indicating coverage (e.g., output from get_total_mask).
        blur (int, optional): Standard deviation for Gaussian blur. Default is 5.

    Returns:
        np.ndarray: The image with the heatmap overlay as a NumPy array (RGB).
    """
    # Normalize the mask to range [0, 1]
    norm_mask = cv2.normalize(mask, None, 0, 1, cv2.NORM_MINMAX)

    # Applying gaussian blur to smooth the mask
    norm_mask = cv2.GaussianBlur(norm_mask, (0, 0), blur)

    # Apply colormap
    heatmap = cv2.applyColorMap((norm_mask * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # Convert heatmap to RGB
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay heatmap on the original image
    overlay = cv2.addWeighted(img, 0.3, heatmap_rgb, 0.7, 0)

    return overlay


def plot_annotation_summary(
    image_filename: str,
    df_shapes: pd.DataFrame,
    base_dir: str = "images",
    condition: str = None,
    figsize: tuple = (15, 5),
):
    """
    Displays a 1x3 side-by-side visualization:
    [Original Image] | [All Labeled Annotations] | [Heatmap Density Overlay]
    """
    import matplotlib.pyplot as plt

    img_path = get_img_path(image_filename, images_directory=base_dir)
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_filename}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    filtered = df_shapes[df_shapes["image_filename"] == os.path.basename(image_filename)]
    if condition:
        filtered = filtered[filtered["condition"] == condition]

    labelled = label_image(img, filtered, fill=True)
    mask = get_total_mask(img, filtered)
    heatmap = create_heatmap(img, mask)

    title_suffix = f" ({condition})" if condition else ""

    fig, axs = plt.subplots(1, 3, figsize=figsize)
    axs[0].imshow(img)
    axs[0].set_title(f"Original: {os.path.basename(image_filename)}")
    axs[0].axis("off")

    axs[1].imshow(labelled)
    axs[1].set_title(f"Annotations{title_suffix} (N={len(filtered)} shapes)")
    axs[1].axis("off")

    axs[2].imshow(heatmap)
    axs[2].set_title(f"Heatmap Density{title_suffix}")
    axs[2].axis("off")

    plt.tight_layout()
    return fig


def compare_conditions_heatmap(
    image_filename: str,
    df_shapes: pd.DataFrame,
    condition_a: str,
    condition_b: str,
    base_dir: str = "images",
    blur: int = 15,
    figsize: tuple = (16, 5),
):
    """
    Compares how participants annotated an image under two different compositional conditions
    (e.g., Balance vs Segregating).

    Displays side-by-side:
    [Original Image] | [Condition A Heatmap] | [Condition B Heatmap]
    """
    import matplotlib.pyplot as plt

    img_path = get_img_path(image_filename, images_directory=base_dir)
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_filename}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    base_name = os.path.basename(image_filename)
    img_shapes = df_shapes[df_shapes["image_filename"] == base_name]
    shapes_a = img_shapes[img_shapes["condition"] == condition_a]
    shapes_b = img_shapes[img_shapes["condition"] == condition_b]

    mask_a = get_total_mask(img, shapes_a)
    mask_b = get_total_mask(img, shapes_b)

    heatmap_a = create_heatmap(img, mask_a, blur=blur)
    heatmap_b = create_heatmap(img, mask_b, blur=blur)

    fig, axs = plt.subplots(1, 3, figsize=figsize)
    axs[0].imshow(img)
    axs[0].set_title(f"Original: {base_name}")
    axs[0].axis("off")

    axs[1].imshow(heatmap_a)
    axs[1].set_title(f"{condition_a} (N={len(shapes_a)} shapes)")
    axs[1].axis("off")

    axs[2].imshow(heatmap_b)
    axs[2].set_title(f"{condition_b} (N={len(shapes_b)} shapes)")
    axs[2].axis("off")

    plt.tight_layout()
    return fig

