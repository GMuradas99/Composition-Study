"""
data_parser.py
==============
Parser for jsPsych experimental data collected in the Composition Study.

Extracts tidy, analysis-ready DataFrames for:
- Participants (demographics, EMS, completion status, durations)
- Annotations (trial-level image annotations, RT, canvas properties, SVG)
- Shapes (granular shape-level coordinates, bounding boxes, polygon vertices)
- Aesthetic Ratings (image rating 1-7, RT)

Also provides synchronization back to `trials.js` for dynamic trial selection in `choose_trial.js`.
"""

import os
import glob
import re
import json
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ExperimentDataParser:
    """Parser for experiment CSV files located in collected_data/data/."""

    def __init__(self, combinations_path: str = "combinations"):
        self.combinations_path = combinations_path
        self._load_combinations()

    def _load_combinations(self):
        """Loads combination mappings (e.g., BC -> Balance x Clustering)."""
        comb_file = os.path.join(self.combinations_path, "all_combinations.csv")
        if os.path.exists(comb_file):
            self.combinations_df = pd.read_csv(comb_file)
            self.acronym_to_categories = {
                row["Acronym"]: (row["Category_A"], row["Category_B"], row["Combination"])
                for _, row in self.combinations_df.iterrows()
            }
        else:
            self.combinations_df = None
            self.acronym_to_categories = {}

    @staticmethod
    def extract_session_id(filepath: str) -> str:
        """Extracts timestamp session ID from filename."""
        basename = os.path.basename(filepath)
        match = re.search(r"PARTICIPANT_SESSION_(.+?)\.csv", basename)
        if match:
            return match.group(1)
        return Path(filepath).stem

    def parse_single_file(self, filepath: str) -> Dict[str, Any]:
        """
        Parses a single session CSV into structured data components.

        Returns a dict:
        {
            'participant': dict or None,
            'annotations': list of dicts,
            'shapes': list of dicts,
            'ratings': list of dicts,
            'is_completed': bool
        }
        """
        session_id = self.extract_session_id(filepath)
        df = pd.read_csv(filepath, low_memory=False)

        # 1. Determine participant ID
        participant_id = None
        if "participant_id" in df.columns:
            valid_pids = df["participant_id"].dropna()
            if len(valid_pids) > 0:
                participant_id = str(valid_pids.iloc[0])

        if not participant_id:
            participant_id = f"P_{session_id}"

        # 2. Basic metadata
        image_type = None
        if "image_type" in df.columns:
            valid_types = df["image_type"].dropna()
            if len(valid_types) > 0:
                image_type = str(valid_types.iloc[0])
        if not image_type and "image" in df.columns:
            imgs = df["image"].dropna().tolist()
            if any("NAT" in img for img in imgs):
                image_type = "Nat"
            elif any("ART" in img for img in imgs) or any("paintings" in img for img in imgs):
                image_type = "Art"

        trial_type = None
        if "trial_type" in df.columns:
            # Check if any value is a recognized acronym
            for val in df["trial_type"].dropna():
                val_str = str(val).strip()
                if val_str in self.acronym_to_categories:
                    trial_type = val_str
                    break
            # If no acronym matched, check if it's not a generic jsPsych plugin
            if trial_type is None:
                valid_vals = df["trial_type"].dropna()
                first_val = str(valid_vals.iloc[0]) if len(valid_vals) > 0 else None
                jspsych_types = {
                    "preload", "pavlovia", "external-html", "fullscreen",
                    "survey-html-form", "instructions", "fiat-trial",
                    "html-button-response", "survey-likert"
                }
                if first_val and first_val.lower() not in jspsych_types:
                    trial_type = first_val
                elif "condition" in df.columns and df["condition"].dropna().nunique() > 2:
                    trial_type = "Pilot_All"

        # Selected images array if present
        selected_images = []
        if "selected_images" in df.columns:
            for s in df["selected_images"].dropna():
                try:
                    loaded = json.loads(s)
                    if isinstance(loaded, list) and loaded:
                        selected_images = loaded
                        break
                except Exception:
                    pass

        # 3. Demographics and Survey (EMS)
        demographics = {}
        ems_score = None
        for _, row in df.iterrows():
            resp_val = row.get("response")
            if pd.notna(resp_val) and isinstance(resp_val, str) and resp_val.startswith("{"):
                try:
                    resp_dict = json.loads(resp_val)
                    if "age" in resp_dict:
                        demographics = resp_dict
                    if "EMS" in resp_dict:
                        ems_score = resp_dict["EMS"]
                except Exception:
                    pass

        # 4. Image Annotations & Shapes
        annotations_list = []
        shapes_list = []

        # Find annotation rows (rows with 'image' and 'condition' populated)
        if "image" in df.columns and "condition" in df.columns:
            anno_rows = df[df["image"].notna() & df["condition"].notna()]
            for _, row in anno_rows.iterrows():
                image_path = str(row["image"])
                image_filename = os.path.basename(image_path)
                condition = str(row["condition"])
                rt = row.get("rt")
                trial_index = row.get("trial_index")
                internal_node_id = row.get("internal_node_id")
                svg_content = row.get("svg") if pd.notna(row.get("svg")) else None
                json_content = row.get("json") if pd.notna(row.get("json")) else None

                # Parse object_coordinates JSON
                obj_coords_raw = row.get("object_coordinates")
                num_objects = 0
                canvas_width = None
                canvas_height = None
                parsed_coords = None

                if pd.notna(obj_coords_raw):
                    try:
                        parsed_coords = json.loads(obj_coords_raw)
                        # Extract canvas dims
                        img_data = parsed_coords.get("image_data", {})
                        if "tr" in img_data:
                            canvas_width = img_data["tr"].get("x")
                        if "bl" in img_data:
                            canvas_height = img_data["bl"].get("y")

                        # Extract shapes
                        shape_objects = parsed_coords.get("object_coordinates", [])
                        num_objects = len(shape_objects)

                        for shape_idx, shape in enumerate(shape_objects):
                            a_coords = shape.get("aCoords", {})
                            tl = a_coords.get("tl", {})
                            br = a_coords.get("br", {})

                            # Extract point_coordinates as [(x, y), ...]
                            raw_pts = shape.get("vertices") or shape.get("points") or []
                            coords_list = [
                                (p["x"], p["y"])
                                for p in raw_pts
                                if isinstance(p, dict) and "x" in p and "y" in p
                            ]

                            shape_info = {
                                "session_id": session_id,
                                "participant_id": participant_id,
                                "image_filename": image_filename,
                                "image_path": image_path,
                                "canvas_width": canvas_width,
                                "canvas_height": canvas_height,
                                "image_type": image_type,
                                "trial_type": trial_type,
                                "condition": condition,
                                "trial_index": trial_index,
                                "shape_index": shape_idx,
                                "figure": shape.get("figure"),
                                "type": shape.get("type"),
                                "label": shape.get("label"),
                                "color": shape.get("color"),
                                "fill": shape.get("fill"),
                                "opacity": shape.get("opacity"),
                                "bbox_tl_x": tl.get("x"),
                                "bbox_tl_y": tl.get("y"),
                                "bbox_br_x": br.get("x"),
                                "bbox_br_y": br.get("y"),
                                "point_coordinates": coords_list,
                                "points": shape.get("points"),
                                "transformed_points": shape.get("transformed_points"),
                                "vertices": shape.get("vertices"),
                            }
                            shapes_list.append(shape_info)
                    except Exception as e:
                        logger.warning(f"Error parsing object_coordinates in session {session_id}, img {image_filename}: {e}")

                anno_info = {
                    "session_id": session_id,
                    "participant_id": participant_id,
                    "image_type": image_type,
                    "trial_type": trial_type,
                    "condition": condition,
                    "trial_index": trial_index,
                    "internal_node_id": internal_node_id,
                    "image_filename": image_filename,
                    "image_path": image_path,
                    "rt": float(rt) if pd.notna(rt) else None,
                    "num_objects": num_objects,
                    "canvas_width": canvas_width,
                    "canvas_height": canvas_height,
                    "raw_object_coordinates": obj_coords_raw if pd.notna(obj_coords_raw) else None,
                    "svg": svg_content,
                    "json": json_content,
                }
                annotations_list.append(anno_info)

        # 5. Aesthetic Ratings
        ratings_list = []
        if "task" in df.columns and "rating" in df.columns:
            rating_rows = df[df["task"] == "aesthetic_rating"]
            for _, row in rating_rows.iterrows():
                stimulus = str(row.get("stimulus"))
                image_filename = os.path.basename(stimulus)
                rating_val = row.get("rating")
                rt = row.get("rt")
                trial_index = row.get("trial_index")

                ratings_list.append({
                    "session_id": session_id,
                    "participant_id": participant_id,
                    "image_type": image_type,
                    "trial_type": trial_type,
                    "trial_index": trial_index,
                    "image_filename": image_filename,
                    "image_path": stimulus,
                    "rating": float(rating_val) if pd.notna(rating_val) else None,
                    "rt": float(rt) if pd.notna(rt) else None,
                })

        # 6. Session Completion Check
        # Supports:
        # - Experiment 1 (data): 40 annotation trials (20 images x 2 conditions) + 20 aesthetic ratings
        # - Pilot study (data_3): 64 annotation trials (8 images x 8 conditions) + 8 aesthetic ratings
        # - General: sessions reaching the final completion stage (success==True near end) with full tasks
        has_succ_tail = False
        if "success" in df.columns:
            has_succ_tail = any(df.tail(3)["success"] == True)

        is_completed = (
            (len(annotations_list) >= 40 and len(ratings_list) >= 20)
            or (len(annotations_list) >= 64 and (len(ratings_list) >= 8 or has_succ_tail))
            or (len(annotations_list) >= 8 and len(ratings_list) >= 8 and has_succ_tail)
        )

        # Total session time in seconds
        total_time_ms = None
        if "time_elapsed" in df.columns:
            time_series = df["time_elapsed"].dropna()
            if len(time_series) > 0:
                total_time_ms = float(time_series.iloc[-1])

        participant_info = {
            "session_id": session_id,
            "participant_id": participant_id,
            "image_type": image_type,
            "trial_type": trial_type,
            "completed": is_completed,
            "total_rows_logged": len(df),
            "num_annotations": len(annotations_list),
            "num_ratings": len(ratings_list),
            "total_time_sec": round(total_time_ms / 1000.0, 2) if total_time_ms is not None else None,
            "age": demographics.get("age"),
            "gender": demographics.get("gender"),
            "colorblindness": demographics.get("colorblindness"),
            "visual_expertise": demographics.get("visual_expertise"),
            "EMS": ems_score,
            "num_selected_images": len(selected_images),
            "selected_images": selected_images,
        }

        # Add category details if available
        if trial_type in self.acronym_to_categories:
            cat_a, cat_b, comb_name = self.acronym_to_categories[trial_type]
            participant_info["category_a"] = cat_a
            participant_info["category_b"] = cat_b
            participant_info["combination"] = comb_name

        return {
            "participant": participant_info,
            "annotations": annotations_list,
            "shapes": shapes_list,
            "ratings": ratings_list,
            "is_completed": is_completed,
        }

    def parse_all(
        self,
        data_dir: str = "collected_data/data",
        only_completed: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Parses all CSV files in the data directory.

        Args:
            data_dir: Directory containing participant CSV files.
            only_completed: If True, filters annotations, shapes, and ratings
                            to only include participants who completed the experiment.

        Returns:
            Dictionary containing four DataFrames:
            - 'participants': One row per session
            - 'annotations': One row per annotated image trial
            - 'shapes': One row per annotated shape/polygon
            - 'ratings': One row per aesthetic rating
        """
        csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
        logger.info(f"Found {len(csv_files)} files in {data_dir}")

        all_participants = []
        all_annotations = []
        all_shapes = []
        all_ratings = []

        for fpath in csv_files:
            try:
                res = self.parse_single_file(fpath)
                p_info = res["participant"]
                all_participants.append(p_info)

                if only_completed and not res["is_completed"]:
                    continue

                all_annotations.extend(res["annotations"])
                all_shapes.extend(res["shapes"])
                all_ratings.extend(res["ratings"])
            except Exception as e:
                logger.error(f"Error processing file {fpath}: {e}")

        participants_df = pd.DataFrame(all_participants)
        annotations_df = pd.DataFrame(all_annotations)
        shapes_df = pd.DataFrame(all_shapes)
        ratings_df = pd.DataFrame(all_ratings)

        logger.info(
            f"Parsed successfully: "
            f"{len(participants_df)} sessions ({participants_df['completed'].sum()} completed), "
            f"{len(annotations_df)} annotation trials, "
            f"{len(shapes_df)} shapes, "
            f"{len(ratings_df)} aesthetic ratings."
        )

        return {
            "participants": participants_df,
            "annotations": annotations_df,
            "shapes": shapes_df,
            "ratings": ratings_df,
        }

    def update_trials_js(
        self,
        data_dir: str = "collected_data/data",
        trials_js_path: str = "trials.js",
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Recomputes participant and image counts based on completed sessions
        and writes the updated dictionary back into `trials.js` for `choose_trial.js`.

        Args:
            data_dir: Directory containing participant session CSVs.
            trials_js_path: Template or current trials.js file path.
            output_path: Destination path (defaults to overwriting trials_js_path).

        Returns:
            The updated trials dictionary.
        """
        if output_path is None:
            output_path = trials_js_path

        # 1. Initialize fresh counts structure using combinations and image lists
        art_img_file = os.path.join(self.combinations_path, "ART_image_names.csv")
        nat_img_file = os.path.join(self.combinations_path, "NAT_image_names.csv")

        art_images = pd.read_csv(art_img_file)["image_name"].tolist() if os.path.exists(art_img_file) else []
        nat_images = pd.read_csv(nat_img_file)["image_name"].tolist() if os.path.exists(nat_img_file) else []
        combinations = self.combinations_df["Acronym"].tolist() if self.combinations_df is not None else []

        trials_dict = {
            "Art": {"Num_Participants": 0, "Trial_Participants": {ac: 0 for ac in combinations}, "Image_Participants": {}},
            "Nat": {"Num_Participants": 0, "Trial_Participants": {ac: 0 for ac in combinations}, "Image_Participants": {}},
        }

        for img in art_images:
            trials_dict["Art"]["Image_Participants"][img] = {"Num_Participants": 0, **{ac: 0 for ac in combinations}}
        for img in nat_images:
            trials_dict["Nat"]["Image_Participants"][img] = {"Num_Participants": 0, **{ac: 0 for ac in combinations}}

        # 2. Parse all completed sessions
        parsed = self.parse_all(data_dir=data_dir, only_completed=True)
        completed_participants = parsed["participants"][parsed["participants"]["completed"] == True]

        for _, p in completed_participants.iterrows():
            img_type = p["image_type"]
            trial_type = p["trial_type"]
            selected_imgs = p["selected_images"]

            if img_type not in trials_dict:
                continue

            trials_dict[img_type]["Num_Participants"] += 1
            if trial_type in trials_dict[img_type]["Trial_Participants"]:
                trials_dict[img_type]["Trial_Participants"][trial_type] += 1

            for img in selected_imgs:
                # normalize basename
                base_img = os.path.basename(img)
                if base_img in trials_dict[img_type]["Image_Participants"]:
                    trials_dict[img_type]["Image_Participants"][base_img]["Num_Participants"] += 1
                    if trial_type in trials_dict[img_type]["Image_Participants"][base_img]:
                        trials_dict[img_type]["Image_Participants"][base_img][trial_type] += 1

        # 3. Save to JS file
        json_str = json.dumps(trials_dict, indent=4)
        js_content = f"export const trials = {json_str};\n"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(js_content)

        logger.info(f"Updated {output_path} with {len(completed_participants)} completed participant sessions.")
        return trials_dict

    def export_parsed_data(
        self,
        data_dir: str = "collected_data/data",
        output_dir: str = "parsed_data",
        only_completed: bool = True,
    ) -> None:
        """
        Parses all sessions and exports tables as individual CSV files into output_dir.
        """
        os.makedirs(output_dir, exist_ok=True)
        data = self.parse_all(data_dir=data_dir, only_completed=only_completed)

        for name, df in data.items():
            out_file = os.path.join(output_dir, f"{name}.csv")
            # Drop complex/large raw json columns for lightweight summary csv if desired, or keep all
            df.to_csv(out_file, index=False)
            logger.info(f"Exported {name} -> {out_file} ({len(df)} rows)")


# Quick helper functions for module-level usage:
def load_all_data(data_dir: str = "collected_data/data", only_completed: bool = True) -> Dict[str, pd.DataFrame]:
    """Convenience wrapper to parse and return all tables as DataFrames."""
    parser = ExperimentDataParser()
    return parser.parse_all(data_dir=data_dir, only_completed=only_completed)


def sync_trials_js(data_dir: str = "collected_data/data", trials_js_path: str = "trials.js") -> Dict[str, Any]:
    """Convenience wrapper to update trials.js with completed participant counts."""
    parser = ExperimentDataParser()
    return parser.update_trials_js(data_dir=data_dir, trials_js_path=trials_js_path)


if __name__ == "__main__":
    import argparse

    cli_parser = argparse.ArgumentParser(description="Parse jsPsych experiment session data.")
    cli_parser.add_argument("--data-dir", default="collected_data/data", help="Path to raw CSV data directory")
    cli_parser.add_argument("--output-dir", default="parsed_data", help="Directory to save parsed CSVs")
    cli_parser.add_argument("--all-sessions", action="store_true", help="Include incomplete/aborted sessions")
    cli_parser.add_argument("--sync-trials", action="store_true", help="Update trials.js with completed counts")

    args = cli_parser.parse_args()

    parser = ExperimentDataParser()
    parser.export_parsed_data(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        only_completed=not args.all_sessions,
    )

    if args.sync_trials:
        parser.update_trials_js(data_dir=args.data_dir)

