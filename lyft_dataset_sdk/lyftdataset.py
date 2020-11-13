# Lyft Dataset SDK.
# Code written by Oscar Beijbom, 2018.
# Licensed under the Creative Commons [see licence.txt]
# Modified by Vladimir Iglovikov 2019.

import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
import yaml
from addict import Dict as Adict
from matplotlib.axes import Axes
from PIL import Image
from pyquaternion import Quaternion
from tqdm import tqdm

from lyft_dataset_sdk.Box import Box
from lyft_dataset_sdk.data_classes import LidarPointCloud
from lyft_dataset_sdk.geometry_utils import BoxVisibility, box_in_image, view_points
from lyft_dataset_sdk.utils.map_mask import MapMask

with open(Path(__file__).parent / "config.yaml") as f:
    config = Adict(yaml.load(f, Loader=yaml.SafeLoader))


class LyftDataset:
    """Database class for Lyft Dataset to help query and retrieve information from the database."""

    def __init__(
        self,
        image_path: str,
        lidar_path: str,
        json_path: str,
        map_path: Optional[str] = None,
        verbose: bool = True,
        map_resolution: float = 0.1,
    ):
        """Loads database and creates reverse indexes and shortcuts.

        Args:
            image_path: Path to the images
            lidar_path: Path to lidar
            json_path: Path to the folder with json files
            map_path: Path to the map file
            verbose: Whether to print status messages during load.
            map_resolution: Resolution of maps (meters).
        """

        self.image_path = Path(image_path).expanduser().absolute()
        self.lidar_path = Path(lidar_path).expanduser().absolute()

        self.json_path = Path(json_path)

        self.table_names = [
            "category",
            "attribute",
            "visibility",
            "instance",
            "sensor",
            "calibrated_sensor",
            "ego_pose",
            "log",
            "scene",
            "sample",
            "sample_data",
            "sample_annotation",
            "map",
        ]

        start_time = time.time()

        # Explicitly assign tables to help the IDE determine valid class members.
        self.category = self.__load_table__("category", verbose)
        self.attribute = self.__load_table__("attribute", verbose)
        self.visibility = self.__load_table__("visibility", verbose)
        self.instance = self.__load_table__("instance", verbose, missing_ok=True)
        self.sensor = self.__load_table__("sensor", verbose)
        self.calibrated_sensor = self.__load_table__("calibrated_sensor", verbose)
        self.ego_pose = self.__load_table__("ego_pose", verbose)
        self.log = self.__load_table__("log", verbose)
        self.scene = self.__load_table__("scene", verbose)
        self.sample = self.__load_table__("sample", verbose)
        self.sample_data = self.__load_table__("sample_data", verbose)
        self.sample_annotation = self.__load_table__("sample_annotation", verbose, missing_ok=True)
        self.map = self.__load_table__("map", verbose)

        # if map_path is not None:
        self.map_mask = MapMask(map_path, resolution=map_resolution)

        if verbose:
            for table in self.table_names:
                print(f"{len(getattr(self, table))} {table},")
            print(f"Done loading in {time.time() - start_time:.1f} seconds.\n======")

        # Make reverse indexes for common lookups.
        self.__make_reverse_index__(verbose)

        # Initialize LyftDatasetExplorer class
        self.explorer = LyftDatasetExplorer(self)

    def __load_table__(self, table_name: str, verbose: bool = False, missing_ok: bool = False) -> Dict[Any, Any]:
        """Loads a table."""
        filepath = self.json_path / f"{table_name}.json"

        if not filepath.is_file() and missing_ok:
            if verbose:
                print(f"JSON file {table_name}.json missing, using empty list")
            return {}

        with open(str(filepath)) as f:
            table = json.load(f)
        return table

    def __make_reverse_index__(self, verbose: bool) -> None:
        """De-normalizes database to create reverse indices for common cases.

        Args:
            verbose: Whether to print outputs.

        """

        start_time = time.time()
        if verbose:
            print("Reverse indexing ...")

        # Store the mapping from token to table index for each table.
        self._token2ind: Dict = dict()
        for table in self.table_names:
            self._token2ind[table] = dict()
            for ind, member in enumerate(getattr(self, table)):
                self._token2ind[table][member["token"]] = ind

        # Decorate (adds short-cut) sample_annotation table with for category name.
        for record in self.sample_annotation:
            inst = self.get("instance", record["instance_token"])
            record["category_name"] = self.get("category", inst["category_token"])["name"]

        # Decorate (adds short-cut) sample_data with sensor information.
        for record in self.sample_data:
            cs_record = self.get("calibrated_sensor", record["calibrated_sensor_token"])
            sensor_record = self.get("sensor", cs_record["sensor_token"])
            record["sensor_modality"] = sensor_record["modality"]
            record["channel"] = sensor_record["channel"]

        # Reverse-index samples with sample_data and annotations.
        for record in self.sample:
            record["data"] = {}
            record["anns"] = []

        for record in self.sample_data:
            if record["is_key_frame"]:
                sample_record = self.get("sample", record["sample_token"])
                sample_record["data"][record["channel"]] = record["token"]

        for ann_record in self.sample_annotation:
            sample_record = self.get("sample", ann_record["sample_token"])
            sample_record["anns"].append(ann_record["token"])

        # Add reverse indices from log records to map records.
        if "log_tokens" not in self.map[0].keys():
            raise Exception("Error: log_tokens not in map table. This code is not compatible with the teaser dataset.")
        log_to_map = dict()
        for map_record in self.map:
            for log_token in map_record["log_tokens"]:
                log_to_map[log_token] = map_record["token"]
        for log_record in self.log:
            log_record["map_token"] = log_to_map[log_record["token"]]

        if verbose:
            print("Done reverse indexing in {:.1f} seconds.\n======".format(time.time() - start_time))

    def get(self, table_name: str, token: str) -> dict:
        """Returns a record from table in constant runtime.

        Args:
            table_name: Table name.
            token: Token of the record.

        Returns: Table record.

        """

        if table_name not in self.table_names:
            raise KeyError(f"Table {table_name} not found")

        return getattr(self, table_name)[self.getind(table_name, token)]

    def getind(self, table_name: str, token: str) -> int:
        """Returns the index of the record in a table in constant runtime.

        Args:
            table_name: Table name.
            token: The index of the record in table, table is an array.

        Returns:

        """
        return self._token2ind[table_name][token]

    def field2token(self, table_name: str, field: str, query: str) -> List[str]:
        """Query all records for a certain field value, and returns the tokens for the matching records.

        Runs in linear time.

        Args:
            table_name: Table name.
            field: Field name.
            query: Query to match against. Needs to type match the content of the query field.

        Returns: List of tokens for the matching records.

        """
        matches = []
        for member in getattr(self, table_name):
            if member[field] == query:
                matches.append(member["token"])
        return matches

    def get_sample_data_path(self, sample_data_token: str, modality: str) -> Path:
        """Returns the path to a sample_data.

        Args:
            sample_data_token:
            modality: 'lidar' or 'camera'

        Returns:

        """

        sd_record = self.get("sample_data", sample_data_token)

        if modality == "lidar":
            data_path = self.lidar_path
        elif modality == "camera":
            data_path = self.image_path
        else:
            raise NotImplementedError()

        return data_path / Path(sd_record["filename"]).name

    def get_sample_data(
        self,
        sample_data_token: str,
        box_vis_level: BoxVisibility = BoxVisibility.ANY,
        selected_anntokens: Optional[List[str]] = None,
        flat_vehicle_coordinates: bool = False,
    ) -> Tuple[Path, List[Box], np.array]:
        """Returns the data path as well as all annotations related to that sample_data.
        The boxes are transformed into the current sensor's coordinate frame.

        Args:
            sample_data_token: Sample_data token.
            box_vis_level: If sample_data is an image, this sets required visibility for boxes.
            selected_anntokens: If provided only return the selected annotation.
            flat_vehicle_coordinates: Instead of current sensor's coordinate frame, use vehicle frame which is
        aligned to z-plane in world

        Returns: (data_path, boxes, camera_intrinsic <np.array: 3, 3>)

        """

        # Retrieve sensor & pose records
        sample_data_record = self.get("sample_data", sample_data_token)
        calibrated_sensor_record = self.get("calibrated_sensor", sample_data_record["calibrated_sensor_token"])
        sensor_record = self.get("sensor", calibrated_sensor_record["sensor_token"])
        pose_record = self.get("ego_pose", sample_data_record["ego_pose_token"])
        modality = sensor_record["modality"]

        data_path = self.get_sample_data_path(sample_data_token, modality)

        cam_intrinsic: Optional[np.ndarray] = None
        image_size: Optional[Tuple[int, int]] = None

        if modality == "camera":
            cam_intrinsic = np.array(calibrated_sensor_record["camera_intrinsic"])
            image_size = (sample_data_record["width"], sample_data_record["height"])

        # Retrieve all sample annotations and map to sensor coordinate system.
        if selected_anntokens is not None:
            boxes = list(map(self.get_box, selected_anntokens))
        else:
            boxes = self.get_boxes(sample_data_token)

        # Make list of Box objects including coord system transforms.
        box_list = []
        for box in boxes:
            if flat_vehicle_coordinates:
                # Move box to ego vehicle coord system parallel to world z plane
                ypr = Quaternion(pose_record["rotation"]).yaw_pitch_roll
                yaw = ypr[0]

                box.translate(-np.array(pose_record["translation"]))
                box.rotate_around_origin(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)

            else:
                # Move box to ego vehicle coord system
                box.translate(-np.array(pose_record["translation"]))
                box.rotate_around_origin(Quaternion(pose_record["rotation"]).inverse)

                #  Move box to sensor coord system
                box.translate(-np.array(calibrated_sensor_record["translation"]))
                box.rotate_around_origin(Quaternion(calibrated_sensor_record["rotation"]).inverse)

            if (
                sensor_record["modality"] == "camera"
                and image_size is not None
                and not box_in_image(box, cam_intrinsic, image_size, vis_level=box_vis_level)
            ):
                continue

            box_list.append(box)

        return data_path, box_list, cam_intrinsic

    def get_box(self, sample_annotation_token: str) -> Box:
        """Instantiates a Box class from a sample annotation record.

        Args:
            sample_annotation_token: Unique sample_annotation identifier.

        Returns:

        """
        record = self.get("sample_annotation", sample_annotation_token)
        return Box(
            record["translation"],
            record["size"],
            Quaternion(record["rotation"]),
            name=record["category_name"],
            token=record["token"],
        )

    def get_boxes(self, sample_data_token: str) -> List[Box]:
        """Instantiates Boxes for all annotation for a particular sample_data record. If the sample_data is a
        keyframe, this returns the annotations for that sample. But if the sample_data is an intermediate
        sample_data, a linear interpolation is applied to estimate the location of the boxes at the time the
        sample_data was captured.

        Args:
            sample_data_token: Unique sample_data identifier.

        Returns:

        """

        # Retrieve sensor & pose records
        sample_data_record = self.get("sample_data", sample_data_token)
        curr_sample_record = self.get("sample", sample_data_record["sample_token"])

        if curr_sample_record["prev"] == "" or sample_data_record["is_key_frame"]:
            # If no previous annotations available, or if sample_data is keyframe just return the current ones.
            boxes = list(map(self.get_box, curr_sample_record["anns"]))

        else:
            prev_sample_record = self.get("sample", curr_sample_record["prev"])

            curr_ann_recs = [self.get("sample_annotation", token) for token in curr_sample_record["anns"]]
            prev_ann_recs = [self.get("sample_annotation", token) for token in prev_sample_record["anns"]]

            # Maps instance tokens to prev_ann records
            prev_inst_map = {entry["instance_token"]: entry for entry in prev_ann_recs}

            t0 = prev_sample_record["timestamp"]
            t1 = curr_sample_record["timestamp"]
            t = sample_data_record["timestamp"]

            # There are rare situations where the timestamps in the DB are off so ensure that t0 < t < t1.
            t = max(t0, min(t1, t))

            boxes = []
            for curr_ann_rec in curr_ann_recs:

                if curr_ann_rec["instance_token"] in prev_inst_map:
                    # If the annotated instance existed in the previous frame, interpolate center & orientation.
                    prev_ann_rec = prev_inst_map[curr_ann_rec["instance_token"]]

                    # Interpolate center.
                    center = [
                        np.interp(t, [t0, t1], [c0, c1])
                        for c0, c1 in zip(prev_ann_rec["translation"], curr_ann_rec["translation"])
                    ]

                    # Interpolate orientation.
                    rotation = Quaternion.slerp(
                        q0=Quaternion(prev_ann_rec["rotation"]),
                        q1=Quaternion(curr_ann_rec["rotation"]),
                        amount=(t - t0) / (t1 - t0),
                    )

                    box = Box(
                        center,
                        curr_ann_rec["size"],
                        rotation,
                        name=curr_ann_rec["category_name"],
                        token=curr_ann_rec["token"],
                    )
                else:
                    # If not, simply grab the current annotation.
                    box = self.get_box(curr_ann_rec["token"])

                boxes.append(box)
        return boxes

    def box_velocity(self, sample_annotation_token: str, max_time_diff: float = 1.5) -> np.ndarray:
        """Estimate the velocity for an annotation.

        If possible, we compute the centered difference between the previous and next frame.
        Otherwise we use the difference between the current and previous/next frame.
        If the velocity cannot be estimated, values are set to np.nan.

        Args:
            sample_annotation_token: Unique sample_annotation identifier.
            max_time_diff: Max allowed time diff between consecutive samples that are used to estimate velocities.


        Returns: <np.float: 3>. Velocity in x/y/z direction in m/s.

        """

        current = self.get("sample_annotation", sample_annotation_token)
        has_prev = current["prev"] != ""
        has_next = current["next"] != ""

        # Cannot estimate velocity for a single annotation.
        if not has_prev and not has_next:
            return np.array([np.nan, np.nan, np.nan])

        if has_prev:
            first = self.get("sample_annotation", current["prev"])
        else:
            first = current

        if has_next:
            last = self.get("sample_annotation", current["next"])
        else:
            last = current

        pos_last = np.array(last["translation"])
        pos_first = np.array(first["translation"])
        pos_diff = pos_last - pos_first

        time_last = 1e-6 * self.get("sample", last["sample_token"])["timestamp"]
        time_first = 1e-6 * self.get("sample", first["sample_token"])["timestamp"]
        time_diff = time_last - time_first

        if has_next and has_prev:
            # If doing centered difference, allow for up to double the max_time_diff.
            max_time_diff *= 2

        if time_diff > max_time_diff:
            # If time_diff is too big, don't return an estimate.
            return np.array([np.nan, np.nan, np.nan])

        return pos_diff / time_diff

    def list_categories(self) -> None:
        self.explorer.list_categories()

    def list_attributes(self) -> None:
        self.explorer.list_attributes()

    def list_scenes(self) -> None:
        self.explorer.list_scenes()

    def list_sample(self, sample_token: str) -> None:
        self.explorer.list_sample(sample_token)

    def render_pointcloud_in_image(
        self,
        sample_token: str,
        dot_size: int = 5,
        pointsensor_channel: str = "LIDAR_TOP",
        camera_channel: str = "CAM_FRONT",
        out_path: Optional[str] = None,
    ) -> None:
        self.explorer.render_pointcloud_in_image(
            sample_token,
            dot_size,
            pointsensor_channel=pointsensor_channel,
            camera_channel=camera_channel,
            out_path=out_path,
        )

    def render_sample(
        self,
        sample_token: str,
        box_vis_level: BoxVisibility = BoxVisibility.ANY,
        num_sweeps: int = 1,
        out_path: Optional[Path] = None,
    ) -> None:
        self.explorer.render_sample(sample_token, box_vis_level, num_sweeps=num_sweeps, out_path=out_path)

    def render_sample_data(
        self,
        sample_data_token: str,
        with_anns: bool = True,
        box_vis_level: BoxVisibility = BoxVisibility.ANY,
        axes_limit: float = 40,
        ax: Optional[Axes] = None,
        nsweeps: int = 1,
        out_path: Optional[Path] = None,
        underlay_map: bool = False,
    ) -> Optional[Path]:
        return self.explorer.render_sample_data(
            sample_data_token,
            with_anns,
            box_vis_level,
            axes_limit,
            ax,
            num_sweeps=nsweeps,
            out_path=out_path,
            underlay_map=underlay_map,
        )

    def render_annotation(
        self,
        sample_annotation_token: str,
        margin: float = 10,
        view: np.ndarray = np.eye(4),
        box_vis_level: BoxVisibility = BoxVisibility.ANY,
        out_path: str = None,
    ) -> None:
        self.explorer.render_annotation(sample_annotation_token, margin, view, box_vis_level, out_path)

    def render_instance(self, instance_token: str, out_path: str = None) -> None:
        self.explorer.render_instance(instance_token, out_path=out_path)

    def render_scene(
        self, scene_token: str, freq: float = 10, image_width: int = 640, out_path: Optional[Path] = None
    ) -> None:
        self.explorer.render_scene(scene_token, freq, image_width=image_width, out_path=out_path)

    def render_scene_channel(
        self,
        scene_token: str,
        channel: str = "CAM_FRONT",
        freq: float = 10,
        imsize: Tuple[float, float] = (640, 360),
        out_path: Path = None,
        interactive: bool = True,
        verbose: bool = False,
    ) -> None:
        self.explorer.render_scene_channel(
            scene_token=scene_token,
            channel=channel,
            freq=freq,
            image_size=imsize,
            out_path=out_path,
            interactive=interactive,
            verbose=verbose,
        )

    def render_egoposes_on_map(
        self, log_location: str, scene_tokens: Optional[List] = None, out_path: Optional[Path] = None
    ) -> None:
        self.explorer.render_egoposes_on_map(log_location, scene_tokens, out_path=out_path)

    def render_sample_3d_interactive(self, sample_id: str, render_sample: bool = True) -> None:
        """Render 3D visualization of the sample using plotly

        Args:
            sample_id: Unique sample identifier.
            render_sample: call self.render_sample (Render all LIDAR and camera sample_data in
                                                                sample along with annotations.)

        """
        import plotly.graph_objects as go

        sample = self.get("sample", sample_id)
        sample_data = self.get("sample_data", sample["data"]["LIDAR_TOP"])
        pc = LidarPointCloud.from_file(self.lidar_path / Path(sample_data["filename"]).name)
        _, boxes, _ = self.get_sample_data(sample["data"]["LIDAR_TOP"], flat_vehicle_coordinates=False)

        if render_sample:
            self.render_sample(sample_id)

        df_tmp = pd.DataFrame(pc.points[:3, :].T, columns=["x", "y", "z"])
        df_tmp["norm"] = np.sqrt(np.power(df_tmp[["x", "y", "z"]].values, 2).sum(axis=1))

        scatter = go.Scatter3d(
            x=df_tmp["x"],
            y=df_tmp["y"],
            z=df_tmp["z"],
            mode="markers",
            # marker=dict(size=1, color=df_tmp["norm"], opacity=0.8, colorscale=[(0.0, 'rgb(179,205,227)'),
            #                                                                    (1, 'rgb(179,205,227)')]),
            marker=dict(
                size=1,
                color=df_tmp["norm"],
                opacity=1,
                colorscale=[
                    (0.0, f"rgb({config.colors[9][0]}, {config.colors[9][1]}, {config.colors[9][2]})"),
                    (1, f"rgb({config.colors[9][0]}, {config.colors[9][1]}, {config.colors[9][2]})"),
                ],
            ),
        )

        ixs_box_0 = [0, 1, 2, 3, 0]
        ixs_box_1 = [4, 5, 6, 7, 4]

        boxes_dict: Dict = {}

        for class_name in config.class_names:
            boxes_dict[class_name] = {"x_lines": [], "y_lines": [], "z_lines": []}

        for box in boxes:
            points = view_points(box.corners(), view=np.eye(3), normalize=False)

            box_name = box.name

            boxes_dict[box_name]["x_lines"].extend(points[0, ixs_box_0])
            boxes_dict[box_name]["y_lines"].extend(points[1, ixs_box_0])
            boxes_dict[box_name]["z_lines"].extend(points[2, ixs_box_0])

            boxes_dict[box_name]["x_lines"].append(None)
            boxes_dict[box_name]["y_lines"].append(None)
            boxes_dict[box_name]["z_lines"].append(None)

            boxes_dict[box_name]["x_lines"].extend(points[0, ixs_box_1])
            boxes_dict[box_name]["y_lines"].extend(points[1, ixs_box_1])
            boxes_dict[box_name]["z_lines"].extend(points[2, ixs_box_1])

            boxes_dict[box_name]["x_lines"].append(None)
            boxes_dict[box_name]["y_lines"].append(None)
            boxes_dict[box_name]["z_lines"].append(None)

            # x_lines.extend(points[0, ixs_box_0])
            # y_lines.extend(points[1, ixs_box_0])
            # z_lines.extend(points[2, ixs_box_0])
            # f_lines_add_nones()
            # x_lines.extend(points[0, ixs_box_1])
            # y_lines.extend(points[1, ixs_box_1])
            # z_lines.extend(points[2, ixs_box_1])
            # f_lines_add_nones()

            for i in range(4):
                boxes_dict[box_name]["x_lines"].extend(points[0, [ixs_box_0[i], ixs_box_1[i]]])
                boxes_dict[box_name]["y_lines"].extend(points[1, [ixs_box_0[i], ixs_box_1[i]]])
                boxes_dict[box_name]["z_lines"].extend(points[2, [ixs_box_0[i], ixs_box_1[i]]])

                boxes_dict[box_name]["x_lines"].append(None)
                boxes_dict[box_name]["y_lines"].append(None)
                boxes_dict[box_name]["z_lines"].append(None)

                # x_lines.extend(points[0, [ixs_box_0[i], ixs_box_1[i]]])
                # y_lines.extend(points[1, [ixs_box_0[i], ixs_box_1[i]]])
                # z_lines.extend(points[2, [ixs_box_0[i], ixs_box_1[i]]])
                # f_lines_add_nones()

        temp = [scatter]

        for class_name in boxes_dict:
            if len(boxes_dict[class_name]["x_lines"]) == 0:
                continue

            lines = go.Scatter3d(
                x=boxes_dict[class_name]["x_lines"],
                y=boxes_dict[class_name]["y_lines"],
                z=boxes_dict[class_name]["z_lines"],
                mode="lines",
                name="lines",
                marker={"color": self.explorer.get_color(class_name)},
            )

            temp += [lines]
        # lines = px.scatter_3d(x=x_lines, y=y_lines, z=z_lines, mode="lines", name="lines")

        fig = go.Figure(data=temp)
        fig.update_layout(scene_aspectmode="data")
        fig.update_layout(showlegend=False)
        fig.show()


class LyftDatasetExplorer:
    """Helper class to list and visualize Lyft Dataset data. These are meant to serve as tutorials and templates for
    working with the data."""

    def __init__(self, lyftd: LyftDataset) -> None:
        self.lyftd = lyftd

    @staticmethod
    def get_color(category_name: Optional[str]) -> Tuple[int, int, int]:
        """Provides the default colors based on the category names.
        This method works for the general Lyft Dataset categories, as well as the Lyft Dataset detection categories.

        Args:
            category_name:

        Returns:

        """
        result = config.colors[0]

        if category_name == config.class_names[1]:
            result = config.colors[1]
        elif category_name == config.class_names[2]:
            result = config.colors[2]
        elif category_name == config.class_names[3]:
            result = config.colors[3]
        elif category_name == config.class_names[4]:
            result = config.colors[4]
        elif category_name == config.class_names[5]:
            result = config.colors[5]
        elif category_name == config.class_names[6]:
            result = config.colors[6]
        elif category_name == config.class_names[7]:
            result = config.colors[7]
        elif category_name == config.class_names[8]:
            result = config.colors[8]

        return result

    def list_categories(self) -> None:
        """Print categories, counts and stats."""

        print("Category stats")

        # Add all annotations
        categories: Dict = dict()
        for record in self.lyftd.sample_annotation:
            if record["category_name"] not in categories:
                categories[record["category_name"]] = []
            categories[record["category_name"]].append(record["size"] + [record["size"][1] / record["size"][0]])

        result: List[Tuple] = []
        # Print stats
        for name, stats in sorted(categories.items()):
            stats = np.array(stats)

            result += [
                (
                    name.strip(),
                    stats.shape[0],
                    stats[:, 0].mean(),
                    stats[:, 0].std(),
                    stats[:, 1].mean(),
                    stats[:, 1].std(),
                    stats[:, 2].mean(),
                    stats[:, 2].std(),
                    stats[:, 3].mean(),
                    stats[:, 3].std(),
                )
            ]

        df = pd.DataFrame(
            result,
            columns=[
                "category",
                "num_annotations",
                "width_mean",
                "width_std",
                "length_mean",
                "length_std",
                "height_mean",
                "height_std",
                "lw_aspect_mean",
                "lw_aspect_std",
            ],
        )

        df["width"] = df["width_mean"].round(3).astype(str) + "\u00B1" + df["width_std"].round(3).astype(str)
        df["length"] = df["length_mean"].round(3).astype(str) + "\u00B1" + df["length_std"].round(3).astype(str)
        df["height"] = df["height_mean"].round(3).astype(str) + "\u00B1" + df["height_std"].round(3).astype(str)
        df["lw_aspect"] = (
            df["lw_aspect_mean"].round(3).astype(str) + "\u00B1" + df["lw_aspect_std"].round(3).astype(str)
        )

        df = df[["category", "num_annotations", "width", "length", "height", "lw_aspect"]]
        print(df)

    def list_attributes(self) -> None:
        """Prints attributes and counts."""
        attribute_counts: Dict[str, int] = dict()
        for record in self.lyftd.sample_annotation:
            for attribute_token in record["attribute_tokens"]:
                att_name = self.lyftd.get("attribute", attribute_token)["name"]
                if att_name not in attribute_counts:
                    attribute_counts[att_name] = 0
                attribute_counts[att_name] += 1

        for name, count in sorted(attribute_counts.items()):
            print(f"{name}: {count}")

    def list_scenes(self) -> None:
        """Lists all scenes with some meta data."""

        def ann_count(record):
            count = 0
            sample = self.lyftd.get("sample", record["first_sample_token"])
            while not sample["next"] == "":
                count += len(sample["anns"])
                sample = self.lyftd.get("sample", sample["next"])
            return count

        recs = [
            (self.lyftd.get("sample", record["first_sample_token"])["timestamp"], record)
            for record in self.lyftd.scene
        ]

        for start_time, record in sorted(recs):
            start_time = self.lyftd.get("sample", record["first_sample_token"])["timestamp"] / 1000000
            length_time = self.lyftd.get("sample", record["last_sample_token"])["timestamp"] / 1000000 - start_time
            location = self.lyftd.get("log", record["log_token"])["location"]
            desc = record["name"] + ", " + record["description"]
            if len(desc) > 55:
                desc = desc[:51] + "..."
            if len(location) > 18:
                location = location[:18]

            print(
                "{:16} [{}] {:4.0f}s, {}, #anns:{}".format(
                    desc,
                    datetime.utcfromtimestamp(start_time).strftime("%y-%m-%d %H:%M:%S"),
                    length_time,
                    location,
                    ann_count(record),
                )
            )

    def list_sample(self, sample_token: str) -> None:
        """Prints sample_data tokens and sample_annotation tokens related to the sample_token."""

        sample_record = self.lyftd.get("sample", sample_token)
        print(f"Sample: {sample_record['token']}\n")
        for sd_token in sample_record["data"].values():
            sd_record = self.lyftd.get("sample_data", sd_token)
            print(
                f"sample_data_token: {sd_token}, mod: {sd_record['sensor_modality']}, channel: {sd_record['channel']}"
            )
        print("")
        for ann_token in sample_record["anns"]:
            ann_record = self.lyftd.get("sample_annotation", ann_token)
            print(f"sample_annotation_token: {ann_record['token']}, category: {ann_record['category_name']}")

    def map_pointcloud_to_image(self, pointsensor_token: str, camera_token: str) -> Tuple:
        """Given a point sensor lidar token and camera sample_data token, load point-cloud and map it to
        the image plane.

        Args:
            pointsensor_token: Lidar/radar sample_data token.
            camera_token: Camera sample_data token.

        Returns: (pointcloud <np.float: 2, n)>, coloring <np.float: n>, image <Image>).

        """

        cam = self.lyftd.get("sample_data", camera_token)
        pointsensor = self.lyftd.get("sample_data", pointsensor_token)
        pointcloud_path = self.lyftd.lidar_path / Path(pointsensor["filename"]).name
        pointcloud = LidarPointCloud.from_file(pointcloud_path)

        image = Image.open(str(self.lyftd.image_path / Path(cam["filename"]).name))

        # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
        # First step: transform the point-cloud to the ego vehicle frame for the timestamp of the sweep.
        cs_record = self.lyftd.get("calibrated_sensor", pointsensor["calibrated_sensor_token"])
        pointcloud.rotate(Quaternion(cs_record["rotation"]).rotation_matrix)
        pointcloud.translate(np.array(cs_record["translation"]))

        # Second step: transform to the global frame.
        poserecord = self.lyftd.get("ego_pose", pointsensor["ego_pose_token"])
        pointcloud.rotate(Quaternion(poserecord["rotation"]).rotation_matrix)
        pointcloud.translate(np.array(poserecord["translation"]))

        # Third step: transform into the ego vehicle frame for the timestamp of the image.
        poserecord = self.lyftd.get("ego_pose", cam["ego_pose_token"])
        pointcloud.translate(-np.array(poserecord["translation"]))
        pointcloud.rotate(Quaternion(poserecord["rotation"]).rotation_matrix.T)

        # Fourth step: transform into the camera.
        cs_record = self.lyftd.get("calibrated_sensor", cam["calibrated_sensor_token"])
        pointcloud.translate(-np.array(cs_record["translation"]))
        pointcloud.rotate(Quaternion(cs_record["rotation"]).rotation_matrix.T)

        # Fifth step: actually take a "picture" of the point cloud.
        # Grab the depths (camera frame z axis points away from the camera).
        depths = pointcloud.points[2, :]

        # Retrieve the color from the depth.
        coloring = depths

        # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
        points = view_points(pointcloud.points[:3, :], np.array(cs_record["camera_intrinsic"]), normalize=True)

        # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > 0)
        mask = np.logical_and(mask, points[0, :] > 1)
        mask = np.logical_and(mask, points[0, :] < image.size[0] - 1)
        mask = np.logical_and(mask, points[1, :] > 1)
        mask = np.logical_and(mask, points[1, :] < image.size[1] - 1)
        points = points[:, mask]
        coloring = coloring[mask]

        return points, coloring, image

    def render_pointcloud_in_image(
        self,
        sample_token: str,
        dot_size: int = 2,
        pointsensor_channel: str = "LIDAR_TOP",
        camera_channel: str = "CAM_FRONT",
        out_path: Optional[str] = None,
    ) -> None:
        """Scatter-plots a point-cloud on top of image.

        Args:
            sample_token: Sample token.
            dot_size: Scatter plot dot size.
            pointsensor_channel: RADAR or LIDAR channel name, e.g. 'LIDAR_TOP'.
            camera_channel: Camera channel name, e.g. 'CAM_FRONT'.
            out_path: Optional path to save the rendered figure to disk.

        Returns:

        """
        sample_record = self.lyftd.get("sample", sample_token)

        # Here we just grab the front camera and the point sensor.
        pointsensor_token = sample_record["data"][pointsensor_channel]
        camera_token = sample_record["data"][camera_channel]

        points, coloring, im = self.map_pointcloud_to_image(pointsensor_token, camera_token)
        plt.figure(figsize=(9, 16))
        plt.imshow(im)
        plt.scatter(points[0, :], points[1, :], c=coloring, s=dot_size)
        plt.axis("off")

        if out_path is not None:
            plt.savefig(out_path)

    def render_sample(
        self,
        token: str,
        box_vis_level: BoxVisibility = BoxVisibility.ANY,
        num_sweeps: int = 1,
        out_path: Optional[Path] = None,
        underlay_map: bool = False,
    ) -> None:
        """Render all LIDAR and camera sample_data in sample along with annotations.

        Args:
            token: Sample token.
            box_vis_level: If sample_data is an image, this sets required visibility for boxes.
            num_sweeps: Number of sweeps for lidar.
            out_path: Optional path to save the rendered figure to disk.
            underlay_map:

        Returns:

        """
        record = self.lyftd.get("sample", token)

        data = {}
        for channel, data_token in record["data"].items():
            data[channel] = data_token

        # Create plots.
        cols = 2
        fig, axes = plt.subplots(int(np.ceil(len(data) / cols)), cols, figsize=(16, 24))

        # Plot camera and lidar in separate subplots.
        for (_, sd_token), ax in zip(data.items(), axes.flatten()):
            # self.render_sample_data(
            #     sd_token, box_vis_level=box_vis_level,
            #     ax=ax, num_sweeps=num_sweeps, underlay_map=underlay_map, out_path=out_path
            # )

            self.save_plots_sample_data(
                sd_token,
                box_vis_level=box_vis_level,
                ax=ax,
                num_sweeps=num_sweeps,
                underlay_map=underlay_map,
                out_path=out_path,
            )

        axes.flatten()[-1].axis("off")
        # plt.tight_layout()
        fig.subplots_adjust(wspace=0.01, hspace=0.1)

    def render_ego_centric_map(
        self, sample_data_token: str, axes_limit: float = 40, ax: Optional[Axes] = None
    ) -> None:
        """Render map centered around the associated ego pose.

        Args:
            sample_data_token: Sample_data token.
            axes_limit: Axes limit measured in meters.
            ax: Axes onto which to render.

        """

        def crop_image(image: np.array, x_px: int, y_px: int, axes_limit_px: int) -> np.array:
            x_min = int(x_px - axes_limit_px)
            x_max = int(x_px + axes_limit_px)
            y_min = int(y_px - axes_limit_px)
            y_max = int(y_px + axes_limit_px)

            cropped_image = image[y_min:y_max, x_min:x_max]

            return cropped_image

        sd_record = self.lyftd.get("sample_data", sample_data_token)

        # Init axes.
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(9, 9))

        pose = self.lyftd.get("ego_pose", sd_record["ego_pose_token"])
        pixel_coords = self.lyftd.map_mask.to_pixel_coords(pose["translation"][0], pose["translation"][1])

        scaled_limit_px = int(axes_limit * (1.0 / self.lyftd.map_mask.resolution))
        mask_raster = self.lyftd.map_mask.mask()

        cropped = crop_image(mask_raster, pixel_coords[0], pixel_coords[1], int(scaled_limit_px * math.sqrt(2)))

        ypr_rad = Quaternion(pose["rotation"]).yaw_pitch_roll
        yaw_deg = -math.degrees(ypr_rad[0])

        rotated_cropped = np.array(Image.fromarray(cropped).rotate(yaw_deg))
        ego_centric_map = crop_image(
            rotated_cropped, rotated_cropped.shape[1] / 2, rotated_cropped.shape[0] / 2, scaled_limit_px
        )
        ax.imshow(
            ego_centric_map, extent=[-axes_limit, axes_limit, -axes_limit, axes_limit], cmap="gray", vmin=0, vmax=150
        )

    def save_plots_sample_data(
        self,
        sample_data_token: str,
        with_annotations: bool = True,
        box_vis_level: BoxVisibility = BoxVisibility.ANY,
        axes_limit: float = 40,
        ax: Optional[Axes] = None,
        num_sweeps: int = 1,
        out_path: Optional[Path] = None,
        underlay_map: bool = False,
    ) -> Optional[Path]:
        """Render sample data onto axis.

        Args:
            sample_data_token: Sample_data token.
            with_annotations: Whether to draw annotations.
            box_vis_level: If sample_data is an image, this sets required visibility for boxes.
            axes_limit: Axes limit for lidar and radar (measured in meters).
            ax: Axes onto which to render.
            num_sweeps: Number of sweeps for lidar and radar.
            out_path: Optional path to save the rendered figure to disk.
            underlay_map: When set to true, LIDAR data is plotted onto the map. This can be slow.

        """

        # Get sensor modality.
        sd_record = self.lyftd.get("sample_data", sample_data_token)
        sensor_modality = sd_record["sensor_modality"]

        channel = sd_record["channel"]

        if sensor_modality == "lidar":
            # Get boxes in lidar frame.
            _, boxes, _ = self.lyftd.get_sample_data(
                sample_data_token, box_vis_level=box_vis_level, flat_vehicle_coordinates=True
            )

            # Get aggregated point cloud in lidar frame.
            sample_rec = self.lyftd.get("sample", sd_record["sample_token"])
            ref_chan = "LIDAR_TOP"
            pointcloud, _ = LidarPointCloud.from_file_multisweep(
                self.lyftd, sample_rec, channel, ref_chan, num_sweeps=num_sweeps
            )

            # Compute transformation matrices for lidar point cloud
            cs_record = self.lyftd.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
            pose_record = self.lyftd.get("ego_pose", sd_record["ego_pose_token"])
            vehicle_from_sensor = np.eye(4)
            vehicle_from_sensor[:3, :3] = Quaternion(cs_record["rotation"]).rotation_matrix
            vehicle_from_sensor[:3, 3] = cs_record["translation"]

            ego_yaw = Quaternion(pose_record["rotation"]).yaw_pitch_roll[0]
            rot_vehicle_flat_from_vehicle = np.dot(
                Quaternion(scalar=np.cos(ego_yaw / 2), vector=[0, 0, np.sin(ego_yaw / 2)]).rotation_matrix,
                Quaternion(pose_record["rotation"]).inverse.rotation_matrix,
            )

            vehicle_flat_from_vehicle = np.eye(4)
            vehicle_flat_from_vehicle[:3, :3] = rot_vehicle_flat_from_vehicle

            # # Init axes.
            # if ax is None:
            # _, ax = plt.subplots(1, 1, figsize=(9, 9))
            _, ax = plt.subplots(1, 1, figsize=(6, 6))

            if underlay_map:
                self.render_ego_centric_map(sample_data_token=sample_data_token, axes_limit=axes_limit, ax=ax)

            # Show point cloud.
            points = view_points(
                pointcloud.points[:3, :], np.dot(vehicle_flat_from_vehicle, vehicle_from_sensor), normalize=False
            )

            ax.scatter(points[0, :], points[1, :], color=np.array(config.colors[9]) / 255, s=0.2)

            # Show ego vehicle.
            ax.plot(0, 0, "x", color="red")

            # Show boxes.
            if with_annotations:
                for box in boxes:
                    c = np.array(self.get_color(box.name)) / 255.0
                    box.render(ax, view=np.eye(4), colors=(c, c, c))

            # Limit visible range.
            ax.set_xlim(-axes_limit, axes_limit)
            ax.set_ylim(-axes_limit, axes_limit)

        elif sensor_modality == "camera":
            # Load boxes and image.
            data_path, boxes, camera_intrinsic = self.lyftd.get_sample_data(
                sample_data_token, box_vis_level=box_vis_level
            )

            data = Image.open(data_path)

            # Init axes.
            _, ax = plt.subplots(1, 1)

            # Show image.
            ax.imshow(data)
            # plt.imshow(data)

            # Show boxes.
            if with_annotations:
                for box in boxes:
                    c = np.array(self.get_color(box.name)) / 255.0
                    box.render(ax, view=camera_intrinsic, normalize=True, colors=(c, c, c))

            # Limit visible range.
            ax.set_xlim(0, data.size[0])
            ax.set_ylim(data.size[1], 0)

        else:
            raise ValueError(f"Error: Unknown sensor modality! Got {sensor_modality}")

        ax.axis("off")
        # ax.set_title(sd_record["channel"], fontsize=15)
        ax.set_aspect("equal")
        plt.tight_layout()

        if out_path is not None:
            print("Channel = ", channel)
            plt.savefig(Path(out_path) / f"{channel}.png", dpi=150, bbox_inches="tight")
            plt.close("all")

            return Path(out_path) / f"{channel}.png"

        return out_path

    def render_sample_data(
        self,
        sample_data_token: str,
        with_annotations: bool = True,
        box_vis_level: BoxVisibility = BoxVisibility.ANY,
        axes_limit: float = 40,
        ax: Optional[Axes] = None,
        num_sweeps: int = 1,
        out_path: Optional[Path] = None,
        underlay_map: bool = False,
    ) -> Optional[Path]:
        """Render sample data onto axis.

        Args:
            sample_data_token: Sample_data token.
            with_annotations: Whether to draw annotations.
            box_vis_level: If sample_data is an image, this sets required visibility for boxes.
            axes_limit: Axes limit for lidar and radar (measured in meters).
            ax: Axes onto which to render.
            num_sweeps: Number of sweeps for lidar and radar.
            out_path: Optional path to save the rendered figure to disk.
            underlay_map: When set to true, LIDAR data is plotted onto the map. This can be slow.

        """

        # Get sensor modality.
        sd_record = self.lyftd.get("sample_data", sample_data_token)
        sensor_modality = sd_record["sensor_modality"]

        channel = sd_record["channel"]

        if sensor_modality == "lidar":
            # Get boxes in lidar frame.
            _, boxes, _ = self.lyftd.get_sample_data(
                sample_data_token, box_vis_level=box_vis_level, flat_vehicle_coordinates=True
            )

            # Get aggregated point cloud in lidar frame.
            sample_rec = self.lyftd.get("sample", sd_record["sample_token"])
            ref_chan = "LIDAR_TOP"
            pointcloud, _ = LidarPointCloud.from_file_multisweep(
                self.lyftd, sample_rec, channel, ref_chan, num_sweeps=num_sweeps
            )

            # Compute transformation matrices for lidar point cloud
            cs_record = self.lyftd.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
            pose_record = self.lyftd.get("ego_pose", sd_record["ego_pose_token"])
            vehicle_from_sensor = np.eye(4)
            vehicle_from_sensor[:3, :3] = Quaternion(cs_record["rotation"]).rotation_matrix
            vehicle_from_sensor[:3, 3] = cs_record["translation"]

            ego_yaw = Quaternion(pose_record["rotation"]).yaw_pitch_roll[0]
            rot_vehicle_flat_from_vehicle = np.dot(
                Quaternion(scalar=np.cos(ego_yaw / 2), vector=[0, 0, np.sin(ego_yaw / 2)]).rotation_matrix,
                Quaternion(pose_record["rotation"]).inverse.rotation_matrix,
            )

            vehicle_flat_from_vehicle = np.eye(4)
            vehicle_flat_from_vehicle[:3, :3] = rot_vehicle_flat_from_vehicle

            # Init axes.
            if ax is None:
                _, ax = plt.subplots(1, 1, figsize=(9, 9))

            if underlay_map:
                self.render_ego_centric_map(sample_data_token=sample_data_token, axes_limit=axes_limit, ax=ax)

            # Show point cloud.
            points = view_points(
                pointcloud.points[:3, :], np.dot(vehicle_flat_from_vehicle, vehicle_from_sensor), normalize=False
            )
            dists = np.sqrt(np.sum(pointcloud.points[:2, :] ** 2, axis=0))
            colors = np.minimum(1, dists / axes_limit / np.sqrt(2))
            ax.scatter(points[0, :], points[1, :], c=colors, s=0.2)

            # Show ego vehicle.
            ax.plot(0, 0, "x", color="red")

            # Show boxes.
            if with_annotations:
                for box in boxes:
                    c = np.array(self.get_color(box.name)) / 255.0
                    box.render(ax, view=np.eye(4), colors=(c, c, c))

            # Limit visible range.
            ax.set_xlim(-axes_limit, axes_limit)
            ax.set_ylim(-axes_limit, axes_limit)

        elif sensor_modality == "camera":
            # Load boxes and image.
            data_path, boxes, camera_intrinsic = self.lyftd.get_sample_data(
                sample_data_token, box_vis_level=box_vis_level
            )

            data = Image.open(data_path)

            # Init axes.
            if ax is None:
                _, ax = plt.subplots(1, 1, figsize=(9, 16))

            # Show image.
            ax.imshow(data)

            # Show boxes.
            if with_annotations:
                for box in boxes:
                    c = np.array(self.get_color(box.name)) / 255.0
                    box.render(ax, view=camera_intrinsic, normalize=True, colors=(c, c, c))

            # Limit visible range.
            ax.set_xlim(0, data.size[0])
            ax.set_ylim(data.size[1], 0)

        else:
            raise ValueError(f"Error: Unknown sensor modality! Got {sensor_modality}")

        ax.axis("off")
        ax.set_title(sd_record["channel"])
        ax.set_aspect("equal")

        if out_path is not None:
            plt.savefig(Path(out_path) / f"{channel}_{sample_data_token}.png")
            plt.close("all")

            return Path(out_path) / f"{channel}_{sample_data_token}.png"

        return out_path

    def render_annotation(
        self,
        ann_token: str,
        margin: float = 10,
        view: np.ndarray = np.eye(4),
        box_vis_level: BoxVisibility = BoxVisibility.ANY,
        out_path: str = None,
    ) -> None:
        """Render selected annotation.

        Args:
            ann_token: Sample_annotation token.
            margin: How many meters in each direction to include in LIDAR view.
            view: LIDAR view point.
            box_vis_level: If sample_data is an image, this sets required visibility for boxes.
            out_path: Optional path to save the rendered figure to disk.

        """

        ann_record = self.lyftd.get("sample_annotation", ann_token)
        sample_record = self.lyftd.get("sample", ann_record["sample_token"])

        if "LIDAR_TOP" not in sample_record["data"].keys():
            raise KeyError("No LIDAR_TOP in data, cant render")

        _, axes = plt.subplots(1, 2, figsize=(18, 9))

        # Figure out which camera the object is fully visible in (this may return nothing)
        boxes: List[Box] = []
        cams = [key for key in sample_record["data"].keys() if "CAM" in key]
        for cam in cams:
            _, boxes, _ = self.lyftd.get_sample_data(
                sample_record["data"][cam], box_vis_level=box_vis_level, selected_anntokens=[ann_token]
            )
            if len(boxes) > 0:
                break  # We found an image that matches. Let's abort.

        if len(boxes) == 0:
            raise ValueError("Could not find image where annotation is visible. Try using e.g. BoxVisibility.ANY.")

        if len(boxes) >= 2:
            raise KeyError(f"Found {len(boxes)} annotations. But we should have only one. Something is wrong!")

        cam = sample_record["data"][cam]

        # Plot LIDAR view
        lidar = sample_record["data"]["LIDAR_TOP"]
        data_path, boxes, camera_intrinsic = self.lyftd.get_sample_data(lidar, selected_anntokens=[ann_token])
        LidarPointCloud.from_file(data_path).render_height(axes[0], view=view)
        for box in boxes:
            c = np.array(self.get_color(box.name)) / 255.0
            box.render(axes[0], view=view, colors=(c, c, c))
            corners = view_points(boxes[0].corners(), view, False)[:2, :]
            axes[0].set_xlim([np.min(corners[0, :]) - margin, np.max(corners[0, :]) + margin])
            axes[0].set_ylim([np.min(corners[1, :]) - margin, np.max(corners[1, :]) + margin])
            axes[0].axis("off")
            axes[0].set_aspect("equal")

        # Plot CAMERA view
        data_path, boxes, camera_intrinsic = self.lyftd.get_sample_data(cam, selected_anntokens=[ann_token])
        im = Image.open(data_path)
        axes[1].imshow(im)
        axes[1].set_title(self.lyftd.get("sample_data", cam)["channel"])
        axes[1].axis("off")
        axes[1].set_aspect("equal")
        for box in boxes:
            c = np.array(self.get_color(box.name)) / 255.0
            box.render(axes[1], view=camera_intrinsic, normalize=True, colors=(c, c, c))

        if out_path is not None:
            plt.savefig(out_path)

    def render_instance(self, instance_token: str, out_path: str = None) -> None:
        """Finds the annotation of the given instance that is closest to the vehicle, and then renders it.

        Args:
            instance_token: The instance token.
            out_path: Optional path to save the rendered figure to disk.

        Returns:

        """

        ann_tokens = self.lyftd.field2token("sample_annotation", "instance_token", instance_token)
        closest = [np.inf, None]
        for ann_token in ann_tokens:
            ann_record = self.lyftd.get("sample_annotation", ann_token)
            sample_record = self.lyftd.get("sample", ann_record["sample_token"])
            sample_data_record = self.lyftd.get("sample_data", sample_record["data"]["LIDAR_TOP"])
            pose_record = self.lyftd.get("ego_pose", sample_data_record["ego_pose_token"])
            dist = np.linalg.norm(np.array(pose_record["translation"]) - np.array(ann_record["translation"]))
            if dist < closest[0]:
                closest[0] = dist
                closest[1] = ann_token
        self.render_annotation(closest[1], out_path=out_path)

    def render_scene(self, scene_token: str, freq: float = 10, image_width: int = 640, out_path: Path = None) -> None:
        """Renders a full scene with all surround view camera channels.

        Args:
            scene_token: Unique identifier of scene to render.
            freq: Display frequency (Hz).
            image_width: Width of image to render. Height is determined automatically to preserve aspect ratio.
            out_path: Optional path to write a video file of the rendered frames.

        """

        if out_path is not None:
            assert out_path.suffix == ".avi"

        # Get records from DB.
        scene_rec = self.lyftd.get("scene", scene_token)
        first_sample_rec = self.lyftd.get("sample", scene_rec["first_sample_token"])
        last_sample_rec = self.lyftd.get("sample", scene_rec["last_sample_token"])

        channels = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]

        horizontal_flip = ["CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]  # Flip these for aesthetic reasons.

        time_step = 1 / freq * 1e6  # Time-stamps are measured in micro-seconds.

        window_name = f"{scene_rec['name']}"
        cv2.namedWindow(window_name)
        cv2.moveWindow(window_name, 0, 0)

        # Load first sample_data record for each channel
        current_recs: Dict = {}  # Holds the current record to be displayed by channel.
        prev_recs: Dict = {}  # Hold the previous displayed record by channel.
        for channel in channels:
            current_recs[channel] = self.lyftd.get("sample_data", first_sample_rec["data"][channel])
            prev_recs[channel] = None

        # We assume that the resolution is the same for all surround view cameras.
        image_height = int(image_width * current_recs[channels[0]]["height"] / current_recs[channels[0]]["width"])
        image_size = (image_width, image_height)

        # Set some display parameters
        layout = {
            "CAM_FRONT_LEFT": (0, 0),
            "CAM_FRONT": (image_size[0], 0),
            "CAM_FRONT_RIGHT": (2 * image_size[0], 0),
            "CAM_BACK_LEFT": (0, image_size[1]),
            "CAM_BACK": (image_size[0], image_size[1]),
            "CAM_BACK_RIGHT": (2 * image_size[0], image_size[1]),
        }

        canvas = np.ones((2 * image_size[1], 3 * image_size[0], 3), np.uint8)
        if out_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            out = cv2.VideoWriter(str(out_path), fourcc, freq, canvas.shape[1::-1])
        else:
            out = None

        current_time = first_sample_rec["timestamp"]

        while current_time < last_sample_rec["timestamp"]:
            current_time += time_step

            # For each channel, find first sample that has time > current_time.
            for channel, sd_rec in current_recs.items():
                while sd_rec["timestamp"] < current_time and sd_rec["next"] != "":
                    sd_rec = self.lyftd.get("sample_data", sd_rec["next"])
                    current_recs[channel] = sd_rec

            # Now add to canvas
            for channel, sd_rec in current_recs.items():

                # Only update canvas if we have not already rendered this one.
                if not sd_rec == prev_recs[channel]:

                    # Get annotations and params from DB.
                    image_path, boxes, camera_intrinsic = self.lyftd.get_sample_data(
                        sd_rec["token"], box_vis_level=BoxVisibility.ANY
                    )

                    # Load and render
                    if not image_path.exists():
                        raise Exception("Error: Missing image %s" % image_path)
                    image = cv2.imread(str(image_path))
                    for box in boxes:
                        c = self.get_color(box.name)
                        box.render_cv2(image, view=camera_intrinsic, normalize=True, colors=(c, c, c))

                    image = cv2.resize(image, image_size)
                    if channel in horizontal_flip:
                        image = image[:, ::-1, :]

                    canvas[
                        layout[channel][1] : layout[channel][1] + image_size[1],
                        layout[channel][0] : layout[channel][0] + image_size[0],
                        :,
                    ] = image

                    prev_recs[channel] = sd_rec  # Store here so we don't render the same image twice.

            # Show updated canvas.
            cv2.imshow(window_name, canvas)
            if out_path is not None:
                out.write(canvas)

            key = cv2.waitKey(1)  # Wait a very short time (1 ms).

            if key == 32:  # if space is pressed, pause.
                key = cv2.waitKey()

            if key == 27:  # if ESC is pressed, exit.
                cv2.destroyAllWindows()
                break

        cv2.destroyAllWindows()
        if out_path is not None:
            out.release()

    def render_scene_channel(
        self,
        scene_token: str,
        channel: str = "CAM_FRONT",
        freq: float = 10,
        image_size: Tuple[float, float] = (640, 360),
        out_path: Path = None,
        interactive: bool = True,
        verbose: bool = False,
    ) -> None:
        """Renders a full scene for a particular camera channel.

        Args:
            scene_token: Unique identifier of scene to render.
            channel: Channel to render.
            freq: Display frequency (Hz).
            image_size: Size of image to render. The larger the slower this will run.
            out_path: Optional path to write a video file of the rendered frames.
            interactive: show video in a cv2 window (True by default).
            verbose: set to True to print currently processed token.

        """

        valid_channels = [
            "CAM_FRONT_LEFT",
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_LEFT",
            "CAM_BACK",
            "CAM_BACK_RIGHT",
        ]

        if image_size[0] / image_size[1] != 16 / 9:
            raise ValueError("Aspect ratio should be 16/9.")

        if channel not in valid_channels:
            raise ValueError(f"Input channel {channel} not valid.")

        if out_path is not None:
            assert out_path.suffix == ".avi"

        # Get records from DB
        scene_rec = self.lyftd.get("scene", scene_token)
        sample_rec = self.lyftd.get("sample", scene_rec["first_sample_token"])
        sd_rec = self.lyftd.get("sample_data", sample_rec["data"][channel])

        # Open CV init
        if interactive:
            name = f"{scene_rec['name']}: {channel} (Space to pause, ESC to exit)"
            cv2.namedWindow(name)
            cv2.moveWindow(name, 0, 0)

        if out_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            out = cv2.VideoWriter(str(out_path), fourcc, freq, image_size)
        else:
            out = None

        has_more_frames = True
        while has_more_frames:
            if verbose:
                print(sd_rec["token"])

            # Get data from DB
            image_path, boxes, camera_intrinsic = self.lyftd.get_sample_data(
                sd_rec["token"], box_vis_level=BoxVisibility.ANY
            )

            # Load and render
            if not image_path.exists():
                raise FileNotFoundError(f"Error: Missing image {image_path}")

            image = cv2.imread(str(image_path))
            for box in boxes:
                c = self.get_color(box.name)
                box.render_cv2(image, view=camera_intrinsic, normalize=True, colors=(c, c, c))

            # Render
            image = cv2.resize(image, image_size)
            if interactive:
                cv2.imshow(name, image)
            if out_path is not None:
                out.write(image)

            key = cv2.waitKey(10)  # Images stored at approx 10 Hz, so wait 10 ms.
            if key == 32:  # If space is pressed, pause.
                key = cv2.waitKey()

            if key == 27:  # if ESC is pressed, exit
                cv2.destroyAllWindows()
                break

            if not sd_rec["next"] == "":
                sd_rec = self.lyftd.get("sample_data", sd_rec["next"])
            else:
                has_more_frames = False

        if interactive:
            cv2.destroyAllWindows()
        if out_path is not None:
            out.release()

    def render_egoposes_on_map(
        self,
        log_location: str,
        scene_tokens: List = None,
        close_dist: float = 100,
        color_fg: Tuple[int, int, int] = (167, 174, 186),
        color_bg: Tuple[int, int, int] = (255, 255, 255),
        out_path: Path = None,
    ) -> None:
        """Renders ego poses a the map. These can be filtered by location or scene.

        Args:
            log_location: Name of the location, e.g. "singapore-onenorth", "singapore-hollandvillage",
                             "singapore-queenstown' and "boston-seaport".
            scene_tokens: Optional list of scene tokens.
            close_dist: Distance in meters for an ego pose to be considered within range of another ego pose.
            color_fg: Color of the semantic prior in RGB format (ignored if map is RGB).
            color_bg: Color of the non-semantic prior in RGB format (ignored if map is RGB).
            out_path: Optional path to save the rendered figure to disk.

        Returns:

        """

        # Get logs by location
        log_tokens = [x["token"] for x in self.lyftd.log if x["location"] == log_location]
        assert len(log_tokens) > 0, "Error: This split has 0 scenes for location %s!" % log_location

        # Filter scenes
        scene_tokens_location = [e["token"] for e in self.lyftd.scene if e["log_token"] in log_tokens]
        if scene_tokens is not None:
            scene_tokens_location = [t for t in scene_tokens_location if t in scene_tokens]
        if len(scene_tokens_location) == 0:
            print("Warning: Found 0 valid scenes for location %s!" % log_location)

        map_poses = []

        print("Adding ego poses to map...")
        for scene_token in tqdm(scene_tokens_location):

            # Get records from the database.
            scene_record = self.lyftd.get("scene", scene_token)
            log_record = self.lyftd.get("log", scene_record["log_token"])
            map_record = self.lyftd.get("map", log_record["map_token"])
            map_mask = map_record["mask"]

            # For each sample in the scene, store the ego pose.
            sample_tokens = self.lyftd.field2token("sample", "scene_token", scene_token)
            for sample_token in sample_tokens:
                sample_record = self.lyftd.get("sample", sample_token)

                # Poses are associated with the sample_data. Here we use the lidar sample_data.
                sample_data_record = self.lyftd.get("sample_data", sample_record["data"]["LIDAR_TOP"])
                pose_record = self.lyftd.get("ego_pose", sample_data_record["ego_pose_token"])

                # Calculate the pose on the map and append
                map_poses.append(
                    np.concatenate(
                        map_mask.to_pixel_coords(pose_record["translation"][0], pose_record["translation"][1])
                    )
                )

        # Compute number of close ego poses.
        print("Creating plot...")
        map_poses = np.vstack(map_poses)
        dists = sklearn.metrics.pairwise.euclidean_distances(map_poses * map_mask.resolution)
        close_poses = np.sum(dists < close_dist, axis=0)

        if len(np.array(map_mask.mask()).shape) == 3 and np.array(map_mask.mask()).shape[2] == 3:
            # RGB Colour maps
            mask = map_mask.mask()
        else:
            # Monochrome maps
            # Set the colors for the mask.
            mask = Image.fromarray(map_mask.mask())
            mask = np.array(mask)

            maskr = color_fg[0] * np.ones(np.shape(mask), dtype=np.uint8)
            maskr[mask == 0] = color_bg[0]
            maskg = color_fg[1] * np.ones(np.shape(mask), dtype=np.uint8)
            maskg[mask == 0] = color_bg[1]
            maskb = color_fg[2] * np.ones(np.shape(mask), dtype=np.uint8)
            maskb[mask == 0] = color_bg[2]
            mask = np.concatenate(
                (np.expand_dims(maskr, axis=2), np.expand_dims(maskg, axis=2), np.expand_dims(maskb, axis=2)), axis=2
            )

        # Plot.
        _, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(mask)
        title = f"Number of ego poses within {close_dist}m in {log_location}"
        ax.set_title(title, color="k")
        sc = ax.scatter(map_poses[:, 0], map_poses[:, 1], s=10, c=close_poses)  # type: ignore
        color_bar = plt.colorbar(sc, fraction=0.025, pad=0.04)
        plt.rcParams["figure.facecolor"] = "black"
        color_bar_ticklabels = plt.getp(color_bar.ax.axes, "yticklabels")
        plt.setp(color_bar_ticklabels, color="k")
        plt.rcParams["figure.facecolor"] = "white"  # Reset for future plots

        if out_path is not None:
            plt.savefig(str(out_path))
            plt.close("all")
