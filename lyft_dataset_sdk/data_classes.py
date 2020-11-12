# Lyft Dataset SDK dev-kit.
# Code written by Oscar Beijbom, 2018.
# Licensed under the Creative Commons [see licence.txt]
# Modified by Vladimir Iglovikov 2019.

from abc import ABC, abstractmethod
from functools import reduce
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from matplotlib.axes import Axes
from pyquaternion import Quaternion

from lyft_dataset_sdk.geometry_utils import transform_matrix, view_points
from lyft_dataset_sdk.lyftdataset import LyftDataset


class PointCloud(ABC):
    """
    Abstract class for manipulating and viewing point clouds.
    Every lidar point cloud consists of points where:
    - Dimensions 0, 1, 2 represent x, y, z coordinates.
        These are modified when the point cloud is rotated or translated.
    - All other dimensions are optional. Hence these have to be manually modified if the reference frame changes.
    """

    def __init__(self, points: np.ndarray):
        """Initialize a point cloud and check it has the correct dimensions.

        Args:
            points: <np.float: d, n>. d-dimensional input point cloud matrix.
        """
        if points.shape[0] != self.nbr_dims():
            raise ValueError(f"Error: Pointcloud points must have format: {self.nbr_dims()} x n")
        self.points = points

    @staticmethod
    @abstractmethod
    def nbr_dims() -> int:
        """Returns the number of dimensions."""
        pass

    @classmethod
    @abstractmethod
    def from_file(cls, file_name: Path) -> "PointCloud":
        """Loads point cloud from disk.

        Args:
            file_name: Path of the pointcloud file on disk.

        Returns: PointCloud instance.

        """
        pass

    @classmethod
    def from_file_multisweep(
        cls,
        lyftd: LyftDataset,
        sample_rec: Dict,
        chan: str,
        ref_chan: str,
        num_sweeps: int = 26,
        min_distance: float = 1.0,
    ) -> Tuple["PointCloud", np.ndarray]:
        """Return a point cloud that aggregates multiple sweeps.
        As every sweep is in a different coordinate frame, we need to map the coordinates to a single reference frame.
        As every sweep has a different timestamp, we need to account for that in the transformations and timestamps.

        Args:
            lyftd: A LyftDataset instance.
            sample_rec: The current sample.
            chan: The radar channel from which we track back n sweeps to aggregate the point cloud.
            ref_chan: The reference channel of the current sample_rec that the point clouds are mapped to.
            num_sweeps: Number of sweeps to aggregated.
            min_distance: Distance below which points are discarded.

        Returns: (all_pc, all_times). The aggregated point cloud and timestamps.

        """

        # Init
        points = np.zeros((cls.nbr_dims(), 0))
        all_pc = cls(points)
        all_times = np.zeros((1, 0))

        # Get reference pose and timestamp
        ref_sd_token = sample_rec["data"][ref_chan]
        ref_sd_rec = lyftd.get("sample_data", ref_sd_token)
        ref_pose_rec = lyftd.get("ego_pose", ref_sd_rec["ego_pose_token"])
        ref_cs_rec = lyftd.get("calibrated_sensor", ref_sd_rec["calibrated_sensor_token"])
        ref_time = 1e-6 * ref_sd_rec["timestamp"]

        # Homogeneous transform from ego car frame to reference frame
        ref_from_car = transform_matrix(ref_cs_rec["translation"], Quaternion(ref_cs_rec["rotation"]), inverse=True)

        # Homogeneous transformation matrix from global to _current_ ego car frame
        car_from_global = transform_matrix(
            ref_pose_rec["translation"], Quaternion(ref_pose_rec["rotation"]), inverse=True
        )

        # Aggregate current and previous sweeps.
        sample_data_token = sample_rec["data"][chan]
        current_sd_rec = lyftd.get("sample_data", sample_data_token)
        for _ in range(num_sweeps):
            # Load up the pointcloud.
            current_pc = cls.from_file(lyftd.lidar_path / Path(current_sd_rec["filename"]).name)

            # Get past pose.
            current_pose_rec = lyftd.get("ego_pose", current_sd_rec["ego_pose_token"])
            global_from_car = transform_matrix(
                current_pose_rec["translation"], Quaternion(current_pose_rec["rotation"]), inverse=False
            )

            # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
            current_cs_rec = lyftd.get("calibrated_sensor", current_sd_rec["calibrated_sensor_token"])
            car_from_current = transform_matrix(
                current_cs_rec["translation"], Quaternion(current_cs_rec["rotation"]), inverse=False
            )

            # Fuse four transformation matrices into one and perform transform.
            trans_matrix = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])
            current_pc.transform(trans_matrix)

            # Remove close points and add timevector.
            current_pc.remove_close(min_distance)
            time_lag = ref_time - 1e-6 * current_sd_rec["timestamp"]  # positive difference
            times = time_lag * np.ones((1, current_pc.nbr_points()))
            all_times = np.hstack((all_times, times))

            # Merge with key pc.
            all_pc.points = np.hstack((all_pc.points, current_pc.points))

            # Abort if there are no previous sweeps.
            if current_sd_rec["prev"] == "":
                break

            current_sd_rec = lyftd.get("sample_data", current_sd_rec["prev"])

        return all_pc, all_times

    def nbr_points(self) -> int:
        """Returns the number of points."""
        return self.points.shape[1]

    def subsample(self, ratio: float) -> None:
        """Sub-samples the pointcloud.

        Args:
            ratio: Fraction to keep.

        """
        selected_ind = np.random.choice(np.arange(0, self.nbr_points()), size=int(self.nbr_points() * ratio))
        self.points = self.points[:, selected_ind]

    def remove_close(self, radius: float) -> None:
        """Removes point too close within a certain radius from origin.

        Args:
            radius: Radius below which points are removed.

        Returns:

        """
        x_filt = np.abs(self.points[0, :]) < radius
        y_filt = np.abs(self.points[1, :]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        self.points = self.points[:, not_close]

    def translate(self, x: np.ndarray) -> None:
        """Applies a translation to the point cloud.

        Args:
            x: <np.float: 3, 1>. Translation in x, y, z.

        """
        for i in range(3):
            self.points[i, :] = self.points[i, :] + x[i]

    def rotate(self, rot_matrix: np.ndarray) -> None:
        """Applies a rotation.

        Args:
            rot_matrix: <np.float: 3, 3>. Rotation matrix.

        Returns:

        """
        self.points[:3, :] = np.dot(rot_matrix, self.points[:3, :])

    def transform(self, transf_matrix: np.ndarray) -> None:
        """Applies a homogeneous transform.

        Args:
            transf_matrix: transf_matrix: <np.float: 4, 4>. Homogenous transformation matrix.

        """
        self.points[:3, :] = transf_matrix.dot(np.vstack((self.points[:3, :], np.ones(self.nbr_points()))))[:3, :]

    def render_height(
        self,
        ax: Axes,
        view: np.ndarray = np.eye(4),
        x_lim: Tuple = (-20, 20),
        y_lim: Tuple = (-20, 20),
        marker_size: float = 1,
    ) -> None:
        """Simple method that applies a transformation and then scatter plots the points colored by height (z-value).

        Args:
            ax: Axes on which to render the points.
            view: <np.float: n, n>. Defines an arbitrary projection (n <= 4).
            x_lim: (min <float>, max <float>). x range for plotting.
            y_lim: (min <float>, max <float>). y range for plotting.
            marker_size: Marker size.

        """
        self._render_helper(2, ax, view, x_lim, y_lim, marker_size)

    def render_intensity(
        self,
        ax: Axes,
        view: np.ndarray = np.eye(4),
        x_lim: Tuple = (-20, 20),
        y_lim: Tuple = (-20, 20),
        marker_size: float = 1,
    ) -> None:
        """Very simple method that applies a transformation and then scatter plots the points colored by intensity.

        Args:
            ax: Axes on which to render the points.
            view: <np.float: n, n>. Defines an arbitrary projection (n <= 4).
            x_lim: (min <float>, max <float>).
            y_lim: (min <float>, max <float>).
            marker_size: Marker size.

        Returns:

        """
        self._render_helper(3, ax, view, x_lim, y_lim, marker_size)

    def _render_helper(
        self, color_channel: int, ax: Axes, view: np.ndarray, x_lim: Tuple, y_lim: Tuple, marker_size: float
    ) -> None:
        """Helper function for rendering.

        Args:
            color_channel: Point channel to use as color.
            ax: Axes on which to render the points.
            view: <np.float: n, n>. Defines an arbitrary projection (n <= 4).
            x_lim: (min <float>, max <float>).
            y_lim: (min <float>, max <float>).
            marker_size: Marker size.

        """
        points = view_points(self.points[:3, :], view, normalize=False)
        ax.scatter(points[0, :], points[1, :], c=self.points[color_channel, :], s=marker_size)
        ax.set_xlim(x_lim[0], x_lim[1])
        ax.set_ylim(y_lim[0], y_lim[1])


class LidarPointCloud(PointCloud):
    @staticmethod
    def nbr_dims() -> int:
        """Returns the number of dimensions."""
        return 4

    @classmethod
    def from_file(cls, file_name: Path) -> "LidarPointCloud":
        """Loads LIDAR data from binary numpy format. Data is stored as (x, y, z, intensity, ring index).

        Args:
            file_name: Path of the pointcloud file on disk.

        Returns: LidarPointCloud instance (x, y, z, intensity).

        """

        assert file_name.suffix == ".bin", f"Unsupported filetype {file_name}"

        scan = np.fromfile(str(file_name), dtype=np.float32)
        points = scan.reshape((-1, 5))[:, : cls.nbr_dims()]
        return cls(points.T)
