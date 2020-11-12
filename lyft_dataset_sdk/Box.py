# Lyft Dataset SDK dev-kit.
# Code written by Oscar Beijbom, 2018.
# Licensed under the Creative Commons [see licence.txt]
# Modified by Vladimir Iglovikov 2019.

import copy
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from matplotlib.axes import Axes
from pyquaternion import Quaternion


class Box:
    """Simple data class representing a 3d box including, label, score and velocity."""

    def __init__(
        self,
        center: Union[List[float], Tuple[float, float, float]],
        size: Union[List[float], Tuple[float, float, float]],
        orientation: Quaternion,
        label: int = np.nan,
        score: float = np.nan,
        velocity: Tuple = (np.nan, np.nan, np.nan),
        name: Optional[str] = None,
        token: Optional[str] = None,
    ):
        """

        Args:
            center: Center of box given as x, y, z.
            size: Size of box in width, length, height.
            orientation: Box orientation.
            label: Integer label, optional.
            score: Classification score, optional.
            velocity: Box velocity in x, y, z direction.
            name: Box name, optional. Can be used e.g. for denote category name.
            token: Unique string identifier from DB.
        """
        if np.any(np.isnan(center)):
            raise ValueError(f"Center coordinates should not have NaN values but we got {center}")

        if np.any(np.isnan(size)):
            raise ValueError(f"Size values should not have NaN values but we got {size}")

        if len(center) != 3:
            raise ValueError(f"Center should be defined by 3 numbers but we got {center}")

        if len(size) != 3:
            raise ValueError(f"Size should be defined by 3 numbers but we got {size}")

        if not isinstance(orientation, Quaternion):
            raise TypeError("The orientation should be a Quaternion but we got {}".format(type(orientation)))

        self.center = np.array(center)
        self.wlh = np.array(size)
        self.orientation = orientation
        self.label = int(label) if not np.isnan(label) else label
        self.score = float(score) if not np.isnan(score) else score
        self.velocity = np.array(velocity)
        self.name = name
        self.token = token

    def __eq__(self, other: "Box") -> bool:  # type: ignore
        if not isinstance(other, Box):
            return NotImplemented

        center = np.allclose(self.center, other.center)
        wlh = np.allclose(self.wlh, other.wlh)
        orientation = np.allclose(self.orientation.elements, other.orientation.elements)
        label = (self.label == other.label) or (np.isnan(self.label) and np.isnan(other.label))
        score = (self.score == other.score) or (np.isnan(self.score) and np.isnan(other.score))
        vel = np.allclose(self.velocity, other.velocity) or (
            np.all(np.isnan(self.velocity)) and np.all(np.isnan(other.velocity))
        )

        return center and wlh and orientation and label and score and vel

    def __repr__(self):
        repr_str = (
            "label: {}, score: {:.2f}, xyz: [{:.2f}, {:.2f}, {:.2f}], wlh: [{:.2f}, {:.2f}, {:.2f}], "
            "rot axis: [{:.2f}, {:.2f}, {:.2f}], ang(degrees): {:.2f}, ang(rad): {:.2f}, "
            "vel: {:.2f}, {:.2f}, {:.2f}, name: {}, token: {}"
        )

        return repr_str.format(
            self.label,
            self.score,
            self.center[0],
            self.center[1],
            self.center[2],
            self.wlh[0],
            self.wlh[1],
            self.wlh[2],
            self.orientation.axis[0],
            self.orientation.axis[1],
            self.orientation.axis[2],
            self.orientation.degrees,
            self.orientation.radians,
            self.velocity[0],
            self.velocity[1],
            self.velocity[2],
            self.name,
            self.token,
        )

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Return a rotation matrix.

        Returns: <np.float: 3, 3>. The box's rotation matrix.

        """
        return self.orientation.rotation_matrix

    def translate(self, x: Union[np.ndarray, List[float], Tuple[float]]) -> "Box":
        """Applies a translation.

        Args:
            x: <np.float: 3, 1>. Translation in x, y, z direction.

        Returns: translated Box

        """
        self.center += x

        return self

    def rotate(self, **kwargs) -> None:  # type: ignore
        raise DeprecationWarning(
            "rotate method is deprecated. Use `rotate_around_origin` " "and `rotate_around_box_center` instead."
        )

    def rotate_around_origin(self, quaternion: Quaternion) -> "Box":
        """Rotates the box around (0, 0, 0).

        Args:
            quaternion: Rotation to apply.

        Returns: rotated box

        """
        rotation_matrix = quaternion.rotation_matrix

        self.center = np.dot(rotation_matrix, self.center)
        self.orientation = quaternion * self.orientation
        self.velocity = np.dot(rotation_matrix, self.velocity)

        return self

    def rotate_around_box_center(self, quaternion: Quaternion) -> "Box":
        """Rotates the box around it's center.

        Args:
            quaternion: Rotation to apply.

        Returns: rotated box

        """
        self.orientation = quaternion * self.orientation
        self.velocity = np.dot(quaternion.rotation_matrix, self.velocity)

        return self

    def corners(self, wlh_factor: float = 1.0) -> np.ndarray:
        """Returns the bounding box corners.

        Args:
            wlh_factor: Multiply width, length, height by a factor to scale the box.

        Returns: First four corners are the ones facing forward.
                The last four are the ones facing backwards.

        """

        width, length, height = self.wlh * wlh_factor

        # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
        x_corners = length / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
        y_corners = width / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
        z_corners = height / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
        corners = np.vstack((x_corners, y_corners, z_corners))

        # Rotate
        corners = np.dot(self.orientation.rotation_matrix, corners)

        # Translate
        x, y, z = self.center
        corners[0, :] = corners[0, :] + x
        corners[1, :] = corners[1, :] + y
        corners[2, :] = corners[2, :] + z

        return corners

    def bottom_corners(self) -> np.ndarray:
        """Returns the four bottom corners.

        Returns: <np.float: 3, 4>. Bottom corners. First two face forward, last two face backwards.

        """
        return self.corners()[:, [2, 3, 7, 6]]

    def render(
        self,
        axis: Axes,
        view: np.ndarray = np.eye(3),
        normalize: bool = False,
        colors: Tuple = ("b", "r", "k"),
        linewidth: float = 2,
    ) -> None:
        """Renders the box in the provided Matplotlib axis.

        Args:
            axis: Axis onto which the box should be drawn.
            view: <np.array: 3, 3>. Define a projection in needed (e.g. for drawing projection in an image).
            normalize: Whether to normalize the remaining coordinate.
            colors: (<Matplotlib.colors>: 3). Valid Matplotlib colors (<str> or normalized RGB tuple) for front,
            back and sides.
            linewidth: Width in pixel of the box sides.

        """
        corners = view_points(self.corners(), view, normalize=normalize)[:2, :]

        def draw_rect(selected_corners, color):
            prev = selected_corners[-1]
            for corner in selected_corners:
                axis.plot([prev[0], corner[0]], [prev[1], corner[1]], color=color, linewidth=linewidth)
                prev = corner

        # Draw the sides
        for i in range(4):
            axis.plot(
                [corners.T[i][0], corners.T[i + 4][0]],
                [corners.T[i][1], corners.T[i + 4][1]],
                color=colors[2],
                linewidth=linewidth,
            )

        # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
        draw_rect(corners.T[:4], colors[0])
        draw_rect(corners.T[4:], colors[1])

        # Draw line indicating the front
        center_bottom_forward = np.mean(corners.T[2:4], axis=0)
        center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
        axis.plot(
            [center_bottom[0], center_bottom_forward[0]],
            [center_bottom[1], center_bottom_forward[1]],
            color=colors[0],
            linewidth=linewidth,
        )

    def render_cv2(
        self,
        image: np.ndarray,
        view: np.ndarray = np.eye(3),
        normalize: bool = False,
        colors: Tuple = ((0, 0, 255), (255, 0, 0), (155, 155, 155)),
        linewidth: int = 2,
    ) -> None:
        """Renders box using OpenCV2.

        Args:
            image: <np.array: width, height, 3>. Image array. Channels are in BGR order.
            view: <np.array: 3, 3>. Define a projection if needed (e.g. for drawing projection in an image).
            normalize: Whether to normalize the remaining coordinate.
            colors: ((R, G, B), (R, G, B), (R, G, B)). Colors for front, side & rear.
            linewidth: Linewidth for plot.

        Returns:

        """
        corners = view_points(self.corners(), view, normalize=normalize)[:2, :]

        def draw_rect(selected_corners, color):
            prev = selected_corners[-1]
            for corner in selected_corners:
                cv2.line(image, (int(prev[0]), int(prev[1])), (int(corner[0]), int(corner[1])), color, linewidth)
                prev = corner

        # Draw the sides
        for i in range(4):
            cv2.line(
                image,
                (int(corners.T[i][0]), int(corners.T[i][1])),
                (int(corners.T[i + 4][0]), int(corners.T[i + 4][1])),
                colors[2][::-1],
                linewidth,
            )

        # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
        draw_rect(corners.T[:4], colors[0][::-1])
        draw_rect(corners.T[4:], colors[1][::-1])

        # Draw line indicating the front
        center_bottom_forward = np.mean(corners.T[2:4], axis=0)
        center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
        cv2.line(
            image,
            (int(center_bottom[0]), int(center_bottom[1])),
            (int(center_bottom_forward[0]), int(center_bottom_forward[1])),
            colors[0][::-1],
            linewidth,
        )

    def copy(self) -> "Box":
        """Create a copy of self.

        Returns: A copy.

        """
        return copy.deepcopy(self)


def view_points(points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
    """This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
    orthographic projections. It first applies the dot product between the points and the view. By convention,
    the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
    normalization along the third dimension.

    For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
    For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
    For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
     all zeros) and normalize=False

    Args:
        points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
        view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
        The projection should be such that the corners are projected onto the first 2 axis.
        normalize: Whether to normalize the remaining coordinate (along the third axis).

    Returns: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.

    """

    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[: view.shape[0], : view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points
