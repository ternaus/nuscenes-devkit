# Lyft Dataset SDK
# Code written by Oscar Beijbom and Alex Lang, 2018.
# Licensed under the Creative Commons [see licence.txt]
# Modified by Vladimir Iglovikov 2019.

from enum import IntEnum
from typing import Tuple

import numpy as np
from pyquaternion import Quaternion

from lyft_dataset_sdk.Box import Box, view_points


class BoxVisibility(IntEnum):
    """Enumerates the various level of box visibility in an image."""

    ALL = 0  # Requires all corners are inside the image.
    ANY = 1  # Requires at least one corner visible in the image.
    NONE = 2  # Requires no corners to be inside, i.e. box can be fully outside the image.


def box_in_image(
    box: Box, intrinsic: np.ndarray, image_size: Tuple[int, int], vis_level: int = BoxVisibility.ANY
) -> bool:
    """Check if a box is visible inside an image without accounting for occlusions.

    Args:
        box: The box to be checked.
        intrinsic: <float: 3, 3>. Intrinsic camera matrix.
        image_size: (width, height)
        vis_level: One of the enumerations of <BoxVisibility>.

    Returns: True if visibility condition is satisfied.

    """

    width, height = image_size

    corners_3d = box.corners()
    corners_img = view_points(corners_3d, intrinsic, normalize=True)[:2, :]

    visible = np.logical_and(corners_img[0, :] > 0, corners_img[0, :] < width)
    visible = np.logical_and(visible, corners_img[1, :] < height)
    visible = np.logical_and(visible, corners_img[1, :] > 0)
    visible = np.logical_and(visible, corners_3d[2, :] > 1)

    in_front = corners_3d[2, :] > 0.1  # True if a corner is at least 0.1 meter in front of the camera.

    if vis_level in [BoxVisibility.ALL, BoxVisibility.ANY]:  # pylint: disable=R1705
        return all(visible) and all(in_front)
    elif vis_level == BoxVisibility.NONE:
        return True

    raise ValueError(f"vis_level: {vis_level} not valid")


def transform_matrix(
    translation: np.ndarray = np.array([0, 0, 0]),
    rotation: Quaternion = Quaternion([1, 0, 0, 0]),
    inverse: bool = False,
) -> np.ndarray:
    """Convert pose to transformation matrix.

    Args:
        translation: <np.float32: 3>. Translation in x, y, z.
        rotation: Rotation in quaternions (w ri rj rk).
        inverse: Whether to compute inverse transform matrix.

    Returns: <np.float32: 4, 4>. Transformation matrix.

    """
    tm = np.eye(4)

    if inverse:
        rot_inv = rotation.rotation_matrix.T
        trans = np.transpose(-np.array(translation))
        tm[:3, :3] = rot_inv
        tm[:3, 3] = rot_inv.dot(trans)
    else:
        tm[:3, :3] = rotation.rotation_matrix
        tm[:3, 3] = np.transpose(np.array(translation))

    return tm


def points_in_box(box: Box, points: np.ndarray, wlh_factor: float = 1.0) -> np.ndarray:
    """Checks whether points are inside the box.

    Picks one corner as reference (p1) and computes the vector to a target point (v).
    Then for each of the 3 axes, project v onto the axis and compare the length.
    Inspired by: https://math.stackexchange.com/a/1552579

    Args:
        box: <Box>.
        points: <np.float: 3, n>.
        wlh_factor: Inflates or deflates the box.

    Returns: <np.bool: n, >.

    """
    corners = box.corners(wlh_factor=wlh_factor)

    p1 = corners[:, 0]
    p_x = corners[:, 4]
    p_y = corners[:, 1]
    p_z = corners[:, 3]

    i = p_x - p1
    j = p_y - p1
    k = p_z - p1

    v = points - p1.reshape((-1, 1))

    iv = np.dot(i, v)
    jv = np.dot(j, v)
    kv = np.dot(k, v)

    mask_x = np.logical_and(iv >= 0, iv <= np.dot(i, i))
    mask_y = np.logical_and(jv >= 0, jv <= np.dot(j, j))
    mask_z = np.logical_and(kv >= 0, kv <= np.dot(k, k))
    mask = np.logical_and(np.logical_and(mask_x, mask_y), mask_z)

    return mask


def quaternion_yaw(q: Quaternion) -> float:
    """Calculate the yaw angle from a quaternion.

    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.

    Args:
        q: Quaternion of interest.

    Returns: Yaw angle in radians.

    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    return np.arctan2(v[1], v[0])
