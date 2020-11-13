# Lyft Dataset SDK
# Code written by Qiang Xu and Oscar Beijbom, 2018.
# Licensed under the Creative Commons [see licence.txt]
# Modified by Vladimir Iglovikov 2019.

from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import numpy as np
from cachetools import LRUCache, cached
from PIL import Image

# Set the maximum loadable image size.
Image.MAX_IMAGE_PIXELS = 400000 * 400000


class MapMask:
    def __init__(self, image_path: Optional[str], resolution: float = 0.1) -> None:
        """Init a map mask object that contains the semantic prior (drivable surface and sidewalks) mask.

        Args:
            image_path: File path to map png file.
            resolution: Map resolution in meters.
        """
        if image_path is None:
            self.is_none = True
        else:
            assert Path(image_path).exists(), f"map mask {image_path} does not exist"
            assert resolution >= 0.1, "Only supports down to 0.1 meter resolution."
            self.img_file = image_path
            self.resolution = resolution
            self.foreground = 255
            self.background = 0
            self.is_none = False

    @cached(cache=LRUCache(maxsize=3))
    def mask(self, dilation: float = 0.0) -> Optional[np.ndarray]:
        """Returns the map mask, optionally dilated.

        Args:
            dilation: Dilation in meters.

        Returns: Dilated map mask.

        """
        if self.is_none:
            return None

        if dilation == 0:
            return self._base_mask

        distance_mask = cv2.distanceTransform(
            (np.ones_like(self._base_mask) * self.foreground - self._base_mask).astype(np.uint8), cv2.DIST_L2, 5
        )
        distance_mask = (distance_mask * self.resolution).astype(np.float32)
        return (distance_mask <= dilation).astype(np.uint8) * self.foreground

    @property
    def transform_matrix(self) -> Optional[np.ndarray]:
        """Generate transform matrix for this map mask.

        Returns: <np.array: 4, 4>. The transformation matrix.

        """
        if self.is_none:
            return None

        if self._base_mask is None:
            return None

        return np.array(
            [
                [1.0 / self.resolution, 0, 0, 0],
                [0, -1.0 / self.resolution, 0, self._base_mask.shape[0]],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

    def is_on_mask(self, x: Any, y: Any, dilation: float = 0) -> Optional[np.ndarray]:
        """Determine whether the given coordinates are on the (optionally dilated) map mask.

        Args:
            x: Global x coordinates. Can be a scalar, list or a numpy array of x coordinates.
            y: Global y coordinates. Can be a scalar, list or a numpy array of x coordinates.
            dilation: Optional dilation of map mask.

        Returns: <np.bool: x.shape>. Whether the points are on the mask.

        """
        if self.mask is None:
            return None

        px, py = self.to_pixel_coords(x, y)

        on_mask = np.ones(px.size, dtype=np.bool)
        this_mask = self.mask(dilation)

        if this_mask is None:
            return None

        on_mask[px < 0] = False
        on_mask[px >= this_mask.shape[1]] = False
        on_mask[py < 0] = False
        on_mask[py >= this_mask.shape[0]] = False

        on_mask[on_mask] = this_mask[py[on_mask], px[on_mask]] == self.foreground

        return on_mask

    def to_pixel_coords(self, x: Any, y: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Maps x, y location in global map coordinates to the map image coordinates.

        Args:
            x: Global x coordinates. Can be a scalar, list or a numpy array of x coordinates.
            y: Global y coordinates. Can be a scalar, list or a numpy array of x coordinates.

        Returns: (px <np.uint8: x.shape>, py <np.uint8: y.shape>). Pixel coordinates in map.

        """
        if self.is_none:
            return np.array([-np.inf]), np.array([-np.inf])

        x = np.array(x)
        y = np.array(y)
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)

        assert x.shape == y.shape
        assert x.ndim == y.ndim == 1

        pts = np.stack([x, y, np.zeros(x.shape), np.ones(x.shape)])
        pixel_coords = np.round(np.dot(self.transform_matrix, pts)).astype(np.int32)

        return pixel_coords[0, :], pixel_coords[1, :]

    @property  # type: ignore
    @cached(cache=LRUCache(maxsize=1))
    def _base_mask(self) -> Optional[np.ndarray]:
        """Returns the original binary mask stored in map png file.

        Returns: <np.int8: image.height, image.width>. The binary mask.

        """
        if self.is_none:
            return None

        # Pillow allows us to specify the maximum image size above, whereas this is more difficult in OpenCV.
        img = Image.open(self.img_file)

        # Resize map mask to desired resolution.
        native_resolution = 0.1
        size_x = int(img.size[0] / self.resolution * native_resolution)
        size_y = int(img.size[1] / self.resolution * native_resolution)
        img = img.resize((size_x, size_y), resample=Image.NEAREST)

        # Convert to numpy.
        raw_mask = np.array(img)
        return raw_mask
