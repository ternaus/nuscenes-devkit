"""
mAP 3D calculation for the data in nuScenes format.


The intput files expected to have the format:

Expected fields:


gt = [{
    'sample_token': '0f0e3ce89d2324d8b45aa55a7b4f8207fbb039a550991a5149214f98cec136ac',
    'translation': [974.2811881299899, 1714.6815014457964, -23.689857123368846],
    'size': [1.796, 4.488, 1.664],
    'rotation': [0.14882026466054782, 0, 0, 0.9888642620837121],
    'name': 'car'
}]

prediction_result = [{
    'sample_token': '0f0e3ce89d2324d8b45aa55a7b4f8207fbb039a550991a5149214f98cec136ac',
    'translation': [971.8343488872263, 1713.6816097857359, -25.82534357061308],
    'size': [2.519726579986132, 7.810161372666739, 3.483438286096803],
    'rotation': [0.10913582721095375, 0.04099572636992043, 0.01927712319721745, 1.029328402625659],
    'name': 'car',
    'score': 0.3077029437237213
}]


input arguments:

--pred_file:  file with predictions
--gt_file: ground truth file
--iou_threshold: IOU threshold


In general we would be interested in average of mAP at thresholds [0.5, 0.55, 0.6, 0.65,...0.95], similar to the
standard COCO => one needs to run this file N times for every IOU threshold independently.

Outputs set of dataframes with scores.
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from iglovikov_helper_functions.utils.geometry_utils import intersection_rectangles
from joblib import Parallel, delayed
from pyquaternion import Quaternion
from tqdm import tqdm

AP_THRESHOLD = 1e-4


class Box3D:
    """Data class used during detection evaluation. Can be a prediction or ground truth."""

    def __init__(
        self,
        sample_token: str,
        translation: Union[List[float], np.ndarray, Tuple[float, float, float]],
        size: List[float],
        rotation: List[float],
        name: str,
        score: float = -1,
    ):

        if not isinstance(sample_token, str):
            raise TypeError("Sample_token must be a string!")

        if not len(translation) == 3:
            raise ValueError("Translation must have 3 elements!")

        if np.any(np.isnan(translation)):
            raise ValueError("Translation may not be NaN!")

        if not len(rotation) == 4:
            raise ValueError("Rotation must have 4 elements!")

        if np.any(np.isnan(rotation)):
            raise ValueError("Rotation may not be NaN!")

        if name is None:
            raise ValueError("Name cannot be empty!")

        # Assign.
        self.sample_token = sample_token
        self.translation = translation
        self.size = size
        self.volume = np.prod(self.size)
        self.score = score

        self.rotation = rotation
        self.name = name
        self.quaternion = Quaternion(self.rotation)

        self.width, self.length, self.height = size

        self.center_x, self.center_y, self.center_z = [float(x) for x in self.translation]

        self.center_xy = np.array([self.center_x, self.center_y], copy=False)
        self.area = self.width * self.length

        self.min_z = self.center_z - self.height / 2
        self.max_z = self.center_z + self.height / 2

        self.ground_bbox_coords = self.get_ground_bbox_coords()

    def get_ground_bbox_coords(self):
        if hasattr(self, "ground_bbox_coords"):
            return self.ground_bbox_coords

        return self.calculate_ground_bbox_coords()

    def box_intersection_with_other(self, other):
        box_a = self.bounding_box
        box_b = other.bounding_box
        return self.box_intersection(box_a, box_b)

    @staticmethod
    def box_intersection(box_a, box_b):
        x_a = max(box_a[0], box_b[0])
        y_a = max(box_a[1], box_b[1])

        x_b = min(box_a[2], box_b[2])
        y_b = min(box_a[3], box_b[3])
        # compute the area of intersection rectangle
        return max(0, x_b - x_a) * max(0, y_b - y_a)

    def calculate_ground_bbox_coords(self):
        """We assume that the 3D box has lower plane parallel to the ground.

        Returns: Polygon with 4 points describing the base.

        """
        if hasattr(self, "ground_bbox_coords"):
            return self.ground_bbox_coords

        rotation_matrix = self.quaternion.rotation_matrix[:2, :2]

        point_0 = self.center_xy + np.dot([+self.length, +self.width], rotation_matrix) / 2
        point_1 = self.center_xy + np.dot([-self.length, +self.width], rotation_matrix) / 2
        point_2 = self.center_xy + np.dot([-self.length, -self.width], rotation_matrix) / 2
        point_3 = self.center_xy + np.dot([+self.length, -self.width], rotation_matrix) / 2

        self.ground_bbox_coords = np.array([point_0, point_1, point_2, point_3], copy=False)
        x_coords = [x[0] for x in [point_0, point_1, point_2, point_3]]
        y_coords = [x[1] for x in [point_0, point_1, point_2, point_3]]

        self.bounding_box = min(x_coords), min(y_coords), max(x_coords), max(y_coords)  # x_min, y_min, x_max, y_max

        return self.ground_bbox_coords

    def get_height_intersection(self, other):
        min_z = max(other.min_z, self.min_z)
        max_z = min(other.max_z, self.max_z)
        return max(0, max_z - min_z)

    def get_area_intersection(self, other: "Box3D") -> float:
        self_rectangle = self.get_ground_bbox_coords()
        other_rectangle = other.get_ground_bbox_coords()

        return intersection_rectangles(self_rectangle, other_rectangle)

    def get_intersection(self, other: "Box3D", iou_threshold: float = 0) -> float:
        height_intersection = self.get_height_intersection(other)

        if height_intersection == 0:
            return 0

        box_intersection = self.box_intersection_with_other(other)
        if box_intersection == 0:
            return 0

        # max intersection will be if :
        # both are aligned toward x,y axis
        # longer side toward longer side
        # bottom left corner matches

        box_a = (0, 0, max(self.length, self.width), min(self.length, self.width))
        box_b = (0, 0, max(other.length, other.width), min(other.length, other.width))

        approx_2d_intersection = self.box_intersection(box_a, box_b)

        approx_3d_intersection = approx_2d_intersection * height_intersection
        appox_iou = approx_3d_intersection / (self.volume + other.volume - approx_3d_intersection)

        if appox_iou < iou_threshold:  # this is hack. Should be rewritten
            return 0

        area_intersection = self.get_area_intersection(other)
        result = height_intersection * area_intersection

        return result

    def get_iou(self, other: "Box3D", iou_threshold: float = 0) -> float:
        intersection = self.get_intersection(other, iou_threshold)
        union = self.volume + other.volume - intersection

        return np.clip(intersection / union, 0, 1)

    def __repr__(self):
        return str(self.serialize())

    def serialize(self) -> Dict[str, Any]:
        """Returns: Serialized instance as dict."""

        return {
            "sample_token": self.sample_token,
            "translation": self.translation,
            "size": self.size,
            "rotation": self.rotation,
            "name": self.name,
            "volume": self.volume,
            "score": self.score,
        }


def group_by_key(detections: List, key: str) -> Dict[Any, List]:
    groups: Dict = defaultdict(list)
    for detection in tqdm(detections):
        groups[detection[key]].append(detection)
    return groups


def wrap_in_box(input_boxes: Dict[str, List[Box3D]]) -> Dict[str, Dict[int, Box3D]]:
    result: Dict[str, Dict[int, Box3D]] = {}
    for sample_token, list_boxes in input_boxes.items():
        result[sample_token] = {}
        for i, x in enumerate(list_boxes):
            box = Box3D(x["sample_token"], x["translation"], x["size"], x["rotation"], x["name"])  # type: ignore
            result[sample_token][i] = box
    return result


def filter_invalid(predictions: List[Dict[str, Any]], min_score: float) -> List[Dict[str, Any]]:
    def helper(x):
        if x["score"] < min_score:  # pylint: disable=R1705
            return False
        elif np.all(np.array(x["size"], copy=False) <= 0) or np.any(np.isnan(x["size"])) or len(x["size"]) != 3:
            return False
        return True

    return [x for x in tqdm(predictions) if helper(x)]


def get_envelope(precisions):
    """Compute the precision envelope.

    Args:
      precisions:

    Returns:

    """
    for i in range(precisions.size - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
    return precisions


def get_ap(recalls: np.ndarray, precisions: np.ndarray) -> np.ndarray:
    """Calculate average precision.

    Args:
      recalls:
      precisions: Returns (float): average precision.

    Returns:

    """
    # correct AP calculation
    # first append sentinel values at the end
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))

    precisions = get_envelope(precisions)

    # to calculate area under PR curve, look for points where X axis (recall) changes value
    i = np.where(recalls[1:] != recalls[:-1])[0]

    # and sum (\Delta recall) * prec
    return np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])


def get_ious(gt_boxes: Dict[int, Box3D], predicted_box: Box3D, iou_threshold: float) -> np.ndarray:
    result = np.zeros(max(gt_boxes.keys()) + 1)
    for box_id in gt_boxes.keys():
        result[box_id] = predicted_box.get_iou(gt_boxes[box_id], iou_threshold)
    return result


def recall_precision(
    gt: List, predictions: List, iou_threshold: float, output_file: Path = Path("output.txt"), reduce: bool = True
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[float, float, float]]:
    num_gts = len(gt)

    image_gts = wrap_in_box(group_by_key(gt, "sample_token"))

    sample_gt_checked = {sample_token: np.zeros(len(boxes)) for sample_token, boxes in image_gts.items()}

    predictions = sorted(predictions, key=lambda x: x["score"], reverse=True)

    def process_prediction(prediction):
        predicted_box = Box3D(
            prediction["sample_token"],
            prediction["translation"],
            prediction["size"],
            prediction["rotation"],
            prediction["name"],
            prediction["score"],
        )
        sample_token = prediction["sample_token"]

        max_overlap = -np.inf
        max_overlap_index = -1

        try:
            gt_checked = sample_gt_checked[sample_token]  # gt flags per sample
            gt_boxes = {}

            for box_id, is_checked in enumerate(gt_checked):
                if is_checked == 0:
                    gt_boxes[box_id] = image_gts[sample_token][box_id]
        except KeyError:
            return 0

        if len(gt_boxes) > 0:
            overlaps = get_ious(gt_boxes, predicted_box, iou_threshold)

            max_overlap = max(overlaps)
            max_overlap_index = np.argmax(overlaps)

        if max_overlap >= iou_threshold:
            gt_checked[max_overlap_index] = 1
            return 1

        return 0

    result = [process_prediction(prediction) for prediction in tqdm(predictions)]

    tp = np.array(result, copy=False)
    fp = 1 - tp

    # compute precision recall
    fp = np.cumsum(fp, axis=0)
    tp = np.cumsum(tp, axis=0)

    recalls = tp / float(num_gts)

    assert np.all(recalls >= 0) & np.all(recalls <= 1)

    # avoid divide by zero in case the first detection matches a difficult ground truth
    precisions = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    assert np.all(precisions <= 0) & np.all(precisions <= 1)

    ap = get_ap(recalls, precisions)

    if reduce:
        with open(output_file, "w") as f:
            f.write(f"Recall {recalls.mean()}\n")
            f.write(f"Precision {precisions.mean()}\n")
            f.write(f"AP {ap.mean()}")

        return recalls.mean(), precisions.mean(), ap.mean()

    return recalls, precisions, ap


def get_class_names(gt: dict) -> list:
    """Get sorted list of class names.

    Args:
        gt:

    Returns: Sorted list of class names.

    """
    return sorted(list({x["name"] for x in tqdm(gt)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-p", "--pred_file", type=str, help="Path to the predictions file.", required=True)
    arg("-g", "--gt_file", type=str, help="Path to the ground truth file.", required=True)
    arg("-j", "--num_workers", type=int, help="The number of CPU workers.", default=1)
    arg("-o", "--output_folder", type=Path, help="Path to the output folder.", required=True)
    arg(
        "-m",
        "--min_prob",
        type=float,
        help="We do not use scores below this threshold in the computation.",
        default=0.1,
    )

    start_time = time.time()

    args = parser.parse_args()

    gt_path = Path(args.gt_file)
    pred_path = Path(args.pred_file)

    print("Reading predictions")
    with open(args.pred_file) as f:
        predictions = json.load(f)

    print("Reading gt")
    with open(args.gt_file) as f:
        gt = json.load(f)

    predictions = filter_invalid(predictions, args.min_prob)
    class_names = get_class_names(gt)
    print("Class_names = ", class_names)

    iou_thresholds = np.arange(0.5, 1, 0.05)

    # class_x_iou = list(product(class_names, iou_thresholds))
    pred_by_class_name = group_by_key(predictions, "name")
    gt_by_class_name = group_by_key(gt, "name")

    output_folder = args.output_folder
    output_folder.mkdir(parents=True, exist_ok=True)

    def process_class(class_name: List[str]) -> None:
        for iou_threshold in iou_thresholds:
            output_file = output_folder / f"{class_name}_{iou_threshold:.2f}.txt"
            if output_file.exists():
                with open(output_file) as f:
                    ap = float(f.readlines()[-1].strip().split()[-1])

                if ap < AP_THRESHOLD:
                    return

                continue

            _, _, ap = recall_precision(
                gt_by_class_name[class_name],
                pred_by_class_name[class_name],
                iou_threshold,
                output_file,
                True,
            )
            if ap < AP_THRESHOLD:
                return

    results = Parallel(n_jobs=args.num_workers)(delayed(process_class)(class_name) for class_name in class_names)

    print(time.time() - start_time)
