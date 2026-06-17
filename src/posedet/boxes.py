"""Pure box-geometry helpers, framework-free (NumPy only).

These operate on **COCO ``(x, y, w, h)``** boxes unless a name says otherwise
(``filter_to_roi`` takes a VOC ``(x1, y1, x2, y2)`` region). They carry no model
or framework state, so the deployment team can lift any of them out as-is. Higher
layers (the detector, the video runner) compose these; keep the math here.
"""

from __future__ import annotations

import numpy as np

EMPTY_BOXES = np.empty((0, 4), dtype=np.float32)


def iou_xywh(first_box: np.ndarray, second_box: np.ndarray) -> float:
    """Intersection-over-Union of two COCO ``(x, y, w, h)`` boxes."""
    first_x1, first_y1, first_w, first_h = first_box
    second_x1, second_y1, second_w, second_h = second_box
    first_x2 = first_x1 + first_w
    first_y2 = first_y1 + first_h
    second_x2 = second_x1 + second_w
    second_y2 = second_y1 + second_h

    overlap_w = max(0.0, min(first_x2, second_x2) - max(first_x1, second_x1))
    overlap_h = max(0.0, min(first_y2, second_y2) - max(first_y1, second_y1))
    overlap_area = overlap_w * overlap_h

    first_area = max(0.0, first_w) * max(0.0, first_h)
    second_area = max(0.0, second_w) * max(0.0, second_h)
    union_area = first_area + second_area - overlap_area
    if union_area <= 0.0:
        return 0.0
    return float(overlap_area / union_area)


def expand_box(
    box: np.ndarray, image_width: int, image_height: int, scale: float
) -> np.ndarray:
    """Scale a COCO box about its center, clamped to the image bounds.

    ``scale`` > 1 grows the box (e.g. padding a person crop); the result never
    leaves ``[0, image_width] x [0, image_height]`` and keeps width/height >= 1.
    """
    x_coord, y_coord, width, height = box
    center_x = x_coord + width * 0.5
    center_y = y_coord + height * 0.5
    new_width = width * scale
    new_height = height * scale
    new_x = max(0.0, center_x - new_width * 0.5)
    new_y = max(0.0, center_y - new_height * 0.5)
    new_width = min(new_width, image_width - new_x)
    new_height = min(new_height, image_height - new_y)
    return np.array(
        [new_x, new_y, max(new_width, 1.0), max(new_height, 1.0)], dtype=np.float32
    )


def filter_to_roi(boxes_coco: np.ndarray, roi_xyxy: np.ndarray | None) -> np.ndarray:
    """Keep only boxes whose center lies inside the VOC ``(x1, y1, x2, y2)`` ROI.

    ``roi_xyxy`` of ``None`` is a no-op (returns the input unchanged), so callers
    can pass an optional stage region without branching.
    """
    if roi_xyxy is None or len(boxes_coco) == 0:
        return boxes_coco

    x1, y1, x2, y2 = roi_xyxy
    centers_x = boxes_coco[:, 0] + boxes_coco[:, 2] * 0.5
    centers_y = boxes_coco[:, 1] + boxes_coco[:, 3] * 0.5
    in_roi = (
        (centers_x >= x1) & (centers_x <= x2) & (centers_y >= y1) & (centers_y <= y2)
    )
    return boxes_coco[in_roi]


def dedupe_indices(
    boxes: np.ndarray,
    scores: np.ndarray,
    max_boxes: int,
    iou_threshold: float = 0.45,
) -> list[int]:
    """Greedy-NMS the boxes and return the kept *indices* into ``boxes``.

    Boxes are considered in descending ``scores`` order. A candidate is dropped if
    its IoU with any already-kept box reaches ``iou_threshold``. At most
    ``max_boxes`` indices are returned. Returning indices (rather than the boxes
    themselves) lets callers keep parallel arrays such as scores aligned.
    """
    selected_indices: list[int] = []
    for index in np.argsort(scores)[::-1]:
        box = boxes[index]
        if any(
            iou_xywh(box, boxes[kept]) >= iou_threshold for kept in selected_indices
        ):
            continue
        selected_indices.append(int(index))
        if len(selected_indices) >= max_boxes:
            break
    return selected_indices


def dedupe_ranked_boxes(
    boxes: np.ndarray,
    scores: np.ndarray,
    max_boxes: int,
    iou_threshold: float = 0.45,
) -> np.ndarray:
    """Greedy NMS: keep highest-scoring boxes, drop those overlapping a kept one.

    Thin wrapper over :func:`dedupe_indices` for callers that only want the boxes.
    """
    if len(boxes) == 0:
        return EMPTY_BOXES.copy()
    kept = dedupe_indices(boxes, scores, max_boxes, iou_threshold)
    if not kept:
        return EMPTY_BOXES.copy()
    return boxes[kept].astype(np.float32)


def smooth_boxes(
    previous_boxes: np.ndarray,
    detected_boxes: np.ndarray,
    smoothing: float,
    min_iou: float = 0.15,
) -> np.ndarray:
    """Temporally blend each detection with its best match from the prior frame.

    For every detected box, find the unused previous box with the highest IoU; if
    that IoU clears ``min_iou`` the two are blended ``smoothing * prev + (1 -
    smoothing) * detected`` to damp skeleton jitter. ``smoothing`` of 0 (or empty
    inputs) returns the detections unchanged. ``smoothing`` is clamped to 0.95 so a
    box can never freeze entirely. This is the one stateful helper's pure core; the
    caller owns the previous-frame buffer.
    """
    if smoothing <= 0.0 or len(previous_boxes) == 0 or len(detected_boxes) == 0:
        return detected_boxes

    smoothing = min(max(smoothing, 0.0), 0.95)
    smoothed_boxes = detected_boxes.copy()
    used_previous_indexes: set[int] = set()

    for detected_index, detected_box in enumerate(detected_boxes):
        best_iou = 0.0
        best_previous_index = -1
        for previous_index, previous_box in enumerate(previous_boxes):
            if previous_index in used_previous_indexes:
                continue
            iou = iou_xywh(previous_box, detected_box)
            if iou > best_iou:
                best_iou = iou
                best_previous_index = previous_index

        if best_previous_index >= 0 and best_iou >= min_iou:
            used_previous_indexes.add(best_previous_index)
            previous_box = previous_boxes[best_previous_index]
            smoothed_boxes[detected_index] = (
                smoothing * previous_box + (1.0 - smoothing) * detected_box
            )

    return smoothed_boxes
