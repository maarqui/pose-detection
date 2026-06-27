import numpy as np
from dataclasses import dataclass

_NOSE = 0
_L_EYE, _R_EYE = 1, 2
_L_EAR, _R_EAR = 3, 4
_L_SHOULDER, _R_SHOULDER = 5, 6
_L_ELBOW, _R_ELBOW = 7, 8
_L_WRIST, _R_WRIST = 9, 10
_L_HIP, _R_HIP = 11, 12

@dataclass(frozen=True)
class ShotCandidate:
    """A proposed subject region before camera-aspect fitting."""
    target_box: np.ndarray
    score: float
    musician_indices: tuple[int, ...]
    shot_type: str
    description: str
    margin: float
    max_zoom: float

def box_from_points(points: list[np.ndarray], pad: float = 0.25) -> np.ndarray | None:
    if not points:
        return None
    arr = np.asarray(points, dtype=float).reshape(-1, 2)
    x1, y1 = arr[:, 0].min(), arr[:, 1].min()
    x2, y2 = arr[:, 0].max(), arr[:, 1].max()
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    grow = max(width, height) * pad
    return np.array(
        [x1 - grow, y1 - grow, width + grow * 2.0, height + grow * 2.0],
        dtype=float,
    )

def union_box(boxes: list[np.ndarray]) -> np.ndarray:
    arr = np.asarray(boxes, dtype=float).reshape(-1, 4)
    x1 = arr[:, 0].min()
    y1 = arr[:, 1].min()
    x2 = (arr[:, 0] + arr[:, 2]).max()
    y2 = (arr[:, 1] + arr[:, 3]).max()
    return np.array([x1, y1, x2 - x1, y2 - y1], dtype=float)

def visible_points(
    musician, indexes: tuple[int, ...], threshold: float
) -> list[np.ndarray]:
    keypoints = np.asarray(musician.pose.keypoints, dtype=float).reshape(-1, 2)
    scores = np.asarray(musician.pose.scores, dtype=float).reshape(-1)
    px, py, pw, ph = (float(v) for v in musician.pose.box)
    margin_x = pw * 0.35
    margin_y = ph * 0.35
    points = []
    for i in indexes:
        if i >= len(scores) or scores[i] < threshold:
            continue
        x, y = keypoints[i]
        inside_x = px - margin_x <= x <= px + pw + margin_x
        inside_y = py - margin_y <= y <= py + ph + margin_y
        if inside_x and inside_y:
            points.append(keypoints[i])
    return points

def relative_box(person_box, x: float, y: float, w: float, h: float) -> np.ndarray:
    px, py, pw, ph = (float(v) for v in person_box)
    return np.array([px + pw * x, py + ph * y, pw * w, ph * h], dtype=float)

def instrument_box(musician) -> np.ndarray | None:
    if musician.instrument is None:
        return None
    return np.asarray(musician.instrument.box, dtype=float)

def instrument_union(musician, extra_boxes: list[np.ndarray]) -> np.ndarray:
    boxes = [np.asarray(box, dtype=float) for box in extra_boxes]
    inst = instrument_box(musician)
    if inst is not None:
        boxes.append(inst)
    return union_box(boxes)

def head_box(musician, kpt_threshold: float) -> np.ndarray | None:
    points = visible_points(
        musician,
        (_NOSE, _L_EYE, _R_EYE, _L_EAR, _R_EAR, _L_SHOULDER, _R_SHOULDER),
        kpt_threshold,
    )
    box = box_from_points(points, pad=0.35)
    if box is not None:
        return box
    return relative_box(musician.pose.box, 0.2, 0.0, 0.6, 0.28)

def hands_box(musician, kpt_threshold: float) -> np.ndarray:
    points = visible_points(
        musician, (_L_ELBOW, _R_ELBOW, _L_WRIST, _R_WRIST), kpt_threshold
    )
    box = box_from_points(points, pad=0.55)
    if box is not None:
        inst = instrument_box(musician)
        if inst is not None:
            box = union_box([box, relative_box(inst, 0.15, 0.15, 0.7, 0.7)])
        return box
    inst = instrument_box(musician)
    if inst is not None:
        return relative_box(inst, 0.15, 0.15, 0.7, 0.7)
    return relative_box(musician.pose.box, 0.15, 0.35, 0.7, 0.35)

def upper_body_box(musician) -> np.ndarray:
    return relative_box(musician.pose.box, 0.05, 0.0, 0.9, 0.66)

def full_body_box(musician) -> np.ndarray:
    return np.asarray(musician.pose.box, dtype=float)

def candidate(
    target_box,
    score: float,
    musician_index: int,
    shot_type: str,
    description: str,
    *,
    margin: float,
    max_zoom: float,
) -> ShotCandidate:
    return ShotCandidate(
        target_box=np.asarray(target_box, dtype=float),
        score=float(score),
        musician_indices=(musician_index,),
        shot_type=shot_type,
        description=description,
        margin=margin,
        max_zoom=max_zoom,
    )
