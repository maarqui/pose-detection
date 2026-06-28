from dataclasses import dataclass

import numpy as np

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
    """Broad hands area."""
    points = visible_points(
        musician, (_L_ELBOW, _R_ELBOW, _L_WRIST, _R_WRIST), kpt_threshold
    )
    box = box_from_points(points, pad=0.6)
    inst = instrument_box(musician)
    if inst is not None:
        # For a close-up, we only want the part of the instrument where the hands are
        inst_focus = relative_box(inst, 0.2, 0.2, 0.6, 0.6)
        if box is not None:
            box = union_box([box, inst_focus])
        else:
            box = inst_focus

    if box is not None:
        return box
    return relative_box(musician.pose.box, 0.1, 0.3, 0.8, 0.4)


def upper_body_box(musician) -> np.ndarray:
    """Waist up (Medium Shot)."""
    # Slightly wider and lower-centered than a tight chest-up
    return relative_box(musician.pose.box, -0.1, 0.0, 1.2, 0.65)


def medium_shot_box(musician) -> np.ndarray:
    """Knees up (Medium Wide)."""
    return relative_box(musician.pose.box, -0.1, 0.0, 1.2, 0.8)


def chest_up_box(musician) -> np.ndarray:
    """Chest up (Medium Close-up)."""
    return relative_box(musician.pose.box, 0.0, 0.0, 1.0, 0.45)


def head_only_box(musician, kpt_threshold: float) -> np.ndarray:
    """Head only."""
    points = visible_points(
        musician, (_NOSE, _L_EYE, _R_EYE, _L_EAR, _R_EAR), kpt_threshold
    )
    box = box_from_points(points, pad=0.5)
    if box is not None:
        return box
    return relative_box(musician.pose.box, 0.3, 0.0, 0.4, 0.2)


def pianist_hands_focused_box(musician, kpt_threshold: float) -> np.ndarray:
    """Focused on pianist hands and keys."""
    points = visible_points(musician, (_L_WRIST, _R_WRIST), kpt_threshold)
    px, py, pw, ph = (float(v) for v in musician.pose.box)

    if points:
        # If wrists are found, we want a wide but short box for the keyboard
        arr = np.asarray(points)
        x1, y1 = arr[:, 0].min(), arr[:, 1].min()
        x2, y2 = arr[:, 0].max(), arr[:, 1].max()

        # Detected width of hands
        dw = x2 - x1
        dh = y2 - y1

        # Target width: at least 80% of person width to see keys,
        # but expanded if hands are far apart.
        target_w = max(dw + pw * 0.4, pw * 0.9)
        # Target height: enough to see hands and keys
        target_h = max(dh + ph * 0.1, ph * 0.2)

        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        return np.array(
            [cx - target_w * 0.5, cy - target_h * 0.5, target_w, target_h], dtype=float
        )

    # Fallback to a region where hands usually are for a pianist
    return relative_box(musician.pose.box, -0.1, 0.4, 1.2, 0.3)


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
