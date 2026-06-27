"""Role-aware shot proposals for jazz musicians.

This module turns a labeled ``Musician`` into candidate subject regions such as
hands, mouthpiece, upper body, full body, or drum kit details. The regions are still
plain COCO boxes; ``framing.fit_aspect`` is responsible for growing the chosen target
to the camera's aspect ratio.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .poseclass import ARMS_RAISED

_NOSE = 0
_L_EYE, _R_EYE = 1, 2
_L_EAR, _R_EAR = 3, 4
_L_SHOULDER, _R_SHOULDER = 5, 6
_L_ELBOW, _R_ELBOW = 7, 8
_L_WRIST, _R_WRIST = 9, 10
_L_HIP, _R_HIP = 11, 12

HORN_ROLES = {"saxophonist", "trumpeter", "trombonist", "clarinetist", "flutist"}


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


def _box_from_points(points: list[np.ndarray], pad: float = 0.25) -> np.ndarray | None:
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


def _union_box(boxes: list[np.ndarray]) -> np.ndarray:
    arr = np.asarray(boxes, dtype=float).reshape(-1, 4)
    x1 = arr[:, 0].min()
    y1 = arr[:, 1].min()
    x2 = (arr[:, 0] + arr[:, 2]).max()
    y2 = (arr[:, 1] + arr[:, 3]).max()
    return np.array([x1, y1, x2 - x1, y2 - y1], dtype=float)


def _visible_points(
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


def _relative_box(person_box, x: float, y: float, w: float, h: float) -> np.ndarray:
    px, py, pw, ph = (float(v) for v in person_box)
    return np.array([px + pw * x, py + ph * y, pw * w, ph * h], dtype=float)


def _instrument_box(musician) -> np.ndarray | None:
    if musician.instrument is None:
        return None
    return np.asarray(musician.instrument.box, dtype=float)


def _instrument_union(musician, extra_boxes: list[np.ndarray]) -> np.ndarray:
    boxes = [np.asarray(box, dtype=float) for box in extra_boxes]
    instrument = _instrument_box(musician)
    if instrument is not None:
        boxes.append(instrument)
    return _union_box(boxes)


def _head_box(musician, kpt_threshold: float) -> np.ndarray | None:
    points = _visible_points(
        musician,
        (_NOSE, _L_EYE, _R_EYE, _L_EAR, _R_EAR, _L_SHOULDER, _R_SHOULDER),
        kpt_threshold,
    )
    box = _box_from_points(points, pad=0.35)
    if box is not None:
        return box
    return _relative_box(musician.pose.box, 0.2, 0.0, 0.6, 0.28)


def _hands_box(musician, kpt_threshold: float) -> np.ndarray:
    points = _visible_points(
        musician, (_L_ELBOW, _R_ELBOW, _L_WRIST, _R_WRIST), kpt_threshold
    )
    box = _box_from_points(points, pad=0.55)
    if box is not None:
        instrument = _instrument_box(musician)
        if instrument is not None:
            box = _union_box([box, _relative_box(instrument, 0.15, 0.15, 0.7, 0.7)])
        return box
    instrument = _instrument_box(musician)
    if instrument is not None:
        return _relative_box(instrument, 0.15, 0.15, 0.7, 0.7)
    return _relative_box(musician.pose.box, 0.15, 0.35, 0.7, 0.35)


def _upper_body_box(musician) -> np.ndarray:
    return _relative_box(musician.pose.box, 0.05, 0.0, 0.9, 0.66)


def _full_body_box(musician) -> np.ndarray:
    return np.asarray(musician.pose.box, dtype=float)


def _candidate(
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


def role_shot_candidates(
    musician,
    musician_index: int,
    salience: float,
    *,
    kpt_threshold: float = 0.3,
) -> list[ShotCandidate]:
    """Return role-specific shot candidates for one musician."""
    role = musician.role
    label = musician.instrument.label.lower() if musician.instrument else ""
    candidates: list[ShotCandidate] = []
    solo_bonus = 0.12 if musician.posture.arms == ARMS_RAISED else 0.0

    def add(target, weight, shot_type, description, margin, max_zoom):
        candidates.append(
            _candidate(
                target,
                salience + weight + solo_bonus,
                musician_index,
                shot_type,
                description,
                margin=margin,
                max_zoom=max_zoom,
            )
        )

    if role == "pianist":
        add(
            _hands_box(musician, kpt_threshold),
            0.24,
            "close_up",
            "pianist hands",
            0.18,
            4.0,
        )
        add(_upper_body_box(musician), 0.15, "medium", "pianist waist up", 0.16, 2.8)
        add(
            _full_body_box(musician),
            0.04,
            "extreme_wide",
            "pianist full body",
            0.22,
            1.8,
        )
    elif role == "drummer" or label == "drum kit":
        kit = _instrument_box(musician)
        if kit is None:
            kit = _relative_box(musician.pose.box, 0.0, 0.25, 1.0, 0.65)
        add(
            _relative_box(kit, 0.15, 0.0, 0.7, 0.55),
            0.20,
            "close_up",
            "cymbals toms hat",
            0.12,
            3.8,
        )
        add(
            _hands_box(musician, kpt_threshold),
            0.22,
            "medium_close",
            "drummer hands",
            0.18,
            3.4,
        )
        add(_upper_body_box(musician), 0.13, "medium", "drummer waist up", 0.18, 2.6)
        add(
            _full_body_box(musician),
            0.03,
            "extreme_wide",
            "drummer full body",
            0.22,
            1.8,
        )
    elif role in HORN_ROLES:
        face = _head_box(musician, kpt_threshold)
        hands = _hands_box(musician, kpt_threshold)
        horn_detail = _instrument_union(musician, [face, hands])
        add(horn_detail, 0.25, "close_up", f"{role} face mouthpiece hands", 0.13, 4.0)
        add(
            _instrument_union(musician, [_upper_body_box(musician)]),
            0.20,
            "medium_close",
            f"{role} torso instrument bell",
            0.15,
            3.2,
        )
        add(
            _instrument_union(
                musician, [_relative_box(musician.pose.box, 0.0, 0.0, 1.0, 0.78)]
            ),
            0.12,
            "medium",
            f"{role} full instrument upper body",
            0.18,
            2.6,
        )
    elif role == "bassist":
        add(
            _hands_box(musician, kpt_threshold),
            0.22,
            "close_up",
            "bassist hands",
            0.18,
            3.8,
        )
        add(
            _instrument_union(musician, [_upper_body_box(musician)]),
            0.17,
            "medium",
            "bassist torso bass neck both hands",
            0.16,
            2.8,
        )
        add(
            _instrument_union(musician, [_full_body_box(musician)]),
            0.08,
            "wide_vertical",
            "bassist full body",
            0.20,
            2.0,
        )
    elif role == "guitarist":
        add(
            _hands_box(musician, kpt_threshold),
            0.21,
            "close_up",
            "guitarist picking hand or fretboard",
            0.16,
            3.8,
        )
        add(
            _instrument_union(musician, [_upper_body_box(musician)]),
            0.18,
            "medium_close",
            "guitarist torso guitar both hands",
            0.15,
            3.0,
        )
        add(
            _instrument_union(
                musician, [_relative_box(musician.pose.box, 0.0, 0.0, 1.0, 0.86)]
            ),
            0.12,
            "medium",
            "guitarist posture with instrument",
            0.18,
            2.5,
        )
    else:
        add(_upper_body_box(musician), 0.10, "medium", f"{role} upper body", 0.18, 2.6)
        add(_full_body_box(musician), 0.02, "wide", f"{role} full body", 0.22, 1.9)

    return candidates
