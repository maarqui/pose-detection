"""Tie instruments to the musicians playing them, and label each by role.

Stage 1 gives person boxes, stage 2 gives their poses, and ``instruments`` gives
labeled instrument boxes — but nothing yet says *who plays what*. This module closes
that gap with pure geometry: an instrument is associated to the person whose body and
hands sit closest to it, and the instrument name maps to a role ("guitar" ->
"guitarist"). The result is one ``Musician`` per pose, carrying its posture and (when
found) its instrument and role.

No model, no framework state — synthetic poses/boxes exercise every rule, and the
deployment team can lift the association out as-is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .instruments import InstrumentDetection
from .pose import PersonPose
from .poseclass import PoseClassification, classify_poses

# COCO-17 indices used to locate where an instrument is held.
_L_WRIST, _R_WRIST = 9, 10
_L_HIP, _R_HIP = 11, 12

# Relative weights of the two association signals (must describe a sensible blend;
# containment dominates, hand proximity rescues instruments that spill outside the
# person box such as a guitar neck or a double bass).
_CONTAINMENT_WEIGHT = 0.6
_CLOSENESS_WEIGHT = 0.4

# Instrument name -> performer role. Lower-cased lookup; unknown maps to DEFAULT_ROLE.
INSTRUMENT_ROLES: dict[str, str] = {
    "saxophone": "saxophonist",
    "trumpet": "trumpeter",
    "trombone": "trombonist",
    "clarinet": "clarinetist",
    "flute": "flutist",
    "piano": "pianist",
    "grand piano": "pianist",
    "guitar": "guitarist",
    "electric guitar": "guitarist",
    "double bass": "bassist",
    "cello": "cellist",
    "violin": "violinist",
    "drum kit": "drummer",
    "microphone": "vocalist",
}
DEFAULT_ROLE = "musician"


@dataclass
class Musician:
    """A performer: a pose, its posture, and the instrument they play (if found).

    Attributes:
        pose: The underlying ``PersonPose`` (keypoints + box).
        posture: Posture classification for the pose (see ``poseclass``).
        instrument: The associated ``InstrumentDetection``, or ``None`` if no
            instrument was close enough.
        role: Performer role derived from the instrument (e.g. ``"guitarist"``);
            ``DEFAULT_ROLE`` when there is no instrument or it is unmapped.
    """

    pose: PersonPose
    posture: PoseClassification
    instrument: InstrumentDetection | None
    role: str


def role_for_instrument(
    label: str | None, roles: dict[str, str] = INSTRUMENT_ROLES
) -> str:
    """Map an instrument label to a performer role, or ``DEFAULT_ROLE``."""
    if not label:
        return DEFAULT_ROLE
    return roles.get(label.lower(), DEFAULT_ROLE)


def _containment(inner: np.ndarray, outer: np.ndarray) -> float:
    """Fraction of the ``inner`` COCO box's area that lies inside ``outer``."""
    ix1, iy1, iw, ih = inner
    ox1, oy1, ow, oh = outer
    overlap_w = max(0.0, min(ix1 + iw, ox1 + ow) - max(ix1, ox1))
    overlap_h = max(0.0, min(iy1 + ih, oy1 + oh) - max(iy1, oy1))
    inner_area = max(0.0, iw) * max(0.0, ih)
    if inner_area <= 0.0:
        return 0.0
    return float(overlap_w * overlap_h / inner_area)


def _hand_closeness(
    pose: PersonPose, instrument_box: np.ndarray, kpt_threshold: float
) -> float:
    """How near the instrument's center is to a visible wrist (else hip), in ``[0, 1]``.

    Distance is normalized by the person's box height so the signal is scale-free;
    ``0`` when neither wrists nor hips are visible.
    """
    keypoints = np.asarray(pose.keypoints, dtype=float).reshape(-1, 2)
    scores = np.asarray(pose.scores, dtype=float).reshape(-1)
    center = np.array(
        [
            instrument_box[0] + instrument_box[2] * 0.5,
            instrument_box[1] + instrument_box[3] * 0.5,
        ]
    )

    anchors = [_L_WRIST, _R_WRIST]
    visible = [i for i in anchors if scores[i] >= kpt_threshold]
    if not visible:  # wrists hidden (common behind instruments) -> fall back to lap
        visible = [i for i in (_L_HIP, _R_HIP) if scores[i] >= kpt_threshold]
    if not visible:
        return 0.0

    scale = max(float(pose.box[3]), 1.0)
    distance = min(float(np.linalg.norm(center - keypoints[i])) for i in visible)
    return max(0.0, 1.0 - distance / scale)


def association_score(
    pose: PersonPose, instrument_box: np.ndarray, kpt_threshold: float = 0.3
) -> float:
    """Likelihood in ``[0, 1]`` that ``pose`` plays this instrument."""
    person_box = np.asarray(pose.box, dtype=float)
    instrument_box = np.asarray(instrument_box, dtype=float)
    containment = _containment(instrument_box, person_box)
    closeness = _hand_closeness(pose, instrument_box, kpt_threshold)
    return _CONTAINMENT_WEIGHT * containment + _CLOSENESS_WEIGHT * closeness


def associate_instruments(
    poses,
    instruments,
    *,
    min_association: float = 0.1,
    kpt_threshold: float = 0.3,
) -> list[InstrumentDetection | None]:
    """Match instruments to poses one-to-one, greedily by association score.

    Args:
        poses: The detected poses.
        instruments: The detected instruments.
        min_association: A pairing below this score is never made.
        kpt_threshold: Keypoint visibility threshold for the hand-proximity signal.

    Returns:
        A list aligned to ``poses``: each entry is the ``InstrumentDetection``
        assigned to that pose, or ``None``. Every instrument is used at most once;
        the highest-scoring pairings win.
    """
    poses = list(poses)
    instruments = list(instruments)
    assigned: list[InstrumentDetection | None] = [None] * len(poses)
    if not poses or not instruments:
        return assigned

    candidates = []
    for pose_index, pose in enumerate(poses):
        for instrument_index, instrument in enumerate(instruments):
            score = association_score(pose, instrument.box, kpt_threshold)
            if score >= min_association:
                candidates.append((score, pose_index, instrument_index))
    # Highest score first; pose/instrument indices break ties deterministically.
    candidates.sort(key=lambda c: (c[0], -c[1], -c[2]), reverse=True)

    used_poses: set[int] = set()
    used_instruments: set[int] = set()
    for _, pose_index, instrument_index in candidates:
        if pose_index in used_poses or instrument_index in used_instruments:
            continue
        assigned[pose_index] = instruments[instrument_index]
        used_poses.add(pose_index)
        used_instruments.add(instrument_index)
    return assigned


def label_musicians(
    poses,
    instruments,
    *,
    classifications: list[PoseClassification] | None = None,
    min_association: float = 0.1,
    kpt_threshold: float = 0.3,
    roles: dict[str, str] = INSTRUMENT_ROLES,
) -> list[Musician]:
    """Build one ``Musician`` per pose: associate an instrument and derive its role.

    Args:
        poses: The detected poses.
        instruments: The detected instruments.
        classifications: Optional precomputed postures aligned to ``poses`` (reused
            from an earlier stage to avoid recomputation); classified here if ``None``.
        min_association: Threshold passed to :func:`associate_instruments`.
        kpt_threshold: Keypoint visibility threshold.
        roles: Instrument-to-role mapping.

    Returns:
        A ``Musician`` per input pose, in the same order.
    """
    poses = list(poses)
    if classifications is None:
        classifications = classify_poses(poses, kpt_threshold)
    elif len(classifications) != len(poses):
        raise ValueError("classifications must align one-to-one with poses")

    assigned = associate_instruments(
        poses, instruments, min_association=min_association, kpt_threshold=kpt_threshold
    )

    musicians: list[Musician] = []
    for pose, instrument, posture in zip(poses, assigned, classifications, strict=True):
        label = instrument.label if instrument is not None else None
        musicians.append(
            Musician(
                pose=pose,
                posture=posture,
                instrument=instrument,
                role=role_for_instrument(label, roles),
            )
        )
    return musicians
