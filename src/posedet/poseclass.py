"""Single-frame posture classification from COCO-17 keypoints.

Rule-based and framework-free: given the joints ViTPose produces for one person,
decide a coarse posture (standing / sitting) and arm state (down / raised) using
scale-invariant geometry. No model, no weights, no global state — the deployment
team can lift these functions out as-is, and the later shot-scoring stage
(``framing``) consumes the labels to bias the chosen shot toward salient performers.

Coordinates follow the OpenCV/image convention used everywhere in this package:
``x`` grows right, ``y`` grows *down*. Keypoint order is COCO-17 (see
``visualization.COCO_KEYPOINTS``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# COCO-17 keypoint indices used by the rules.
_L_SHOULDER, _R_SHOULDER = 5, 6
_L_WRIST, _R_WRIST = 9, 10
_L_HIP, _R_HIP = 11, 12
_L_KNEE, _R_KNEE = 13, 14
_L_ANKLE, _R_ANKLE = 15, 16

# Posture labels (mutually exclusive primary class).
UNKNOWN = "unknown"
STANDING = "standing"
SITTING = "sitting"
POSTURES = (UNKNOWN, STANDING, SITTING)

# Arm-state labels (orthogonal to posture).
ARMS_UNKNOWN = "unknown"
ARMS_DOWN = "down"
ARMS_RAISED = "raised"
ARM_STATES = (ARMS_UNKNOWN, ARMS_DOWN, ARMS_RAISED)

# A knee angle at/above this (degrees) reads as an essentially straight leg.
KNEE_STRAIGHT_DEG = 150.0
# A wrist this fraction of the torso length above the shoulder counts as "raised".
WRIST_RAISE_MARGIN = 0.1


@dataclass(frozen=True)
class PoseClassification:
    """Coarse posture of one person, derived from joint geometry.

    Attributes:
        posture: One of ``POSTURES``. ``UNKNOWN`` when the legs are not visible
            enough to separate standing from sitting.
        arms: One of ``ARM_STATES``; orthogonal to ``posture``.
        confidence: Mean confidence of the keypoints that drove the posture
            decision, in ``[0, 1]``. ``0`` when posture is ``UNKNOWN``.
    """

    posture: str
    arms: str
    confidence: float


def _visible(scores: np.ndarray, idx: int, threshold: float) -> bool:
    return bool(scores[idx] >= threshold)


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Interior angle at ``b`` from points a-b-c, in degrees (``nan`` if degenerate)."""
    ba = a - b
    bc = c - b
    nba = float(np.linalg.norm(ba))
    nbc = float(np.linalg.norm(bc))
    if nba == 0.0 or nbc == 0.0:
        return float("nan")
    cos = float(np.dot(ba, bc) / (nba * nbc))
    cos = max(-1.0, min(1.0, cos))
    return float(np.degrees(np.arccos(cos)))


def _center(
    keypoints: np.ndarray, scores: np.ndarray, i: int, j: int, threshold: float
) -> np.ndarray | None:
    """Midpoint of two keypoints, or the single visible one, or ``None``."""
    vi = _visible(scores, i, threshold)
    vj = _visible(scores, j, threshold)
    if vi and vj:
        return (keypoints[i] + keypoints[j]) / 2.0
    if vi:
        return keypoints[i]
    if vj:
        return keypoints[j]
    return None


def _torso_len(keypoints: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    """Scale reference: vertical shoulder-to-hip span, else shoulder width, else 0."""
    shoulder = _center(keypoints, scores, _L_SHOULDER, _R_SHOULDER, threshold)
    hip = _center(keypoints, scores, _L_HIP, _R_HIP, threshold)
    if shoulder is not None and hip is not None:
        return float(abs(shoulder[1] - hip[1]))
    if _visible(scores, _L_SHOULDER, threshold) and _visible(
        scores, _R_SHOULDER, threshold
    ):
        return float(np.linalg.norm(keypoints[_L_SHOULDER] - keypoints[_R_SHOULDER]))
    return 0.0


def _leg_angle(
    keypoints: np.ndarray,
    scores: np.ndarray,
    hip_i: int,
    knee_i: int,
    ankle_i: int,
    threshold: float,
) -> tuple[float, float] | None:
    """Knee angle and the mean confidence of (hip, knee, ankle), if all are visible."""
    if not (
        _visible(scores, hip_i, threshold)
        and _visible(scores, knee_i, threshold)
        and _visible(scores, ankle_i, threshold)
    ):
        return None
    angle = _angle(keypoints[hip_i], keypoints[knee_i], keypoints[ankle_i])
    if np.isnan(angle):
        return None
    mean_score = float((scores[hip_i] + scores[knee_i] + scores[ankle_i]) / 3.0)
    return angle, mean_score


def _arm_state(
    keypoints: np.ndarray, scores: np.ndarray, threshold: float, torso_len: float
) -> str:
    """``raised`` if a wrist is well above its shoulder, else ``down``/``unknown``."""
    raised = False
    seen = False
    margin = WRIST_RAISE_MARGIN * torso_len if torso_len > 0 else 0.0
    for shoulder_i, wrist_i in ((_L_SHOULDER, _L_WRIST), (_R_SHOULDER, _R_WRIST)):
        if _visible(scores, shoulder_i, threshold) and _visible(
            scores, wrist_i, threshold
        ):
            seen = True
            # y grows downward, so "above" means a smaller y.
            if keypoints[wrist_i][1] < keypoints[shoulder_i][1] - margin:
                raised = True
    if not seen:
        return ARMS_UNKNOWN
    return ARMS_RAISED if raised else ARMS_DOWN


def classify_pose(pose, kpt_threshold: float = 0.3) -> PoseClassification:
    """Classify the posture of a single pose from its joint geometry.

    Args:
        pose: Any object with ``.keypoints`` ``(17, 2)`` and ``.scores`` ``(17,)``
            in COCO-17 order — e.g. a ``PersonPose``.
        kpt_threshold: Keypoints scoring below this are treated as not visible.

    Returns:
        A ``PoseClassification``. Posture is ``UNKNOWN`` when no leg (hip-knee-ankle)
        is visible enough to tell standing from sitting; arm state is independent and
        may still be resolved.
    """
    keypoints = np.asarray(pose.keypoints, dtype=float).reshape(-1, 2)
    scores = np.asarray(pose.scores, dtype=float).reshape(-1)
    if keypoints.shape[0] < 17 or scores.shape[0] < 17:
        return PoseClassification(UNKNOWN, ARMS_UNKNOWN, 0.0)

    torso_len = _torso_len(keypoints, scores, kpt_threshold)
    arms = _arm_state(keypoints, scores, kpt_threshold, torso_len)

    legs = [
        leg
        for leg in (
            _leg_angle(keypoints, scores, _L_HIP, _L_KNEE, _L_ANKLE, kpt_threshold),
            _leg_angle(keypoints, scores, _R_HIP, _R_KNEE, _R_ANKLE, kpt_threshold),
        )
        if leg is not None
    ]
    if not legs:
        return PoseClassification(UNKNOWN, arms, 0.0)

    avg_angle = float(np.mean([angle for angle, _ in legs]))
    confidence = float(np.mean([score for _, score in legs]))
    posture = STANDING if avg_angle >= KNEE_STRAIGHT_DEG else SITTING
    return PoseClassification(posture, arms, confidence)


def classify_poses(poses, kpt_threshold: float = 0.3) -> list[PoseClassification]:
    """Classify every pose in ``poses`` (see :func:`classify_pose`)."""
    return [classify_pose(pose, kpt_threshold) for pose in poses]
