"""Tests for the rule-based pose classifier. Pure geometry on synthetic COCO-17
keypoints, so no model weights are involved."""

from __future__ import annotations

import numpy as np

from posedet import PersonPose
from posedet.poseclass import (
    ARMS_DOWN,
    ARMS_RAISED,
    ARMS_UNKNOWN,
    SITTING,
    STANDING,
    UNKNOWN,
    classify_pose,
    classify_poses,
)

# Indices into COCO-17, mirroring posedet.poseclass.
L_SHOULDER, R_SHOULDER = 5, 6
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

# A standing figure (image coords, y grows downward), arms hanging down.
_STANDING = {
    0: (50, 0),  # nose
    1: (47, -2),
    2: (53, -2),
    3: (45, 0),
    4: (55, 0),
    L_SHOULDER: (40, 20),
    R_SHOULDER: (60, 20),
    7: (38, 40),
    8: (62, 40),
    L_WRIST: (37, 60),
    R_WRIST: (63, 60),
    L_HIP: (42, 60),
    R_HIP: (58, 60),
    L_KNEE: (42, 90),
    R_KNEE: (58, 90),
    L_ANKLE: (42, 120),
    R_ANKLE: (58, 120),
}


def _standing_keypoints() -> np.ndarray:
    kpts = np.zeros((17, 2), dtype=float)
    for i, (x, y) in _STANDING.items():
        kpts[i] = (x, y)
    return kpts


def _pose(keypoints: np.ndarray, scores: np.ndarray) -> PersonPose:
    return PersonPose(
        keypoints=keypoints,
        scores=scores,
        box=np.asarray([0.0, 0.0, 100.0, 120.0], dtype=float),
    )


def test_standing_arms_down():
    pose = _pose(_standing_keypoints(), np.full(17, 0.9))
    result = classify_pose(pose)
    assert result.posture == STANDING
    assert result.arms == ARMS_DOWN
    assert result.confidence > 0.0


def test_sitting_from_bent_knees():
    kpts = _standing_keypoints()
    # Bend both knees forward so the knee angle drops well below straight.
    kpts[L_KNEE] = (60, 66)
    kpts[L_ANKLE] = (60, 96)
    kpts[R_KNEE] = (40, 66)
    kpts[R_ANKLE] = (40, 96)
    result = classify_pose(_pose(kpts, np.full(17, 0.9)))
    assert result.posture == SITTING


def test_arms_raised():
    kpts = _standing_keypoints()
    kpts[L_WRIST] = (37, 5)  # above the shoulder (smaller y)
    kpts[R_WRIST] = (63, 5)
    result = classify_pose(_pose(kpts, np.full(17, 0.9)))
    assert result.posture == STANDING
    assert result.arms == ARMS_RAISED


def test_all_low_scores_is_unknown():
    pose = _pose(_standing_keypoints(), np.full(17, 0.1))
    result = classify_pose(pose, kpt_threshold=0.3)
    assert result.posture == UNKNOWN
    assert result.arms == ARMS_UNKNOWN
    assert result.confidence == 0.0


def test_legs_hidden_keeps_arm_state_but_unknown_posture():
    scores = np.full(17, 0.9)
    for i in (L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE):
        scores[i] = 0.0
    result = classify_pose(_pose(_standing_keypoints(), scores))
    assert result.posture == UNKNOWN
    assert result.arms == ARMS_DOWN
    assert result.confidence == 0.0


def test_one_visible_leg_is_enough():
    scores = np.full(17, 0.9)
    for i in (R_HIP, R_KNEE, R_ANKLE):  # hide the right leg entirely
        scores[i] = 0.0
    result = classify_pose(_pose(_standing_keypoints(), scores))
    assert result.posture == STANDING


def test_confidence_reflects_leg_scores():
    scores = np.full(17, 0.9)
    for i in (L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE):
        scores[i] = 0.6
    result = classify_pose(_pose(_standing_keypoints(), scores))
    assert result.confidence == 0.6


def test_classify_poses_maps_over_list():
    pose = _pose(_standing_keypoints(), np.full(17, 0.9))
    results = classify_poses([pose, pose])
    assert len(results) == 2
    assert all(r.posture == STANDING for r in results)
