"""Drawing tests. ``draw_pose`` needs OpenCV, so the suite skips cleanly when it is
not installed, keeping the pure-logic suite green everywhere."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from posedet import COCO_SKELETON, KEYPOINT_COLORS, PersonPose, draw_pose

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("cv2") is None, reason="opencv not installed"
)


def _center_pose() -> PersonPose:
    # All keypoints at (50, 50), all confident, so everything is drawn.
    return PersonPose(
        keypoints=np.full((17, 2), 50.0),
        scores=np.ones(17),
        box=np.array([10.0, 10.0, 80.0, 80.0]),
    )


def test_palette_has_one_color_per_keypoint():
    assert len(KEYPOINT_COLORS) == 17


def test_draw_pose_does_not_mutate_input():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    out = draw_pose(image, [_center_pose()])
    assert out is not image
    assert image.sum() == 0  # original untouched


def test_draw_pose_uses_per_keypoint_color():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    # Only keypoint 0 is confident, so its color is unambiguous at (50, 50).
    scores = np.zeros(17)
    scores[0] = 1.0
    pose = PersonPose(
        keypoints=np.full((17, 2), 50.0),
        scores=scores,
        box=np.array([10.0, 10.0, 80.0, 80.0]),
    )
    out = draw_pose(
        image, [pose], point_colors=KEYPOINT_COLORS, point_radius=4, draw_boxes=False
    )
    assert tuple(int(c) for c in out[50, 50]) == KEYPOINT_COLORS[0]


def test_draw_pose_thresholds_hide_low_keypoints():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    pose = PersonPose(
        keypoints=np.full((17, 2), 50.0),
        scores=np.zeros(17),  # all below threshold
        box=np.array([10.0, 10.0, 80.0, 80.0]),
    )
    out = draw_pose(image, [pose], kpt_threshold=0.3, draw_boxes=False)
    assert out.sum() == 0  # nothing drawn


def test_skeleton_is_single_source():
    # Slice 5 removed the script's inline edge list; COCO_SKELETON is canonical.
    for a, b in COCO_SKELETON:
        assert 0 <= a < 17 and 0 <= b < 17
