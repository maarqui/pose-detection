"""Pure-logic tests. These run in milliseconds and need no model weights, so they
are the fast feedback loop the agent should use after every change."""

from __future__ import annotations

import numpy as np

from posedet import COCO_KEYPOINTS, COCO_SKELETON, voc_to_coco


def test_voc_to_coco_basic():
    voc = np.array([[10, 20, 110, 220]], dtype=float)  # x1,y1,x2,y2
    coco = voc_to_coco(voc)
    np.testing.assert_allclose(coco[0], [10, 20, 100, 200])


def test_voc_to_coco_empty():
    assert voc_to_coco(np.empty((0, 4))).shape == (0, 4)


def test_coco_keypoints_count():
    assert len(COCO_KEYPOINTS) == 17


def test_skeleton_indices_in_range():
    for a, b in COCO_SKELETON:
        assert 0 <= a < 17 and 0 <= b < 17
