"""Pure-logic tests for the YOLO-free performer-selection policy."""

from __future__ import annotations

import numpy as np

from posedet import select_performers

# A 1000x1000 frame with three people:
#   A: high score, mid-frame      (a clear performer)
#   B: low score, mid-frame
#   C: mid score, feet near bottom (foreground-audience candidate)
W = H = 1000
BOXES = np.array(
    [
        [400.0, 300.0, 100.0, 300.0],  # A: bottom edge at y=600
        [100.0, 300.0, 100.0, 300.0],  # B: bottom edge at y=600
        [700.0, 600.0, 100.0, 350.0],  # C: bottom edge at y=950
    ],
    dtype=np.float32,
)
SCORES = np.array([0.9, 0.4, 0.6], dtype=np.float32)


def test_defaults_sort_by_score_keep_all():
    boxes, scores = select_performers(BOXES, SCORES, W, H)
    np.testing.assert_allclose(scores, [0.9, 0.6, 0.4])  # A, C, B
    np.testing.assert_allclose(boxes[0], BOXES[0])


def test_max_people_caps_after_ranking():
    boxes, scores = select_performers(BOXES, SCORES, W, H, max_people=2)
    assert boxes.shape == (2, 4)
    np.testing.assert_allclose(scores, [0.9, 0.6])  # A then C


def test_stage_roi_drops_outside_center():
    # ROI covering only the right half drops A and B (centered left), keeps C.
    boxes, scores = select_performers(
        BOXES, SCORES, W, H, stage_roi=(0.55, 0.0, 1.0, 1.0)
    )
    assert boxes.shape == (1, 4)
    np.testing.assert_allclose(boxes[0], BOXES[2])


def test_audience_suppression_demotes_low_box():
    # Penalize C (feet at y=950 > 0.82*1000) hard enough to fall below B.
    boxes, scores = select_performers(
        BOXES, SCORES, W, H, audience_suppression=0.3, audience_band=0.82
    )
    # Ranking scores: A=0.9, B=0.4, C=0.6-0.3=0.3 -> order A, B, C.
    np.testing.assert_allclose(scores, [0.9, 0.4, 0.6])
    np.testing.assert_allclose(boxes[2], BOXES[2])  # C demoted to last


def test_audience_suppression_is_soft_not_a_filter():
    # Even penalized, C is still returned (ranking only, never removed).
    boxes, _ = select_performers(BOXES, SCORES, W, H, audience_suppression=0.5)
    assert boxes.shape == (3, 4)


def test_dedupe_removes_overlap():
    boxes = np.array(
        [
            [0.0, 0.0, 100.0, 100.0],
            [5.0, 5.0, 100.0, 100.0],  # near-duplicate of the first
            [500.0, 500.0, 100.0, 100.0],
        ],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    kept, kept_scores = select_performers(boxes, scores, W, H, dedupe_iou=0.5)
    assert kept.shape == (2, 4)
    np.testing.assert_allclose(kept_scores, [0.9, 0.7])


def test_empty_input():
    boxes, scores = select_performers(np.empty((0, 4)), np.empty((0,)), W, H)
    assert boxes.shape == (0, 4)
    assert scores.shape == (0,)


def test_roi_emptied_returns_empty():
    boxes, scores = select_performers(
        BOXES, SCORES, W, H, stage_roi=(0.0, 0.0, 0.01, 0.01)
    )
    assert boxes.shape == (0, 4)
    assert scores.shape == (0,)
