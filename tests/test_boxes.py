"""Pure-logic tests for the box-geometry helpers. No weights, milliseconds to run."""

from __future__ import annotations

import numpy as np

from posedet import (
    dedupe_ranked_boxes,
    expand_box,
    filter_to_roi,
    iou_xywh,
    smooth_boxes,
)


def test_iou_identical_is_one():
    box = np.array([10.0, 10.0, 20.0, 20.0])
    assert iou_xywh(box, box) == 1.0


def test_iou_disjoint_is_zero():
    a = np.array([0.0, 0.0, 10.0, 10.0])
    b = np.array([100.0, 100.0, 10.0, 10.0])
    assert iou_xywh(a, b) == 0.0


def test_iou_half_overlap():
    # Two 10x10 boxes sharing exactly half their area -> 1/3 IoU.
    a = np.array([0.0, 0.0, 10.0, 10.0])
    b = np.array([5.0, 0.0, 10.0, 10.0])
    assert iou_xywh(a, b) == 1.0 / 3.0


def test_expand_box_grows_and_clamps():
    # Centered box expanded 2x but clamped so it cannot leave the image.
    box = np.array([40.0, 40.0, 20.0, 20.0])
    grown = expand_box(box, image_width=100, image_height=100, scale=2.0)
    np.testing.assert_allclose(grown, [30.0, 30.0, 40.0, 40.0])

    edge = np.array([0.0, 0.0, 20.0, 20.0])
    clamped = expand_box(edge, image_width=100, image_height=100, scale=3.0)
    assert clamped[0] == 0.0 and clamped[1] == 0.0  # cannot go negative


def test_filter_to_roi_keeps_centers_inside():
    boxes = np.array(
        [
            [10.0, 10.0, 10.0, 10.0],  # center (15, 15) -> inside
            [90.0, 90.0, 10.0, 10.0],  # center (95, 95) -> outside
        ],
        dtype=np.float32,
    )
    roi = np.array([0.0, 0.0, 50.0, 50.0])
    kept = filter_to_roi(boxes, roi)
    assert kept.shape == (1, 4)
    np.testing.assert_allclose(kept[0], boxes[0])


def test_filter_to_roi_none_is_noop():
    boxes = np.array([[10.0, 10.0, 10.0, 10.0]], dtype=np.float32)
    assert filter_to_roi(boxes, None) is boxes


def test_dedupe_drops_overlap_and_caps():
    boxes = np.array(
        [
            [0.0, 0.0, 10.0, 10.0],  # high score
            [1.0, 1.0, 10.0, 10.0],  # near-duplicate of the first
            [50.0, 50.0, 10.0, 10.0],  # distinct
        ],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    kept = dedupe_ranked_boxes(boxes, scores, max_boxes=5, iou_threshold=0.45)
    assert kept.shape == (2, 4)
    np.testing.assert_allclose(kept[0], boxes[0])  # highest score wins
    np.testing.assert_allclose(kept[1], boxes[2])  # the distinct box

    capped = dedupe_ranked_boxes(boxes, scores, max_boxes=1, iou_threshold=0.45)
    assert capped.shape == (1, 4)


def test_dedupe_empty():
    assert dedupe_ranked_boxes(np.empty((0, 4)), np.empty((0,)), max_boxes=5).shape == (
        0,
        4,
    )


def test_smooth_boxes_blends_matched_box():
    previous = np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32)
    detected = np.array([[2.0, 2.0, 10.0, 10.0]], dtype=np.float32)
    smoothed = smooth_boxes(previous, detected, smoothing=0.5, min_iou=0.15)
    np.testing.assert_allclose(smoothed[0], [1.0, 1.0, 10.0, 10.0])


def test_smooth_boxes_zero_smoothing_is_passthrough():
    previous = np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32)
    detected = np.array([[2.0, 2.0, 10.0, 10.0]], dtype=np.float32)
    assert smooth_boxes(previous, detected, smoothing=0.0) is detected


def test_smooth_boxes_unmatched_box_unchanged():
    # No previous box overlaps the detection -> returned as-is.
    previous = np.array([[100.0, 100.0, 10.0, 10.0]], dtype=np.float32)
    detected = np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32)
    smoothed = smooth_boxes(previous, detected, smoothing=0.5, min_iou=0.15)
    np.testing.assert_allclose(smoothed[0], detected[0])
