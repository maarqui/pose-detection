"""Tests for instrument detection. The post-processing (label mapping, VOC->COCO,
threshold, NMS, cap) is pure and tested on synthetic arrays; one wiring test drives
the OWLv2 detector with injected fakes so no model weights are needed."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from posedet import Config
from posedet.instruments import (
    InstrumentDetection,
    InstrumentDetector,
    build_instrument_detections,
)

PROMPTS = ("guitar", "saxophone", "drum kit")


def test_maps_labels_and_converts_to_coco():
    # Two non-overlapping boxes; labels index into PROMPTS.
    labels = [0, 1]
    scores = [0.4, 0.9]
    boxes_voc = [[0, 0, 10, 20], [100, 100, 130, 160]]  # x1,y1,x2,y2
    dets = build_instrument_detections(PROMPTS, labels, scores, boxes_voc)

    # Sorted by descending score: saxophone (0.9) first.
    assert [d.label for d in dets] == ["saxophone", "guitar"]
    assert dets[0].score == 0.9
    # VOC (100,100,130,160) -> COCO (100,100,30,60).
    np.testing.assert_array_equal(dets[0].box, [100, 100, 30, 60])
    np.testing.assert_array_equal(dets[1].box, [0, 0, 10, 20])


def test_threshold_filters_low_scores():
    labels = [0, 1]
    scores = [0.05, 0.5]
    boxes_voc = [[0, 0, 10, 10], [50, 50, 70, 70]]
    dets = build_instrument_detections(
        PROMPTS, labels, scores, boxes_voc, threshold=0.1
    )
    assert [d.label for d in dets] == ["saxophone"]


def test_dedupe_drops_overlapping_lower_score():
    labels = [0, 0, 2]
    scores = [0.9, 0.8, 0.7]
    # First two boxes are the same region (IoU 1.0); third is elsewhere.
    boxes_voc = [[0, 0, 100, 100], [0, 0, 100, 100], [200, 200, 260, 260]]
    dets = build_instrument_detections(
        PROMPTS, labels, scores, boxes_voc, dedupe_iou=0.5
    )
    assert len(dets) == 2
    assert dets[0].score == 0.9
    assert dets[1].label == "drum kit"


def test_dedupe_disabled_keeps_overlaps():
    labels = [0, 0]
    scores = [0.9, 0.8]
    boxes_voc = [[0, 0, 100, 100], [0, 0, 100, 100]]
    dets = build_instrument_detections(
        PROMPTS, labels, scores, boxes_voc, dedupe_iou=0.0
    )
    assert len(dets) == 2


def test_max_instruments_caps_by_score():
    labels = [0, 1, 2]
    scores = [0.3, 0.9, 0.6]
    boxes_voc = [[0, 0, 10, 10], [50, 0, 60, 10], [90, 0, 100, 10]]
    dets = build_instrument_detections(
        PROMPTS, labels, scores, boxes_voc, max_instruments=2
    )
    assert [d.label for d in dets] == ["saxophone", "drum kit"]


def test_size_filter_drops_tiny_background_hits():
    labels = [0, 1]
    scores = [0.9, 0.8]
    boxes_voc = [
        [0, 0, 5, 5],  # too tiny in a 1000x1000 frame
        [100, 100, 180, 180],
    ]
    dets = build_instrument_detections(
        PROMPTS,
        labels,
        scores,
        boxes_voc,
        image_size=(1000, 1000),
        min_area_fraction=0.001,
    )
    assert [d.label for d in dets] == ["saxophone"]


def test_aspect_filter_drops_skinny_microphone_stand_like_boxes():
    labels = [0, 1]
    scores = [0.9, 0.8]
    boxes_voc = [
        [20, 20, 24, 220],  # aspect ratio 50:1
        [100, 100, 180, 180],
    ]
    dets = build_instrument_detections(
        PROMPTS,
        labels,
        scores,
        boxes_voc,
        image_size=(1000, 1000),
        max_aspect_ratio=8.0,
    )
    assert [d.label for d in dets] == ["saxophone"]


def test_empty_input_returns_empty():
    assert build_instrument_detections(PROMPTS, [], [], np.empty((0, 4))) == []


def test_label_index_out_of_range_falls_back():
    dets = build_instrument_detections(("guitar",), [5], [0.9], [[0, 0, 10, 10]])
    assert dets[0].label == "5"


def test_detector_wiring_with_fakes():
    """Drive InstrumentDetector.detect end-to-end with stubs (no weights)."""
    torch = pytest.importorskip("torch")

    class FakeProcessor:
        def __call__(self, *, text, images, return_tensors):
            return SimpleNamespace(to=lambda device: {})

        def post_process_grounded_object_detection(
            self, *, outputs, target_sizes, threshold
        ):
            return [
                {
                    "labels": torch.tensor([0, 1]),
                    "scores": torch.tensor([0.5, 0.4]),
                    "boxes": torch.tensor(
                        [[0.0, 0.0, 10.0, 20.0], [5.0, 5.0, 15.0, 25.0]]
                    ),
                }
            ]

    class FakeModel:
        def __call__(self, **kwargs):
            return object()

    detector = InstrumentDetector(Config(device="cpu", instrument_prompts=PROMPTS))
    detector._processor = FakeProcessor()
    detector._model = FakeModel()  # non-None so _ensure_loaded() skips real loading

    image = SimpleNamespace(height=100, width=200)
    dets = detector.detect(image)

    assert [d.label for d in dets] == ["guitar", "saxophone"]
    assert isinstance(dets[0], InstrumentDetection)
    np.testing.assert_array_equal(dets[0].box, [0, 0, 10, 20])
