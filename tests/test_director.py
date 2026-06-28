"""Tests for the auto-shot director. Fake runner/instrument-detector drive the
composition, striding, and shot-smoothing logic without any model weights."""

from __future__ import annotations

import numpy as np
import pytest

from posedet import InstrumentDetection, PersonPose
from posedet.director import DirectorFrame, ShotDirector
from posedet.framing import choose_shot

# process() builds a PIL image on instrument-update frames.
pytest.importorskip("PIL")

FRAME_W, FRAME_H = 1920, 1080


def _frame():
    return np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)


def _pose(box):
    return PersonPose(
        keypoints=np.zeros((17, 2), dtype=float),
        scores=np.ones(17, dtype=float),
        box=np.asarray(box, dtype=float),
    )


class FakeRunner:
    def __init__(self, pose_lists):
        self.pose_lists = pose_lists
        self.calls = 0

    def process(self, frame_bgr):
        poses = self.pose_lists[min(self.calls, len(self.pose_lists) - 1)]
        self.calls += 1
        return poses


class FakeInstrumentDetector:
    def __init__(self, instruments):
        self.instruments = instruments
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        return self.instruments


def test_instrument_striding_and_runner_calls():
    runner = FakeRunner([[_pose((100, 100, 200, 400))]])
    instruments = FakeInstrumentDetector([])
    director = ShotDirector(
        runner=runner, instrument_detector=instruments, instrument_stride=2
    )

    for _ in range(3):
        director.process(_frame())

    assert runner.calls == 3  # pose runner every frame
    assert instruments.calls == 2  # instrument detector on frames 0 and 2


def test_process_returns_labeled_musicians_and_shot():
    pose = _pose((800, 300, 300, 500))
    instr = InstrumentDetection("guitar", 0.8, np.array([820.0, 600.0, 200.0, 200.0]))
    director = ShotDirector(
        runner=FakeRunner([[pose]]),
        instrument_detector=FakeInstrumentDetector([instr]),
        shot_smoothing=0.0,
    )

    result = director.process(_frame())

    assert isinstance(result, DirectorFrame)
    assert len(result.musicians) == 1
    assert result.instruments == [instr]
    # No smoothing -> shot equals the raw framing of the labeled musicians.
    expected = choose_shot(result.musicians, FRAME_W, FRAME_H)
    np.testing.assert_allclose(result.shot.box, expected.box)


def test_unassociated_instruments_are_not_exposed_for_overlay():
    pose = _pose((800, 300, 300, 500))
    near = InstrumentDetection("guitar", 0.8, np.array([820.0, 600.0, 200.0, 200.0]))
    far = InstrumentDetection("saxophone", 0.9, np.array([20.0, 20.0, 40.0, 40.0]))
    director = ShotDirector(
        runner=FakeRunner([[pose]]),
        instrument_detector=FakeInstrumentDetector([near, far]),
        shot_smoothing=0.0,
    )

    result = director.process(_frame())

    assert result.instruments == [near]
    assert result.musicians[0].instrument is near


def test_shot_smoothing_blends_with_previous_frame():
    runner = FakeRunner([[_pose((100, 100, 200, 400))], [_pose((1500, 400, 300, 500))]])
    director = ShotDirector(
        runner=runner,
        instrument_detector=FakeInstrumentDetector([]),
        shot_smoothing=0.9,
        instrument_stride=100,  # keep instruments out of the way
    )

    first = director.process(_frame())
    second = director.process(_frame())

    raw_second = choose_shot(second.musicians, FRAME_W, FRAME_H)
    expected = 0.9 * first.shot.box + 0.1 * raw_second.box
    np.testing.assert_allclose(second.shot.box, expected)
    # Smoothing actually moved the box away from the raw framing.
    assert not np.allclose(second.shot.box, raw_second.box)


def test_invalid_instrument_stride_raises():
    with pytest.raises(ValueError):
        ShotDirector(
            runner=FakeRunner([[]]),
            instrument_detector=FakeInstrumentDetector([]),
            instrument_stride=0,
        )
