"""Tests for the stateful video runner. Uses fake detector/estimator so the
striding / smoothing / scaling logic runs without any model weights."""

from __future__ import annotations

import numpy as np

from posedet import PersonPose, VideoPoseRunner
from posedet.runner import resize_for_inference, scale_pose


def _pose(box=(10.0, 20.0, 30.0, 40.0)) -> PersonPose:
    return PersonPose(
        keypoints=np.ones((17, 2), dtype=float),
        scores=np.ones(17, dtype=float),
        box=np.asarray(box, dtype=float),
    )


class FakeDetector:
    def __init__(self, box=(0.0, 0.0, 10.0, 10.0)):
        self.box = np.asarray([box], dtype=np.float32)
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        return self.box.copy()


class FakeEstimator:
    def __init__(self):
        self.calls = 0

    def estimate(self, image, boxes):
        self.calls += 1
        return [_pose() for _ in range(len(boxes))]


def _frames(n, size=8):
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    return [(i, frame.copy()) for i in range(n)]


def test_detector_stride_runs_detection_periodically():
    det, est = FakeDetector(), FakeEstimator()
    runner = VideoPoseRunner(
        detector_stride=2, pose_stride=1, detector=det, estimator=est
    )
    list(runner.run(_frames(4)))
    assert det.calls == 2  # frames 0 and 2
    assert est.calls == 4  # every frame


def test_pose_stride_reuses_previous_poses():
    det, est = FakeDetector(), FakeEstimator()
    runner = VideoPoseRunner(pose_stride=2, detector=det, estimator=est)
    outputs = [poses for _, _, poses in runner.run(_frames(4))]
    assert est.calls == 2  # frames 0 and 2 only
    # Skipped frames reuse the exact previous pose list object.
    assert outputs[0] is outputs[1]
    assert outputs[2] is outputs[3]


def test_reset_clears_state():
    det, est = FakeDetector(), FakeEstimator()
    runner = VideoPoseRunner(detector=det, estimator=est)
    list(runner.run(_frames(3)))
    runner.reset()
    assert runner._frame_index == 0
    assert runner._last_boxes.shape == (0, 4)
    assert runner._last_poses == []


def test_invalid_strides_rejected():
    for kwargs in ({"detector_stride": 0}, {"pose_stride": 0}):
        try:
            VideoPoseRunner(
                detector=FakeDetector(), estimator=FakeEstimator(), **kwargs
            )
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")


def test_scale_pose_scales_keypoints_and_box():
    pose = _pose(box=(10.0, 20.0, 30.0, 40.0))
    scaled = scale_pose(pose, 2.0, 3.0)
    np.testing.assert_allclose(scaled.keypoints[:, 0], 2.0)
    np.testing.assert_allclose(scaled.keypoints[:, 1], 3.0)
    np.testing.assert_allclose(scaled.box, [20.0, 60.0, 60.0, 120.0])


def test_scale_pose_unit_is_passthrough():
    pose = _pose()
    assert scale_pose(pose, 1.0, 1.0) is pose


def test_resize_for_inference_noop_when_disabled():
    frame = np.zeros((10, 20, 3), dtype=np.uint8)
    out, sx, sy = resize_for_inference(frame, 0)
    assert out is frame and sx == 1.0 and sy == 1.0
    # A width that would upscale is also a no-op.
    out2, sx2, sy2 = resize_for_inference(frame, 40)
    assert out2 is frame and sx2 == 1.0 and sy2 == 1.0
