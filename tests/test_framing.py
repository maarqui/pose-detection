"""Tests for shot scoring and aspect-preserving framing. Pure geometry on synthetic
musicians/boxes; the OpenCV-backed apply_zoom is checked only for shape."""

from __future__ import annotations

import numpy as np
import pytest

from posedet import InstrumentDetection, Musician, PersonPose
from posedet.framing import (
    Shot,
    apply_zoom,
    choose_shot,
    fit_aspect,
    score_musician,
    select_target,
)
from posedet.poseclass import PoseClassification

FRAME_W, FRAME_H = 1920, 1080
ASPECT = FRAME_W / FRAME_H


def _musician(box, *, posture="standing", arms="down", conf=0.9, instrument=False):
    pose = PersonPose(
        keypoints=np.zeros((17, 2), dtype=float),
        scores=np.ones(17, dtype=float),
        box=np.asarray(box, dtype=float),
    )
    instr = (
        InstrumentDetection("guitar", 0.8, np.asarray(box, dtype=float))
        if instrument
        else None
    )
    return Musician(
        pose=pose,
        posture=PoseClassification(posture, arms, conf),
        instrument=instr,
        role="guitarist" if instrument else "musician",
    )


def test_soloist_scores_higher_than_seated_bystander():
    soloist = _musician(
        (0, 0, FRAME_W * 0.5, FRAME_H * 0.5),
        posture="standing",
        arms="raised",
        instrument=True,
    )
    bystander = _musician((0, 0, 50, 50), posture="sitting", arms="down")
    assert score_musician(soloist, FRAME_W, FRAME_H) > score_musician(
        bystander, FRAME_W, FRAME_H
    )


def test_fit_aspect_preserves_frame_aspect_ratio():
    crop = fit_aspect((900, 400, 100, 80), FRAME_W, FRAME_H)
    assert abs(crop[2] / crop[3] - ASPECT) < 1e-6
    # Inside the frame bounds.
    assert crop[0] >= 0 and crop[1] >= 0
    assert crop[0] + crop[2] <= FRAME_W + 1e-6
    assert crop[1] + crop[3] <= FRAME_H + 1e-6


def test_fit_aspect_expands_tall_target_to_aspect():
    crop = fit_aspect((900, 200, 50, 600), FRAME_W, FRAME_H)
    assert abs(crop[2] / crop[3] - ASPECT) < 1e-6
    assert crop[2] > 50  # widened to match the aspect ratio


def test_fit_aspect_respects_max_zoom_floor():
    crop = fit_aspect((10, 10, 10, 10), FRAME_W, FRAME_H, max_zoom=2.0)
    # Never zooms in past frame / max_zoom in either dimension.
    assert crop[2] >= FRAME_W / 2.0 - 1e-6
    assert crop[3] >= FRAME_H / 2.0 - 1e-6


def test_fit_aspect_huge_target_returns_full_frame():
    crop = fit_aspect((0, 0, FRAME_W - 5, FRAME_H - 5), FRAME_W, FRAME_H)
    np.testing.assert_allclose(crop, [0, 0, FRAME_W, FRAME_H])


def test_select_target_isolates_dominant_soloist():
    soloist = _musician(
        (100, 100, 400, 600), posture="standing", arms="raised", instrument=True
    )
    weak = _musician((1500, 800, 80, 80), posture="sitting", arms="down")
    _target, indices, _top = select_target([soloist, weak], FRAME_W, FRAME_H)
    assert indices == (0,)


def test_select_target_groups_near_equal_performers():
    a = _musician((100, 100, 300, 500), posture="standing", instrument=True)
    b = _musician((1400, 100, 300, 500), posture="standing", instrument=True)
    _target, indices, _top = select_target([a, b], FRAME_W, FRAME_H)
    assert indices == (0, 1)


def test_choose_shot_with_no_musicians_is_full_frame():
    shot = choose_shot([], FRAME_W, FRAME_H)
    assert isinstance(shot, Shot)
    np.testing.assert_allclose(shot.box, [0, 0, FRAME_W, FRAME_H])
    assert shot.score == 0.0
    assert shot.musician_indices == ()


def test_choose_shot_frames_single_musician_with_correct_aspect():
    musician = _musician((800, 300, 300, 500), posture="standing", instrument=True)
    shot = choose_shot([musician], FRAME_W, FRAME_H)
    assert shot.musician_indices == (0,)
    assert abs(shot.box[2] / shot.box[3] - ASPECT) < 1e-6


def test_apply_zoom_keeps_frame_size():
    pytest.importorskip("cv2")
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    out = apply_zoom(frame, (10, 10, 50, 50))
    assert out.shape == frame.shape
