"""Tests for role-aware shot proposals."""

from __future__ import annotations

import numpy as np

from posedet import InstrumentDetection, Musician, PersonPose, choose_shot
from posedet.poseclass import PoseClassification
from posedet.shot_profiles import role_shot_candidates

FRAME_W, FRAME_H = 1920, 1080
ASPECT = FRAME_W / FRAME_H

L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12


def _pose(box, *, wrists=((0, 0), (0, 0))) -> PersonPose:
    keypoints = np.zeros((17, 2), dtype=float)
    scores = np.zeros(17, dtype=float)
    x, y, w, h = box
    keypoints[L_SHOULDER] = (x + w * 0.35, y + h * 0.20)
    keypoints[R_SHOULDER] = (x + w * 0.65, y + h * 0.20)
    keypoints[L_ELBOW] = (x + w * 0.38, y + h * 0.42)
    keypoints[R_ELBOW] = (x + w * 0.62, y + h * 0.42)
    keypoints[L_WRIST], keypoints[R_WRIST] = wrists
    keypoints[L_HIP] = (x + w * 0.38, y + h * 0.62)
    keypoints[R_HIP] = (x + w * 0.62, y + h * 0.62)
    scores[[L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST]] = 0.9
    scores[[L_HIP, R_HIP]] = 0.9
    return PersonPose(
        keypoints=keypoints, scores=scores, box=np.asarray(box, dtype=float)
    )


def _musician(role, label, box, instrument_box, *, arms="down") -> Musician:
    x, y, w, h = box
    pose = _pose(
        box,
        wrists=((x + w * 0.45, y + h * 0.50), (x + w * 0.62, y + h * 0.50)),
    )
    instrument = InstrumentDetection(
        label, 0.8, np.asarray(instrument_box, dtype=float)
    )
    return Musician(
        pose=pose,
        posture=PoseClassification("standing", arms, 0.9),
        instrument=instrument,
        role=role,
    )


def test_pianist_prefers_hands_close_up_candidate():
    musician = _musician(
        "pianist",
        "piano",
        (780, 260, 420, 620),
        (700, 560, 620, 180),
    )

    candidates = role_shot_candidates(musician, 0, 0.8)
    best = max(candidates, key=lambda candidate: candidate.score)

    assert best.shot_type == "close_up"
    assert "hands" in best.description
    assert best.target_box[3] < musician.pose.box[3]


def test_horn_solo_shot_names_mouthpiece_and_keeps_camera_aspect():
    musician = _musician(
        "saxophonist",
        "saxophone",
        (700, 220, 360, 700),
        (760, 360, 220, 360),
        arms="raised",
    )

    shot = choose_shot([musician], FRAME_W, FRAME_H)

    assert shot.shot_type == "close_up"
    assert "mouthpiece" in shot.description
    assert abs(shot.box[2] / shot.box[3] - ASPECT) < 1e-6


def test_near_equal_musicians_still_produce_wide_ensemble_shot():
    guitarist = _musician(
        "guitarist",
        "guitar",
        (260, 260, 360, 620),
        (300, 490, 320, 180),
    )
    saxophonist = _musician(
        "saxophonist",
        "saxophone",
        (1280, 240, 360, 640),
        (1340, 380, 210, 340),
    )

    shot = choose_shot([guitarist, saxophonist], FRAME_W, FRAME_H)

    assert shot.shot_type == "wide"
    assert shot.musician_indices == (0, 1)
    assert abs(shot.box[2] / shot.box[3] - ASPECT) < 1e-6
