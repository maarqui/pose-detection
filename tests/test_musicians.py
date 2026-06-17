"""Tests for instrument<->musician association and role labeling. Pure geometry on
synthetic poses and instrument boxes, so no model weights are involved."""

from __future__ import annotations

import numpy as np

from posedet import InstrumentDetection, PersonPose
from posedet.musicians import (
    DEFAULT_ROLE,
    Musician,
    associate_instruments,
    label_musicians,
    role_for_instrument,
)
from posedet.poseclass import ARMS_DOWN, STANDING, PoseClassification

L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12


def _pose(box, wrists, hips=None, score=0.9) -> PersonPose:
    keypoints = np.zeros((17, 2), dtype=float)
    scores = np.full(17, score, dtype=float)
    keypoints[L_WRIST] = wrists[0]
    keypoints[R_WRIST] = wrists[1]
    if hips is not None:
        keypoints[L_HIP] = hips[0]
        keypoints[R_HIP] = hips[1]
    return PersonPose(
        keypoints=keypoints, scores=scores, box=np.asarray(box, dtype=float)
    )


def _instrument(label, box, score=0.8) -> InstrumentDetection:
    return InstrumentDetection(
        label=label, score=score, box=np.asarray(box, dtype=float)
    )


def test_overlapping_instrument_is_associated():
    pose = _pose((0, 0, 100, 200), wrists=[(40, 120), (60, 120)])
    instr = _instrument("guitar", (30, 100, 40, 40))
    assigned = associate_instruments([pose], [instr])
    assert assigned[0] is instr


def test_far_instrument_is_not_associated():
    pose = _pose((0, 0, 100, 200), wrists=[(40, 120), (60, 120)])
    instr = _instrument("guitar", (300, 300, 40, 40))
    assigned = associate_instruments([pose], [instr])
    assert assigned[0] is None


def test_instrument_outside_box_but_near_wrist_associates():
    # A guitar neck spilling past the person box is rescued by hand proximity.
    pose = _pose((0, 0, 100, 200), wrists=[(80, 120), (80, 120)])
    instr = _instrument("guitar", (90, 110, 60, 30))
    assigned = associate_instruments([pose], [instr])
    assert assigned[0] is instr


def test_greedy_one_to_one_assignment():
    near = _pose((0, 0, 100, 200), wrists=[(50, 120), (50, 120)])
    far = _pose((300, 0, 100, 200), wrists=[(350, 120), (350, 120)])
    instr = _instrument("saxophone", (40, 100, 40, 40))
    assigned = associate_instruments([near, far], [instr])
    assert assigned[0] is instr
    assert assigned[1] is None


def test_two_instruments_match_their_players():
    person_a = _pose((0, 0, 100, 200), wrists=[(50, 120), (50, 120)])
    person_b = _pose((300, 0, 100, 200), wrists=[(350, 120), (350, 120)])
    guitar = _instrument("guitar", (40, 100, 40, 40))
    sax = _instrument("saxophone", (340, 100, 40, 40))
    assigned = associate_instruments([person_a, person_b], [guitar, sax])
    assert assigned[0] is guitar
    assert assigned[1] is sax


def test_min_association_blocks_weak_pairings():
    pose = _pose((0, 0, 100, 200), wrists=[(80, 120), (80, 120)])
    instr = _instrument("guitar", (90, 110, 60, 30))  # ~0.42 association
    assert associate_instruments([pose], [instr], min_association=0.5)[0] is None


def test_role_mapping():
    assert role_for_instrument("guitar") == "guitarist"
    assert role_for_instrument("Guitar") == "guitarist"  # case-insensitive
    assert role_for_instrument("kazoo") == DEFAULT_ROLE  # unmapped
    assert role_for_instrument(None) == DEFAULT_ROLE


def test_label_musicians_composes_with_injected_classifications():
    player = _pose((0, 0, 100, 200), wrists=[(50, 120), (50, 120)])
    bystander = _pose((300, 0, 100, 200), wrists=[(350, 120), (350, 120)])
    guitar = _instrument("guitar", (40, 100, 40, 40))
    postures = [
        PoseClassification(STANDING, ARMS_DOWN, 0.9),
        PoseClassification(STANDING, ARMS_DOWN, 0.9),
    ]

    musicians = label_musicians([player, bystander], [guitar], classifications=postures)

    assert all(isinstance(m, Musician) for m in musicians)
    assert musicians[0].instrument is guitar
    assert musicians[0].role == "guitarist"
    assert musicians[0].posture.posture == STANDING
    assert musicians[1].instrument is None
    assert musicians[1].role == DEFAULT_ROLE


def test_label_musicians_classifies_when_not_provided():
    pose = _pose((0, 0, 100, 200), wrists=[(50, 120), (50, 120)])
    musicians = label_musicians([pose], [])
    assert len(musicians) == 1
    assert musicians[0].instrument is None
    assert isinstance(musicians[0].posture, PoseClassification)


def test_classifications_length_mismatch_raises():
    pose = _pose((0, 0, 100, 200), wrists=[(50, 120), (50, 120)])
    try:
        label_musicians(
            [pose, pose],
            [],
            classifications=[PoseClassification(STANDING, ARMS_DOWN, 0.9)],
        )
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_empty_inputs():
    assert associate_instruments([], []) == []
    assert label_musicians([], []) == []
