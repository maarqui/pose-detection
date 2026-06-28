"""Role-aware shot proposals for jazz musicians.

This module turns a labeled ``Musician`` into candidate subject regions such as
hands, mouthpiece, upper body, full body, or drum kit details. The regions are still
plain COCO boxes; ``framing.fit_aspect`` is responsible for growing the chosen target
to the camera's aspect ratio.

Refactored to use musician-specific profile modules.
"""

from __future__ import annotations

from .poseclass import ARMS_RAISED
from .profiles import (
    bassist_shots,
    drummer_shots,
    guitarist_shots,
    horn_shots,
    pianist_shots,
)
from .profiles.utils import (
    ShotCandidate,
    candidate,
    full_body_box,
    upper_body_box,
)

HORN_ROLES = {"saxophonist", "trumpeter", "trombonist", "clarinetist", "flutist"}


def role_shot_candidates(
    musician,
    musician_index: int,
    musicians: list,
    salience: float,
    *,
    kpt_threshold: float = 0.3,
) -> list[ShotCandidate]:
    """Return role-specific shot candidates for one musician."""
    role = musician.role
    solo_bonus = 0.12 if musician.posture.arms == ARMS_RAISED else 0.0

    if role == "pianist":
        return pianist_shots(
            musician, musician_index, musicians, salience, solo_bonus, kpt_threshold
        )

    if role == "drummer":
        return drummer_shots(
            musician, musician_index, musicians, salience, solo_bonus, kpt_threshold
        )

    if role in HORN_ROLES:
        return horn_shots(
            musician,
            musician_index,
            musicians,
            salience,
            solo_bonus,
            kpt_threshold,
            role,
        )

    if role == "bassist":
        return bassist_shots(
            musician, musician_index, musicians, salience, solo_bonus, kpt_threshold
        )

    if role == "guitarist":
        return guitarist_shots(
            musician, musician_index, musicians, salience, solo_bonus, kpt_threshold
        )

    # Fallback for unknown roles
    candidates: list[ShotCandidate] = []
    candidates.append(
        candidate(
            upper_body_box(musician),
            salience + 0.10 + solo_bonus,
            musician_index,
            "medium",
            f"{role} upper body",
            margin=0.18,
            max_zoom=2.6,
        )
    )
    candidates.append(
        candidate(
            full_body_box(musician),
            salience + 0.02 + solo_bonus,
            musician_index,
            "wide",
            f"{role} full body",
            margin=0.22,
            max_zoom=1.9,
        )
    )
    return candidates
