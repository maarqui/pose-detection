from .utils import (
    hands_box,
    upper_body_box,
    full_body_box,
    instrument_union,
    candidate,
    ShotCandidate
)

def bassist_shots(
    musician,
    musician_index: int,
    salience: float,
    solo_bonus: float,
    kpt_threshold: float,
) -> list[ShotCandidate]:
    candidates = []

    # Close-up: hands
    candidates.append(
        candidate(
            hands_box(musician, kpt_threshold),
            salience + 0.23 + solo_bonus,
            musician_index,
            "close_up",
            "bassist hands",
            margin=0.18,
            max_zoom=3.8,
        )
    )

    # Medium shot: torso, bass neck, both hand
    candidates.append(
        candidate(
            instrument_union(musician, [upper_body_box(musician)]),
            salience + 0.18 + solo_bonus,
            musician_index,
            "medium",
            "bassist torso bass neck both hands",
            margin=0.16,
            max_zoom=2.8,
        )
    )

    # Wide vertical shot: full body
    candidates.append(
        candidate(
            instrument_union(musician, [full_body_box(musician)]),
            salience + 0.09 + solo_bonus,
            musician_index,
            "wide_vertical",
            "bassist full body",
            margin=0.20,
            max_zoom=2.0,
        )
    )

    return candidates
