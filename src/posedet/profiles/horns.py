from .utils import (
    head_box,
    hands_box,
    upper_body_box,
    chest_up_box,
    head_only_box,
    relative_box,
    instrument_union,
    candidate,
    ShotCandidate
)

def horn_shots(
    musician,
    musician_index: int,
    musicians: list,
    salience: float,
    solo_bonus: float,
    kpt_threshold: float,
    role: str
) -> list[ShotCandidate]:
    candidates = []

    face = head_box(musician, kpt_threshold)
    hands = hands_box(musician, kpt_threshold)

    # Close-up: face, mouthpiece, hands during solo
    horn_detail = instrument_union(musician, [face, hands])
    candidates.append(
        candidate(
            horn_detail,
            salience + 0.26 + solo_bonus,
            musician_index,
            "close_up",
            f"{role} face mouthpiece hands",
            margin=0.13,
            max_zoom=4.0,
        )
    )

    # Medium close-up: head, torso, instrument bell (solo)
    candidates.append(
        candidate(
            instrument_union(musician, [chest_up_box(musician)]),
            salience + 0.22 + solo_bonus,
            musician_index,
            "medium_close",
            f"{role} torso instrument bell",
            margin=0.15,
            max_zoom=3.2,
        )
    )

    # Medium shot: full instrument and upper body
    candidates.append(
        candidate(
            instrument_union(musician, [upper_body_box(musician)]),
            salience + 0.18 + solo_bonus,
            musician_index,
            "medium",
            f"{role} full instrument upper body",
            margin=0.18,
            max_zoom=2.6,
        )
    )

    # Close-up: just the head/mouthpiece area
    candidates.append(
        candidate(
            head_only_box(musician, kpt_threshold),
            salience + 0.14 + solo_bonus,
            musician_index,
            "close_up",
            f"{role} headshot",
            margin=0.25,
            max_zoom=3.5
        )
    )

    return candidates
