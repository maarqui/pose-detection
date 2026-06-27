from .utils import (
    head_box,
    hands_box,
    upper_body_box,
    relative_box,
    instrument_union,
    candidate,
    ShotCandidate
)

def horn_shots(
    musician,
    musician_index: int,
    salience: float,
    solo_bonus: float,
    kpt_threshold: float,
    role: str
) -> list[ShotCandidate]:
    candidates = []

    face = head_box(musician, kpt_threshold)
    hands = hands_box(musician, kpt_threshold)

    # Close-up: face, mouthpiece, hands during solo
    # We use a union of face and hands to capture the "action" area
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
            instrument_union(musician, [upper_body_box(musician)]),
            salience + 0.21 + solo_bonus,
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
            instrument_union(
                musician, [relative_box(musician.pose.box, 0.0, 0.0, 1.0, 0.78)]
            ),
            salience + 0.13 + solo_bonus,
            musician_index,
            "medium",
            f"{role} full instrument upper body",
            margin=0.18,
            max_zoom=2.6,
        )
    )

    return candidates
