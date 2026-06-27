from .utils import (
    hands_box,
    upper_body_box,
    relative_box,
    instrument_union,
    candidate,
    ShotCandidate
)

def guitarist_shots(
    musician,
    musician_index: int,
    salience: float,
    solo_bonus: float,
    kpt_threshold: float,
) -> list[ShotCandidate]:
    candidates = []

    # Close-up: picking hand or fretboard
    candidates.append(
        candidate(
            hands_box(musician, kpt_threshold),
            salience + 0.22 + solo_bonus,
            musician_index,
            "close_up",
            "guitarist picking hand or fretboard",
            margin=0.16,
            max_zoom=3.8,
        )
    )

    # Medium close-up: torso, guitar, both hands
    candidates.append(
        candidate(
            instrument_union(musician, [upper_body_box(musician)]),
            salience + 0.19 + solo_bonus,
            musician_index,
            "medium_close",
            "guitarist torso guitar both hands",
            margin=0.15,
            max_zoom=3.0,
        )
    )

    # Medium shot: seated/standing posture with instrument
    candidates.append(
        candidate(
            instrument_union(
                musician, [relative_box(musician.pose.box, 0.0, 0.0, 1.0, 0.86)]
            ),
            salience + 0.13 + solo_bonus,
            musician_index,
            "medium",
            "guitarist posture with instrument",
            margin=0.18,
            max_zoom=2.5,
        )
    )

    return candidates
