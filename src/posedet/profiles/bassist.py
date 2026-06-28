from .utils import (
    ShotCandidate,
    candidate,
    chest_up_box,
    full_body_box,
    hands_box,
    head_only_box,
    instrument_union,
    upper_body_box,
)


def bassist_shots(
    musician,
    musician_index: int,
    musicians: list,
    salience: float,
    solo_bonus: float,
    kpt_threshold: float,
) -> list[ShotCandidate]:
    candidates = []

    # Close-up: hands
    candidates.append(
        candidate(
            hands_box(musician, kpt_threshold),
            salience + 0.25 + solo_bonus,
            musician_index,
            "close_up",
            "bassist hands",
            margin=0.18,
            max_zoom=3.8,
        )
    )

    # Medium close-up: torso, bass neck, hands
    candidates.append(
        candidate(
            instrument_union(musician, [chest_up_box(musician)]),
            salience + 0.22 + solo_bonus,
            musician_index,
            "medium_close",
            "bassist torso and neck",
            margin=0.15,
            max_zoom=3.2,
        )
    )

    # Medium shot: torso, bass neck, both hand
    candidates.append(
        candidate(
            instrument_union(musician, [upper_body_box(musician)]),
            salience + 0.20 + solo_bonus,
            musician_index,
            "medium",
            "bassist waist up",
            margin=0.16,
            max_zoom=2.8,
        )
    )

    # Close-up: head shot
    candidates.append(
        candidate(
            head_only_box(musician, kpt_threshold),
            salience + 0.14 + solo_bonus,
            musician_index,
            "close_up",
            "bassist headshot",
            margin=0.25,
            max_zoom=3.5,
        )
    )

    # Wide vertical shot: full body
    candidates.append(
        candidate(
            instrument_union(musician, [full_body_box(musician)]),
            salience + 0.08 + solo_bonus,
            musician_index,
            "wide_vertical",
            "bassist full body",
            margin=0.20,
            max_zoom=2.0,
        )
    )

    return candidates
