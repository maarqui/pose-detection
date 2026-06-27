from .utils import (
    hands_box,
    upper_body_box,
    full_body_box,
    relative_box,
    instrument_box,
    candidate,
    ShotCandidate
)

def drummer_shots(
    musician,
    musician_index: int,
    salience: float,
    solo_bonus: float,
    kpt_threshold: float,
) -> list[ShotCandidate]:
    candidates = []

    kit = instrument_box(musician)
    if kit is None:
        kit = relative_box(musician.pose.box, 0.0, 0.25, 1.0, 0.65)

    # close up cymbals, toms or hat
    candidates.append(
        candidate(
            relative_box(kit, 0.1, 0.0, 0.8, 0.5),
            salience + 0.22 + solo_bonus,
            musician_index,
            "close_up",
            "cymbals toms hat",
            margin=0.12,
            max_zoom=3.8,
        )
    )

    # medium close up area hands
    candidates.append(
        candidate(
            hands_box(musician, kpt_threshold),
            salience + 0.20 + solo_bonus,
            musician_index,
            "medium_close",
            "drummer hands area",
            margin=0.18,
            max_zoom=3.4,
        )
    )

    # medium shot from waist up
    candidates.append(
        candidate(
            upper_body_box(musician),
            salience + 0.14 + solo_bonus,
            musician_index,
            "medium",
            "drummer waist up",
            margin=0.18,
            max_zoom=2.6,
        )
    )

    # Extreme wide shot full body (centered at drummer)
    candidates.append(
        candidate(
            full_body_box(musician),
            salience + 0.04 + solo_bonus,
            musician_index,
            "extreme_wide",
            "drummer full body",
            margin=0.22,
            max_zoom=1.8,
        )
    )

    return candidates
