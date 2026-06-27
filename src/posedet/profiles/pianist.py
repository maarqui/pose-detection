from .utils import (
    hands_box,
    upper_body_box,
    full_body_box,
    candidate,
    ShotCandidate
)

def pianist_shots(
    musician,
    musician_index: int,
    salience: float,
    solo_bonus: float,
    kpt_threshold: float,
) -> list[ShotCandidate]:
    candidates = []

    # Close up: hands
    candidates.append(
        candidate(
            hands_box(musician, kpt_threshold),
            salience + 0.25 + solo_bonus,
            musician_index,
            "close_up",
            "pianist hands",
            margin=0.18,
            max_zoom=4.0,
        )
    )

    # Medium shot: from waist up
    candidates.append(
        candidate(
            upper_body_box(musician),
            salience + 0.15 + solo_bonus,
            musician_index,
            "medium",
            "pianist waist up",
            margin=0.16,
            max_zoom=2.8,
        )
    )

    # Extreme wide shot: full body (centered at pianist)
    candidates.append(
        candidate(
            full_body_box(musician),
            salience + 0.05 + solo_bonus,
            musician_index,
            "extreme_wide",
            "pianist full body",
            margin=0.22,
            max_zoom=1.8,
        )
    )

    return candidates
