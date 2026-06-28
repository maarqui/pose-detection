from .utils import (
    ShotCandidate,
    candidate,
    chest_up_box,
    full_body_box,
    hands_box,
    head_only_box,
    pianist_hands_focused_box,
    upper_body_box,
)


def pianist_shots(
    musician,
    musician_index: int,
    musicians: list,
    salience: float,
    solo_bonus: float,
    kpt_threshold: float,
) -> list[ShotCandidate]:
    candidates = []

    # Close up: hands (broad)
    candidates.append(
        candidate(
            hands_box(musician, kpt_threshold),
            salience + 0.24 + solo_bonus,
            musician_index,
            "close_up",
            "pianist hands",
            margin=0.18,
            max_zoom=4.0,
        )
    )

    # Close up: hands focused (tight on keys)
    candidates.append(
        candidate(
            pianist_hands_focused_box(musician, kpt_threshold),
            salience + 0.26 + solo_bonus,
            musician_index,
            "close_up",
            "pianist hands focused",
            margin=0.12,
            max_zoom=4.5,
        )
    )

    # Medium close-up: chest up
    candidates.append(
        candidate(
            chest_up_box(musician),
            salience + 0.22 + solo_bonus,
            musician_index,
            "medium_close",
            "pianist chest up",
            margin=0.15,
            max_zoom=3.2,
        )
    )

    # Medium shot: from waist up
    candidates.append(
        candidate(
            upper_body_box(musician),
            salience + 0.20 + solo_bonus,
            musician_index,
            "medium",
            "pianist waist up",
            margin=0.16,
            max_zoom=2.8,
        )
    )

    # Close up: head only
    candidates.append(
        candidate(
            head_only_box(musician, kpt_threshold),
            salience + 0.15 + solo_bonus,
            musician_index,
            "close_up",
            "pianist headshot",
            margin=0.25,
            max_zoom=3.5,
        )
    )

    # Extreme wide shot: full body
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
