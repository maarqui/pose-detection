from .utils import (
    hands_box,
    upper_body_box,
    chest_up_box,
    head_only_box,
    relative_box,
    instrument_union,
    union_box,
    candidate,
    ShotCandidate
)

def guitarist_shots(
    musician,
    musician_index: int,
    musicians: list,
    salience: float,
    solo_bonus: float,
    kpt_threshold: float,
) -> list[ShotCandidate]:
    candidates = []

    # Close-up: picking hand or fretboard
    candidates.append(
        candidate(
            hands_box(musician, kpt_threshold),
            salience + 0.25 + solo_bonus,
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
            instrument_union(musician, [chest_up_box(musician)]),
            salience + 0.22 + solo_bonus,
            musician_index,
            "medium_close",
            "guitarist torso and guitar",
            margin=0.15,
            max_zoom=3.0,
        )
    )

    # Medium shot: waist up with instrument
    candidates.append(
        candidate(
            instrument_union(musician, [upper_body_box(musician)]),
            salience + 0.20 + solo_bonus,
            musician_index,
            "medium",
            "guitarist waist up with instrument",
            margin=0.18,
            max_zoom=2.5,
        )
    )

    # Close-up: head shot
    candidates.append(
        candidate(
            head_only_box(musician, kpt_threshold),
            salience + 0.14 + solo_bonus,
            musician_index,
            "close_up",
            "guitarist headshot",
            margin=0.25,
            max_zoom=3.5
        )
    )

    # Wide shot: guitarist with nearby soloist
    for i, other in enumerate(musicians):
        if i == musician_index:
            continue

        # Check if "other" is a soloist (arms raised)
        from ..poseclass import ARMS_RAISED
        if other.posture.arms == ARMS_RAISED:
            g_center = musician.pose.box[0] + musician.pose.box[2] * 0.5
            o_center = other.pose.box[0] + other.pose.box[2] * 0.5
            dist = abs(g_center - o_center)

            if dist < musician.pose.box[2] * 1.8:
                combined_box = union_box([musician.pose.box, other.pose.box])
                candidates.append(
                    ShotCandidate(
                        target_box=combined_box,
                        score=salience + 0.15 + 0.10,
                        musician_indices=(musician_index, i),
                        shot_type="wide",
                        description=f"guitarist with nearby soloist ({other.role})",
                        margin=0.20,
                        max_zoom=2.0
                    )
                )

    return candidates
