"""Shot scoring and aspect-preserving zoom framing.

Given the musicians in a frame, decide which rectangle a camera *should* show. The
choice is two steps, both pure geometry:

1. **Score & select** — rank musicians by salience (a soloing, standing performer
   holding an instrument and filling the frame scores higher than a seated figure in
   the back) and pick the subject(s) to frame. A clear soloist is framed alone; an
   even ensemble is framed together.
2. **Fit aspect** — grow the subjects' bounding region to the *frame's own aspect
   ratio* (never square), clamp it inside the frame, and bound how far it may zoom in.
   The result is a sub-rectangle with the same shape as the source, so scaling it up
   reads as a clean optical-style zoom.

``apply_zoom`` (the only OpenCV-touching function) crops and rescales a frame to that
rectangle; everything else is framework-free and unit-tested on synthetic boxes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .shot_profiles import ShotCandidate, role_shot_candidates

# Salience weights (sum to 1). Arms-raised is weighted high so a soloing gesture
# pulls the shot toward the soloist; size favours performers closer to camera.
_W_POSTURE = 0.2
_W_ARMS = 0.3
_W_INSTRUMENT = 0.25
_W_SIZE = 0.15
_W_CONFIDENCE = 0.1

# Person-box area (as a fraction of the frame) at which the size signal saturates.
_SIZE_SATURATION = 0.15

_POSTURE_SCORE = {"standing": 1.0, "sitting": 0.6, "unknown": 0.4}
_ARMS_SCORE = {"raised": 1.0, "down": 0.4, "unknown": 0.3}


@dataclass
class Shot:
    """A chosen camera shot.

    Attributes:
        box: COCO ``(x, y, w, h)`` crop rectangle, with the *frame's* aspect ratio.
        score: Salience of the framed subject(s); ``0`` for a default full-frame shot.
        musician_indices: Indices (into the input list) of the framed musicians.
        shot_type: Camera grammar label such as ``medium_close`` or ``wide``.
        description: Human-readable reason for the shot choice.
    """

    box: np.ndarray
    score: float
    musician_indices: tuple[int, ...] = field(default=())
    shot_type: str = "wide"
    description: str = "full frame"


def score_musician(musician, frame_width: int, frame_height: int) -> float:
    """Salience of a musician for shot selection, in ``[0, 1]`` (pure)."""
    posture = _POSTURE_SCORE.get(musician.posture.posture, 0.4)
    arms = _ARMS_SCORE.get(musician.posture.arms, 0.3)
    instrument = 1.0 if musician.instrument is not None else 0.3

    box = np.asarray(musician.pose.box, dtype=float)
    frame_area = max(float(frame_width) * float(frame_height), 1.0)
    area_fraction = max(0.0, box[2]) * max(0.0, box[3]) / frame_area
    size = min(area_fraction / _SIZE_SATURATION, 1.0)

    confidence = float(musician.posture.confidence)

    return (
        _W_POSTURE * posture
        + _W_ARMS * arms
        + _W_INSTRUMENT * instrument
        + _W_SIZE * size
        + _W_CONFIDENCE * confidence
    )


def _union_box(boxes: list[np.ndarray]) -> np.ndarray:
    """Smallest COCO box covering all given COCO boxes."""
    arr = np.asarray(boxes, dtype=float).reshape(-1, 4)
    x1 = arr[:, 0].min()
    y1 = arr[:, 1].min()
    x2 = (arr[:, 0] + arr[:, 2]).max()
    y2 = (arr[:, 1] + arr[:, 3]).max()
    return np.array([x1, y1, x2 - x1, y2 - y1], dtype=float)


def select_target(
    musicians,
    frame_width: int,
    frame_height: int,
    *,
    group_ratio: float = 0.8,
) -> tuple[np.ndarray | None, tuple[int, ...], float]:
    """Pick the subject region to frame: the top musician plus any near-equal peers.

    Returns ``(target_box | None, framed_indices, top_score)``. A musician joins the
    group when their salience is at least ``group_ratio`` of the top score, so a
    dominant soloist is framed alone while an even ensemble is framed together.
    """
    musicians = list(musicians)
    if not musicians:
        return None, (), 0.0

    scores = [score_musician(m, frame_width, frame_height) for m in musicians]
    top_score = max(scores)
    cutoff = group_ratio * top_score
    indices = tuple(i for i, s in enumerate(scores) if s >= cutoff)
    target = _union_box([musicians[i].pose.box for i in indices])
    return target, indices, top_score


def fit_aspect(
    target_box,
    frame_width: int,
    frame_height: int,
    *,
    margin: float = 0.15,
    max_zoom: float = 2.5,
) -> np.ndarray:
    """Grow a target box to the frame's aspect ratio, clamped and zoom-bounded (pure).

    Args:
        target_box: COCO ``(x, y, w, h)`` region to keep in shot.
        frame_width, frame_height: Source frame size.
        margin: Padding added around the target, as a fraction of its size.
        max_zoom: Hard limit on zoom-in; the crop is never smaller than
            ``frame / max_zoom`` in either dimension.

    Returns:
        A COCO crop rectangle with exactly the frame's aspect ratio, fully inside the
        frame. Returns the full frame when the target needs (nearly) all of it.
    """
    frame_w = float(frame_width)
    frame_h = float(frame_height)
    aspect = frame_w / frame_h

    x, y, w, h = (float(v) for v in target_box)
    center_x = x + w * 0.5
    center_y = y + h * 0.5

    w *= 1.0 + margin
    h *= 1.0 + margin

    # Don't zoom in past the max-zoom floor.
    w = max(w, frame_w / max_zoom)
    h = max(h, frame_h / max_zoom)

    # Expand the deficient dimension so the crop matches the frame's aspect ratio.
    if w / h < aspect:
        w = h * aspect
    else:
        h = w / aspect

    # The crop can never exceed the frame; aspect already matches, so cap to full.
    if w >= frame_w or h >= frame_h:
        return np.array([0.0, 0.0, frame_w, frame_h], dtype=float)

    # Center on the subject, then slide back inside the frame.
    crop_x = min(max(center_x - w * 0.5, 0.0), frame_w - w)
    crop_y = min(max(center_y - h * 0.5, 0.0), frame_h - h)
    return np.array([crop_x, crop_y, w, h], dtype=float)


def choose_shot(
    musicians,
    frame_width: int,
    frame_height: int,
    *,
    margin: float = 0.15,
    max_zoom: float = 2.5,
    group_ratio: float = 0.8,
    kpt_threshold: float = 0.3,
    shot_history: list[str] | None = None,
) -> Shot:
    """Select subjects and return the aspect-correct ``Shot`` to frame them.

    Role-specific candidates are generated first (hands, mouthpiece, cymbals,
    upper-body, full-body, etc.) and then fitted to the frame aspect ratio.
    Ensemble-wide shots are also added as candidates. The best candidate is
    selected, potentially influenced by shot history to encourage variety.
    """
    musicians = list(musicians)
    target, indices, top_score = select_target(
        musicians, frame_width, frame_height, group_ratio=group_ratio
    )
    if target is None:
        full = np.array([0.0, 0.0, float(frame_width), float(frame_height)])
        return Shot(box=full, score=0.0, musician_indices=())

    scores = [score_musician(m, frame_width, frame_height) for m in musicians]
    candidates: list[ShotCandidate] = []

    # 1. Add role-specific candidates
    for index, musician in enumerate(musicians):
        candidates.extend(
            role_shot_candidates(
                musician,
                index,
                musicians,
                scores[index],
                kpt_threshold=kpt_threshold,
            )
        )

    # 2. Add ensemble candidates if multiple performers
    if len(indices) > 1:
        candidates.append(
            ShotCandidate(
                target_box=target,
                score=top_score + 0.05, # Slight bias towards ensemble if salient
                musician_indices=indices,
                shot_type="wide",
                description="near-equal ensemble",
                margin=margin,
                max_zoom=max_zoom
            )
        )

    if not candidates:
        crop = fit_aspect(
            target, frame_width, frame_height, margin=margin, max_zoom=max_zoom
        )
        return Shot(box=crop, score=top_score, musician_indices=indices)

    # 3. Variety: penalize candidates that match recent shot types too closely
    def variety_score(c: ShotCandidate) -> float:
        penalty = 0.0
        if shot_history:
            # Penalize the exact same shot description heavily
            # Looking at a longer history (last 10 entries)
            last_n = shot_history[-10:]

            # Count exact matches for description
            desc_matches = last_n.count(c.description)
            penalty += 0.3 * desc_matches  # Increased penalty from 0.15

            # Count matches for shot type
            type_matches = last_n.count(c.shot_type)
            penalty += 0.1 * type_matches  # Increased penalty from 0.05

            # Heavily penalize if the last shot is identical to this one
            if shot_history[-1] == c.description:
                penalty += 0.5

        return float(c.score - penalty)

    best = max(candidates, key=variety_score)
    crop = fit_aspect(
        best.target_box,
        frame_width,
        frame_height,
        margin=best.margin,
        max_zoom=best.max_zoom,
    )
    return Shot(
        box=crop,
        score=best.score,
        musician_indices=best.musician_indices,
        shot_type=best.shot_type,
        description=best.description,
    )


def apply_zoom(frame_bgr: np.ndarray, crop_box) -> np.ndarray:
    """Crop ``frame_bgr`` to ``crop_box`` and rescale to the original size (OpenCV).

    The crop keeps the frame aspect ratio (see :func:`fit_aspect`), so the output has
    the same resolution as the input and reads as a zoom. The crop is clamped to the
    frame bounds defensively. This is the one I/O-edge function here.
    """
    import cv2

    height, width = frame_bgr.shape[:2]
    x, y, w, h = (int(round(v)) for v in crop_box)
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    crop = frame_bgr[y : y + h, x : x + w]
    return cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)
