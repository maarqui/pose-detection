"""Performer selection: refine raw person detections toward on-stage musicians.

This is the **license-clean, YOLO-free** replacement for the musician-finding role
that an instrument-aware detector used to play. Given person boxes + scores from a
``transformers`` detector, it favors performers over audience using *geometry only*:

1. an optional stage region of interest (drop people outside the performance area),
2. a foreground-audience penalty (demote people standing low in the frame), and
3. greedy dedupe + a hard cap on the number of people.

There are no instrument classes and no ``ultralytics`` dependency, so everything
here stays inside the Apache/MIT-licensed set the deployment team can actually ship.

All functions are pure (NumPy in, NumPy out). Boxes are COCO ``(x, y, w, h)``; the
ROI is VOC ``(x1, y1, x2, y2)`` expressed as *fractions of the image* in ``[0, 1]``
so it is resolution-independent. The detector and the video runner compose this.
"""

from __future__ import annotations

import numpy as np

from .boxes import dedupe_indices


def roi_fractions_to_pixels(
    roi: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """Convert a normalized ``(x1, y1, x2, y2)`` ROI in ``[0, 1]`` to pixel coords."""
    x1, y1, x2, y2 = roi
    return np.array(
        [x1 * image_width, y1 * image_height, x2 * image_width, y2 * image_height],
        dtype=np.float32,
    )


def select_performers(
    boxes_coco: np.ndarray,
    scores: np.ndarray,
    image_width: int,
    image_height: int,
    *,
    stage_roi: tuple[float, float, float, float] | None = None,
    audience_suppression: float = 0.0,
    audience_band: float = 0.82,
    dedupe_iou: float = 0.0,
    max_people: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Rank and trim person detections toward likely on-stage performers.

    With all knobs at their defaults this is just "sort by score, keep everyone",
    so a caller that configures nothing gets the raw detector ranking unchanged.

    Args:
        boxes_coco: ``(N, 4)`` person boxes in COCO ``(x, y, w, h)``.
        scores: ``(N,)`` detector confidence per box.
        image_width: Width of the image the boxes are in, in pixels.
        image_height: Height of the image the boxes are in, in pixels.
        stage_roi: Optional ``(x1, y1, x2, y2)`` in ``[0, 1]`` fractions; boxes whose
            center falls outside are dropped. ``None`` keeps the whole frame.
        audience_suppression: Score penalty subtracted from people whose feet sit
            below ``audience_band`` (the typical near-crowd position). ``0`` disables
            it. This only changes *ranking*, never removes a box outright, so an
            unusually low performer can still survive on raw confidence.
        audience_band: Fraction of image height below which a box's bottom edge marks
            it as a foreground-audience candidate.
        dedupe_iou: If > 0, greedily drop boxes overlapping a higher-ranked one at or
            above this IoU. ``0`` skips dedupe (transformers detectors already NMS).
        max_people: Keep at most this many people (highest ranked first). ``0`` keeps
            all.

    Returns:
        ``(boxes, scores)`` in the selected order, carrying the *original* detector
        scores (not the audience-adjusted ranking scores).
    """
    boxes_coco = np.asarray(boxes_coco, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    empty = (
        np.empty((0, 4), dtype=np.float32),
        np.empty((0,), dtype=np.float32),
    )
    if boxes_coco.shape[0] == 0:
        return empty

    # 1. Stage ROI: keep only boxes whose center lies in the performance area.
    if stage_roi is not None:
        x1, y1, x2, y2 = roi_fractions_to_pixels(stage_roi, image_width, image_height)
        centers_x = boxes_coco[:, 0] + boxes_coco[:, 2] * 0.5
        centers_y = boxes_coco[:, 1] + boxes_coco[:, 3] * 0.5
        inside = (
            (centers_x >= x1)
            & (centers_x <= x2)
            & (centers_y >= y1)
            & (centers_y <= y2)
        )
        boxes_coco = boxes_coco[inside]
        scores = scores[inside]
        if boxes_coco.shape[0] == 0:
            return empty

    # 2. Foreground-audience penalty: demote people standing low in the frame.
    ranking_scores = scores.copy()
    if audience_suppression > 0.0:
        y_bottom = boxes_coco[:, 1] + boxes_coco[:, 3]
        is_foreground = (y_bottom > audience_band * image_height).astype(np.float32)
        ranking_scores = ranking_scores - is_foreground * audience_suppression

    # 3. Order by adjusted score, optionally deduping, then cap the count.
    if dedupe_iou > 0.0:
        cap = max_people if max_people > 0 else len(ranking_scores)
        order = dedupe_indices(boxes_coco, ranking_scores, cap, dedupe_iou)
    else:
        order = np.argsort(ranking_scores)[::-1]
        if max_people > 0:
            order = order[:max_people]

    return boxes_coco[order], scores[order]
