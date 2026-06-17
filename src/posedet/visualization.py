"""Drawing utilities. Kept separate from model logic so the inference core stays
pure and the deployment team can swap in their own renderer.

Operates on NumPy image arrays (H, W, 3). Uses OpenCV only.
"""

from __future__ import annotations

import numpy as np

# COCO-17 keypoint names, in index order.
COCO_KEYPOINTS = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

# Per-keypoint BGR colors (OpenCV order), indexed by COCO-17 keypoint. A warm-to-cool
# ramp head->feet so left/right limbs read at a glance. Opt-in via ``draw_pose``.
KEYPOINT_COLORS = (
    (0, 255, 255),
    (0, 220, 255),
    (0, 220, 255),
    (0, 180, 255),
    (0, 180, 255),
    (0, 255, 120),
    (0, 255, 120),
    (80, 220, 0),
    (80, 220, 0),
    (160, 180, 0),
    (160, 180, 0),
    (255, 180, 0),
    (255, 180, 0),
    (255, 90, 0),
    (255, 90, 0),
    (255, 0, 80),
    (255, 0, 80),
)

# Skeleton as pairs of keypoint indices to connect with a line.
COCO_SKELETON = (
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
)


def draw_pose(
    image: np.ndarray,
    poses,
    kpt_threshold: float = 0.3,
    draw_boxes: bool = True,
    *,
    skeleton_color: tuple[int, int, int] = (0, 165, 255),
    point_color: tuple[int, int, int] = (0, 0, 255),
    point_colors=None,
    line_thickness: int = 2,
    point_radius: int = 3,
) -> np.ndarray:
    """Draw skeletons (and optionally boxes) for each person onto a copy of ``image``.

    Edges always come from ``COCO_SKELETON`` (the single skeleton definition); the
    colors and sizes are styling knobs.

    Args:
        image: ``(H, W, 3)`` BGR array (OpenCV convention).
        poses: Iterable of objects with ``.keypoints (17,2)``, ``.scores (17,)``,
            and ``.box (4,)`` in COCO ``(x, y, w, h)`` — e.g. ``PersonPose``.
        kpt_threshold: Keypoints below this score are not drawn.
        draw_boxes: Whether to draw the person bounding box.
        skeleton_color: BGR color for skeleton lines.
        point_color: BGR color for keypoints when ``point_colors`` is ``None``.
        point_colors: Optional per-keypoint BGR palette (e.g. ``KEYPOINT_COLORS``),
            indexed by keypoint and wrapped modulo its length. Overrides
            ``point_color`` when given.
        line_thickness: Skeleton line thickness in pixels.
        point_radius: Keypoint circle radius in pixels.

    Returns:
        A new annotated image array (the input is not mutated).
    """
    import cv2

    canvas = image.copy()
    for person in poses:
        kpts = np.asarray(person.keypoints, dtype=float)
        scores = np.asarray(person.scores, dtype=float)

        if draw_boxes:
            x, y, w, h = (int(round(v)) for v in person.box)
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)

        for a, b in COCO_SKELETON:
            if scores[a] < kpt_threshold or scores[b] < kpt_threshold:
                continue
            pa = tuple(int(round(v)) for v in kpts[a])
            pb = tuple(int(round(v)) for v in kpts[b])
            cv2.line(canvas, pa, pb, skeleton_color, line_thickness, cv2.LINE_AA)

        for i, (x, y) in enumerate(kpts):
            if scores[i] < kpt_threshold:
                continue
            color = point_colors[i % len(point_colors)] if point_colors else point_color
            cv2.circle(
                canvas,
                (int(round(x)), int(round(y))),
                point_radius,
                color,
                -1,
                cv2.LINE_AA,
            )

    return canvas
