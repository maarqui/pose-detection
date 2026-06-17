"""Video / camera frame iteration. Isolated from model logic so that moving from
file-based testing to a live stream later only touches this module.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def iter_video_frames(
    source: str | int,
    stride: int = 1,
    max_frames: int | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(frame_index, frame_bgr)`` from a video file or camera.

    Args:
        source: Path to a video file, or an integer camera index.
        stride: Yield every ``stride``-th frame (downsampling for slow CPU runs).
        max_frames: Stop after yielding this many frames. ``None`` means no limit.

    Yields:
        Tuples of original frame index and the BGR frame array.
    """
    import cv2

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video source: {source!r}")

    yielded = 0
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % stride == 0:
                yield index, frame
                yielded += 1
                if max_frames is not None and yielded >= max_frames:
                    break
            index += 1
    finally:
        capture.release()
