"""Stateful video runner: the per-frame loop that turns a frame stream into poses.

This is the one component that deliberately holds state. It owns the *temporal*
concerns that must not leak into the pure, single-image estimator:

- **Striding** — run the detector / pose model only every N frames and reuse the
  previous result in between (a big deal on CPU, where each model call costs seconds).
- **Box smoothing** — blend each person box with its match from the prior frame to
  damp skeleton jitter (delegated to ``boxes.smooth_boxes``).
- **Inference downscale** — optionally shrink frames for the models, then scale the
  resulting keypoints/boxes back to the original resolution.

Model inference itself stays in ``PersonDetector`` / ``PoseEstimator``; drawing and
file I/O stay at the edges (the caller). The detector and estimator are injectable so
the loop logic is testable without model weights and swappable by the deployment team.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np

from .boxes import smooth_boxes
from .config import Config
from .detection import PersonDetector
from .pose import PersonPose, PoseEstimator


def resize_for_inference(
    frame_bgr: np.ndarray, inference_width: int
) -> tuple[np.ndarray, float, float]:
    """Downscale a BGR frame to ``inference_width`` for the models.

    Returns ``(resized_bgr, scale_x, scale_y)`` where the scales map a coordinate in
    the resized frame *back* to the original frame (multiply by the scale). A width
    of ``0``, or one that would upscale, is a no-op: the frame and unit scales are
    returned and OpenCV is never touched.
    """
    height, width = frame_bgr.shape[:2]
    if inference_width <= 0 or inference_width >= width:
        return frame_bgr, 1.0, 1.0

    import cv2

    scale = inference_width / width
    inference_height = int(round(height * scale))
    resized = cv2.resize(
        frame_bgr, (inference_width, inference_height), interpolation=cv2.INTER_AREA
    )
    return resized, width / inference_width, height / inference_height


def scale_pose(pose: PersonPose, scale_x: float, scale_y: float) -> PersonPose:
    """Return a copy of ``pose`` with keypoints and box scaled by the given factors.

    Used to lift poses estimated on a downscaled inference frame back to the
    original video resolution. Unit scales return the input unchanged.
    """
    if scale_x == 1.0 and scale_y == 1.0:
        return pose
    keypoints = pose.keypoints.copy()
    keypoints[:, 0] *= scale_x
    keypoints[:, 1] *= scale_y
    box = pose.box.copy()
    box[[0, 2]] *= scale_x
    box[[1, 3]] *= scale_y
    return PersonPose(keypoints=keypoints, scores=pose.scores, box=box)


class VideoPoseRunner:
    """Drive detection + pose estimation over a stream of frames, with temporal state.

    Args:
        config: Model / selection configuration. A default ``Config`` is used if
            omitted.
        detector_stride: Run person detection every Nth processed frame, reusing the
            previous boxes in between. ``1`` detects on every frame.
        pose_stride: Run pose estimation every Nth processed frame, reusing the
            previous skeletons in between. Detection only runs on pose-update frames.
        box_smoothing: Temporal box blend in ``[0, 0.95]``; ``0`` disables smoothing.
        box_smoothing_min_iou: Minimum IoU for a box to be matched across frames.
        inference_width: Downscale frames to this width for inference; ``0`` keeps
            full resolution. Keypoints are scaled back to the original size.
        detector: Optional detector with ``.detect(image) -> (N,4) COCO boxes``.
            Defaults to ``PersonDetector(config)``. Injectable for testing/swapping.
        estimator: Optional estimator with ``.estimate(image, boxes) -> [PersonPose]``.
            Defaults to ``PoseEstimator(config)``.
    """

    def __init__(
        self,
        config: Config | None = None,
        *,
        detector_stride: int = 1,
        pose_stride: int = 1,
        box_smoothing: float = 0.0,
        box_smoothing_min_iou: float = 0.15,
        inference_width: int = 0,
        detector=None,
        estimator=None,
    ) -> None:
        if detector_stride < 1:
            raise ValueError("detector_stride must be >= 1")
        if pose_stride < 1:
            raise ValueError("pose_stride must be >= 1")

        self.config = config or Config()
        self.detector = (
            detector if detector is not None else PersonDetector(self.config)
        )
        self.estimator = (
            estimator if estimator is not None else PoseEstimator(self.config)
        )
        self.detector_stride = detector_stride
        self.pose_stride = pose_stride
        self.box_smoothing = box_smoothing
        self.box_smoothing_min_iou = box_smoothing_min_iou
        self.inference_width = inference_width
        self.reset()

    def reset(self) -> None:
        """Clear temporal state so the next frame starts a fresh sequence."""
        self._frame_index = 0
        self._last_boxes = np.empty((0, 4), dtype=np.float32)
        self._last_poses: list[PersonPose] = []

    def process(self, frame_bgr: np.ndarray) -> list[PersonPose]:
        """Process one BGR frame and return poses in *original* frame coordinates.

        Advances the internal frame counter, so frames must be fed in order. On
        strided (skipped) frames the previous poses are returned unchanged.
        """
        from PIL import Image

        inference_bgr, scale_x, scale_y = resize_for_inference(
            frame_bgr, self.inference_width
        )
        index = self._frame_index
        should_update_pose = index % self.pose_stride == 0

        if should_update_pose:
            image = Image.fromarray(inference_bgr[:, :, ::-1])  # BGR -> RGB
            if index % self.detector_stride == 0:
                detected = self.detector.detect(image)
                self._last_boxes = smooth_boxes(
                    self._last_boxes,
                    detected,
                    self.box_smoothing,
                    self.box_smoothing_min_iou,
                )
            poses = self.estimator.estimate(image, self._last_boxes)
            self._last_poses = [scale_pose(p, scale_x, scale_y) for p in poses]

        self._frame_index += 1
        return self._last_poses

    def run(
        self, frames: Iterable[tuple[int, np.ndarray]]
    ) -> Iterator[tuple[int, np.ndarray, list[PersonPose]]]:
        """Yield ``(index, frame_bgr, poses)`` for each ``(index, frame_bgr)`` in.

        Pairs with ``iter_video_frames``; the caller draws and writes. Expects
        consecutive frames (don't also stride the source, or strides compound).
        """
        for index, frame_bgr in frames:
            yield index, frame_bgr, self.process(frame_bgr)
