"""Stage 2: pose estimation with ViTPose, plus the end-to-end pipeline.

ViTPose receives the person boxes from stage 1 and returns COCO-17 keypoints per
person. Results are returned as plain Python/NumPy structures (no framework objects
leak out) so the deployment team can consume them without importing transformers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config, quantize_dynamic_if_cpu
from .detection import PersonDetector


@dataclass
class PersonPose:
    """Pose of a single person.

    Attributes:
        keypoints: Array ``(17, 2)`` of (x, y) pixel coordinates.
        scores: Array ``(17,)`` of per-keypoint confidence.
        box: The COCO ``(x, y, w, h)`` box this pose was estimated from.
    """

    keypoints: np.ndarray
    scores: np.ndarray
    box: np.ndarray


class PoseEstimator:
    """Loads ViTPose once and estimates keypoints for given person boxes."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self._processor = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoProcessor, VitPoseForPoseEstimation

        self._processor = AutoProcessor.from_pretrained(self.config.pose_model)
        self._model = VitPoseForPoseEstimation.from_pretrained(
            self.config.pose_model
        ).to(self.config.device)
        self._model.eval()
        self._model = quantize_dynamic_if_cpu(self._model, self.config)

    def estimate(self, image, person_boxes: np.ndarray) -> list[PersonPose]:
        """Estimate poses for the persons described by ``person_boxes``.

        Args:
            image: A ``PIL.Image.Image`` in RGB.
            person_boxes: ``(N, 4)`` boxes in COCO ``(x, y, w, h)`` format.

        Returns:
            One ``PersonPose`` per input box. Empty list if no boxes.
        """
        import torch

        person_boxes = np.asarray(person_boxes, dtype=float).reshape(-1, 4)
        if person_boxes.shape[0] == 0:
            return []

        self._ensure_loaded()

        proc_kwargs = {"boxes": [person_boxes], "return_tensors": "pt"}
        inputs = self._processor(image, **proc_kwargs).to(self.config.device)

        forward_kwargs = {}
        if self.config.is_moe_pose:
            # vitpose-plus-* expect a dataset index per person (0 == COCO).
            forward_kwargs["dataset_index"] = torch.zeros(
                person_boxes.shape[0], dtype=torch.int64, device=self.config.device
            )

        with torch.no_grad():
            outputs = self._model(**inputs, **forward_kwargs)

        results = self._processor.post_process_pose_estimation(
            outputs, boxes=[person_boxes]
        )[0]

        poses: list[PersonPose] = []
        for person, box in zip(results, person_boxes, strict=False):
            poses.append(
                PersonPose(
                    keypoints=np.asarray(person["keypoints"], dtype=float),
                    scores=np.asarray(person["scores"], dtype=float),
                    box=np.asarray(box, dtype=float),
                )
            )
        return poses


class PosePipeline:
    """Convenience wrapper: detect persons, then estimate their poses."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self.detector = PersonDetector(self.config)
        self.estimator = PoseEstimator(self.config)

    def __call__(self, image) -> list[PersonPose]:
        boxes = self.detector.detect(image)
        return self.estimator.estimate(image, boxes)
