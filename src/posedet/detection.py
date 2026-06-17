"""Stage 1: person detection with a transformers object detector.

ViTPose is top-down and carries no detector, so we must produce person boxes first.
Any ``transformers`` detector with the standard object-detection head works here
(RT-DETR is the default; D-FINE is selected by passing its model id) — they share
the ``AutoModelForObjectDetection`` call shape, so the model id is the only knob.

These detectors return boxes in VOC format ``(x1, y1, x2, y2)``; ViTPose expects
COCO format ``(x, y, w, h)``. The conversion is the single most common source of
silent bugs in this pipeline, so it is isolated and unit-tested.
"""

from __future__ import annotations

import numpy as np

from .config import Config, quantize_dynamic_if_cpu
from .selection import select_performers


def voc_to_coco(boxes: np.ndarray) -> np.ndarray:
    """Convert VOC ``(x1, y1, x2, y2)`` boxes to COCO ``(x, y, w, h)``.

    Args:
        boxes: Array of shape ``(N, 4)`` in VOC format.

    Returns:
        Array of shape ``(N, 4)`` in COCO format. Empty input returns ``(0, 4)``.
    """
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    coco = boxes.copy()
    coco[:, 2] = boxes[:, 2] - boxes[:, 0]
    coco[:, 3] = boxes[:, 3] - boxes[:, 1]
    return coco


class PersonDetector:
    """Loads a transformers object detector once and returns person boxes (COCO)."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self._processor = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForObjectDetection, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self.config.detector_model)
        self._model = AutoModelForObjectDetection.from_pretrained(
            self.config.detector_model
        ).to(self.config.device)
        self._model.eval()
        self._model = quantize_dynamic_if_cpu(self._model, self.config)

    def _person_mask(self, labels: np.ndarray) -> np.ndarray:
        """Select person detections by class name, falling back to ``person_label``.

        Matching ``id2label`` against ``person_label_name`` is robust across detector
        vocabularies where the person index differs; only when the model exposes no
        such mapping do we fall back to the numeric index.
        """
        id2label = getattr(self._model.config, "id2label", None)
        if id2label:
            wanted = self.config.person_label_name.lower()
            return np.array(
                [
                    str(id2label.get(int(label), "")).lower() == wanted
                    for label in labels
                ]
            )
        return labels == self.config.person_label

    def detect(self, image) -> np.ndarray:
        """Detect persons in a PIL image.

        Args:
            image: A ``PIL.Image.Image`` in RGB.

        Returns:
            Array of shape ``(N, 4)`` person boxes in COCO ``(x, y, w, h)`` format,
            refined by the performer-selection policy in ``Config`` (stage ROI,
            audience suppression, dedupe) and capped at ``Config.max_people``.
        """
        import torch

        self._ensure_loaded()
        inputs = self._processor(images=image, return_tensors="pt").to(
            self.config.device
        )
        with torch.no_grad():
            outputs = self._model(**inputs)

        target_sizes = torch.tensor([(image.height, image.width)])
        results = self._processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.config.det_threshold
        )[0]

        labels = results["labels"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        boxes_voc = results["boxes"].cpu().numpy()

        keep = self._person_mask(labels)
        person_boxes = voc_to_coco(boxes_voc[keep])
        person_scores = scores[keep]

        boxes, _ = select_performers(
            person_boxes,
            person_scores,
            image.width,
            image.height,
            stage_roi=self.config.stage_roi,
            audience_suppression=self.config.audience_suppression,
            audience_band=self.config.audience_band,
            dedupe_iou=self.config.dedupe_iou,
            max_people=self.config.max_people,
        )
        return boxes
