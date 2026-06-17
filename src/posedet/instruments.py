"""Instrument detection with an open-vocabulary detector (OWLv2).

The person stage (``detection.py``) deliberately can't find instruments: its COCO
detector has no instrument classes, and YOLO (whose vocabulary did) is excluded for
licensing. Instead we run a *text-queried* detector — OWLv2 (Apache-2.0) — with the
instrument names from ``Config.instrument_prompts``. Any jazz instrument is then
detectable just by adding a prompt, with no fixed class table.

This stage produces only labeled instrument boxes; tying each instrument to the
musician playing it (and turning that into a role label) is the next stage. As with
the rest of the package, results are plain NumPy/Python — no framework objects leak.

Boxes come out of the detector in VOC ``(x1, y1, x2, y2)`` and are converted to the
COCO ``(x, y, w, h)`` the rest of the pipeline uses, exactly like the person stage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .boxes import dedupe_indices
from .config import Config, quantize_dynamic_if_cpu
from .detection import voc_to_coco


@dataclass
class InstrumentDetection:
    """A single detected instrument.

    Attributes:
        label: The matched text prompt (e.g. ``"saxophone"``).
        score: Detection confidence in ``[0, 1]``.
        box: COCO ``(x, y, w, h)`` box.
    """

    label: str
    score: float
    box: np.ndarray


def build_instrument_detections(
    prompts,
    labels: np.ndarray,
    scores: np.ndarray,
    boxes_voc: np.ndarray,
    *,
    threshold: float = 0.0,
    dedupe_iou: float = 0.5,
    max_instruments: int = 0,
) -> list[InstrumentDetection]:
    """Turn raw detector output into ranked, de-duplicated ``InstrumentDetection``s.

    Pure (NumPy only) so the post-processing is testable without model weights.

    Args:
        prompts: The text queries, indexed by ``labels``.
        labels: ``(N,)`` indices into ``prompts``.
        scores: ``(N,)`` confidences.
        boxes_voc: ``(N, 4)`` boxes in VOC ``(x1, y1, x2, y2)``.
        threshold: Drop detections scoring below this.
        dedupe_iou: NMS IoU for overlapping boxes; ``<= 0`` keeps all (still ranked
            and capped).
        max_instruments: Keep at most this many, by descending score; ``0`` = no cap.

    Returns:
        ``InstrumentDetection``s in descending score order, boxes in COCO format.
    """
    labels = np.asarray(labels).reshape(-1).astype(int)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    boxes_voc = np.asarray(boxes_voc, dtype=float).reshape(-1, 4)
    if labels.shape[0] == 0:
        return []

    keep = scores >= threshold
    labels, scores, boxes_voc = labels[keep], scores[keep], boxes_voc[keep]
    if labels.shape[0] == 0:
        return []

    boxes_coco = voc_to_coco(boxes_voc)
    cap = (
        max_instruments if max_instruments and max_instruments > 0 else len(boxes_coco)
    )
    if dedupe_iou and dedupe_iou > 0:
        # dedupe_indices ranks by score, NMS-filters, and caps in one pass.
        kept = dedupe_indices(boxes_coco, scores, cap, dedupe_iou)
    else:
        # Dedupe disabled: keep all, still ranked by descending score and capped.
        kept = [int(index) for index in np.argsort(scores)[::-1][:cap]]

    prompts = list(prompts)
    detections: list[InstrumentDetection] = []
    for index in kept:
        label_index = int(labels[index])
        name = (
            prompts[label_index]
            if 0 <= label_index < len(prompts)
            else str(label_index)
        )
        detections.append(
            InstrumentDetection(
                label=name,
                score=float(scores[index]),
                box=boxes_coco[index].astype(float),
            )
        )
    return detections


class InstrumentDetector:
    """Loads OWLv2 once and detects instruments by text prompt."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self._processor = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self._processor = Owlv2Processor.from_pretrained(self.config.instrument_model)
        self._model = Owlv2ForObjectDetection.from_pretrained(
            self.config.instrument_model
        ).to(self.config.device)
        self._model.eval()
        self._model = quantize_dynamic_if_cpu(self._model, self.config)

    def detect(self, image) -> list[InstrumentDetection]:
        """Detect instruments in a PIL image.

        Args:
            image: A ``PIL.Image.Image`` in RGB.

        Returns:
            ``InstrumentDetection``s (COCO boxes), filtered by
            ``Config.instrument_threshold``, de-duplicated, and capped at
            ``Config.max_instruments``. Empty when no prompts are configured.
        """
        import torch

        prompts = list(self.config.instrument_prompts)
        if not prompts:
            return []

        self._ensure_loaded()
        inputs = self._processor(text=[prompts], images=image, return_tensors="pt").to(
            self.config.device
        )
        with torch.no_grad():
            outputs = self._model(**inputs)

        # OWLv2 post-processing expects target sizes as (height, width). The
        # Owlv2Processor only exposes the *grounded* variant; it still returns
        # "labels" as indices into the prompt list, which is what we map below.
        target_sizes = torch.tensor([(image.height, image.width)])
        results = self._processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=self.config.instrument_threshold,
        )[0]

        return build_instrument_detections(
            prompts,
            results["labels"].cpu().numpy(),
            results["scores"].cpu().numpy(),
            results["boxes"].cpu().numpy(),
            threshold=self.config.instrument_threshold,
            dedupe_iou=self.config.instrument_dedupe_iou,
            max_instruments=self.config.max_instruments,
        )
