"""Named speed/accuracy presets.

Bundles a detector + pose model with the runner's temporal knobs (striding, box
smoothing) and CPU quantization into three ready-made points on the speed/accuracy
curve. They are *defaults*, not policy: callers (and the CLI) override any field.

- ``accurate`` — base models, no striding. The quality reference; real-time on the
  deployment GPU, slow on CPU. This matches the historical default behavior.
- ``balanced`` — small detector + small (MoE) pose model, striding 2 with smoothing
  to hide the reuse. The recommended starting point.
- ``fast`` — same small models, striding 3 + int8 quantization. Maximizes throughput;
  the skeleton may jitter or drop joints.

The model ids are real HuggingFace checkpoints (all Apache/MIT, no YOLO). ``balanced``
and ``fast`` use ``vitpose-plus-small``, which is mixture-of-experts — handled
automatically via ``Config.is_moe_pose``.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Config


@dataclass(frozen=True)
class Preset:
    """A named bundle of model + runner settings. See module docstring."""

    name: str
    detector_model: str
    pose_model: str
    detector_stride: int = 1
    pose_stride: int = 1
    inference_width: int = 0
    box_smoothing: float = 0.0
    quantize: bool = False

    def to_config(self, **overrides) -> Config:
        """Build a ``Config`` from this preset, with optional keyword overrides."""
        params = {
            "detector_model": self.detector_model,
            "pose_model": self.pose_model,
            "quantize": self.quantize,
        }
        params.update(overrides)
        return Config(**params)

    @property
    def runner_kwargs(self) -> dict:
        """Keyword arguments for ``VideoPoseRunner`` (the temporal knobs)."""
        return {
            "detector_stride": self.detector_stride,
            "pose_stride": self.pose_stride,
            "inference_width": self.inference_width,
            "box_smoothing": self.box_smoothing,
        }


PRESETS: dict[str, Preset] = {
    "accurate": Preset(
        name="accurate",
        detector_model="PekingU/rtdetr_r50vd_coco_o365",
        pose_model="usyd-community/vitpose-base",
    ),
    "balanced": Preset(
        name="balanced",
        detector_model="PekingU/rtdetr_r18vd_coco_o365",
        pose_model="usyd-community/vitpose-plus-small",
        detector_stride=2,
        pose_stride=2,
        box_smoothing=0.25,
    ),
    "fast": Preset(
        name="fast",
        detector_model="PekingU/rtdetr_r18vd_coco_o365",
        pose_model="usyd-community/vitpose-plus-small",
        detector_stride=3,
        pose_stride=3,
        box_smoothing=0.30,
        quantize=True,
    ),
}
