"""Runtime configuration for the pose-detection pipeline.

Everything tunable lives here so that callers (including the deployment team that
integrates this subproject) configure behaviour by passing a ``Config`` instance
instead of editing module internals. No global state, no hardcoded paths.
"""

from __future__ import annotations

from dataclasses import dataclass


def _default_device() -> str:
    """Pick CUDA when present, otherwise CPU. Dev hardware here is CPU-only."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


@dataclass(frozen=True)
class Config:
    """Immutable pipeline configuration.

    Attributes:
        detector_model: HuggingFace id of the person detector (stage 1).
        pose_model: HuggingFace id of the ViTPose model (stage 2).
        device: "cuda" or "cpu". Auto-detected by default.
        det_threshold: Min confidence to keep a person detection.
        kpt_threshold: Min confidence to draw/keep a keypoint.
        person_label: COCO class index for "person" (0 for RT-DETR COCO). Used as a
            fallback when the model exposes no ``id2label`` name matching
            ``person_label_name``.
        person_label_name: Class name treated as "person". Preferred over
            ``person_label`` because the index differs across detector vocabularies
            (RT-DETR COCO vs. D-FINE COCO vs. Objects365).
        max_people: Cap on persons returned per image, kept by descending score.
            ``0`` means no cap. ViTPose cost scales with this, so it matters on CPU.
        stage_roi: Optional ``(x1, y1, x2, y2)`` performance area as fractions of the
            image in ``[0, 1]``; person boxes centered outside it are dropped. ``None``
            uses the whole frame (default — the concert camera moves/zooms).
        audience_suppression: Score penalty applied to people standing low in the
            frame (likely foreground audience) when ranking/capping. ``0`` disables.
        audience_band: Fraction of image height below which a box's feet mark it as a
            foreground-audience candidate. Only used when ``audience_suppression`` > 0.
        dedupe_iou: If > 0, drop person boxes overlapping a higher-ranked one at this
            IoU. ``0`` (default) trusts the detector's own NMS.
        quantize: Apply int8 dynamic quantization to the models' linear layers. CPU
            only (a no-op on CUDA); roughly 1.5-2x faster for some accuracy loss.

    The selection knobs are all off by default, so the out-of-the-box detector
    behavior is unchanged; they are the YOLO-free way to bias toward on-stage
    performers and need per-venue tuning. See ``posedet.presets`` for ready-made
    speed/accuracy configurations.
    """

    detector_model: str = "PekingU/rtdetr_r50vd_coco_o365"
    pose_model: str = "usyd-community/vitpose-base"
    device: str = ""  # empty -> auto-detect in __post_init__
    det_threshold: float = 0.3
    kpt_threshold: float = 0.3
    person_label: int = 0
    person_label_name: str = "person"
    max_people: int = 0
    stage_roi: tuple[float, float, float, float] | None = None
    audience_suppression: float = 0.0
    audience_band: float = 0.82
    dedupe_iou: float = 0.0
    quantize: bool = False

    def __post_init__(self) -> None:
        if not self.device:
            object.__setattr__(self, "device", _default_device())

    @property
    def is_moe_pose(self) -> bool:
        """vitpose-plus-* are mixture-of-experts and need a dataset_index."""
        return "vitpose-plus" in self.pose_model


def quantize_dynamic_if_cpu(model, config: Config):
    """Int8-quantize a model's linear layers when ``config.quantize`` and on CPU.

    Dynamic quantization only helps (and is only supported) on CPU, so this is a
    no-op on CUDA or when disabled. Returns the (possibly wrapped) model.
    """
    if not config.quantize or config.device != "cpu":
        return model
    import torch

    try:
        from torch.ao.quantization import quantize_dynamic
    except ImportError:  # older torch
        from torch.quantization import quantize_dynamic

    return quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
