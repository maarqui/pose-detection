"""Runtime configuration for the pose-detection pipeline.

Everything tunable lives here so that callers (including the deployment team that
integrates this subproject) configure behaviour by passing a ``Config`` instance
instead of editing module internals. No global state, no hardcoded paths.
"""

from __future__ import annotations

from dataclasses import dataclass

# Default open-vocabulary text queries for the instrument detector (OWLv2). Bare
# nouns work well; the set leans toward a jazz ensemble. Override via
# ``Config.instrument_prompts`` to add/remove instruments without code changes.
DEFAULT_INSTRUMENT_PROMPTS: tuple[str, ...] = (
    "saxophone",
    "trumpet",
    "trombone",
    "clarinet",
    "flute",
    "piano",
    "grand piano",
    "guitar",
    "electric guitar",
    "double bass",
    "cello",
    "violin",
    "drum kit",
)


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
        instrument_model: HuggingFace id of the open-vocabulary detector used for
            instruments (OWLv2 by default). Separate from ``detector_model`` (person
            stage); it is text-queried, so any jazz instrument is detectable without
            a YOLO-style fixed vocabulary.
        instrument_prompts: Text queries fed to the instrument detector. See
            ``DEFAULT_INSTRUMENT_PROMPTS``. The matched prompt becomes the label.
        instrument_threshold: Min confidence to keep an instrument detection. OWLv2
            scores run low, so this is smaller than ``det_threshold``.
        instrument_dedupe_iou: NMS IoU for overlapping instrument boxes; ``<= 0``
            disables dedupe (boxes are still score-sorted and capped).
        max_instruments: Cap on instrument detections returned, by descending score.
            ``0`` means no cap.
        instrument_min_area_fraction: Drop instrument boxes smaller than this
            fraction of the frame. This rejects tiny background/crowd hits.
        instrument_max_area_fraction: Drop instrument boxes larger than this
            fraction of the frame. This rejects implausibly broad stage/background
            regions.
        instrument_max_aspect_ratio: Drop very skinny/wide boxes, which are common
            false positives for microphone stands and cables.

    Defaults are tuned for a real concert feed: keep a small set of likely stage
    performers, demote foreground audience, and keep only plausible instrument boxes.
    Set caps/filters to ``0`` or restore lower thresholds for exploratory debugging.
    See ``posedet.presets`` for ready-made speed/accuracy configurations.
    """

    detector_model: str = "PekingU/rtdetr_r50vd_coco_o365"
    pose_model: str = "usyd-community/vitpose-base"
    device: str = ""  # empty -> auto-detect in __post_init__
    det_threshold: float = 0.35
    kpt_threshold: float = 0.3
    person_label: int = 0
    person_label_name: str = "person"
    max_people: int = 8
    stage_roi: tuple[float, float, float, float] | None = None
    audience_suppression: float = 0.35
    audience_band: float = 0.80
    dedupe_iou: float = 0.45
    quantize: bool = False
    instrument_model: str = "google/owlv2-base-patch16-ensemble"
    instrument_prompts: tuple[str, ...] = DEFAULT_INSTRUMENT_PROMPTS
    instrument_threshold: float = 0.18
    instrument_dedupe_iou: float = 0.4
    max_instruments: int = 12
    instrument_min_area_fraction: float = 0.0004
    instrument_max_area_fraction: float = 0.25
    instrument_max_aspect_ratio: float = 8.0

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
