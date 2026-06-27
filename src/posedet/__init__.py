"""posedet — top-down human pose detection on images and video.

This module's public names form the boundary the deployment team imports against.
Keep it small and stable; treat everything not listed in ``__all__`` as internal.
"""

from __future__ import annotations

from .boxes import (
    dedupe_ranked_boxes,
    expand_box,
    filter_to_roi,
    iou_xywh,
    smooth_boxes,
)
from .config import Config
from .detection import PersonDetector, voc_to_coco
from .director import DirectorFrame, ShotDirector
from .framing import Shot, apply_zoom, choose_shot, fit_aspect
from .instruments import InstrumentDetection, InstrumentDetector
from .musicians import (
    Musician,
    associate_instruments,
    label_musicians,
    role_for_instrument,
)
from .pose import PersonPose, PoseEstimator, PosePipeline
from .poseclass import (
    ARM_STATES,
    POSTURES,
    PoseClassification,
    classify_pose,
    classify_poses,
)
from .presets import PRESETS, Preset
from .runner import VideoPoseRunner
from .selection import select_performers
from .shot_profiles import ShotCandidate, role_shot_candidates
from .video import iter_video_frames
from .visualization import (
    COCO_KEYPOINTS,
    COCO_SKELETON,
    KEYPOINT_COLORS,
    draw_instruments,
    draw_musician_labels,
    draw_pose,
    draw_shot,
)

__all__ = [
    "Config",
    "PersonDetector",
    "InstrumentDetector",
    "InstrumentDetection",
    "Musician",
    "label_musicians",
    "associate_instruments",
    "role_for_instrument",
    "PoseEstimator",
    "PosePipeline",
    "PersonPose",
    "PoseClassification",
    "classify_pose",
    "classify_poses",
    "POSTURES",
    "ARM_STATES",
    "VideoPoseRunner",
    "ShotDirector",
    "DirectorFrame",
    "Shot",
    "ShotCandidate",
    "choose_shot",
    "fit_aspect",
    "apply_zoom",
    "role_shot_candidates",
    "Preset",
    "PRESETS",
    "draw_pose",
    "draw_instruments",
    "draw_musician_labels",
    "draw_shot",
    "iter_video_frames",
    "voc_to_coco",
    "iou_xywh",
    "expand_box",
    "filter_to_roi",
    "dedupe_ranked_boxes",
    "smooth_boxes",
    "select_performers",
    "COCO_KEYPOINTS",
    "COCO_SKELETON",
    "KEYPOINT_COLORS",
]

__version__ = "0.1.0"
