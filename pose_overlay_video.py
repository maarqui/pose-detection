"""Render a concert video with multi-person pose-skeleton overlays.

Thin CLI over the ``posedet`` package: it parses arguments, builds a ``Config`` and a
``VideoPoseRunner``, then reads frames, draws skeletons, and writes the output video.
No model, box, or selection logic lives here — that all sits behind ``posedet`` so the
deployment team can lift it out. Person detection uses a ``transformers`` detector
(RT-DETR by default; pass a D-FINE id to ``--detector-model``). There is no YOLO path:
``ultralytics`` has no commercial license, so on-stage-vs-audience filtering is done
geometrically via the selection knobs below.

Examples:
    python pose_overlay_video.py --input input/concert.mov --output out.mp4
    python pose_overlay_video.py --input in.mov --output out.mp4 \\
        --inference-width 1280 --pose-stride 2 --detector-stride 2 \\
        --max-people 8 --box-smoothing 0.25 --stage-roi 0.0,0.1,1.0,0.9
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from posedet import PRESETS, Config, VideoPoseRunner, draw_pose
from posedet.visualization import KEYPOINT_COLORS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("input") / "concertVideo.mov",
        help="Input concert video path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output") / "concertVideo_skeletons.mp4",
        help="Output annotated video path.",
    )
    parser.add_argument(
        "--preset",
        choices=tuple(PRESETS),
        default="accurate",
        help="Speed/accuracy preset (sets models, striding, smoothing, quantize). "
        "Explicit flags below override the preset. 'balanced' is recommended.",
    )
    parser.add_argument(
        "--detector-model",
        default=None,
        help="HuggingFace object detector id. Overrides the preset.",
    )
    parser.add_argument(
        "--pose-model",
        default=None,
        help="HuggingFace ViTPose checkpoint. Overrides the preset.",
    )
    parser.add_argument(
        "--device",
        default="",
        help="Inference device: 'cpu', 'cuda', or '' to auto-detect.",
    )
    parser.add_argument(
        "--person-threshold", type=float, default=0.3, help="Person score threshold."
    )
    parser.add_argument(
        "--keypoint-threshold",
        type=float,
        default=0.3,
        help="Keypoint score threshold.",
    )
    parser.add_argument(
        "--max-people",
        type=int,
        default=0,
        help="Max people to pose-estimate per frame (by score). 0 means no cap.",
    )
    parser.add_argument(
        "--stage-roi",
        default="",
        help=(
            "Performance area as x1,y1,x2,y2 in 0..1 fractions of the frame. People "
            "centered outside are dropped. Empty disables (camera may move/zoom)."
        ),
    )
    parser.add_argument(
        "--audience-suppression",
        type=float,
        default=0.0,
        help="Score penalty for people standing low in the frame (foreground crowd).",
    )
    parser.add_argument(
        "--audience-band",
        type=float,
        default=0.82,
        help="Frame-height fraction below which feet mark a foreground-audience box.",
    )
    parser.add_argument(
        "--dedupe-iou",
        type=float,
        default=0.0,
        help="Drop person boxes overlapping a higher-ranked one at this IoU. 0 = off.",
    )
    parser.add_argument(
        "--inference-width",
        type=int,
        default=None,
        help="Resize frames to this width for inference, then scale skeletons back. "
        "0 keeps full resolution. (Note: has little speed effect; see README.)",
    )
    parser.add_argument(
        "--pose-stride",
        type=int,
        default=None,
        help="Run pose estimation every N frames; reuse skeletons in between. "
        "Overrides the preset.",
    )
    parser.add_argument(
        "--detector-stride",
        type=int,
        default=None,
        help="Run person detection every N frames; reuse boxes in between. "
        "Overrides the preset.",
    )
    parser.add_argument(
        "--box-smoothing",
        type=float,
        default=None,
        help="Temporal box smoothing in 0..0.95. Higher is steadier but laggier. "
        "Overrides the preset.",
    )
    parser.add_argument(
        "--quantize",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Int8-quantize models on CPU (~1.5-2x, some accuracy loss). "
        "Overrides the preset.",
    )
    parser.add_argument(
        "--limit-frames",
        type=int,
        default=0,
        help="Debug limit on frames processed. 0 processes the whole video.",
    )
    parser.add_argument(
        "--draw-boxes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw person bounding boxes (on by default; --no-draw-boxes to disable).",
    )
    return parser.parse_args()


def parse_roi(value: str) -> tuple[float, float, float, float] | None:
    """Parse a ``x1,y1,x2,y2`` fraction string into a tuple, or ``None`` if empty."""
    value = value.strip()
    if not value:
        return None
    parts = tuple(float(part.strip()) for part in value.split(","))
    if len(parts) != 4:
        raise ValueError(f"--stage-roi needs four comma-separated values, got: {value}")
    x1, y1, x2, y2 = parts
    if not (x2 > x1 and y2 > y1):
        raise ValueError(f"--stage-roi must satisfy x2>x1 and y2>y1, got: {value}")
    return parts


def format_duration(seconds: float) -> str:
    whole_seconds = int(round(seconds))
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def main() -> None:
    import cv2

    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input video not found: {args.input}")
    if args.limit_frames < 0:
        raise ValueError("--limit-frames must be >= 0")

    # A preset supplies the defaults; any explicit flag (non-None) overrides it.
    preset = PRESETS[args.preset]

    def pick(value, fallback):
        return fallback if value is None else value

    config = Config(
        detector_model=pick(args.detector_model, preset.detector_model),
        pose_model=pick(args.pose_model, preset.pose_model),
        device=args.device,
        det_threshold=args.person_threshold,
        kpt_threshold=args.keypoint_threshold,
        max_people=args.max_people,
        stage_roi=parse_roi(args.stage_roi),
        audience_suppression=args.audience_suppression,
        audience_band=args.audience_band,
        dedupe_iou=args.dedupe_iou,
        quantize=pick(args.quantize, preset.quantize),
    )
    runner = VideoPoseRunner(
        config,
        detector_stride=pick(args.detector_stride, preset.detector_stride),
        pose_stride=pick(args.pose_stride, preset.pose_stride),
        box_smoothing=pick(args.box_smoothing, preset.box_smoothing),
        inference_width=pick(args.inference_width, preset.inference_width),
    )
    print(f"Preset: {preset.name}  (quantize={config.quantize})")
    print(f"Detector: {config.detector_model}")
    print(f"Pose model: {config.pose_model} on {config.device}")

    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {args.input}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_goal = (
        min(args.limit_frames, total_frames) if args.limit_frames else total_frames
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open output video writer: {args.output}")

    frame_index = 0
    started_at = time.perf_counter()
    try:
        while True:
            if args.limit_frames and frame_index >= args.limit_frames:
                break
            ok, frame_bgr = capture.read()
            if not ok:
                break

            poses = runner.process(frame_bgr)
            annotated = draw_pose(
                frame_bgr,
                poses,
                kpt_threshold=args.keypoint_threshold,
                draw_boxes=args.draw_boxes,
                skeleton_color=(60, 255, 60),
                point_colors=KEYPOINT_COLORS,
                line_thickness=3,
                point_radius=4,
            )
            writer.write(annotated)
            frame_index += 1
            if frame_index % 25 == 0:
                elapsed = max(time.perf_counter() - started_at, 1e-6)
                print(
                    f"Processed {frame_index}/{frame_goal or '?'} frames "
                    f"({frame_index / elapsed:.2f} fps)."
                )
    finally:
        capture.release()
        writer.release()

    total_elapsed = time.perf_counter() - started_at
    print(f"Done. Wrote annotated video to {args.output}")
    print(
        f"Timing: {frame_index} frames in {format_duration(total_elapsed)} "
        f"({frame_index / max(total_elapsed, 1e-6):.2f} fps average)."
    )


if __name__ == "__main__":
    main()
