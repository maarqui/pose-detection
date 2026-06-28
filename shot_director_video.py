"""The concert-video CLI: detect people, draw pose skeletons and person boxes, label
musicians by instrument, score the best shot, and either overlay everything or output
the chosen crop as a zoom.

This is the single entry point for the pipeline. In its default **overlay** mode it
draws the full annotation stack — skeletons, person bounding boxes (``--draw-boxes``),
instrument boxes, musician labels, and the chosen shot rectangle. In **zoom** mode
(``--zoom``) it instead emits the framed crop as an automatic-camera cut. Every stage
is configurable from the flags below.

Thin CLI over the ``posedet`` package: it parses arguments, builds a ``Config`` and a
``ShotDirector`` (which wraps the pose runner + OWLv2 instrument detector + framing),
then reads frames and writes the result. No model, framing, or labeling logic lives
here — it all sits behind ``posedet`` so the deployment team can lift it out.

Person detection uses a ``transformers`` detector (RT-DETR by default; pass a D-FINE id
to ``--detector-model``). Instrument detection uses OWLv2 (open-vocabulary, Apache-
licensed) — there is no YOLO path. OWLv2 is heavy on CPU, so it runs on a stride
(``--instrument-stride``). A ``--preset`` supplies sensible defaults; any explicit
model/striding/smoothing flag overrides it.

Examples:
    # Full overlay (skeletons + boxes + instruments + labels + shot)
    python shot_director_video.py --input in.mov --output out.mp4 --draw-boxes
    # Automatic-camera zoom cut
    python shot_director_video.py --input in.mov --output zoom.mp4 --zoom \\
        --preset balanced --instrument-stride 30 --max-zoom 3.0 --shot-smoothing 0.85
    # Override the preset's models / striding for a sharper skeleton pass
    python shot_director_video.py --input in.mov --output out.mp4 \\
        --pose-stride 2 --detector-stride 2 --inference-width 1280
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from posedet import (
    PRESETS,
    Config,
    ShotDirector,
    VideoPoseRunner,
    apply_zoom,
    draw_instruments,
    draw_musician_labels,
    draw_pose,
    draw_shot,
)
from posedet.config import DEFAULT_INSTRUMENT_PROMPTS
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
        default=Path("output") / "concertVideo_director.mp4",
        help="Output video path.",
    )
    parser.add_argument(
        "--preset",
        choices=tuple(PRESETS),
        default="balanced",
        help="Speed/accuracy preset for the pose stage (models, striding, smoothing). "
        "Explicit model/striding/smoothing flags below override it.",
    )
    parser.add_argument("--device", default="", help="'cpu', 'cuda', or '' to auto.")
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
        "--inference-width",
        type=int,
        default=None,
        help="Resize frames to this width for inference, then scale results back. "
        "0 keeps full resolution. Overrides the preset.",
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
        help="Temporal person-box smoothing in 0..0.95. Higher is steadier but "
        "laggier. Overrides the preset.",
    )
    parser.add_argument(
        "--quantize",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Int8-quantize models on CPU (~1.5-2x, some accuracy loss). "
        "Overrides the preset.",
    )
    parser.add_argument(
        "--instrument-stride",
        type=int,
        default=15,
        help="Run OWLv2 instrument detection every N frames (it is slow on CPU).",
    )
    parser.add_argument(
        "--instrument-threshold",
        type=float,
        default=0.18,
        help="Min confidence to keep an instrument detection.",
    )
    parser.add_argument(
        "--max-instruments",
        type=int,
        default=12,
        help="Max instrument boxes to keep per detection pass. 0 means no cap.",
    )
    parser.add_argument(
        "--instrument-min-area",
        type=float,
        default=0.0004,
        help="Drop instrument boxes smaller than this frame-area fraction. 0 disables.",
    )
    parser.add_argument(
        "--instrument-max-area",
        type=float,
        default=0.25,
        help="Drop instrument boxes larger than this frame-area fraction. 0 disables.",
    )
    parser.add_argument(
        "--instrument-max-aspect",
        type=float,
        default=8.0,
        help="Drop very skinny/wide instrument boxes. 0 disables.",
    )
    parser.add_argument(
        "--include-microphones",
        action="store_true",
        help=(
            "Also prompt for microphones. Off by default because mic stands "
            "over-detect."
        ),
    )
    parser.add_argument(
        "--person-threshold", type=float, default=0.35, help="Person score threshold."
    )
    parser.add_argument(
        "--keypoint-threshold",
        type=float,
        default=0.3,
        help="Keypoint score threshold for labeling and overlays.",
    )
    parser.add_argument(
        "--max-people",
        type=int,
        default=8,
        help="Max people to pose-estimate per frame, after audience-aware ranking.",
    )
    parser.add_argument(
        "--stage-roi",
        default="",
        help=(
            "Performance area as x1,y1,x2,y2 in 0..1 fractions of the frame. People "
            "centered outside are dropped. Empty disables."
        ),
    )
    parser.add_argument(
        "--audience-suppression",
        type=float,
        default=0.35,
        help="Score penalty for people standing low in the frame.",
    )
    parser.add_argument(
        "--audience-band",
        type=float,
        default=0.80,
        help="Frame-height fraction below which feet mark a foreground-audience box.",
    )
    parser.add_argument(
        "--dedupe-iou",
        type=float,
        default=0.45,
        help="Drop person boxes overlapping a higher-ranked one at this IoU. 0 = off.",
    )
    parser.add_argument(
        "--shot-smoothing",
        type=float,
        default=0.8,
        help="EMA smoothing of the chosen crop in 0..0.95 (higher = steadier camera).",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.15,
        help="Padding around the framed subjects, as a fraction of their size.",
    )
    parser.add_argument(
        "--max-zoom",
        type=float,
        default=2.5,
        help="Maximum zoom-in factor; the crop is never smaller than frame/max-zoom.",
    )
    parser.add_argument(
        "--group-ratio",
        type=float,
        default=0.8,
        help="Frame peers whose salience is >= this fraction of the top performer's.",
    )
    parser.add_argument(
        "--min-association",
        type=float,
        default=0.1,
        help="Min geometric score to tie an instrument to a musician.",
    )
    parser.add_argument(
        "--zoom",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Output the zoomed crop instead of an overlay of the chosen shot.",
    )
    parser.add_argument(
        "--draw-boxes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draw person bounding boxes in overlay mode (off by default to keep the "
        "overlay readable; ignored in --zoom mode).",
    )
    parser.add_argument(
        "--limit-frames",
        type=int,
        default=0,
        help="Debug limit on frames processed. 0 processes the whole video.",
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
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError(
            "--stage-roi must be normalized x1,y1,x2,y2 with "
            f"0<=x1<x2<=1 and 0<=y1<y2<=1, got: {value}"
        )
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


def render(frame_bgr, director_frame, args):
    """Build the output frame: a zoom of the chosen shot, or an annotated overlay."""
    if args.zoom:
        return apply_zoom(frame_bgr, director_frame.shot.box)
    annotated = draw_pose(
        frame_bgr,
        [m.pose for m in director_frame.musicians],
        kpt_threshold=args.keypoint_threshold,
        draw_boxes=args.draw_boxes,
        skeleton_color=(60, 255, 60),
        point_colors=KEYPOINT_COLORS,
        line_thickness=3,
        point_radius=4,
    )
    annotated = draw_instruments(annotated, director_frame.instruments)
    annotated = draw_musician_labels(annotated, director_frame.musicians)
    return draw_shot(annotated, director_frame.shot)


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

    instrument_prompts = DEFAULT_INSTRUMENT_PROMPTS
    if args.include_microphones:
        instrument_prompts = (*instrument_prompts, "microphone")
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
        instrument_prompts=instrument_prompts,
        instrument_threshold=args.instrument_threshold,
        max_instruments=args.max_instruments,
        instrument_min_area_fraction=args.instrument_min_area,
        instrument_max_area_fraction=args.instrument_max_area,
        instrument_max_aspect_ratio=args.instrument_max_aspect,
        quantize=pick(args.quantize, preset.quantize),
    )
    runner = VideoPoseRunner(
        config,
        detector_stride=pick(args.detector_stride, preset.detector_stride),
        pose_stride=pick(args.pose_stride, preset.pose_stride),
        box_smoothing=pick(args.box_smoothing, preset.box_smoothing),
        inference_width=pick(args.inference_width, preset.inference_width),
    )
    director = ShotDirector(
        config,
        runner=runner,
        instrument_stride=args.instrument_stride,
        shot_smoothing=args.shot_smoothing,
        margin=args.margin,
        max_zoom=args.max_zoom,
        group_ratio=args.group_ratio,
        min_association=args.min_association,
    )
    print(f"Preset: {preset.name}  (quantize={config.quantize})")
    print(f"Detector: {config.detector_model}")
    print(f"Pose model: {config.pose_model} on {config.device}")
    print(f"Instrument model: {config.instrument_model}")
    print(f"Mode: {'zoom' if args.zoom else 'overlay'}")

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

            director_frame = director.process(frame_bgr)
            writer.write(render(frame_bgr, director_frame, args))
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
    print(f"Done. Wrote video to {args.output}")
    print(
        f"Timing: {frame_index} frames in {format_duration(total_elapsed)} "
        f"({frame_index / max(total_elapsed, 1e-6):.2f} fps average)."
    )


if __name__ == "__main__":
    main()
