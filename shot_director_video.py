"""Render a concert video as an automatic-camera cut: detect musicians, label them by
instrument, score the best shot, and either overlay the chosen crop or output it as a
zoom.

Thin CLI over the ``posedet`` package: it parses arguments, builds a ``Config`` and a
``ShotDirector`` (which wraps the pose runner + OWLv2 instrument detector + framing),
then reads frames and writes the result. No model, framing, or labeling logic lives
here — it all sits behind ``posedet`` so the deployment team can lift it out.

Instrument detection uses OWLv2 (open-vocabulary, Apache-licensed) — there is no YOLO
path. OWLv2 is heavy on CPU, so it runs on a stride (``--instrument-stride``).

Examples:
    python shot_director_video.py --input input/concert.mov --output out.mp4
    python shot_director_video.py --input in.mov --output zoom.mp4 --zoom \\
        --preset balanced --instrument-stride 30 --max-zoom 3.0 --shot-smoothing 0.85
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
        help="Speed/accuracy preset for the pose stage (models, striding, smoothing).",
    )
    parser.add_argument("--device", default="", help="'cpu', 'cuda', or '' to auto.")
    parser.add_argument(
        "--instrument-stride",
        type=int,
        default=15,
        help="Run OWLv2 instrument detection every N frames (it is slow on CPU).",
    )
    parser.add_argument(
        "--instrument-threshold",
        type=float,
        default=0.1,
        help="Min confidence to keep an instrument detection.",
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
        "--limit-frames",
        type=int,
        default=0,
        help="Debug limit on frames processed. 0 processes the whole video.",
    )
    return parser.parse_args()


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
        kpt_threshold=args.instrument_threshold,
        draw_boxes=False,
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

    preset = PRESETS[args.preset]
    config = Config(
        detector_model=preset.detector_model,
        pose_model=preset.pose_model,
        device=args.device,
        instrument_threshold=args.instrument_threshold,
        quantize=preset.quantize,
    )
    runner = VideoPoseRunner(
        config,
        detector_stride=preset.detector_stride,
        pose_stride=preset.pose_stride,
        box_smoothing=preset.box_smoothing,
        inference_width=preset.inference_width,
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
    print(f"Instrument model: {config.instrument_model} on {config.device}")
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
