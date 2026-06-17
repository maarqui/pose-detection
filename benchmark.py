"""Benchmark the pipeline's per-stage cost and effective fps for the speed presets.

Times stage 1 (detection) and stage 2 (pose) separately on a handful of real frames,
then reports both the raw per-frame cost and the *effective* fps once the preset's
striding is amortized in. Use it to pick a preset for your hardware, or to check
whether a model/quantization change actually helped.

Examples:
    python benchmark.py --input input/s-video-test_1.mov
    python benchmark.py --input in.mov --preset all --frames 8 --max-people 8
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from posedet import PRESETS, PersonDetector, PoseEstimator


def read_frames(path: Path, count: int):
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {path}")
    frames = []
    while len(frames) < count:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"No frames read from {path}")
    return frames


def bench_preset(name: str, frames, max_people: int, quantize: bool | None):
    import cv2
    from PIL import Image

    preset = PRESETS[name]
    overrides = {"max_people": max_people}
    if quantize is not None:
        overrides["quantize"] = quantize
    config = preset.to_config(**overrides)

    detector = PersonDetector(config)
    estimator = PoseEstimator(config)

    det_times, pose_times, people = [], [], []
    for i, frame_bgr in enumerate(frames):
        image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        t0 = time.perf_counter()
        boxes = detector.detect(image)
        t1 = time.perf_counter()
        estimator.estimate(image, boxes)
        t2 = time.perf_counter()
        if i == 0:
            continue  # first frame warms up lazy model loading
        det_times.append(t1 - t0)
        pose_times.append(t2 - t1)
        people.append(len(boxes))

    n = max(len(det_times), 1)
    det = sum(det_times) / n
    pose = sum(pose_times) / n
    avg_people = sum(people) / n if people else 0.0
    raw_fps = 1.0 / max(det + pose, 1e-6)
    # Effective per-frame cost once striding amortizes detector/pose across frames.
    eff_cost = det / preset.detector_stride + pose / preset.pose_stride
    eff_fps = 1.0 / max(eff_cost, 1e-6)
    return {
        "name": name,
        "quantize": config.quantize,
        "stride": f"{preset.detector_stride}/{preset.pose_stride}",
        "people": avg_people,
        "det": det,
        "pose": pose,
        "raw_fps": raw_fps,
        "eff_fps": eff_fps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input", type=Path, default=Path("input") / "s-video-test_1.mov"
    )
    parser.add_argument(
        "--preset", default="all", choices=(*PRESETS.keys(), "all"), help="Preset(s)."
    )
    parser.add_argument(
        "--frames", type=int, default=4, help="Frames to time (+1 warmup)."
    )
    parser.add_argument(
        "--max-people", type=int, default=0, help="Cap people; 0 = no cap."
    )
    parser.add_argument(
        "--quantize",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force quantization on/off, overriding the preset.",
    )
    args = parser.parse_args()

    frames = read_frames(args.input, args.frames + 1)
    names = list(PRESETS.keys()) if args.preset == "all" else [args.preset]

    print(f"input: {args.input}  frames timed: {len(frames) - 1}\n")
    header = (
        f"{'preset':9} {'stride':6} {'q':3} {'people':6} "
        f"{'det(s)':7} {'pose(s)':7} {'raw fps':8} {'eff fps':8}"
    )
    print(header)
    print("-" * len(header))
    for name in names:
        r = bench_preset(name, frames, args.max_people, args.quantize)
        print(
            f"{r['name']:9} {r['stride']:6} {('y' if r['quantize'] else 'n'):3} "
            f"{r['people']:6.1f} {r['det']:7.2f} {r['pose']:7.2f} "
            f"{r['raw_fps']:8.2f} {r['eff_fps']:8.2f}"
        )
    print(
        "\nraw fps = per-frame cost with no striding; eff fps = with preset striding."
    )


if __name__ == "__main__":
    main()
