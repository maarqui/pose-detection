"""Demo runner: detect persons, estimate poses, draw skeletons + boxes.

This is the modular replacement for an ad-hoc test script. It only wires together
the public API from `posedet` — no model logic lives here.

Examples:
    python run_demo.py --image input/jazz.jpg --out out.jpg
    python run_demo.py --video input/concert.mp4 --out out.mp4 --stride 5
"""

from __future__ import annotations

import argparse

import numpy as np
from PIL import Image

from posedet import Config, PosePipeline, draw_pose, iter_video_frames


def _pil_to_bgr(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"))[:, :, ::-1].copy()


def run_image(pipeline: PosePipeline, path: str, out: str, kpt_thr: float) -> None:
    import cv2

    image = Image.open(path).convert("RGB")
    poses = pipeline(image)
    annotated = draw_pose(_pil_to_bgr(image), poses, kpt_threshold=kpt_thr)
    cv2.imwrite(out, annotated)
    print(f"{len(poses)} person(s) -> {out}")


def run_video(
    pipeline: PosePipeline, path: str, out: str, stride: int, kpt_thr: float
) -> None:
    import cv2

    writer = None
    for index, frame_bgr in iter_video_frames(path, stride=stride):
        image = Image.fromarray(frame_bgr[:, :, ::-1])
        poses = pipeline(image)
        annotated = draw_pose(frame_bgr, poses, kpt_threshold=kpt_thr)
        if writer is None:
            h, w = annotated.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out, fourcc, 25 / max(stride, 1), (w, h))
        writer.write(annotated)
        print(f"frame {index}: {len(poses)} person(s)")
    if writer is not None:
        writer.release()
    print(f"-> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ViTPose top-down demo")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="path to an input image")
    src.add_argument("--video", help="path to an input video")
    parser.add_argument("--out", required=True, help="output path")
    parser.add_argument("--stride", type=int, default=1, help="video frame stride")
    parser.add_argument("--kpt-thr", type=float, default=0.3, help="keypoint threshold")
    parser.add_argument("--pose-model", default=Config().pose_model)
    args = parser.parse_args()

    config = Config(pose_model=args.pose_model, kpt_threshold=args.kpt_thr)
    pipeline = PosePipeline(config)

    if args.image:
        run_image(pipeline, args.image, args.out, args.kpt_thr)
    else:
        run_video(pipeline, args.video, args.out, args.stride, args.kpt_thr)


if __name__ == "__main__":
    main()
