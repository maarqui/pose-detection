"""Generate the regression baseline for one image. Run once, with weights available
and only when you trust the current pipeline output.

    python -m tests.make_baseline tests/fixtures/jazz.jpg
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

from posedet import Config, PosePipeline


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m tests.make_baseline <image_path>")
    img_path = Path(sys.argv[1])
    poses = PosePipeline(Config())(Image.open(img_path).convert("RGB"))
    kpts = np.concatenate([p.keypoints for p in poses], axis=0)
    out = img_path.with_suffix(".keypoints.npy")
    np.save(out, kpts)
    print(f"saved baseline: {out}  shape={kpts.shape}")


if __name__ == "__main__":
    main()
