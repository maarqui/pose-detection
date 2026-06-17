"""Model-level regression test.

This is the check that catches "a change silently broke detection". It runs the
real pipeline on a fixed image and asserts the keypoints have not drifted from a
saved baseline. It is marked ``slow`` and skips cleanly when the model weights or
the fixture baseline are not available, so the fast suite stays green everywhere.

To create the baseline once (with weights available and the pipeline trusted):

    python -m tests.make_baseline tests/fixtures/jazz.jpg

That writes ``tests/fixtures/jazz.keypoints.npy``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

FIXTURE_IMG = Path(__file__).parent / "fixtures" / "jazz.jpg"
FIXTURE_KPTS = Path(__file__).parent / "fixtures" / "jazz.keypoints.npy"

_have_transformers = importlib.util.find_spec("transformers") is not None


@pytest.mark.slow
@pytest.mark.skipif(not _have_transformers, reason="transformers not installed")
@pytest.mark.skipif(
    not (FIXTURE_IMG.exists() and FIXTURE_KPTS.exists()),
    reason="fixture image or baseline keypoints missing (see module docstring)",
)
def test_keypoints_do_not_drift():
    from PIL import Image

    from posedet import Config, PosePipeline

    pipeline = PosePipeline(Config())
    poses = pipeline(Image.open(FIXTURE_IMG).convert("RGB"))

    current = np.concatenate([p.keypoints for p in poses], axis=0)
    baseline = np.load(FIXTURE_KPTS)

    assert current.shape == baseline.shape, "person/keypoint count changed"
    # Allow sub-pixel jitter; flag real drift.
    np.testing.assert_allclose(current, baseline, atol=2.0)
