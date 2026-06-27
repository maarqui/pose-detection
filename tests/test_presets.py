"""Pure-data tests for the speed/accuracy presets. No weights needed."""

from __future__ import annotations

from posedet import PRESETS, Config, Preset


def test_expected_presets_exist():
    assert {"accurate", "balanced", "fast"} <= set(PRESETS)
    assert all(isinstance(p, Preset) for p in PRESETS.values())


def test_accurate_matches_historical_default():
    # The 'accurate' preset must reproduce the package's default behavior.
    p = PRESETS["accurate"]
    default = Config()
    assert p.detector_model == default.detector_model
    assert p.pose_model == default.pose_model
    assert p.detector_stride == 1 and p.pose_stride == 1
    assert not p.quantize


def test_default_config_is_concert_filtered():
    cfg = Config()
    assert cfg.max_people == 8
    assert cfg.audience_suppression > 0
    assert cfg.dedupe_iou > 0
    assert cfg.max_instruments == 12
    assert "microphone" not in cfg.instrument_prompts
    assert cfg.instrument_min_area_fraction > 0
    assert cfg.instrument_max_aspect_ratio > 0


def test_to_config_applies_preset_and_overrides():
    p = PRESETS["fast"]
    cfg = p.to_config(max_people=8)
    assert isinstance(cfg, Config)
    assert cfg.detector_model == p.detector_model
    assert cfg.pose_model == p.pose_model
    assert cfg.quantize == p.quantize  # fast enables quantization
    assert cfg.max_people == 8  # override took effect


def test_runner_kwargs_shape():
    p = PRESETS["balanced"]
    kw = p.runner_kwargs
    assert set(kw) == {
        "detector_stride",
        "pose_stride",
        "inference_width",
        "box_smoothing",
    }
    assert kw["pose_stride"] == p.pose_stride


def test_fast_is_not_slower_than_balanced_in_striding():
    # Sanity: fast strides at least as aggressively as balanced.
    fast, balanced = PRESETS["fast"], PRESETS["balanced"]
    assert fast.pose_stride >= balanced.pose_stride
    assert fast.detector_stride >= balanced.detector_stride
