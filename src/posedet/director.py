"""Stateful auto-shot director: the top-level loop that turns a frame stream into
framed shots.

It composes the pieces built in earlier stages and owns the *temporal* concerns that
make an automatic camera watchable:

- **Pose** — delegated to ``VideoPoseRunner`` (which already handles detector/pose
  striding and box smoothing).
- **Instruments** — ``InstrumentDetector`` is expensive (OWLv2 on CPU), so it runs
  only every ``instrument_stride`` frames; the last instruments are reused between.
- **Labeling** — ``label_musicians`` ties instruments to poses and assigns roles.
- **Framing** — ``choose_shot`` picks the aspect-correct crop, and the director
  **smooths that crop across frames** (EMA) so the shot pans/zooms instead of
  snapping. Drawing and file I/O stay with the caller.

The runner and instrument detector are injectable, so the whole loop is testable with
fakes and no model weights, and the deployment team can swap either component.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np

from .config import Config
from .framing import Shot, choose_shot


@dataclass
class DirectorFrame:
    """The director's output for one frame.

    Attributes:
        musicians: Labeled musicians (pose + posture + instrument + role).
        instruments: Instruments associated with musicians this frame. Raw detections
            are still used internally for association, but unassigned boxes are not
            exposed to the overlay/shot layer.
        shot: The smoothed ``Shot`` to crop to (see ``framing``).
    """

    musicians: list
    instruments: list
    shot: Shot


class ShotDirector:
    """Drive pose + instrument + framing over a frame stream, with temporal state.

    Args:
        config: Model / selection configuration. A default ``Config`` is used if
            omitted.
        instrument_stride: Run instrument detection every Nth processed frame,
            reusing the previous instruments in between. ``1`` runs it every frame.
        shot_smoothing: EMA blend of the crop box with the previous frame's, in
            ``[0, 0.95]``; ``0`` disables smoothing (the shot snaps each frame).
        margin, max_zoom, group_ratio: Framing knobs forwarded to ``choose_shot``.
        min_association: Threshold forwarded to ``label_musicians``.
        runner: Optional object with ``.process(frame_bgr) -> [PersonPose]``.
            Defaults to ``VideoPoseRunner(config)``. Injectable for testing/swapping.
        instrument_detector: Optional object with ``.detect(image) ->
            [InstrumentDetection]``. Defaults to ``InstrumentDetector(config)``.
    """

    def __init__(
        self,
        config: Config | None = None,
        *,
        instrument_stride: int = 15,
        shot_smoothing: float = 0.8,
        margin: float = 0.15,
        max_zoom: float = 2.5,
        group_ratio: float = 0.8,
        min_association: float = 0.1,
        runner=None,
        instrument_detector=None,
    ) -> None:
        if instrument_stride < 1:
            raise ValueError("instrument_stride must be >= 1")

        self.config = config or Config()
        self.instrument_stride = instrument_stride
        self.shot_smoothing = min(max(shot_smoothing, 0.0), 0.95)
        self.margin = margin
        self.max_zoom = max_zoom
        self.group_ratio = group_ratio
        self.min_association = min_association

        if runner is None:
            from .runner import VideoPoseRunner

            runner = VideoPoseRunner(self.config)
        self.runner = runner

        if instrument_detector is None:
            from .instruments import InstrumentDetector

            instrument_detector = InstrumentDetector(self.config)
        self.instrument_detector = instrument_detector

        self.reset()

    def reset(self) -> None:
        """Clear temporal state so the next frame starts a fresh sequence."""
        self._frame_index = 0
        self._last_instruments: list = []
        self._shot_box: np.ndarray | None = None
        self._shot_history: list[str] = []

    def _smooth_shot(self, shot: Shot) -> Shot:
        """EMA-blend the new crop box with the previous frame's to damp shot motion."""
        if self.shot_smoothing <= 0.0 or self._shot_box is None:
            self._shot_box = np.asarray(shot.box, dtype=float)
            return shot
        blended = self.shot_smoothing * self._shot_box + (
            1.0 - self.shot_smoothing
        ) * np.asarray(shot.box, dtype=float)
        self._shot_box = blended
        return Shot(
            box=blended,
            score=shot.score,
            musician_indices=shot.musician_indices,
            shot_type=shot.shot_type,
            description=shot.description,
        )

    def process(self, frame_bgr: np.ndarray) -> DirectorFrame:
        """Process one BGR frame and return the labeled musicians and smoothed shot.

        Frames must be fed in order; the internal counters drive instrument striding
        and shot smoothing.
        """
        from PIL import Image

        from .musicians import label_musicians

        poses = self.runner.process(frame_bgr)

        if self._frame_index % self.instrument_stride == 0:
            image = Image.fromarray(frame_bgr[:, :, ::-1])  # BGR -> RGB
            self._last_instruments = self.instrument_detector.detect(image)
        instruments = self._last_instruments

        musicians = label_musicians(
            poses,
            instruments,
            min_association=self.min_association,
            kpt_threshold=self.config.kpt_threshold,
        )
        associated_instruments = [
            musician.instrument for musician in musicians if musician.instrument
        ]

        height, width = frame_bgr.shape[:2]
        raw_shot = choose_shot(
            musicians,
            width,
            height,
            margin=self.margin,
            max_zoom=self.max_zoom,
            group_ratio=self.group_ratio,
            kpt_threshold=self.config.kpt_threshold,
            shot_history=self._shot_history,
        )
        shot = self._smooth_shot(raw_shot)

        # Update history every ~1 second (assuming 30fps) to avoid rapid switching
        # but still encourage variety over time.
        if self._frame_index % 30 == 0:
            self._shot_history.append(raw_shot.description)
            if len(self._shot_history) > 20:
                self._shot_history.pop(0)

        self._frame_index += 1
        return DirectorFrame(
            musicians=musicians, instruments=associated_instruments, shot=shot
        )

    def run(
        self, frames: Iterable[tuple[int, np.ndarray]]
    ) -> Iterator[tuple[int, np.ndarray, DirectorFrame]]:
        """Yield ``(index, frame_bgr, DirectorFrame)`` for each ``(index, frame)`` in.

        Pairs with ``iter_video_frames``; the caller draws overlays and/or applies the
        zoom and writes the result.
        """
        for index, frame_bgr in frames:
            yield index, frame_bgr, self.process(frame_bgr)
