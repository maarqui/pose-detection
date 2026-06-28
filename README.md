# pose-detection: how it works and how to run it

Top-down human **pose & action detection** for jazz-concert images and video, built on
HuggingFace `transformers` (ViTPose). This document explains the pipeline and shows how
to run it on **video** (one CLI) and on **single images** (via the library).

> Domain note: ViTPose is **top-down** and has **no detector**. Poses are only ever
> estimated *inside* person boxes produced by a separate detector. The pipeline is
> always two stages.

---

## 1. How the pipeline works

```
image / video frame
   │
   ▼
PersonDetector        stage 1: a transformers object detector (RT-DETR by default,
   │  person boxes             D-FINE optional) finds people. Boxes are converted
   │  (COCO x,y,w,h)           from VOC (x1,y1,x2,y2) to COCO (x,y,w,h).
   ▼
select_performers     (optional) bias boxes toward on-stage musicians using geometry:
   │  refined boxes              stage ROI, foreground-audience suppression, dedupe, cap.
   ▼
PoseEstimator         stage 2: ViTPose estimates 17 COCO keypoints per person box.
   │  PersonPose[]             Results cross the API as plain NumPy (PersonPose), never
   ▼                           framework tensors.
draw_pose             draw skeleton + keypoints (+ optional box) onto the frame.
```

For **video**, `VideoPoseRunner` wraps stages 1–2 and adds the temporal concerns:
running the models only every N frames (striding), smoothing boxes across frames, and
downscaling frames for inference then scaling keypoints back to full resolution.

### Beyond skeletons: musician labeling and the auto-shot director

On top of the pose stages, the pipeline can work out *who plays what* and frame a shot
automatically:

```
poses + frame
   │
   ▼
classify_pose      posture from joints (standing / sitting + arms raised). Pure geometry.
   │
   ▼
InstrumentDetector OWLv2 (open-vocabulary, text-prompted) finds instrument boxes — any
   │  instruments      jazz instrument by name, no YOLO.
   ▼
label_musicians    tie each instrument to its player by geometry and assign a role
   │  Musician[]       ("guitar" → "guitarist"), wrapping pose + posture + instrument.
   ▼
choose_shot        score performers (a standing soloist with raised arms wins) and
   │  Shot             return the best crop, grown to the frame's aspect ratio.
   ▼
ShotDirector       per-frame loop: runs instruments on a stride and smooths the crop
                   over time, so the shot pans/zooms instead of snapping.
```

The chosen shot **keeps the camera's aspect ratio** (never square) and acts as a
**zoom** into the original resolution — it never distorts the image.

### Keypoints
17 COCO keypoints per person: nose, eyes, ears, shoulders, elbows, wrists, hips, knees,
ankles. Names in `posedet.COCO_KEYPOINTS`; skeleton edges in `posedet.COCO_SKELETON`.

### Hardware reality
Development assumes **CPU / integrated graphics - no CUDA**. With the `accurate`
preset (`vitpose-base`) expect **~1.5 s per frame**. Real-time is a long-term goal,
**not achievable on this hardware**, it targets the deployment GPU. To trade accuracy
for speed on CPU use a faster `--preset` (see [Section 4c](#4c-performance-presets)), the
biggest levers are model size and striding, not resolution.

---

## 2. Installation

```bash
pip install -e ".[dev]"
```

This installs `torch`, `transformers`, `opencv-python`, `pillow`, `numpy` (plus
`pytest`/`ruff` for dev). Model weights download automatically from HuggingFace on
first run and are cached locally. 

---

## 3. Running on a single image

There is no standalone photo CLI — the project ships **one script**, and it targets
video. For a single still, call the library directly (full example in
[Section 5](#5-using-posedet-as-a-library)):

```python
from PIL import Image
from posedet import Config, PosePipeline, draw_pose
import numpy as np, cv2

image = Image.open("input/jazz.jpg").convert("RGB")
poses = PosePipeline(Config())(image)            # list[PersonPose]
bgr = np.array(image)[:, :, ::-1].copy()
cv2.imwrite("out.jpg", draw_pose(bgr, poses))    # skeletons + boxes

---

## 4. Running on a video

There is a **single CLI**, `shot_director_video.py` — a thin wrapper over
`ShotDirector`. It runs the whole pipeline (person detection → pose skeletons →
instrument detection → musician labeling → shot scoring) and writes one of two outputs:

- **overlay** (default): the frame annotated with pose skeletons, optional person boxes
  (`--draw-boxes`), instrument boxes, role/posture labels, and the chosen shot rectangle.
- **zoom** (`--zoom`): the chosen shot cropped and scaled back to full resolution, as an
  automatic-camera cut.

A `--preset` supplies sensible defaults for the pose stage; any explicit
model/striding/smoothing flag overrides it (see [Section 4c](#4c-performance-presets)).

### 4a. Overlay (skeletons + boxes + instruments + musicians + shot)

Minimal run (skeletons, instruments, labels and the shot rectangle):

```bash
python shot_director_video.py --input input/concert.mov --output out.mp4
```

Add the person bounding boxes to the overlay:

```bash
python shot_director_video.py --input input/concert.mov --output out.mp4 --draw-boxes
```

Faster run with a **performance preset** (recommended for long clips):

```bash
python shot_director_video.py --input input/concert.mov --output out.mp4 --preset fast
```

Tuned run (cap people, smooth boxes, restrict to a stage region):

```bash
python shot_director_video.py \
    --input input/concert.mov --output out.mp4 \
    --preset fast \
    --max-people 8 \
    --box-smoothing 0.25 \
    --stage-roi 0.0,0.1,1.0,0.9
```

Override the preset's models / striding — e.g. a **D-FINE** detector and a sharper pose
pass:

```bash
python shot_director_video.py --input in.mov --output out.mp4 \
    --detector-model ustc-community/dfine-small-coco \
    --pose-stride 2 --detector-stride 2 --inference-width 1280
```

Quick debug run (first 10 frames only):

```bash
python shot_director_video.py --input input/concert.mov --output out.mp4 --limit-frames 10
```

### 4b. Zoom (automatic-camera cut)

With `--zoom` the script outputs the chosen shot as a crop zoomed back to full
resolution instead of the annotated overlay:

```bash
python shot_director_video.py --input in.mov --output zoom.mp4 --zoom \
    --preset balanced --instrument-stride 30 --max-zoom 3.0 --shot-smoothing 0.85
```

> Instrument detection uses **OWLv2** (open-vocabulary, Apache-licensed — no YOLO), so
> any instrument is found by name. It is heavy on CPU, so it runs on a stride; raise
> `--instrument-stride` on long clips. The chosen shot **keeps the camera's aspect
> ratio** and never distorts the image.

### 4c. Performance presets

`--preset` bundles a model pair with striding, smoothing and CPU quantization at three
points on the speed/accuracy curve. Explicit flags override any preset field.

| Preset      | Detector | Pose model | Striding | Quantize | Use it for |
|-------------|----------|------------|----------|----------|------------|
| `accurate`  | RT-DETR-r50 | vitpose-base | 1/1 | no | Quality reference, real-time on the deployment **GPU**. |
| `balanced`  | RT-DETR-r18 | vitpose-plus-small | 2/2 | no | **Recommended.** Best CPU speed/quality trade-off. |
| `fast`      | RT-DETR-r18 | vitpose-plus-small | 3/3 | int8 | Max CPU throughput, skeleton may jitter / drop joints. |

> **Reality check (CPU vs GPU).** True live-stream rates (~25 fps)   are **not reachable on this CPU/iGPU** - they are the job of the deployment GPU, where `accurate` runs in real time. On CPU, `fast` roughly 5×'s the `accurate` throughput but is still a preview tool, not live. Also note **`--inference-width` barely affects speed**: both models resize internally (RT-DETR to ~640, ViTPose to 256×192 per person), so input resolution changes only pre/post-processing cost. The real levers are the **model size**, **striding**, and **number of people** (`--max-people`, `--stage-roi`).

### 4d. Benchmarking your hardware

`benchmark.py` times detection vs pose per stage and reports raw and striding-adjusted fps, so you can pick a preset or verify a change helped:

```bash
python benchmark.py --input input/your_clip.mov --preset all --frames 8 --max-people 8
```

### 4e. All CLI options

**Input / output**

| Flag                   | Meaning                       | Default |
|------------------------|-------------------------------|---------|
| `--input` / `--output` | Input video / output video    | `input/concertVideo.mov` / `output/concertVideo_director.mp4` |

**Models / device** (override the `--preset`)

| Flag                           | Meaning                                            | Default |
|--------------------------------|----------------------------------------------------|---------|
| `--preset`                     | Pose-stage speed/accuracy preset                   | `balanced` |
| `--detector-model`             | HuggingFace object detector id (RT-DETR or D-FINE) | from preset |
| `--pose-model`                 | ViTPose checkpoint                                 | from preset |
| `--device`                     | `cpu`, `cuda`, or `` to auto-detect                | auto    |
| `--quantize` / `--no-quantize` | Int8-quantize models on CPU (~1.5-2×, some loss)   | from preset |

**Video / performance** (trade accuracy for speed on long clips)

| Flag                | Meaning                                                     | Default |
|---------------------|-------------------------------------------------------------|---------|
| `--inference-width` | Resize width for inference, scale results back; `0` = full  | from preset |
| `--pose-stride`     | Run pose estimation every N frames                          | from preset |
| `--detector-stride` | Run person detection every N frames                         | from preset |
| `--box-smoothing`   | Temporal person-box smoothing in 0..0.95                    | from preset |

**Detection & performer selection**

| Flag                     | Meaning                                                       | Default |
|--------------------------|---------------------------------------------------------------|---------|
| `--person-threshold`     | Min person-detection score                                    | `0.35`  |
| `--max-people`           | Cap likely performers per frame; `0` = no cap                 | `8`     |
| `--stage-roi`            | `x1,y1,x2,y2` in 0..1 fractions; drop people centered outside  | off     |
| `--audience-suppression` | Score penalty for people low in the frame                     | `0.35`  |
| `--audience-band`        | Frame-height fraction below which feet mark foreground crowd   | `0.80`  |
| `--dedupe-iou`           | Drop overlapping person boxes at this IoU; `0` = off           | `0.45`  |

**Instrument detection (OWLv2)**

| Flag                      | Meaning                                                        | Default |
|---------------------------|----------------------------------------------------------------|---------|
| `--instrument-stride`     | Run OWLv2 instrument detection every N frames                  | `15`    |
| `--instrument-threshold`  | Min confidence to keep an instrument detection                 | `0.18`  |
| `--max-instruments`       | Max instrument boxes per detection pass; `0` = no cap          | `12`    |
| `--instrument-min-area`   | Drop tiny instrument boxes by frame-area fraction              | `0.0004`|
| `--instrument-max-area`   | Drop huge instrument boxes by frame-area fraction              | `0.25`  |
| `--instrument-max-aspect` | Drop very skinny/wide instrument boxes                         | `8.0`   |
| `--include-microphones`   | Also prompt for microphones; off avoids mic-stand over-detect  | off     |

**Shot framing**

| Flag                | Meaning                                                       | Default |
|---------------------|---------------------------------------------------------------|---------|
| `--shot-smoothing`  | EMA crop smoothing in 0..0.95 (higher = steadier camera)      | `0.8`   |
| `--margin`          | Padding around framed subjects, as a fraction of their size   | `0.15`  |
| `--max-zoom`        | Max zoom-in; the crop is never smaller than frame/max-zoom    | `2.5`   |
| `--group-ratio`     | Frame peers with salience >= this fraction of the top one's   | `0.8`   |
| `--min-association` | Min geometric score to tie an instrument to a musician        | `0.1`   |

**Output & drawing**

| Flag                               | Meaning                                            | Default |
|------------------------------------|----------------------------------------------------|---------|
| `--zoom` / `--no-zoom`             | Output the zoomed crop vs. the annotated overlay   | overlay |
| `--draw-boxes` / `--no-draw-boxes` | Draw person bounding boxes in overlay mode         | off     |
| `--keypoint-threshold`             | Min keypoint score to draw / label                 | `0.3`   |
| `--limit-frames`                   | Process at most N frames (`0` = all)               | `0`     |

> Telling on-stage musicians apart from the audience is done geometrically with
> `--stage-roi`, `--audience-suppression`, `--dedupe-iou`, and `--max-people`. The
> defaults are conservative for real concert footage, but a stable venue should still
> get a tuned `--stage-roi`.

> Instruments are detected by text prompt; the default jazz set lives in
> `posedet.config.DEFAULT_INSTRUMENT_PROMPTS`. Add or remove instruments through
> `Config.instrument_prompts` (library use) — no code change needed.

---

> Microphones are intentionally excluded by default because mic stands over-detect in
> real concert footage; pass `--include-microphones` or add `"microphone"` only when
> vocals are important. Instrument boxes are also filtered by size/aspect before
> association, so tiny crowd hits and very skinny stand-like boxes do not drive shots.

## 5. Using posedet as a library

The deployment team imports from `posedet`. Everything is configured via `Config`
(no global state, no hardcoded paths) and returns plain data.

### Single image

```python
from PIL import Image
from posedet import Config, PosePipeline, draw_pose
import numpy as np, cv2

pipeline = PosePipeline(Config())
image = Image.open("jazz.jpg").convert("RGB")

poses = pipeline(image)                       # list[PersonPose]
for p in poses:
    print(p.keypoints.shape, p.scores.shape)  # (17, 2), (17,)

bgr = np.array(image)[:, :, ::-1].copy()
annotated = draw_pose(bgr, poses)
cv2.imwrite("out.jpg", annotated)
```

### Two stages explicitly (e.g. to inspect/modify boxes)

```python
from posedet import Config, PersonDetector, PoseEstimator

cfg = Config(detector_model="ustc-community/dfine-small-coco", max_people=8)
detector = PersonDetector(cfg)
estimator = PoseEstimator(cfg)

boxes = detector.detect(image)          # (N, 4) COCO boxes, already selected/capped
poses = estimator.estimate(image, boxes)
```

### Video with the runner

```python
import cv2
from posedet import Config, VideoPoseRunner, draw_pose, iter_video_frames
from posedet.visualization import KEYPOINT_COLORS

runner = VideoPoseRunner(
    Config(max_people=8, stage_roi=(0.0, 0.1, 1.0, 0.9)),
    detector_stride=2, pose_stride=2, box_smoothing=0.25, inference_width=1280,
)

writer = None
for index, frame_bgr, poses in runner.run(iter_video_frames("concert.mp4")):
    annotated = draw_pose(frame_bgr, poses, point_colors=KEYPOINT_COLORS)
    if writer is None:
        h, w = annotated.shape[:2]
        writer = cv2.VideoWriter("out.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
    writer.write(annotated)
writer.release()
```

### Auto-shot director

`ShotDirector` runs the whole chain over a frame stream — poses, instruments (on a
stride), musician labeling, shot scoring — and smooths the crop across frames. Each
`DirectorFrame` exposes `.musicians`, `.instruments`, and the chosen `.shot`.

```python
import cv2
from posedet import Config, ShotDirector, apply_zoom, iter_video_frames

director = ShotDirector(
    Config(), instrument_stride=15, shot_smoothing=0.85, max_zoom=3.0,
)

writer = None
for index, frame_bgr, df in director.run(iter_video_frames("concert.mp4")):
    zoomed = apply_zoom(frame_bgr, df.shot.box)   # or draw df.musicians/df.shot instead
    if writer is None:
        h, w = zoomed.shape[:2]
        writer = cv2.VideoWriter("out.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
    writer.write(zoomed)
writer.release()
```

The pieces are usable on their own too: `classify_pose(pose)` for posture,
`InstrumentDetector(cfg).detect(image)` for instrument boxes, and
`label_musicians(poses, instruments)` to tie them together into `Musician`s.

### Public API surface

`Config`, `PersonDetector`, `PoseEstimator`, `PosePipeline`, `PersonPose`,
`VideoPoseRunner`, `draw_pose`, `select_performers`, `iter_video_frames`,
`voc_to_coco`, the box helpers (`iou_xywh`, `expand_box`, `filter_to_roi`,
`dedupe_ranked_boxes`, `smooth_boxes`), and `COCO_SKELETON` / `COCO_KEYPOINTS` /
`KEYPOINT_COLORS`.

The musician-understanding and auto-shot layer adds: `classify_pose`,
`classify_poses`, `PoseClassification`, `InstrumentDetector`, `InstrumentDetection`,
`label_musicians`, `associate_instruments`, `role_for_instrument`, `Musician`,
`ShotDirector`, `DirectorFrame`, `Shot`, `choose_shot`, `fit_aspect`, `apply_zoom`,
and the renderers `draw_instruments`, `draw_musician_labels`, `draw_shot`.

### Choosing models via `Config`

- Default detector: `PekingU/rtdetr_r50vd_coco_o365`. D-FINE
  (`ustc-community/dfine-small-coco`) works by just setting `detector_model`.
- Default pose model: `usyd-community/vitpose-base`. The `vitpose-plus-*` models are
  mixture-of-experts and need a `dataset_index`; `Config.is_moe_pose` detects this and
  `PoseEstimator` supplies it automatically.
- Default instrument detector: `google/owlv2-base-patch16-ensemble` (open-vocabulary).
  Set `Config.instrument_prompts` to control which instruments are searched for, and
  `instrument_threshold` / `max_instruments` to tune precision.

---

## 6. Tips & troubleshooting

- **Too slow.** Use `--preset balanced` or `--preset fast`, set `--max-people` to a small number, and add a `--stage-roi` to skip the audience. (Lowering `--inference-width` barely helps - see [Section 4c](#4c-performance-presets).)
- **Skeletons jitter.** Raise `--box-smoothing` toward `0.4 (steadier but laggier on fast camera moves).
- **Audience members get skeletons.** Set a `--stage-roi`, or add
  `--audience-suppression 0.3`.
- **Missing faint joints.** Lower `--keypoint-threshold` (e.g. `0.2`).
- **No people found.** Lower `--person-threshold`; confirm the detector vocabulary has a `person` class.
- **`ultralytics`/YOLO.** Not used and not required - by design (licensing).
- **Auto-shot too slow.** OWLv2 is the cost; raise `--instrument-stride` (instruments
  barely move between frames) and use a faster `--preset` for the pose stage.
- **Instrument missed / mislabeled.** Lower `--instrument-threshold`, or add the
  instrument to `Config.instrument_prompts`. A wrong player match? Tune `--min-association`.
- **Shot jumps around.** Raise `--shot-smoothing` (toward `0.9`) and/or `--group-ratio`
  so near-equal performers are framed together instead of cutting between them.

---

## 7. Running the tests

```bash
pytest                      # fast units run; model-level test skips without weights
pytest -m "not slow"        # only the fast pure-logic + loop-logic suite
ruff check . && ruff format .
```

The fast suite covers box geometry, performer selection, the video-runner loop (via
injected fakes), drawing, pose classification, instrument post-processing,
instrument↔musician association, shot scoring/framing, and the director loop (via
injected fakes) - no model weights needed.
