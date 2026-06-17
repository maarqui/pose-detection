# pose-detection: how it works and how to run it

Top-down human **pose & action detection** for jazz-concert images and video, built on
HuggingFace `transformers` (ViTPose). This document explains the pipeline and shows how
to run it on **photos** and **videos**.

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

## 3. Running on a photo

Use `run_demo.py`:

```bash
python run_demo.py --image input/jazz.jpg --out out.jpg
```

It detects people, estimates poses, draws skeletons + boxes, and writes `out.jpg`. It
prints how many people were found.

Useful flags:

| Flag           | Meaning                                            | Default          |
|----------------|----------------------------------------------------|------------------|
| `--image`      | Input image path (mutually exclusive with `--video`) | —              |
| `--out`        | Output image path                                  | required         |
| `--kpt-thr`    | Keypoint score threshold (lower = draw more)       | `0.3`            |
| `--pose-model` | ViTPose checkpoint                                 | `vitpose-base`   |

---

## 4. Running on a video

There are two entry points.

### 4a. `run_demo.py`: simplest video path

```bash
python run_demo.py --video input/concert.mp4 --out out.mp4 --stride 5
```

`--stride N` processes every Nth frame (downsampling for slow CPU runs). Good for a quick look, no smoothing or performer selection.

### 4b. `pose_overlay_video.py`: the concert CLI (recommended for video)

A thin CLI over `VideoPoseRunner` with all the tuning knobs. Minimal run:

```bash
python pose_overlay_video.py --input input/concert.mov --output out.mp4
```

Faster run with a **performance preset** (recommended for video - see
[Section 4c](#4c-performance-presets)):

```bash
python pose_overlay_video.py --input input/concert.mov --output out.mp4 --preset balanced
```

Tuned run (preset + cap people, smooth boxes, restrict to a stage region):

```bash
python pose_overlay_video.py \
    --input input/concert.mov --output out.mp4 \
    --preset fast \
    --max-people 8 \
    --box-smoothing 0.25 \
    --stage-roi 0.0,0.1,1.0,0.9
```

Quick debug run (first 10 frames only):

```bash
python pose_overlay_video.py --input input/concert.mov --output out.mp4 --limit-frames 10
```

Use a **D-FINE** detector instead of the default RT-DETR:

```bash
python pose_overlay_video.py --input in.mov --output out.mp4 \
    --detector-model ustc-community/dfine-small-coco
```

#### All CLI options

**Input / output**

| Flag         | Meaning                     | Default |
|--------------|-----------------------------|---------|
| `--input`    | Input video path            | `input/concertVideo.mov` |
| `--output`   | Output annotated video path | `output/concertVideo_skeletons.mp4` |

**Models / device**

| Flag               | Meaning                                           | Default |
|--------------------|---------------------------------------------------|---------|
| `--detector-model` | HuggingFace object detector id (RT-DETR or D-FINE) | RT-DETR |
| `--pose-model`     | ViTPose checkpoint                                | `vitpose-base` |
| `--device`         | `cpu`, `cuda`, or `` to auto-detect               | auto    |

**Detection & performer selection**

| Flag                     | Meaning                                                          | Default |
|--------------------------|------------------------------------------------------------------|---------|
| `--person-threshold`     | Min person-detection score                                       | `0.3`   |
| `--max-people`           | Cap people per frame (by score); `0` = no cap                    | `0`     |
| `--stage-roi`            | `x1,y1,x2,y2` in 0..1 fractions; drop people centered outside    | off     |
| `--audience-suppression` | Score penalty for people low in the frame (foreground crowd)     | `0.0`   |
| `--audience-band`        | Frame-height fraction below which feet mark foreground audience  | `0.82`  |
| `--dedupe-iou`           | Drop boxes overlapping a higher-ranked one at this IoU; `0` = off| `0.0`   |

**Video / performance** (trade accuracy for speed on long clips)

| Flag                | Meaning                                                      | Default |
|---------------------|--------------------------------------------------------------|---------|
| `--inference-width` | Resize width for inference, scale skeletons back; `0` = full | `0`     |
| `--pose-stride`     | Run pose estimation every N frames                           | `1`     |
| `--detector-stride` | Run detection every N frames                                 | `1`     |
| `--box-smoothing`   | Temporal box smoothing in 0..0.95                            | `0.0`   |

**Drawing & debug**

| Flag                               | Meaning                              | Default |
|------------------------------------|--------------------------------------|---------|
| `--keypoint-threshold`             | Min keypoint score to draw           | `0.3`   |
| `--draw-boxes` / `--no-draw-boxes` | Draw person bounding boxes           | on      |
| `--limit-frames`                   | Process at most N frames (`0` = all) | `0`     |

> Telling on-stage musicians apart from the audience is done geometrically with `--stage-roi`, `--audience-suppression`, and `--dedupe-iou`. These default to off and need **per-venue tuning** - start with a stage ROI if the camera framing is stable.

### 4c. Performance presets

`--preset` bundles a model pair with striding, smoothing and CPU quantization at three
points on the speed/accuracy curve. Explicit flags override any preset field.

| Preset      | Detector | Pose model | Striding | Quantize | Use it for |
|-------------|----------|------------|----------|----------|------------|
| `accurate`  | RT-DETR-r50 | vitpose-base | 1/1 | no | Quality reference, real-time on the deployment **GPU**. The default. |
| `balanced`  | RT-DETR-r18 | vitpose-plus-small | 2/2 | no | **Recommended.** Best CPU speed/quality trade-off. |
| `fast`      | RT-DETR-r18 | vitpose-plus-small | 3/3 | int8 | Max CPU throughput, skeleton may jitter / drop joints. |

> **Reality check (CPU vs GPU).** True live-stream rates (~25 fps)   are **not reachable on this CPU/iGPU** - they are the job of the deployment GPU, where `accurate` runs in real time. On CPU, `fast` roughly 5×'s the `accurate` throughput but is still a preview tool, not live. Also note **`--inference-width` barely affects speed**: both models resize internally (RT-DETR to ~640, ViTPose to 256×192 per person), so input resolution changes only pre/post-processing cost. The real levers are the **model size**, **striding**, and **number of people** (`--max-people`, `--stage-roi`).

### 4d. Benchmarking your hardware

`benchmark.py` times detection vs pose per stage and reports raw and striding-adjusted fps, so you can pick a preset or verify a change helped:

```bash
python benchmark.py --input input/your_clip.mov --preset all --frames 8 --max-people 8
```

---

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

### Public API surface

`Config`, `PersonDetector`, `PoseEstimator`, `PosePipeline`, `PersonPose`,
`VideoPoseRunner`, `draw_pose`, `select_performers`, `iter_video_frames`,
`voc_to_coco`, the box helpers (`iou_xywh`, `expand_box`, `filter_to_roi`,
`dedupe_ranked_boxes`, `smooth_boxes`), and `COCO_SKELETON` / `COCO_KEYPOINTS` /
`KEYPOINT_COLORS`.

### Choosing models via `Config`

- Default detector: `PekingU/rtdetr_r50vd_coco_o365`. D-FINE
  (`ustc-community/dfine-small-coco`) works by just setting `detector_model`.
- Default pose model: `usyd-community/vitpose-base`. The `vitpose-plus-*` models are
  mixture-of-experts and need a `dataset_index`; `Config.is_moe_pose` detects this and
  `PoseEstimator` supplies it automatically.

---

## 6. Tips & troubleshooting

- **Too slow.** Use `--preset balanced` or `--preset fast`, set `--max-people` to a small number, and add a `--stage-roi` to skip the audience. (Lowering `--inference-width` barely helps - see [Section 4c](#4c-performance-presets).)
- **Skeletons jitter.** Raise `--box-smoothing` toward `0.4 (steadier but laggier on fast camera moves).
- **Audience members get skeletons.** Set a `--stage-roi`, or add
  `--audience-suppression 0.3`.
- **Missing faint joints.** Lower `--keypoint-threshold` (e.g. `0.2`).
- **No people found.** Lower `--person-threshold`; confirm the detector vocabulary has a `person` class.
- **`ultralytics`/YOLO.** Not used and not required - by design (licensing).

---

## 7. Running the tests

```bash
pytest                      # fast units run; model-level test skips without weights
pytest -m "not slow"        # only the fast pure-logic + loop-logic suite
ruff check . && ruff format .
```

The fast suite covers box geometry, performer selection, the video-runner loop (via injected fakes), and drawing - no model weights needed.
