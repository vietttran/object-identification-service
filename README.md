# Object Identification Service

A Python microservice that accepts an uploaded video, runs a multi-stage computer vision pipeline asynchronously, and exposes the results through a REST API. Given a video clip, the service detects every person and object present, assigns each a stable tracking ID, classifies each tracked entity's motion history as moving or stationary over time, and identifies every episode in which a person sustains physical proximity with an object. Results are persisted in SQLite and returned as structured JSON.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Setup](#setup)
3. [Running the Service](#running-the-service)
4. [Full Workflow Walkthrough](#full-workflow-walkthrough)
5. [API Reference](#api-reference)
6. [Output Schema](#output-schema)
7. [Design Decisions](#design-decisions)
8. [Testing](#testing)
9. [Project Structure](#project-structure)
10. [Known Limitations](#known-limitations)

---

## Architecture

The pipeline is split into four sequential stages. Each stage is a self-contained module with no HTTP or database dependencies, making them independently testable.

```
Upload (POST /tasks)
       │
       ▼
┌─────────────────────────────────────────────────────┐
│  Stage 1 — Detection + Tracking                      │
│  app/pipeline/detection.py                           │
│  YOLOv8n runs inference on each frame. ByteTrack    │
│  assigns stable integer IDs to detections across    │
│  frames. OOD objects are relabeled "object".        │
└──────────────────────────┬──────────────────────────┘
                           │ raw_tracks: {id: TrackRecord}
                           ▼
┌─────────────────────────────────────────────────────┐
│  Stage 2 — Motion Classification                     │
│  app/pipeline/motion.py                             │
│  Per-track bbox-center displacement (px/frame),     │
│  threshold comparison, run-length segmentation,     │
│  and short-segment smoothing.                       │
└──────────────────────────┬──────────────────────────┘
                           │ motion_history per track
                           ▼
┌─────────────────────────────────────────────────────┐
│  Stage 3 — Interaction Detection                     │
│  app/pipeline/interaction.py                        │
│  For every (person, object) pair: evaluate edge     │
│  distance on shared frames, bridge detection gaps,  │
│  filter by minimum duration.                        │
└──────────────────────────┬──────────────────────────┘
                           │ interactions per object
                           ▼
┌─────────────────────────────────────────────────────┐
│  Stage 4 — Schema Assembly + Persistence             │
│  app/processor.py  →  app/database.py               │
│  Build TaskResult Pydantic model, serialize to      │
│  JSON, write to SQLite tasks table.                 │
└─────────────────────────────────────────────────────┘
```

### Module Summary

| Module | Responsibility |
|---|---|
| `app/main.py` | FastAPI app, routes, `BackgroundTasks` wiring |
| `app/database.py` | SQLite schema, per-call connections, task CRUD |
| `app/models.py` | Pydantic schemas for API request/response types |
| `app/processor.py` | Orchestrates the four pipeline stages |
| `app/pipeline/detection.py` | `Detector` class: YOLOv8 + ByteTrack, OOD handling |
| `app/pipeline/motion.py` | `classify_motion`: displacement analysis + smoothing |
| `app/pipeline/interaction.py` | `detect_interactions`: proximity, gap bridging, duration filter |

### Tech Stack

- **Runtime**: Python 3.11
- **Web framework**: [FastAPI](https://fastapi.tiangolo.com/) on [Uvicorn](https://www.uvicorn.org/)
- **Validation**: [Pydantic v2](https://docs.pydantic.dev/latest/)
- **Detection model**: [Ultralytics YOLOv8](https://docs.ultralytics.com/) (`yolov8n`)
- **Tracker**: [ByteTrack](https://arxiv.org/abs/2110.06864) (via Ultralytics)
- **Video decoding**: OpenCV (`cv2`)
- **Persistence**: SQLite (stdlib `sqlite3`)
- **Tests**: pytest + FastAPI TestClient

---

## Setup

```bash
# 1. Clone the repository
git clone <repo-url>
cd object-identification-service

# 2. Create a virtual environment
python -m venv venv

# 3. Activate it
#    PowerShell (Windows)
venv\Scripts\Activate.ps1
#    Bash (macOS / Linux / Git Bash)
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

> **Note:** YOLOv8n model weights (`yolov8n.pt`, ~6 MB) are downloaded automatically from Ultralytics on first run. No manual download is needed. If you are offline, copy the weights file to the project root before starting.

---

## Running the Service

```bash
uvicorn app.main:app --reload
```

The server starts at **http://localhost:8000**.

- **Interactive API docs (Swagger UI):** http://localhost:8000/docs
- **Alternative docs (ReDoc):** http://localhost:8000/redoc

The Swagger UI is the easiest way to explore and test the endpoints — it lets you upload a video file, copy the returned `task_id`, and call the status and result endpoints directly in the browser.

---

## Full Workflow Walkthrough

The service follows an **upload → poll → fetch** pattern. The POST handler returns immediately (in milliseconds); the pipeline runs in the background.

### Step 1 — Upload a video

```bash
curl -X POST http://localhost:8000/tasks \
  -F "file=@/path/to/your/video.mp4"
```

Response (`201 Created`):

```json
{
  "task_id": "a3f9e12c-7b4d-4e2a-9f01-c3d5a6b8e7f2",
  "status": "queued"
}
```

### Step 2 — Poll for status

```bash
curl http://localhost:8000/tasks/a3f9e12c-7b4d-4e2a-9f01-c3d5a6b8e7f2
```

Response while processing:

```json
{
  "task_id": "a3f9e12c-7b4d-4e2a-9f01-c3d5a6b8e7f2",
  "status": "processing",
  "created_at": "2024-11-15T10:32:01.482910"
}
```

Status values: `queued` → `processing` → `complete` (or `failed` with an `error` field).

### Step 3 — Fetch the result

Once `status` is `complete`:

```bash
curl http://localhost:8000/tasks/a3f9e12c-7b4d-4e2a-9f01-c3d5a6b8e7f2/result
```

Response (`200 OK`) — see [Output Schema](#output-schema) for the full structure.

---

## API Reference

| Method | Path | Description |
|---|---|---|
| `POST` | `/tasks` | Upload a video file (`multipart/form-data`, field `file`). Returns `task_id` and initial status `queued`. |
| `GET` | `/tasks/{task_id}` | Return the current lifecycle status (`queued`, `processing`, `complete`, `failed`) and `created_at` timestamp. |
| `GET` | `/tasks/{task_id}/result` | Return the full `TaskResult` JSON when `status == complete`; return `{status, result: null}` while in-flight; return `{status, error}` on failure. |

All endpoints return `404` for an unknown `task_id`.

---

## Output Schema

```json
{
  "videoMetadata": {
    "duration": 8.0,
    "frame_count": 192,
    "resolution": "1280x720",
    "fps": 24.0
  },
  "objectsDetected": [
    {
      "object_id": 1,
      "class": "person",
      "motion_history": [
        { "frame_range": [0, 136],  "state": "stationary" },
        { "frame_range": [136, 191], "state": "moving" }
      ],
      "interactions": []
    },
    {
      "object_id": 2,
      "class": "object",
      "motion_history": [
        { "frame_range": [0, 191], "state": "stationary" }
      ],
      "interactions": [
        {
          "interacted_by_person": 1,
          "frame_start": 0,
          "frame_end": 136
        }
      ]
    }
  ],
  "keyFrames": [
    {
      "object_id": 2,
      "frame_number": 68,
      "reason": "interaction_peak",
      "image_path": "a3f9e12c-7b4d-4e2a-9f01-c3d5a6b8e7f2/obj2_frame000068_interaction_peak.jpg"
    },
    {
      "object_id": 1,
      "frame_number": 136,
      "reason": "stationary_to_moving",
      "image_path": "a3f9e12c-7b4d-4e2a-9f01-c3d5a6b8e7f2/obj1_frame000136_stationary_to_moving.jpg"
    }
  ]
}
```

> **Sample video:** 8 s, 192 frames, 1280×720, 24 fps. Person (object\_id 1) interacts with a lab instrument (object\_id 2) from frame 0 to 136 (~5.7 s), then moves away. The interaction peak keyframe is saved at the interval midpoint (frame 68); the stationary→moving keyframe fires when the person starts walking at frame 136.

### Field Reference

| Field | Type | Description |
|---|---|---|
| `videoMetadata.duration` | float | Video length in seconds |
| `videoMetadata.frame_count` | int | Total frames decoded |
| `videoMetadata.resolution` | string | `"WIDTHxHEIGHT"` e.g. `"1920x1080"` |
| `videoMetadata.fps` | float | Frames per second from the container header |
| `objectsDetected[].object_id` | int | Stable tracker-assigned integer ID |
| `objectsDetected[].class` | string | `"person"` or `"object"` (trusted label only) |
| `objectsDetected[].motion_history` | array | Ordered, non-overlapping segments covering the object's full visible lifetime |
| `motion_history[].frame_range` | [int, int] | Inclusive start and end frame indices |
| `motion_history[].state` | string | `"moving"` or `"stationary"` |
| `interactions[].interacted_by_person` | int | `object_id` of the person in the interaction |
| `interactions[].frame_start` | int | First frame of the sustained contact episode |
| `interactions[].frame_end` | int | Last frame of the sustained contact episode |
| `keyFrames` | array or null | JPEG stills at notable moments; `null` if no transitions or interactions found |
| `keyFrames[].object_id` | int | The tracked object this still was extracted for |
| `keyFrames[].frame_number` | int | Absolute 0-based frame index in the source video |
| `keyFrames[].reason` | string | `"stationary_to_moving"` or `"interaction_peak"` |
| `keyFrames[].image_path` | string | Path relative to `data/outputs/`, e.g. `"{task_id}/obj2_frame000068_interaction_peak.jpg"` |

> **Note on `class` naming:** `"class"` is a Python reserved keyword. Internally the field is named `object_class`; the JSON serialization uses `"class"` via a Pydantic field alias.

---

## Design Decisions

### Out-of-Distribution Object Handling

**The problem.** YOLOv8n is pretrained on COCO, an 80-class dataset of everyday objects (vehicles, furniture, animals, kitchenware). Lab equipment — microscopes, centrifuges, circuit boards, custom fixtures — was never in the training distribution. When the model encounters an unknown object, it does not produce a clean "unknown" signal; it forces the patch through its softmax head and emits a confidently wrong COCO label such as `"laptop"`, `"cell phone"`, or `"car"`.

**Two-layer mitigation** in `app/pipeline/detection.py`:

1. **Confidence threshold (default 0.5).** OOD objects tend to score lower because probability mass spreads across several vaguely similar classes. The threshold removes most hallucinated detections before they reach ByteTrack.

2. **Trusted-class allowlist (default: `{"person"}`).** Detections that survive the confidence gate are labeled with their COCO class name only if that class appears in `trusted_classes`. All others are relabeled `"object"`. The raw COCO guess is preserved in an internal `raw_coco_label` field for transparency.

**Why trust `"person"` but nothing else?** `"person"` is COCO's most heavily annotated class (~264 k instances), and human body shape generalises robustly across novel environments. No piece of lab equipment has comparable training coverage.

**Production upgrade path.** Fine-tune YOLOv8 on a dataset that includes the specific equipment classes you care about, then expand `trusted_classes`. No pipeline changes are required — motion classification, interaction detection, and the API layer are entirely label-agnostic.

---

### Interaction as Bounding-Box Proximity

True hand-level interaction detection would require pose estimation (e.g. MediaPipe Hands or OpenPose) to verify that fingers or wrists actually contact the object. That is the production upgrade path.

This service uses a pragmatic proxy: a person and object are considered to be interacting when their YOLO bounding boxes are in **sustained spatial proximity** — either overlapping or within a configurable pixel gap (`proximity_threshold`, default 30 px). Three post-processing steps reduce false positives:

- **Gap bridging (`gap_tolerance`, default 5 frames).** YOLO occasionally drops a detection for 1–3 frames due to motion blur or confidence dips. Without bridging, one real 10-second interaction with a 2-frame detection gap would appear as two 5-second events. A tolerance of 5 frames bridges these dropouts.
- **Minimum duration filter (`min_interaction_frames`, default 5 frames).** Interactions shorter than this are discarded as walk-by proximity events, not genuine engagements.

This approach is documented as an approximation. Proximity in 2-D image space is not the same as physical contact; camera angle affects apparent distance. The tradeoff is chosen deliberately: useful output with known limitations is better than no output while waiting for pose estimation.

---

### Why ByteTrack

ByteTrack is a proven multi-object tracker that uses a two-stage assignment strategy (high-confidence detections first, then lower-confidence ones) to maintain stable IDs across occlusion and brief detection loss. It requires no re-identification network or appearance embedding — it tracks by motion alone using a Kalman filter — which makes it fast and GPU-optional.

Alternatives such as DeepSORT or BoT-SORT offer stronger re-identification via appearance features, which matters for long occlusions. ByteTrack is the right default for short lab clips where tracks are rarely fully occluded.

A fresh `Detector` instance is created per `process_video` call rather than shared across concurrent requests. ByteTrack's Kalman filter state is stored inside the `YOLO` model object (`persist=True`). Sharing one instance across threads would corrupt that state; per-call instantiation eliminates the risk at the cost of redundant weight loading. The production fix is a worker-process pool (e.g. Celery) where each worker owns exactly one `Detector`.

---

### Minimum Track Length Filter

Raw ByteTrack output includes short-lived tracks from partial occlusions, shadows, reflections, and people briefly entering the frame edge. These produce noise in the output: a 2-frame "person" track with 0 interactions and a 1-segment stationary history adds nothing and confuses downstream consumers.

`processor.py` drops any track with fewer than `min_track_observations` observations (default: 8) before motion classification or interaction detection. At 30 fps this requires ~0.27 s of continuous visibility — enough to exclude single-frame spurious detections while retaining all tracks that could plausibly represent a real interaction.

---

### Motion Classification via Bbox-Center Displacement

The motion classifier (`app/pipeline/motion.py`) measures the Euclidean displacement of a bounding box's centre point between consecutive observations and normalises by the frame gap:

```
per_frame_displacement = sqrt(Δcx² + Δcy²) / (frame_j − frame_i)
```

**Why the centre?** Box size wobbles slightly per frame due to detection jitter (the CNN predicts slightly different corners each pass). Tracking a corner amplifies that wobble; tracking the centre averages it out.

**Why normalise by frame gap?** The same physical speed produces different raw displacements at different sampling strides. Normalisation makes `motion_threshold` stride-independent: a threshold of 2.0 always means "2 px/frame" regardless of whether the video was sampled every frame or every fifth frame.

**Smoothing.** A 1-interval "moving" blip between two long stationary periods (e.g. a single jittery YOLO box) is absorbed into the surrounding stationary state when its interval count is below `min_segment_frames` (default: 3). This prevents one-frame twitches from generating spurious state transitions.

---

### Frame-Stride Configurability

`process_video` accepts a `frame_stride` parameter (default: 1 — every frame). At stride 5, the pipeline processes the video at 1/5 the observation density, reducing both ByteTrack state updates and motion/interaction computation. For a 30 fps clip this still gives 6 observations/second — adequate for most interaction detection use cases. The motion threshold and gap tolerance remain meaningful because both are normalised per frame.

---

### Keyframe Extraction (Bonus)

Implemented in `app/pipeline/keyframes.py`. Two categories of notable moment are captured as JPEG stills and written to `data/outputs/{task_id}/`:

**1. Stationary → Moving transitions.** The first frame of each new `"moving"` segment is saved with `reason="stationary_to_moving"`. This is the frame where the displacement threshold was crossed, making it the most useful still for validating that the motion classifier is correctly calibrated — a reviewer can open it and immediately check whether the object really is in motion at that frame.

**2. Interaction peaks.** The temporal midpoint of each sustained interaction interval is saved with `reason="interaction_peak"`. The midpoint is used as a proxy for peak engagement because our proximity signal is binary: a frame is either in contact or not, with no within-interval confidence gradient. A pose estimator (e.g. MediaPipe Hands) would produce a continuous wrist-contact probability from which we could select the maximum; the midpoint heuristic is a documented stand-in until that upgrade is made. The frame at `(frame_start + frame_end) // 2` is past the approach phase and before the withdrawal phase — the most plausible moment of full engagement given the information available.

**Non-fatal design.** Keyframe extraction is wrapped in a `try/except` in `processor.py`. If it fails for any reason — corrupt GOP, full disk, OpenCV codec error — the failure is logged as a warning and the `TaskResult` is returned with `keyFrames: null`. The core motion and interaction data are always preserved; the bonus stills degrade gracefully rather than aborting the task.

---

## Testing

### Run the test suite

```bash
# From the project root with the virtualenv active:
pytest tests/ -v
```

Expected output: **31 tests pass** in under 5 seconds. No video files or GPU required.

```
tests/test_api.py::test_upload_creates_task_returns_201_with_task_id    PASSED
tests/test_api.py::test_get_task_status_returns_queued_for_newly_created_task  PASSED
...
tests/test_motion.py::test_smoothing_absorbs_brief_moving_blip_between_stationary_periods  PASSED
...
======================== 31 passed in 3.34s ========================
```

### Saving results for evidence

```bash
pytest tests/ -v 2>&1 | Tee-Object -FilePath test_results.txt   # PowerShell
pytest tests/ -v 2>&1 | tee test_results.txt                    # Bash
```

### What is covered

| File | Tests | What is verified |
|---|---|---|
| `tests/test_motion.py` | 11 | `classify_motion`: empty input, single observation, pure states, state transition with correct frame boundaries, per-frame normalisation across two strides, smoothing absorption and contrast, adjacent-state invariant (parametrised × 4 inputs) |
| `tests/test_interaction.py` | 10 | `_box_edge_distance` geometry (overlapping, touching, side-by-side, stacked, diagonal); `detect_interactions` happy path, no-contact rejection, edge-proximity contact, brief-contact filter, gap bridging, gap rejection, empty tracks, non-overlapping windows |
| `tests/test_api.py` | 6 | POST → 201 + task_id, GET status → 200 + queued, GET result while queued → null result field, GET unknown task → 404, GET result for unknown task → 404, two tasks produce distinct IDs |

### Testing philosophy

- **Pure-logic modules** (`motion.py`, `interaction.py`) are tested with hand-crafted synthetic observations. No video file, no YOLO weights, no GPU needed. Inputs are designed to target one code path each so failures are immediately diagnosable.
- **API layer** (`main.py`) is tested via FastAPI's `TestClient`. Two fixtures provide isolation: `DATABASE_PATH` is redirected to a per-test `tmp_path` temp file, and `_run_pipeline` is replaced with a no-op lambda so YOLO is never invoked. This tests routing, request parsing, status codes, and the full DB read/write cycle without touching the real pipeline.

---

## Project Structure

```
object-identification-service/
│
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI app, routes, background task wiring
│   ├── database.py      # SQLite schema init, task CRUD, per-call connections
│   ├── models.py        # Pydantic v2 schemas (TaskStatus, TaskResult, DetectedObject, …)
│   ├── processor.py     # Orchestrates the four pipeline stages; returns TaskResult dict
│   └── pipeline/
│       ├── __init__.py
│       ├── detection.py    # Detector class: YOLOv8 + ByteTrack, OOD relabeling
│       ├── motion.py       # classify_motion: displacement analysis, segmentation, smoothing
│       ├── interaction.py  # detect_interactions: proximity, gap bridging, duration filter
│       ├── keyframes.py    # extract_keyframes: seek + save JPEGs at notable moments
│       └── tracking.py     # (reserved for future standalone tracking utilities)
│
├── tests/
│   ├── __init__.py
│   ├── test_motion.py      # 11 unit tests for classify_motion
│   ├── test_interaction.py # 10 unit tests for detect_interactions and helpers
│   └── test_api.py         # 6 integration tests via FastAPI TestClient
│
├── data/
│   ├── uploads/            # Uploaded video files (git-ignored)
│   └── outputs/            # Per-task keyframe JPEG stills (git-ignored)
│
├── test_detector.py        # Standalone CLI smoke test: runs detector on a video file
├── test_motion.py          # Standalone CLI smoke test: prints per-track motion history
├── test_interaction.py     # Standalone CLI smoke test: prints interaction timeline
│
├── requirements.txt        # Python dependencies
├── .gitignore
├── tasks.db                # SQLite database (created on first run, git-ignored)
└── yolov8n.pt              # YOLOv8n weights (downloaded on first run, git-ignored)
```

---

## Known Limitations

**COCO vocabulary ceiling.** The service can detect and track domain-specific objects reliably, but it cannot name them. Any object outside COCO's 80 classes is reported as `"object"`. Accurate labels for lab equipment require fine-tuning on a domain-specific dataset.

**Bounding-box proximity is not physical contact.** A person standing beside a machine with arms at their sides may trigger proximity; a person reaching across a surface to touch an off-screen instrument will not. The interaction signal is a strong proxy for many common lab manipulation scenarios, but should not be treated as ground-truth contact detection without validation on the specific environment.

**Camera angle dependence.** Proximity thresholds are in pixel space and scale with both resolution and camera angle. A threshold tuned for a side-view camera will not transfer directly to a top-down camera without re-calibration.

**Single-camera, single-scene.** The pipeline processes one video file per task. Multi-camera scene reconstruction and cross-camera tracking are not supported.

**Short clip optimisation.** Default parameters (`min_track_observations=8`, `frame_stride=1`) are tuned for clips of a few seconds to a few minutes at 30 fps. Very long videos (>10 minutes) will be slow; increase `frame_stride` and `min_track_observations` proportionally.
