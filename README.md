# object-identification-service

A FastAPI service that accepts uploaded videos and runs a multi-stage CV pipeline to detect objects, track them across frames, classify motion, and identify person-object interactions.

## Project structure

```
app/
  main.py          — FastAPI routes and server entry point
  database.py      — SQLite task persistence
  models.py        — Pydantic request/response schemas
  processor.py     — End-to-end pipeline orchestration
  pipeline/
    detection.py   — YOLO object + person detection
    tracking.py    — Cross-frame object ID tracking
    motion.py      — Stationary vs moving classification
    interaction.py — Person-object interaction detection
tests/             — pytest test suite
data/
  uploads/         — Uploaded video files (git-ignored)
  outputs/         — Result JSON + keyframes (git-ignored)
```

## Setup

```bash
# Create and activate virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Running the service

```bash
uvicorn app.main:app --reload
```

## Design Decisions

### Out-of-distribution (OOD) object handling

**The problem.** The detector uses a YOLOv8 model pretrained on COCO, a dataset of 80 everyday object classes (people, cars, furniture, etc.). Lab equipment — microscopes, centrifuges, pipettes, circuit boards — was never in the training distribution. When the model encounters these objects, it does not produce a clean "unknown"; instead it forces the image patch through its softmax head and emits a confidently wrong COCO label such as "laptop", "cell phone", or "car".

**Two-layer mitigation.** The `Detector` class applies two sequential filters:

1. **Confidence threshold (default: 0.5).** OOD objects tend to produce lower confidence scores because the classification softmax spreads probability mass across several vaguely-similar COCO classes. Setting a threshold of 0.5 removes most hallucinated false positives before they ever reach the ByteTrack tracker. The threshold is configurable — lowering it recovers more OOD detections (more boxes, more noise); raising it keeps only high-certainty sightings.

2. **Trusted-class filter.** Detections that survive the confidence gate are labeled with the COCO class name only if that class is in the `trusted_classes` set (default: `{"person"}`). Every other detection is relabeled as the generic string `"object"`. The raw COCO prediction is preserved in a `raw_coco_label` field for debugging and transparency, but all downstream pipeline modules (motion, interaction) use only `class_label`.

**Why trust "person" but nothing else?** "Person" is the most heavily annotated class in COCO (~64,000 instances) and human body shape is visually invariant enough that the model generalises reliably to novel environments. No lab equipment class enjoys that breadth of training coverage.

**The honest trade-off.** This approach detects and tracks every sufficiently-confident foreground object as a distinct entity, but deliberately declines to name it if the model is not trustworthy for that domain. A result that says "there are 4 tracked objects interacting with 2 people" is honest and useful. A result that says "a car, a cell phone, and 2 laptops" interacted with people would be precise-looking but wrong.

**Production upgrade path.** The correct long-term fix is to fine-tune YOLOv8 on a dataset that includes the specific equipment classes you care about, then expand `trusted_classes` accordingly. Because the labeling policy is entirely encapsulated in `Detector.__init__` and `Detector._resolve_label`, no other module needs to change — motion classification, interaction detection, and the API layer are all label-agnostic.
