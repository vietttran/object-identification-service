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
