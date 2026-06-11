"""Pydantic schemas defining request/response shapes for the API."""

# Pydantic is FastAPI's built-in validation layer. Every field you declare here
# is validated at runtime — wrong type from a client → automatic 422 Unprocessable
# Entity response before your route handler even runs. It also powers the
# auto-generated OpenAPI/Swagger schema at /docs.

from __future__ import annotations

import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Sub-schemas used inside DetectedObject
# ---------------------------------------------------------------------------

class MotionInterval(BaseModel):
    """One contiguous time window during which a tracked object held the same motion state.

    Example: {"frame_range": [0, 45], "state": "stationary"} means the object
    did not move between frame 0 and frame 45.
    """

    # A two-element list [start_frame, end_frame]. List[int] is permissive about
    # length; in a hardened service you'd add a @field_validator to assert
    # len(frame_range) == 2, but keeping it simple here for clarity.
    frame_range: List[int]

    # "moving" or "stationary". Using plain str instead of Literal["moving",
    # "stationary"] keeps the Swagger schema readable without importing Literal,
    # but either would work fine.
    state: str


class Interaction(BaseModel):
    """A single episode where a person was proximate to / interacting with an object."""

    # The tracker assigns a unique integer ID to each person it follows. This
    # links back to one of the DetectedObject entries whose class == "person".
    interacted_by_person: int

    # Inclusive frame indices bounding the interaction window.
    frame_start: int
    frame_end: int


# ---------------------------------------------------------------------------
# Core per-object schema
# ---------------------------------------------------------------------------

class DetectedObject(BaseModel):
    """All tracking and analysis data for one uniquely identified object in the video."""

    # --- Pydantic v2 config ---
    # ConfigDict replaces the inner `class Config:` block from Pydantic v1.
    model_config = ConfigDict(
        # populate_by_name=True lets callers create instances using *either* the
        # Python field name ("object_class") *or* the JSON alias ("class").
        # Without this, only the alias works as input, which makes internal code
        # awkward (you'd have to write DetectedObject(**{"class": "chair"}) everywhere).
        populate_by_name=True,
    )

    # Unique integer ID assigned by the tracker module, stable across all frames.
    object_id: int

    # -----------------------------------------------------------------------
    # Handling the "class" keyword conflict
    # -----------------------------------------------------------------------
    # "class" is a reserved Python keyword so it cannot be a bare identifier.
    # Solution: name the field "object_class" in Python, but attach
    # Field(alias="class") so that:
    #   • Pydantic accepts "class" when parsing incoming JSON
    #   • model_dump(by_alias=True) outputs "class" in the JSON response
    #   • Python code uses obj.object_class — perfectly valid syntax
    object_class: str = Field(alias="class")

    # Ordered list of motion segments covering the object's full lifetime in the
    # video. Adjacent segments should not have the same state (they'd be merged).
    motion_history: List[MotionInterval]

    # Every person-object interaction detected for this object. May be empty.
    interactions: List[Interaction]


# ---------------------------------------------------------------------------
# Video-level metadata
# ---------------------------------------------------------------------------

class VideoMetadata(BaseModel):
    """Top-level facts about the analysed video file extracted by OpenCV."""

    duration: float    # Total length in seconds.
    frame_count: int   # Total frames decoded (may differ from container header).
    resolution: str    # Width × Height, e.g. "1920x1080".
    fps: float         # Frames per second reported by the container.


# ---------------------------------------------------------------------------
# Keyframe schema
# ---------------------------------------------------------------------------

class KeyFrame(BaseModel):
    """A single extracted still image that illustrates a notable moment in the video."""

    object_id: int      # Which tracked object triggered this keyframe.
    frame_number: int   # Absolute frame index (0-based) in the video.
    reason: str         # Human-readable label, e.g. "interaction_start" or "first_appearance".
    image_path: str     # Relative path inside data/outputs/, e.g. "task-uuid/frame_042.jpg".


# ---------------------------------------------------------------------------
# Top-level result and status schemas
# ---------------------------------------------------------------------------

class TaskResult(BaseModel):
    """The complete analysis payload stored in the DB and returned by GET /tasks/{id}/result.

    Field names are camelCase to match the spec. Pydantic serialises them as-is,
    so the JSON output keys are camelCase automatically — no alias needed.
    """

    videoMetadata: VideoMetadata
    objectsDetected: List[DetectedObject]

    # Optional — the pipeline may skip keyframe extraction in lightweight mode.
    # Default None serialises as JSON null rather than omitting the key, which
    # is more explicit about "we know this field exists but it has no value".
    keyFrames: Optional[List[KeyFrame]] = None


class TaskStatus(BaseModel):
    """Lightweight status record returned by GET /tasks/{task_id}.

    Clients poll this until status reaches "complete" or "failed", then call
    GET /tasks/{task_id}/result to fetch the full payload.
    """

    task_id: str

    # Four lifecycle states as strings. An Enum would add compile-time safety
    # but requires slightly more boilerplate; str is simpler for an interview demo.
    #   queued     → task created, processor not yet started
    #   processing → processor is actively analysing the video
    #   complete   → result_json has been written, result endpoint is ready
    #   failed     → an exception occurred, error field is populated
    status: str

    # Pydantic v2 automatically serialises datetime to ISO-8601 strings in JSON
    # ("2024-01-15T12:34:56.789012"), which every HTTP client can parse.
    created_at: datetime.datetime

    # Only present when status == "failed". Optional[str] serialises to JSON null
    # when absent, making it clear that the field exists but has no value yet.
    error: Optional[str] = None
