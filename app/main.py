"""FastAPI application entry point — defines routes and starts the ASGI server."""

# FastAPI is built on Starlette (the ASGI web framework) and Pydantic (validation).
# Uvicorn is the ASGI server that actually binds to a port and handles TCP connections.
# The relationship is: uvicorn → starlette (ASGI) → FastAPI (routing + validation).

import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.database import create_task, get_task, init_db, update_task_status
from app.models import TaskResult, TaskStatus


# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

# Resolve the project root at import time so it is correct regardless of the
# working directory when uvicorn starts. __file__ is app/main.py; .parent is
# app/; .parent.parent is the project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# All uploaded videos land here. The directory is git-ignored so binary files
# never get committed.
UPLOAD_DIR = _PROJECT_ROOT / "data" / "uploads"

# parents=True creates any missing parent directories; exist_ok=True means no
# error if the directory already exists. Equivalent to `mkdir -p`.
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run setup code at startup and teardown code at shutdown.

    The asynccontextmanager pattern replaces the deprecated @app.on_event("startup")
    decorator. Everything before `yield` executes when uvicorn starts; everything
    after `yield` executes when uvicorn shuts down (Ctrl-C or SIGTERM).

    Why call init_db() here and not at module level?
    Calling it at module level would run during `import app.main`, which happens
    in test environments where you may not want a real DB. Deferring to startup
    keeps side effects explicit and controllable.
    """
    # init_db() is idempotent (uses IF NOT EXISTS) so restarting the server never
    # wipes existing task rows.
    init_db()
    yield
    # Nothing to tear down for SQLite, but you'd close connection pools, flush
    # caches, or drain background queues here in a production service.


# ---------------------------------------------------------------------------
# FastAPI application instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Object Identification Service",
    description=(
        "Upload a video file and receive structured data describing every detected "
        "object, its motion history across frames, and any person-object interactions."
    ),
    version="0.1.0",
    # Swagger UI (interactive browser UI) lives at /docs.
    # ReDoc (read-only docs) lives at /redoc.
    # Both are auto-generated from the OpenAPI schema FastAPI builds by introspecting
    # your route signatures, response_model declarations, and Pydantic models.
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# POST /tasks — upload a video and enqueue analysis
# ---------------------------------------------------------------------------

@app.post(
    "/tasks",
    summary="Upload a video and create an analysis task",
    response_description="The newly created task_id and its initial 'queued' status",
    # 201 Created is semantically more precise than 200 OK when a new resource
    # has been created as a result of the request.
    status_code=201,
)
def create_analysis_task(
    # UploadFile wraps the multipart file from the client. FastAPI reads the
    # Content-Type boundary automatically when `python-multipart` is installed
    # (it's in requirements.txt). The File(...) call marks the field as required.
    file: UploadFile = File(..., description="Video file to analyse (e.g. MP4, AVI)"),
):
    """Accept an uploaded video, save it to disk, and create a queued task record.

    The actual video processing pipeline is not yet implemented. This endpoint
    just persists the file and returns a task_id. Clients should then poll
    GET /tasks/{task_id} until the status reaches 'complete' or 'failed'.
    """

    # --- 1. Generate a globally unique task ID ---
    # UUID version 4 is randomly generated. With 122 bits of randomness the
    # probability of a collision across 1 billion tasks is ~10^-19 — negligible.
    task_id = str(uuid.uuid4())

    # --- 2. Determine the destination file path ---
    # Preserve the original file extension so OpenCV can select the right codec
    # when opening the file later. Fall back to ".mp4" if no extension is present.
    original_ext = Path(file.filename).suffix if file.filename else ".mp4"
    video_path = UPLOAD_DIR / f"{task_id}{original_ext}"

    # --- 3. Stream the upload to disk ---
    try:
        # file.file is a SpooledTemporaryFile — small files are kept in RAM,
        # large files are automatically spooled to a temp file. We read in 1 MB
        # chunks to avoid loading the entire video into memory at once.
        with open(video_path, "wb") as dest:
            while chunk := file.file.read(1024 * 1024):  # walrus operator: assign + test
                dest.write(chunk)
    except Exception as exc:
        # Surface upload errors as 500 so the client knows the task was NOT created.
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save uploaded file: {exc}",
        )

    # --- 4. Create the task record in SQLite ---
    # This must happen AFTER the file is safely on disk; we don't want a task row
    # pointing to a file that doesn't exist.
    create_task(task_id=task_id, video_path=str(video_path))

    # --- 5. Return the task_id ---
    # In a production service you would also kick off background processing here,
    # e.g. enqueue a Celery task or use FastAPI's built-in BackgroundTasks.
    # For now the task stays 'queued' until the processor is implemented.
    return {"task_id": task_id, "status": "queued"}


# ---------------------------------------------------------------------------
# GET /tasks/{task_id} — poll for task status
# ---------------------------------------------------------------------------

@app.get(
    "/tasks/{task_id}",
    response_model=TaskStatus,
    summary="Get the current status of an analysis task",
    response_description="TaskStatus object with lifecycle state and optional error",
)
def get_task_status(task_id: str):
    """Return the current lifecycle status of a task.

    Clients should poll this endpoint until status is 'complete' or 'failed',
    then call GET /tasks/{task_id}/result. A typical polling interval is 2–5 s.

    FastAPI path parameters (the {task_id} in the URL template) are automatically
    extracted and injected as function arguments. No manual parsing needed.
    """
    row = get_task(task_id)

    # get_task() returns None when the primary key does not exist.
    # HTTP 404 Not Found is the correct status code for a missing resource.
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Task '{task_id}' not found.",
        )

    # Construct the response model from the DB row dict. Because we declared
    # response_model=TaskStatus, FastAPI will validate and serialise this object
    # using Pydantic before sending it over the wire — any extra fields are
    # stripped, and missing required fields would raise an internal error.
    return TaskStatus(
        task_id=row["task_id"],
        status=row["status"],
        created_at=row["created_at"],  # already a datetime object thanks to PARSE_DECLTYPES
        error=row["error"],
    )


# ---------------------------------------------------------------------------
# GET /tasks/{task_id}/result — retrieve the full analysis result
# ---------------------------------------------------------------------------

@app.get(
    "/tasks/{task_id}/result",
    summary="Retrieve the full analysis result for a completed task",
    response_description=(
        "TaskResult payload when complete; status dict when still in-flight; "
        "status + error dict when failed."
    ),
)
def get_task_result(task_id: str):
    """Return the TaskResult JSON if the task is complete.

    Rather than forcing clients to hit two endpoints (status then result), this
    endpoint is self-describing: it always tells you what is going on regardless
    of the current state. That makes client polling loops simpler.

    Why no response_model here?
    The return type varies by status (partial dict vs full TaskResult), which
    doesn't map cleanly to a single Pydantic model. We return a plain dict in
    all cases and let FastAPI serialise it with its default JSONResponse.
    """
    row = get_task(task_id)

    if row is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")

    status = row["status"]

    # --- Task is still being processed: tell the client to keep waiting ---
    if status in ("queued", "processing"):
        return {"task_id": task_id, "status": status, "result": None}

    # --- Task failed: surface the error so the client can act on it ---
    if status == "failed":
        return {"task_id": task_id, "status": "failed", "error": row["error"]}

    # --- Task is complete: deserialise and return the full result ---
    if status == "complete":
        if row["result_json"] is None:
            # Defensive guard: should never happen if the processor uses
            # save_task_result() correctly, but we handle it explicitly rather
            # than returning a confusing null body.
            raise HTTPException(
                status_code=500,
                detail="Task is marked complete but result_json is missing — data integrity error.",
            )

        # json.loads converts the stored TEXT back to a Python dict.
        result_dict = json.loads(row["result_json"])

        # Validate through the Pydantic model. This catches schema drift: if the
        # pipeline wrote a result that no longer matches the current model
        # definition, we get a clear ValidationError rather than silently returning
        # malformed data.
        result = TaskResult.model_validate(result_dict)

        # model_dump(by_alias=True) is essential: it tells Pydantic to use the
        # Field(alias=...) values as JSON keys. Without it, DetectedObject would
        # output "object_class" instead of the required "class" key.
        return result.model_dump(by_alias=True)

    # Catch-all: if a status value was written to the DB that we don't recognise,
    # fail loudly rather than silently returning an empty response.
    raise HTTPException(
        status_code=500,
        detail=f"Unexpected task status '{status}' — check the database for corruption.",
    )
