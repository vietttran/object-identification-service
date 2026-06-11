"""FastAPI application entry point — defines routes and starts the ASGI server."""

# FastAPI is built on Starlette (the ASGI web framework) and Pydantic (validation).
# Uvicorn is the ASGI server that actually binds to a port and handles TCP connections.
# The relationship is: uvicorn → starlette (ASGI) → FastAPI (routing + validation).

import json
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile

from app.database import create_task, get_task, init_db, save_task_result, update_task_status
from app.models import TaskResult, TaskStatus
from app.processor import process_video

logger = logging.getLogger(__name__)


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
# Background pipeline job
# ---------------------------------------------------------------------------

def _run_pipeline(task_id: str, video_path: str) -> None:
    """Run the full analysis pipeline and persist the result.

    This function is invoked by FastAPI's BackgroundTasks mechanism after the
    POST /tasks HTTP response has already been sent to the client. The client
    gets an immediate 201 with a task_id and then polls GET /tasks/{task_id}
    until this function finishes and sets status to 'complete' or 'failed'.

    Why background processing instead of blocking the POST handler?

    Video analysis is slow — YOLO inference on a 60-second 1080p clip takes
    30–120 seconds depending on hardware. Blocking the HTTP request for that
    long would:
      - Time out most HTTP clients (default timeout is 30 s)
      - Tie up a thread-pool thread for the entire duration, reducing
        the server's ability to handle concurrent requests
      - Give the client no progress visibility

    The async task-id / polling pattern solves all three: the API returns in
    milliseconds, the client can show a progress indicator, and the server
    remains responsive for other requests while the pipeline runs in the
    background.

    Production upgrade: replace BackgroundTasks with Celery + Redis so that
    tasks survive server restarts, can be distributed across multiple workers,
    and can be queued without blocking any server thread at all.

    Why synchronous (def) rather than async (async def)?

    All pipeline operations — YOLO inference, OpenCV decoding, SQLite writes —
    are blocking I/O and CPU work. There is no benefit to making this async
    because we have nothing to await; every call needs the CPU until it returns.
    FastAPI's BackgroundTasks runs sync functions in the same thread-pool it uses
    for sync route handlers, which is the correct execution model here.

    Thread safety:
    Each call to _run_pipeline uses short-lived DB connections (see database.py),
    so multiple concurrent tasks running on different threads never share a
    connection. The Detector creates its own YOLO instance per call (see
    processor.py) so YOLO's ByteTrack internal state is never shared.
    """
    logger.info("[%s] Background pipeline starting.", task_id)

    try:
        # Immediately flip status so polling clients see "processing" rather
        # than "queued" while the video is being analysed.
        # If the server crashes between this line and save_task_result(), the
        # task will be stuck in "processing" on restart. A production system
        # would have a watchdog that re-queues stalled tasks — out of scope here.
        update_task_status(task_id, "processing")

        result_dict = process_video(task_id=task_id, video_path=video_path)

        # save_task_result writes result_json and atomically sets status="complete"
        # in the same UPDATE statement (see database.py), so there is no window
        # where status is "complete" but result_json is still NULL.
        save_task_result(task_id=task_id, result=result_dict)

        logger.info("[%s] Pipeline complete, result persisted.", task_id)

    except Exception as exc:
        # Catch everything: a corrupt video file, an unsupported codec, a YOLO
        # assertion error, or any bug in the pipeline. Setting status="failed"
        # with a readable error message means:
        #   - The server keeps running (the exception doesn't propagate up)
        #   - GET /tasks/{task_id} returns a useful error string to the client
        #   - We can diagnose the issue from the tasks.db error column
        error_msg = f"{type(exc).__name__}: {exc}"
        logger.exception("[%s] Pipeline failed: %s", task_id, error_msg)
        update_task_status(task_id, "failed", error=error_msg)


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
    # BackgroundTasks is a FastAPI special type — it is injected automatically
    # by the dependency system, not read from the request body. Listing it as
    # a parameter is sufficient; no decorator needed.
    background_tasks: BackgroundTasks,
    # UploadFile wraps the multipart file. FastAPI parses the Content-Type
    # boundary automatically when python-multipart is installed.
    file: UploadFile = File(..., description="Video file to analyse (e.g. MP4, AVI)"),
):
    """Accept an uploaded video, persist it, enqueue background analysis, and return immediately.

    The pipeline runs asynchronously — this handler returns in milliseconds.
    Clients should poll GET /tasks/{task_id} until status is 'complete' or 'failed',
    then fetch the full result from GET /tasks/{task_id}/result.
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
            while chunk := file.file.read(1024 * 1024):  # walrus: assign + test
                dest.write(chunk)
    except Exception as exc:
        # Surface upload errors as 500 so the client knows the task was NOT created.
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save uploaded file: {exc}",
        )

    # --- 4. Create the task record in SQLite ---
    # Must happen AFTER the file is safely on disk; we don't want a DB row
    # pointing to a file path that doesn't exist yet.
    create_task(task_id=task_id, video_path=str(video_path))

    # --- 5. Enqueue the background pipeline job ---
    # add_task registers _run_pipeline to be called after this HTTP response
    # is fully sent. The arguments are passed through as positional args.
    background_tasks.add_task(_run_pipeline, task_id, str(video_path))

    logger.info("[%s] Task created, pipeline queued. file=%s", task_id, file.filename)

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
