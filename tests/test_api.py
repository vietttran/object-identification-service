"""
pytest tests for app/main.py — FastAPI endpoints via TestClient.

These tests exercise the HTTP layer (routing, request parsing, status codes,
response bodies) without invoking YOLO or touching the real task database.

Two isolation mechanisms are used:
  1. Database isolation: DATABASE_PATH is redirected to a temp file so tests
     never read or write the production tasks.db file.
  2. Pipeline isolation: _run_pipeline is replaced with a no-op lambda so YOLO
     is never loaded. The task stays in 'queued' status throughout the test, which
     is sufficient to verify the route handler and status/result endpoint logic.

Why TestClient (not httpx or requests against a live server)?
TestClient drives the ASGI app in-process, with no network I/O. This makes tests:
  - Fast (no port binding, no kernel TCP roundtrip)
  - Reliable (no race conditions from async background processing)
  - Portable (no free port required, works in CI sandboxes)
TestClient executes background tasks synchronously before returning the response,
so by the time client.post() returns, the mocked _run_pipeline has already been
called (and done nothing, since it is a lambda no-op).

Run:
    pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient

import app.database as db_mod
import app.main as main_mod
from app.main import app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    """
    Return an isolated TestClient.

    Steps:
      1. Redirect DATABASE_PATH to a temp file so every test starts with a
         fresh, empty schema and never contaminates tasks.db.
      2. Create the uploads directory inside tmp_path so the route handler
         can write the uploaded file (even for our tiny fake payloads).
      3. Redirect UPLOAD_DIR in app.main so the route handler writes to the
         temp directory rather than data/uploads/.
      4. Replace _run_pipeline with a no-op so YOLO is never loaded.
         The task record is created in the DB by the route handler BEFORE
         add_task is called, so the record exists with status='queued' even
         though the pipeline never runs.
      5. Enter the TestClient context manager, which triggers the lifespan
         startup → init_db() → CREATE TABLE IF NOT EXISTS in the temp DB.

    monkeypatch is a pytest built-in that reverts all patches after each test,
    so tests are independent even though they share this fixture.
    """
    # 1. Redirect the SQLite database to a per-test temp file.
    #    db_mod.DATABASE_PATH is read by _get_connection() at call time (not at
    #    import time), so patching the module attribute here is sufficient.
    monkeypatch.setattr(db_mod, "DATABASE_PATH", str(tmp_path / "test_tasks.db"))

    # 2 & 3. Redirect the upload directory so the route handler writes
    #         uploaded files to our temp dir instead of the real data/uploads/.
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    monkeypatch.setattr(main_mod, "UPLOAD_DIR", upload_dir)

    # 4. Mock _run_pipeline so YOLO is never invoked.
    #    When create_analysis_task runs `background_tasks.add_task(_run_pipeline, ...)`,
    #    Python looks up '_run_pipeline' in app.main's global namespace at call time.
    #    Patching app.main._run_pipeline here updates that namespace entry, so the
    #    lambda is passed to add_task instead of the real function.
    monkeypatch.setattr(main_mod, "_run_pipeline", lambda task_id, video_path: None)

    # 5. TestClient's context manager triggers the lifespan startup (init_db).
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _fake_video_upload(filename: str = "test_video.mp4", content: bytes = b"fake video content"):
    """
    Build the files dict for a multipart upload to POST /tasks.

    TestClient's files= parameter accepts the same format as the `requests` library:
        {"field_name": (filename, content_bytes, content_type)}

    Why use fake (non-video) content?
    The route handler only saves the file to disk — it doesn't validate the
    content. Actual video decoding happens inside _run_pipeline, which we have
    mocked to a no-op. So sending bytes b"fake video content" is enough to
    exercise the upload path without needing a real MP4 file in the test suite.
    """
    return {"file": (filename, content, "video/mp4")}


# ---------------------------------------------------------------------------
# POST /tasks — upload a video and create a task
# ---------------------------------------------------------------------------

def test_upload_creates_task_returns_201_with_task_id(client):
    """
    Uploading a file to POST /tasks should return HTTP 201 Created with a JSON
    body containing a non-empty 'task_id' string and 'status': 'queued'.

    This verifies:
      - The route is reachable.
      - The file upload is handled without error.
      - The task is persisted (task_id is generated, DB row created).
      - The initial status is 'queued' before any processing starts.
      - The status code is 201, not 200 (resource creation semantics).
    """
    response = client.post("/tasks", files=_fake_video_upload())

    assert response.status_code == 201

    body = response.json()
    assert "task_id" in body
    assert isinstance(body["task_id"], str) and len(body["task_id"]) > 0
    assert body["status"] == "queued"


# ---------------------------------------------------------------------------
# GET /tasks/{task_id} — poll task status
# ---------------------------------------------------------------------------

def test_get_task_status_returns_queued_for_newly_created_task(client):
    """
    After creating a task, GET /tasks/{task_id} should return a valid TaskStatus
    object with status='queued'.

    Why does it stay 'queued' and not advance to 'processing' or 'complete'?
    TestClient runs background tasks synchronously, so _run_pipeline is called
    before client.post() returns. But our mocked _run_pipeline is a no-op that
    does NOT call update_task_status(), so the row in the DB retains 'queued'.
    This is the expected behaviour when the pipeline is mocked out.

    This test verifies:
      - GET /tasks/{id} returns 200 for a known task ID.
      - The response body contains all required TaskStatus fields.
      - The 'status' value is one of the valid lifecycle states.
    """
    create_response = client.post("/tasks", files=_fake_video_upload())
    task_id = create_response.json()["task_id"]

    status_response = client.get(f"/tasks/{task_id}")
    assert status_response.status_code == 200

    body = status_response.json()
    assert body["task_id"] == task_id
    assert body["status"] in {"queued", "processing", "complete", "failed"}
    # For a mocked pipeline the status should be 'queued'.
    assert body["status"] == "queued"
    # created_at must be present (auto-set by create_task in database.py).
    assert "created_at" in body


# ---------------------------------------------------------------------------
# GET /tasks/{task_id}/result — retrieve the analysis result
# ---------------------------------------------------------------------------

def test_get_result_while_queued_returns_null_result_field(client):
    """
    Hitting GET /tasks/{task_id}/result before the pipeline finishes should return
    a 200 response indicating the task is still in-flight, with result=null.

    This is the in-flight status path in get_task_result():
        if status in ('queued', 'processing'):
            return {'task_id': ..., 'status': ..., 'result': None}

    Clients should interpret result=null as 'keep polling'. The status field tells
    them whether to expect the result soon ('processing') or whether processing has
    not even started ('queued').
    """
    create_response = client.post("/tasks", files=_fake_video_upload())
    task_id = create_response.json()["task_id"]

    result_response = client.get(f"/tasks/{task_id}/result")
    assert result_response.status_code == 200

    body = result_response.json()
    assert body["task_id"] == task_id
    assert body["status"] == "queued"
    assert body["result"] is None


# ---------------------------------------------------------------------------
# 404 cases — unknown task ID
# ---------------------------------------------------------------------------

def test_get_unknown_task_id_returns_404(client):
    """
    GET /tasks/{task_id} with a task_id that does not exist in the DB should
    return HTTP 404 Not Found.

    This exercises the 'row is None → raise HTTPException(404)' branch in
    get_task_status(). It ensures the API gives a clear error rather than
    crashing or returning an empty 200.

    The all-zeros UUID is used as a sentinel value that is almost certainly not
    in any real database, without needing to guess a random UUID.
    """
    nonexistent_id = "00000000-0000-0000-0000-000000000000"
    response = client.get(f"/tasks/{nonexistent_id}")

    assert response.status_code == 404


def test_get_result_for_unknown_task_returns_404(client):
    """
    GET /tasks/{task_id}/result for a missing task should also return 404.

    This verifies both the status endpoint and the result endpoint consistently
    raise 404 for unknown IDs, rather than one of them returning a confusing
    200 with an empty body.
    """
    nonexistent_id = "00000000-0000-0000-0000-000000000000"
    response = client.get(f"/tasks/{nonexistent_id}/result")

    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Multiple independent tasks — no cross-contamination
# ---------------------------------------------------------------------------

def test_two_tasks_have_different_task_ids(client):
    """
    Creating two tasks in sequence should yield two distinct task IDs.

    This verifies that task IDs are generated freshly per request (via uuid4)
    rather than reused or incremented from a shared counter that could conflict
    across concurrent requests.
    """
    r1 = client.post("/tasks", files=_fake_video_upload("video1.mp4"))
    r2 = client.post("/tasks", files=_fake_video_upload("video2.mp4"))

    assert r1.status_code == 201
    assert r2.status_code == 201

    id1 = r1.json()["task_id"]
    id2 = r2.json()["task_id"]
    assert id1 != id2, "Each task must receive a unique ID"
