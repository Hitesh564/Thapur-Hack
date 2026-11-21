"""Tiny in-process background jobs for long Lab tasks (Hook Surgeon, seeding plan)."""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from typing import Any, Callable

_JOBS: dict[str, dict] = {}
_lock = threading.Lock()


def start(kind: str, fn: Callable[..., Any], **kwargs: Any) -> str:
    """Run ``fn(progress=..., **kwargs)`` in a thread; returns a job id."""
    job_id = f"{kind}-{uuid.uuid4().hex[:8]}"
    job = {"id": job_id, "kind": kind, "state": "running", "progress": "starting", "result": None,
           "error": None, "started": time.time(), "finished": None}
    with _lock:
        _JOBS[job_id] = job

    def progress(msg: str) -> None:
        job["progress"] = msg

    def run() -> None:
        try:
            job["result"] = fn(progress=progress, **kwargs)
            job["state"] = "done"
        except Exception as e:  # noqa: BLE001 - reported to the client
            job["state"] = "error"
            job["error"] = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        job["finished"] = time.time()

    threading.Thread(target=run, daemon=True).start()
    return job_id


def get(job_id: str) -> dict | None:
    with _lock:
        job = _JOBS.get(job_id)
    if job is None:
        return None
    out = dict(job)
    out["elapsed_s"] = round((job["finished"] or time.time()) - job["started"], 1)
    return out
