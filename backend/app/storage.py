"""File-system storage: everything per video lives in cache/videos/<video_id>/."""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

from app.config import BACKEND_DIR, get_config


def cache_root() -> Path:
    override = os.environ.get("VIRALYST_CACHE_DIR")
    root = Path(override) if override else BACKEND_DIR / str(get_config().get("video.cache_dir"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def video_dir(video_id: str) -> Path:
    d = cache_root() / "videos" / video_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_json(path: Path, obj: Any) -> Path:
    """Atomic write via a temp file. On Windows another process (a reader, antivirus) can hold the
    target briefly and make the rename fail, so retry, then fall back to a direct write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=1, default=str)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    for attempt in range(10):
        try:
            tmp.replace(path)
            return path
        except PermissionError:
            time.sleep(0.05 * (attempt + 1))
    path.write_text(text, encoding="utf-8")
    tmp.unlink(missing_ok=True)
    return path


def read_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
