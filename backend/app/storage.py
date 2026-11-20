"""File-system storage: everything per video lives in cache/videos/<video_id>/."""
from __future__ import annotations

import json
import os
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
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    tmp.replace(path)
    return path


def read_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
