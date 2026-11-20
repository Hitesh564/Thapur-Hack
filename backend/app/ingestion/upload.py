"""Upload validation and storage (MUST): type, decode, duration, dimensions, fps, audio."""
from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from pathlib import Path

import cv2

from app.config import get_config
from app.storage import read_json, video_dir, write_json


class IngestionError(ValueError):
    pass


def ffmpeg_exe() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _ffmpeg_info(path: Path) -> str:
    proc = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    return proc.stderr


def probe(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IngestionError("Could not open the file as a video.")
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise IngestionError("The video could not be decoded.")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    h, w = frame.shape[:2]
    cap.release()
    info = _ffmpeg_info(path)
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", info)
    if m:
        duration = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    elif fps > 0 and n_frames > 0:
        duration = n_frames / fps
    else:
        raise IngestionError("Could not determine the video duration.")
    has_audio = bool(re.search(r"Stream #\S+.*Audio:", info))
    return {"duration_s": round(duration, 3), "width": int(w), "height": int(h), "fps": round(fps, 3),
            "has_audio": has_audio, "n_frames": n_frames}


def ingest_file(src: Path, filename: str | None = None) -> dict:
    """Validate and store a video. Returns metadata with ``video_id`` (content hash prefix)."""
    cfg = get_config()
    src = Path(src)
    ext = Path(filename or src.name).suffix.lower()
    if ext not in cfg.get("video.allowed_ext"):
        raise IngestionError(f"Unsupported file type {ext!r}. Use one of {cfg.get('video.allowed_ext')}.")
    h = hashlib.sha256()
    with open(src, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    file_hash = h.hexdigest()
    video_id = file_hash[:16]
    d = video_dir(video_id)
    meta_path = d / "meta.json"
    cached = read_json(meta_path)
    if cached:
        return cached
    dest = d / f"source{ext}"
    if not dest.exists():
        shutil.copyfile(src, dest)
    try:
        meta = probe(dest)
    except IngestionError:
        shutil.rmtree(d, ignore_errors=True)
        raise
    lo, hi = float(cfg.get("video.min_seconds")), float(cfg.get("video.max_seconds"))
    if not lo <= meta["duration_s"] <= hi:
        shutil.rmtree(d, ignore_errors=True)
        raise IngestionError(f"Video is {meta['duration_s']:.1f}s; supported length is {lo:.0f} to {hi:.0f}s.")
    meta.update({"video_id": video_id, "file_hash": file_hash, "filename": filename or src.name,
                 "source_path": str(dest), "vertical": meta["height"] >= meta["width"]})
    write_json(meta_path, meta)
    return meta


def get_meta(video_id: str) -> dict:
    meta = read_json(video_dir(video_id) / "meta.json")
    if not meta:
        raise KeyError(video_id)
    return meta
