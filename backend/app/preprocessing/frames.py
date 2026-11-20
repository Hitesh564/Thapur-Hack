"""Preprocessing (D1): frames (uniform + dense opening), cuts, motion, audio, transcript.

All results cached in the video's folder; a second call reads the cache.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.config import get_config
from app.ingestion.upload import ffmpeg_exe, get_meta
from app.storage import read_json, video_dir, write_json

PREPROCESS_VERSION = "1"


def _resize(frame: np.ndarray, max_px: int) -> np.ndarray:
    h, w = frame.shape[:2]
    s = max_px / max(h, w)
    return cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else frame


def _grab_at(cap: cv2.VideoCapture, t_s: float) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_MSEC, t_s * 1000.0)
    ok, frame = cap.read()
    return frame if ok else None


def extract_frames(src: Path, out_dir: Path, duration: float) -> dict:
    cfg = get_config()
    n_uni, n_open = int(cfg.get("video.uniform_frames")), int(cfg.get("video.opening_frames"))
    open_s = min(float(cfg.get("video.opening_seconds")), duration)
    max_px, q = int(cfg.get("vlm.max_frame_px")), int(cfg.get("vlm.jpeg_quality"))
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(src))
    res: dict[str, list] = {"frame_paths": [], "frame_times_s": [], "opening_frame_paths": [], "opening_frame_times_s": []}
    plan = [("u", (i + 0.5) * duration / n_uni) for i in range(n_uni)]
    plan += [("o", (j + 0.5) * open_s / n_open) for j in range(n_open)]
    last = None
    for kind, t in plan:
        frame = _grab_at(cap, t)
        if frame is None:
            frame = last
        if frame is None:
            continue
        last = frame
        p = out_dir / f"{kind}_{t:07.3f}.jpg"
        cv2.imwrite(str(p), _resize(frame, max_px), [cv2.IMWRITE_JPEG_QUALITY, q])
        key = "frame" if kind == "u" else "opening_frame"
        res[f"{key}_paths"].append(str(p))
        res[f"{key}_times_s"].append(round(t, 3))
    cap.release()
    return res


def scan_video(src: Path, duration: float) -> dict:
    """Cut detection (HSV histogram change) and motion energy (mean abs grey difference)."""
    cfg = get_config()
    scan_fps = float(cfg.get("video.cut_scan_fps"))
    thr, gap = float(cfg.get("video.cut_threshold")), float(cfg.get("video.cut_min_gap_s"))
    open_s = min(float(cfg.get("video.opening_seconds")), duration)
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or scan_fps
    step = max(1, int(round(fps / scan_fps)))
    prev_hist = prev_grey = None
    cuts: list[float] = []
    motion, motion_t, bright = [], [], []
    idx = 0
    while True:
        if not cap.grab():
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            t = idx / fps
            small = cv2.resize(frame, (96, 96), interpolation=cv2.INTER_AREA)
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            bright.append(float(grey.mean()))
            if prev_hist is not None:
                change = 1.0 - float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL))
                if change > thr and (not cuts or t - cuts[-1] >= gap) and t >= gap:
                    cuts.append(round(t, 3))
                motion.append(float(np.abs(grey - prev_grey).mean()))
                motion_t.append(t)
            prev_hist, prev_grey = hist, grey
        idx += 1
    cap.release()
    motion_a, mt = np.array(motion or [0.0]), np.array(motion_t or [0.0])
    open_mask = mt <= open_s
    n_open_cuts = sum(1 for c in cuts if c <= open_s)
    return {
        "cut_times_s": cuts,
        "features": {
            "duration_s": float(duration),
            "cut_rate": len(cuts) / max(duration, 1e-9),
            "opening_cut_rate": n_open_cuts / max(open_s, 1e-9),
            "motion_energy": float(motion_a.mean()),
            "opening_motion": float(motion_a[open_mask].mean()) if open_mask.any() else float(motion_a.mean()),
            "brightness": float(np.mean(bright)) if bright else 0.0,
        },
    }


def extract_audio(src: Path, out: Path) -> Optional[Path]:
    if out.exists():
        return out
    proc = subprocess.run([ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", "-i", str(src), "-vn",
                           "-ac", "1", "-ar", "16000", "-b:a", "48k", str(out)], capture_output=True)
    return out if proc.returncode == 0 and out.exists() and out.stat().st_size > 0 else None


def transcribe(audio: Path, d: Path) -> tuple[str, str]:
    """Returns (transcript, note). Cached; never raises."""
    cached = read_json(d / "transcript.json")
    if cached is not None:
        return cached["text"], cached.get("note", "")
    from app.video_analysis.vlm import VLMUnavailable, get_client
    model = str(get_config().get("vlm.transcribe_model"))
    try:
        client = get_client()
        with open(audio, "rb") as fh:
            text = client.audio.transcriptions.create(model=model, file=fh).text.strip()
        write_json(d / "transcript.json", {"text": text, "model": model})
        return text, ""
    except VLMUnavailable as e:
        return "", f"Transcript unavailable ({e})."
    except Exception as e:  # noqa: BLE001 - external call; fall back to no transcript
        return "", f"Transcript unavailable ({type(e).__name__})."


def preprocess(video_id: str, with_transcript: bool = True) -> dict:
    d = video_dir(video_id)
    cached = read_json(d / "preprocess.json")
    if cached and cached.get("version") == PREPROCESS_VERSION and (cached.get("transcript_done") or not with_transcript):
        cached["cache_hit"] = True
        return cached
    meta = get_meta(video_id)
    src = Path(meta["source_path"])
    out = {"version": PREPROCESS_VERSION, **meta}
    out.update(extract_frames(src, d / "frames", meta["duration_s"]))
    out.update(scan_video(src, meta["duration_s"]))
    out["features"]["has_audio"] = float(meta["has_audio"])
    out["transcript"], notes = "", []
    if meta["has_audio"] and with_transcript:
        audio = extract_audio(src, d / "audio.mp3")
        if audio:
            out["transcript"], note = transcribe(audio, d)
            if note:
                notes.append(note)
        else:
            notes.append("Audio could not be extracted.")
    out["notes"] = notes
    out["transcript_done"] = with_transcript
    write_json(d / "preprocess.json", out)
    out["cache_hit"] = False
    return out
