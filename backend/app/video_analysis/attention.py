"""Attention-risk timeline (D6, SHOULD): where the rating is sensitive.

Split the video into ``video.attention_windows`` windows. For each window, re-run the same
rubric prompt with that window's frames removed and report the change in the rated percentile:
    change_i = p(all frames) - p(without window i)
Positive = the window supports the rating; negative = the rating is higher without it.
This shows where the *model's rating* is sensitive, not where viewers drop off.
"""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from app.config import get_config
from app.prior.prior import RubricResponse, frame_items, rubric_prompt
from app.storage import read_json, video_dir, write_json
from app.video_analysis.vlm import VLMUnavailable, call_json
from schemas import PriorResult, VideoProfile


def _window_of(t: float, duration: float, n: int) -> int:
    return min(int(t / duration * n), n - 1)


def attention_timeline(profile: VideoProfile, baseline: PriorResult) -> dict:
    cfg = get_config()
    n_win = int(cfg.get("video.attention_windows"))
    n_samples = int(cfg.get("video.attention_samples"))
    rmax = float(cfg.get("prior.rubric_max"))
    prompt = rubric_prompt(profile)
    key = hashlib.sha256(json.dumps({"h": profile.file_hash, "p": prompt, "m": cfg.get("vlm.model"), "w": n_win,
                                     "s": n_samples, "base": baseline.ensemble}).encode()).hexdigest()[:12]
    path = video_dir(profile.video_id) / f"attention_{key}.json"
    cached = read_json(path)
    if cached:
        return cached
    if baseline.method != "vlm_rubric":
        raise VLMUnavailable("attention timeline needs the VLM rubric prior")
    items = frame_items(profile)
    times = list(profile.opening_frame_times_s) + list(profile.frame_times_s)
    win = [_window_of(t, profile.duration_s, n_win) for t in times]
    jobs = [(w, s) for w in range(n_win) for s in range(n_samples)]

    def run(job):
        w, _ = job
        kept = [it for it, wi in zip(items, win) if wi != w]
        r, _m = call_json(prompt, kept, RubricResponse)
        return w, float(np.mean(list(r.scores.values())) / rmax)

    with ThreadPoolExecutor(max_workers=min(len(jobs), 12)) as ex:
        results = list(ex.map(run, jobs))
    base_p = float(np.mean(baseline.ensemble))
    noise = float(np.std(baseline.ensemble)) if len(baseline.ensemble) > 1 else 0.0
    step = profile.duration_s / n_win
    windows = []
    for w in range(n_win):
        vals = [v for wi, v in results if wi == w]
        change = base_p - float(np.mean(vals))
        windows.append({
            "window": w, "start_s": round(w * step, 3), "end_s": round((w + 1) * step, 3),
            "frames_removed": int(sum(1 for wi in win if wi == w)),
            "p_without": float(np.mean(vals)), "change": change,
            "within_noise": bool(abs(change) <= max(noise, 1.0 / rmax / len(cfg.get("prior.rubric")))),
        })
    top = max(windows, key=lambda x: x["change"])
    low = min(windows, key=lambda x: x["change"])
    msgs = []
    if top["change"] > 0 and not top["within_noise"]:
        msgs.append(f"The rating leans most on {top['start_s']:.1f}-{top['end_s']:.1f}s: without those frames the "
                    f"model's rating drops by {top['change']:.3f}.")
    if low["change"] < 0 and not low["within_noise"]:
        msgs.append(f"The rating is higher without {low['start_s']:.1f}-{low['end_s']:.1f}s "
                    f"(+{-low['change']:.3f}); the model sees that section as a drag.")
    out = {
        "windows": windows, "baseline_p": base_p, "noise": noise, "most_sensitive_window": top["window"],
        "drag_window": low["window"] if low["change"] < 0 else None,
        "message": " ".join(msgs) or "No single window changes the model's rating beyond its own sampling noise.",
        "label": "Where the rating is sensitive (not where viewers drop off).",
        "evidence_tag": "MODEL_RATED",
    }
    write_json(path, out)
    return out
