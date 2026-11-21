"""Hook Surgeon (USP): generate real re-cut variants of the video from its own shots, run each
through the identical pipeline (GPT rubric prior, CLIP relevance) and a matched experiment against
the original, and rank the edits by robustness.

Edit recipes (deterministic, from detected cuts and the cached attention timeline):
* payoff_first      open on the final shot, then the rest in order
* teaser            flash-forward ``surgeon.teaser_seconds`` of the payoff, then the full video
* drop_drag         remove the shot the attention timeline flags as a drag
* best_shot_first   move the shot the rating leans on most to the front
* trim_slow_open    remove a low-motion first shot
* swap_first_two    swap the first two shots
Every result is a simulated what-if under stated assumptions, labelled as such.
"""
from __future__ import annotations

import glob
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

from app.config import get_config
from app.ingestion.upload import ffmpeg_exe, get_meta, ingest_file
from app.storage import read_json, video_dir, write_json

Seg = tuple[float, float]


def shots(cuts: list[float], duration: float, min_len: float) -> list[Seg]:
    bounds = [0.0] + [c for c in sorted(cuts) if 0 < c < duration] + [duration]
    segs: list[Seg] = []
    for s, e in zip(bounds, bounds[1:]):
        if segs and (e - s < min_len or segs[-1][1] - segs[-1][0] < min_len):
            segs[-1] = (segs[-1][0], e)
        else:
            segs.append((s, e))
    return segs


def _seg_at(segs: list[Seg], t: float) -> int:
    return next((i for i, (s, e) in enumerate(segs) if s <= t < e), len(segs) - 1)


def edit_recipes(video_id: str, pre: dict) -> list[dict]:
    cfg = get_config()
    dur = float(pre["duration_s"])
    segs = shots(pre.get("cut_times_s", []), dur, float(cfg.get("surgeon.min_segment_s")))
    min_s = float(cfg.get("video.min_seconds"))
    out: list[dict] = []
    if len(segs) >= 2:
        out.append({"name": "payoff_first", "title": "Payoff first",
                    "description": "Open on the final shot, then play the rest in order.",
                    "segments": [segs[-1]] + segs[:-1]})
        t = float(cfg.get("surgeon.teaser_seconds"))
        last = segs[-1]
        out.append({"name": "teaser", "title": f"{t:g}s teaser",
                    "description": f"Flash-forward {t:g}s of the payoff, then the full video.",
                    "segments": [(max(last[0], last[1] - t), last[1])] + [(0.0, dur)]})
    att = _latest_attention(video_id)
    if att and len(segs) >= 2:
        drag = [w for w in att["windows"] if w["change"] < 0 and not w["within_noise"]]
        if drag:
            w = min(drag, key=lambda x: x["change"])
            i = _seg_at(segs, (w["start_s"] + w["end_s"]) / 2)
            kept = segs[:i] + segs[i + 1:]
            if sum(e - s for s, e in kept) >= min_s:
                out.append({"name": "drop_drag", "title": "Cut the drag",
                            "description": f"Remove the shot at {segs[i][0]:.1f}-{segs[i][1]:.1f}s that the rating sees as a drag.",
                            "segments": kept})
        sup = [w for w in att["windows"] if w["change"] > 0 and not w["within_noise"]]
        if sup:
            w = max(sup, key=lambda x: x["change"])
            i = _seg_at(segs, (w["start_s"] + w["end_s"]) / 2)
            if i > 0:
                out.append({"name": "best_shot_first", "title": "Strongest shot first",
                            "description": f"Move the shot at {segs[i][0]:.1f}-{segs[i][1]:.1f}s, which the rating leans on most, to the front.",
                            "segments": [segs[i]] + segs[:i] + segs[i + 1:]})
    if len(segs) >= 2 and pre.get("features", {}).get("opening_motion", 1) < float(cfg.get("surgeon.slow_open_motion")):
        kept = segs[1:]
        if sum(e - s for s, e in kept) >= min_s:
            out.append({"name": "trim_slow_open", "title": "Trim the slow start",
                        "description": f"Remove the low-motion first shot (0-{segs[0][1]:.1f}s).", "segments": kept})
    if len(segs) >= 3:
        out.append({"name": "swap_first_two", "title": "Swap first two shots",
                    "description": "Swap the order of the first two shots.", "segments": [segs[1], segs[0]] + segs[2:]})
    seen, uniq = set(), []
    for r in out:
        key = tuple((round(s, 2), round(e, 2)) for s, e in r["segments"])
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    return uniq[: int(cfg.get("surgeon.max_variants"))]


def _latest_attention(video_id: str) -> Optional[dict]:
    files = sorted(glob.glob(str(video_dir(video_id) / "attention_*.json")), key=lambda f: Path(f).stat().st_mtime)
    return read_json(Path(files[-1])) if files else None


def render(src: Path, segments: list[Seg], out: Path, has_audio: bool) -> Path:
    if out.exists():
        return out
    parts, labels = [], []
    for i, (s, e) in enumerate(segments):
        parts.append(f"[0:v]trim=start={s:.3f}:end={e:.3f},setpts=PTS-STARTPTS[v{i}]")
        labels.append(f"[v{i}]")
        if has_audio:
            parts.append(f"[0:a]atrim=start={s:.3f}:end={e:.3f},asetpts=PTS-STARTPTS[a{i}]")
            labels.append(f"[a{i}]")
    k = len(segments)
    parts.append("".join(labels) + f"concat=n={k}:v=1:a={1 if has_audio else 0}[v]" + ("[a]" if has_audio else ""))
    cmd = [ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
           "-filter_complex", ";".join(parts), "-map", "[v]"]
    if has_audio:
        cmd += ["-map", "[a]", "-c:a", "aac"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(out)]
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True, capture_output=True)
    return out


def run_surgeon(video_id: str, progress: Optional[Callable[[str], None]] = None) -> dict:
    from app.api.service import run_experiment
    from app.pipeline import analyze_video, base_profile
    cfg = get_config()
    say = progress or (lambda _m: None)
    meta = get_meta(video_id)
    base_profile(video_id)  # ensure preprocessing exists
    pre = read_json(video_dir(video_id) / "preprocess.json") or {}
    recipes = edit_recipes(video_id, pre)
    if not recipes:
        raise ValueError("Not enough distinct shots to build edits (needs at least 2 detected cuts).")
    say(f"Rendering {len(recipes)} edits")
    stem = Path(meta["filename"]).stem
    for r in recipes:
        out = render(Path(meta["source_path"]), r["segments"], video_dir(video_id) / "edits" / f"{r['name']}.mp4", meta["has_audio"])
        vmeta = ingest_file(out, f"{stem}__{r['name']}.mp4")
        vmeta.update({"derived_from": video_id, "edit": r["name"], "edit_title": r["title"]})
        write_json(video_dir(vmeta["video_id"]) / "meta.json", vmeta)
        r["video_id"] = vmeta["video_id"]
        r["duration_s"] = vmeta["duration_s"]

    say(f"Analysing {len(recipes)} edits with the same rubric prompt and CLIP")
    with ThreadPoolExecutor(max_workers=len(recipes)) as ex:
        bundles = list(ex.map(lambda r: analyze_video(r["video_id"], with_analysis=False), recipes))
    for i, (r, b) in enumerate(zip(recipes, bundles)):
        r["prior"] = b["prior"]["percentile"]
        r["prior_method"] = b["prior"]["method"]
        r["rubric_scores"] = b["prior"]["rubric_scores"]
        say(f"Matched experiment {i + 1} of {len(recipes)}: {r['title']}")
        rec = run_experiment("hook_clip", video_id, r["video_id"], n_seeds=int(cfg.get("surgeon.n_seeds")),
                             n_samples=int(cfg.get("surgeon.n_samples")))
        r["experiment_id"] = rec["result"]["experiment_id"]
        r["result"] = rec["result"]
        r["voi_recommendation"] = rec["voi"]["recommendation"]
        r["segments"] = [list(s) for s in r["segments"]]

    rank = {"Robust": 0, "Likely": 1, "Inconclusive": 2}
    recipes.sort(key=lambda r: (rank[r["result"]["label"]],
                                -((r["result"]["milestone_variant"] or 0) - (r["result"]["milestone_baseline"] or 0)),
                                -r["result"]["win_rate"]))
    base = recipes[0]["result"]["milestone_baseline"] if recipes else None
    top = recipes[0]
    gain = (top["result"]["milestone_variant"] or 0) - (base or 0)
    if gain <= 0:
        msg = "None of the automatic edits improves on the original, in this simulation."
    elif top["result"]["label"] == "Inconclusive":
        msg = (f"Most promising edit: {top['title']} (milestone chance {base:.0%} to "
               f"{top['result']['milestone_variant']:.0%}, wins {top['result']['win_rate']:.0%} of paired runs), "
               f"but it is not robust across the assumption ranges yet, in this simulation.")
    else:
        msg = (f"Best edit: {top['title']}. The chance of reaching the milestone goes from {base:.0%} to "
               f"{top['result']['milestone_variant']:.0%} ({top['result']['label']}: wins {top['result']['win_rate']:.0%} "
               f"of paired runs), in this simulation.")
    out = {"video_id": video_id, "edits": recipes, "baseline_milestone": base, "message": msg,
           "note": "Edits are re-cuts of your own shots, each rated by the same prompt and tested on the same audience "
                   "and random draws. Simulated what-ifs, not predictions.",
           "evidence_tag": "SIMULATED"}
    write_json(video_dir(video_id) / "surgeon.json", out)
    return out


def latest_surgeon(video_id: str) -> Optional[dict]:
    return read_json(video_dir(video_id) / "surgeon.json")


__all__ = ["edit_recipes", "render", "run_surgeon", "latest_surgeon", "shots", "json"]
