"""The prior p (D3). Interface: prior(profile) -> PriorResult. The engine only uses
``percentile`` and ``ensemble``.

* vlm_rubric (MODEL_RATED): rubric prompt x ``prior.vlm_samples``; p_i = mean(scores)/rubric_max;
  percentile = clip(mean(p_i)); ensemble = [p_i].
* heuristic (ASSUMED): fallback from basic features, centred near ``prior.heuristic.base``:
    p = base + w_duration * (+1 if duration in ideal band else -1)
             + w_cuts   * tanh(opening_cut_rate / ref_cut_rate - 1)
             + w_motion * tanh(opening_motion   / ref_motion   - 1)
             + w_audio  * (+1 if audio else -1)
  ensemble = [p - spread, p, p + spread], all clipped.
"""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
from pydantic import BaseModel, model_validator

from app.config import get_config
from app.storage import read_json, video_dir, write_json
from app.video_analysis.vlm import VLMUnavailable, call_json, load_prompt
from schemas import EvidenceTag, PriorResult, VideoProfile

PRIOR_VERSION = "1"


def _clip(p: float) -> float:
    cfg = get_config()
    return float(np.clip(p, cfg.get("prior.clip_min"), cfg.get("prior.clip_max")))


def heuristic_prior(profile: VideoProfile, note: str = "") -> PriorResult:
    cfg = get_config()
    H = lambda k: cfg.get(f"prior.heuristic.{k}")  # noqa: E731
    f = profile.features
    lo, hi = H("ideal_duration_s")
    dur = f.get("duration_s", profile.duration_s)
    p = float(H("base"))
    p += float(H("w_duration")) * (1 if lo <= dur <= hi else -1)
    p += float(H("w_cuts")) * np.tanh(f.get("opening_cut_rate", 0.0) / float(H("ref_cut_rate")) - 1)
    p += float(H("w_motion")) * np.tanh(f.get("opening_motion", 0.0) / float(H("ref_motion")) - 1)
    p += float(H("w_audio")) * (1 if profile.has_audio else -1)
    s = float(H("spread"))
    return PriorResult(percentile=_clip(p), ensemble=[_clip(p - s), _clip(p), _clip(p + s)], method="heuristic",
                       evidence_tag=EvidenceTag.ASSUMED, model="heuristic",
                       note=note or "Heuristic fallback from basic video features; not a model rating.")


class RubricResponse(BaseModel):
    scores: dict[str, float]
    rationale: str = ""

    @model_validator(mode="after")
    def _check(self):
        cfg = get_config()
        dims = list(cfg.get("prior.rubric"))
        rmin, rmax = float(cfg.get("prior.rubric_min")), float(cfg.get("prior.rubric_max"))
        if sorted(self.scores) != sorted(dims):
            raise ValueError(f"rubric keys {sorted(self.scores)} != {sorted(dims)}")
        if any(not rmin <= v <= rmax for v in self.scores.values()):
            raise ValueError("rubric score out of range")
        return self


def frame_items(profile: VideoProfile) -> list[tuple[str, str]]:
    items = [(f"Opening frame at {t:.2f}s:", p) for t, p in zip(profile.opening_frame_times_s, profile.opening_frame_paths)]
    items += [(f"Frame at {t:.2f}s:", p) for t, p in zip(profile.frame_times_s, profile.frame_paths)]
    return items


def rubric_prompt(profile: VideoProfile) -> str:
    cfg = get_config()
    return load_prompt("rubric.md").format(
        rmin=cfg.get("prior.rubric_min"), rmax=cfg.get("prior.rubric_max"), duration=profile.duration_s,
        n_cuts=len(profile.cut_times_s), transcript=profile.transcript or "(no speech)")


def _cache_key(profile: VideoProfile, prompt: str) -> str:
    cfg = get_config()
    payload = json.dumps({"v": PRIOR_VERSION, "h": profile.file_hash, "prompt": prompt, "model": cfg.get("vlm.model"),
                          "n": cfg.get("prior.vlm_samples")}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def vlm_rubric_prior(profile: VideoProfile) -> PriorResult:
    cfg = get_config()
    prompt = rubric_prompt(profile)
    path = video_dir(profile.video_id) / f"prior_vlm_{_cache_key(profile, prompt)}.json"
    cached = read_json(path)
    if cached:
        return PriorResult.model_validate(cached)
    n = int(cfg.get("prior.vlm_samples"))
    items = frame_items(profile)
    with ThreadPoolExecutor(max_workers=n) as ex:
        results = list(ex.map(lambda _: call_json(prompt, items, RubricResponse), range(n)))
    rmax = float(cfg.get("prior.rubric_max"))
    ens = [float(np.mean(list(r.scores.values())) / rmax) for r, _ in results]
    dims = list(cfg.get("prior.rubric"))
    mean_scores = {d: float(np.mean([r.scores[d] for r, _ in results])) for d in dims}
    models = sorted({m for _, m in results})
    out = PriorResult(percentile=_clip(float(np.mean(ens))), ensemble=[_clip(e) for e in ens], method="vlm_rubric",
                      rubric_scores=mean_scores, evidence_tag=EvidenceTag.MODEL_RATED, model=",".join(models),
                      rationale=results[0][0].rationale,
                      note=f"A model's judgement ({n} samples); not trained or validated against real outcomes.")
    write_json(path, out.model_dump(mode="json"))
    return out


def prior(profile: VideoProfile, method: Optional[str] = None) -> PriorResult:
    method = method or str(get_config().get("prior.method"))
    if method == "heuristic":
        return heuristic_prior(profile)
    try:
        return vlm_rubric_prior(profile)
    except VLMUnavailable as e:
        return heuristic_prior(profile, note=f"VLM unavailable, heuristic fallback used ({str(e)[:160]}).")
