"""Orchestration: upload -> preprocess -> (analysis || prior) -> relevance; world; runs.

Everything here is preprocessing or bookkeeping; the engine itself stays headless.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Optional

import numpy as np

from app.config import get_config
from app.engine import World
from app.graph import build_graph
from app.personas import sample_population
from app.preprocessing.frames import preprocess
from app.prior import prior as compute_prior
from app.relevance import compute_relevance, load_relevance, save_relevance
from app.storage import read_json, video_dir, write_json
from app.video_analysis import analyze
from schemas import PriorResult, RelevanceResult, VideoProfile


@lru_cache(maxsize=1)
def get_world() -> World:
    cfg = get_config()
    pop = sample_population(cfg)
    return World(pop, build_graph(pop, cfg))


def base_profile(video_id: str, with_transcript: bool = True) -> VideoProfile:
    pre = preprocess(video_id, with_transcript=with_transcript)
    fields = set(VideoProfile.model_fields)
    return VideoProfile.model_validate({k: v for k, v in pre.items() if k in fields})


def analyze_video(video_id: str, prior_method: Optional[str] = None, relevance_method: Optional[str] = None,
                  with_vlm: bool = True) -> dict:
    """Full preprocessing for one video. The alternate clip goes through this identical path."""
    profile = base_profile(video_id, with_transcript=with_vlm)
    if with_vlm:
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_an = ex.submit(analyze, profile)
            f_pr = ex.submit(compute_prior, profile, prior_method)
            profile, pr = f_an.result(), f_pr.result()
    else:
        from app.prior import heuristic_prior
        pr = heuristic_prior(profile, note="VLM not used for this analysis; heuristic prior.")
        profile.notes = profile.notes + ["Qualitative analysis skipped (offline)."]
    rel, rel_res = compute_relevance(profile, get_world().population, method=relevance_method)
    save_relevance(video_id, rel, rel_res)
    bundle = {"profile": profile.model_dump(mode="json"), "prior": pr.model_dump(mode="json"),
              "relevance": rel_res.model_dump(mode="json")}
    write_json(video_dir(video_id) / "analysis_bundle.json", bundle)
    return bundle


def get_analysis(video_id: str) -> Optional[tuple[VideoProfile, PriorResult, np.ndarray, RelevanceResult]]:
    bundle = read_json(video_dir(video_id) / "analysis_bundle.json")
    rel = load_relevance(video_id)
    if not bundle or rel is None:
        return None
    return (VideoProfile.model_validate(bundle["profile"]), PriorResult.model_validate(bundle["prior"]), rel[0], rel[1])
