"""Qualitative analysis (D2): one VLM call, validated, cached. Display and explanation only:
hook fields never feed the engine."""
from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, Field

from app.config import get_config
from app.personas.population import load_archetypes
from app.storage import read_json, video_dir, write_json
from app.video_analysis.vlm import VLMUnavailable, call_json, load_prompt
from schemas import HookAnalysis, VideoProfile

ANALYSIS_VERSION = "1"


class AnalysisResponse(BaseModel):
    topic: str
    category: str
    subtopics: list[str] = []
    summary: str
    format: str = ""
    narrative_structure: str = ""
    audience_interest_tags: list[str] = Field(default_factory=list, max_length=12)
    hook: HookAnalysis


def community_names() -> list[str]:
    arcs, _ = load_archetypes()
    names: dict[int, str] = {a.community: a.community_name for a in arcs}
    return [names[k] for k in sorted(names)]


def analysis_prompt(profile: VideoProfile) -> str:
    return load_prompt("analysis.md").format(
        categories=", ".join(f'"{c}"' for c in community_names()), duration=profile.duration_s,
        cuts=", ".join(f"{c:.1f}s" for c in profile.cut_times_s[:30]) or "none",
        transcript=profile.transcript or "(no speech)")


def analyze(profile: VideoProfile) -> VideoProfile:
    """Return a copy of ``profile`` with qualitative fields filled (or a note if unavailable)."""
    from app.prior.prior import frame_items
    prompt = analysis_prompt(profile)
    key = hashlib.sha256(json.dumps({"v": ANALYSIS_VERSION, "h": profile.file_hash, "p": prompt,
                                     "m": get_config().get("vlm.model")}).encode()).hexdigest()[:12]
    path = video_dir(profile.video_id) / f"analysis_{key}.json"
    cached = read_json(path)
    source = "cache"
    if cached:
        res = AnalysisResponse.model_validate(cached["analysis"])
        source = f"cache:{cached['model']}"
    else:
        try:
            res, model = call_json(prompt, frame_items(profile), AnalysisResponse)
        except VLMUnavailable as e:
            out = profile.model_copy(deep=True)
            out.analysis_source = "none"
            out.notes = out.notes + [f"Qualitative analysis unavailable; showing basic features only ({str(e)[:160]})."]
            return out
        write_json(path, {"model": model, "analysis": res.model_dump(mode="json")})
        source = f"vlm:{model}"
    out = profile.model_copy(deep=True, update={**res.model_dump(exclude={"hook"}), "hook": res.hook})
    out.analysis_source = source
    return out
