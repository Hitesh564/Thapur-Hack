"""Generate archetypes once with an LLM (D4), validate, save to data/archetypes.json.

Preprocessing only (never inside the simulation). The hand-written fallback file stays as
the offline source. Run:  python -m app.personas.generate
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import date

from pydantic import BaseModel

from app.config import get_config
from app.personas.population import ARCHETYPES_PATH, FALLBACK_PATH, load_archetypes
from app.storage import write_json
from schemas import TRAIT_KEYS, Archetype

SENSITIVE = ("ethnic", "race", "racial", "religio", "church", "mosque", "temple", "politic", "party",
             "health", "disease", "illness", "disab", "sexual", "gender", "caste", "pregnan", "medical")

PROMPT = """Create {n} fictional audience archetypes for a short-form video simulation.
They must cover these {k} interest communities (community index: name): {communities}.
Give each community between {lo} and {hi} archetypes, {n} in total, covering a broad mix of content
(not only tech). Each archetype describes viewing habits only.

Rules:
- No sensitive traits of any kind: no ethnicity, religion, politics, health or medical conditions, sexuality, gender, age.
- "interests": {imin} to {imax} short content-topic phrases (2-4 words), each with a strength in [0, 1].
- "traits": ranges [lo, hi] with lo < hi for: attention (in [0, 1]: how patiently they watch), and
  like, comment, save, share tendencies as multipliers around 1.0 (each in [{tmin}, {tmax}]).
  Vary them: some lurkers (low like/comment/share), engaged commenters, heavy sharers, focused savers, fast scrollers.
- "influence_class": "low", "medium" or "high" (most are low or medium).
- Names are short role descriptions, unique, e.g. "Weeknight Home Cook".

Return JSON only: {{"archetypes": [{{"name": str, "community": int, "community_name": str,
 "interests": [{{"phrase": str, "strength": float}}], "traits": {{"attention": [lo, hi], "like": [lo, hi],
 "comment": [lo, hi], "save": [lo, hi], "share": [lo, hi]}}, "influence_class": str}}]}}"""


class ArchetypeSet(BaseModel):
    archetypes: list[Archetype]


def validate_set(arcs: list[Archetype], communities: list[str]) -> list[str]:
    """Return a list of problems (empty = valid)."""
    cfg = get_config()
    C = lambda k: cfg.get(f"personas.{k}")  # noqa: E731
    probs = []
    n = int(cfg.get("population.n_archetypes"))
    if len(arcs) != n:
        probs.append(f"expected {n} archetypes, got {len(arcs)}")
    counts = Counter(a.community for a in arcs)
    for i, name in enumerate(communities):
        if not C("min_per_community") <= counts.get(i, 0) <= C("max_per_community"):
            probs.append(f"community {i} has {counts.get(i, 0)} archetypes")
    if set(counts) - set(range(len(communities))):
        probs.append("unknown community index")
    if len({a.name for a in arcs}) != len(arcs):
        probs.append("duplicate names")
    for a in arcs:
        if a.community < len(communities) and a.community_name != communities[a.community]:
            probs.append(f"{a.name}: community_name mismatch")
        if not C("min_interests") <= len(a.interests) <= C("max_interests"):
            probs.append(f"{a.name}: {len(a.interests)} interests")
        if set(a.traits) != set(TRAIT_KEYS):
            probs.append(f"{a.name}: trait keys {sorted(a.traits)}")
            continue
        for k, (lo, hi) in a.traits.items():
            bound = (0.0, 1.0) if k == "attention" else (float(C("tendency_min")), float(C("tendency_max")))
            if not (bound[0] <= lo < hi <= bound[1]):
                probs.append(f"{a.name}: {k} range {lo}-{hi}")
    text = json.dumps([a.model_dump() for a in arcs]).lower()
    bad = [w for w in SENSITIVE if w in text]
    if bad:
        probs.append(f"sensitive words: {bad}")
    return probs


def generate(save: bool = True) -> tuple[list[Archetype], dict]:
    from app.video_analysis.vlm import call_json
    cfg = get_config()
    fallback, _ = load_archetypes(FALLBACK_PATH)
    names = {a.community: a.community_name for a in fallback}
    communities = [names[i] for i in sorted(names)]
    prompt = PROMPT.format(
        n=cfg.get("population.n_archetypes"), k=len(communities),
        communities="; ".join(f"{i}: {c}" for i, c in enumerate(communities)),
        lo=cfg.get("personas.min_per_community"), hi=cfg.get("personas.max_per_community"),
        imin=cfg.get("personas.min_interests"), imax=cfg.get("personas.max_interests"),
        tmin=cfg.get("personas.tendency_min"), tmax=cfg.get("personas.tendency_max"))
    last: list[str] = []
    for attempt in range(int(cfg.get("personas.generation_attempts"))):
        p = prompt if not last else prompt + "\n\nYour previous answer had these problems, fix them: " + "; ".join(last[:15])
        res, model = call_json(p, [], ArchetypeSet)
        last = validate_set(res.archetypes, communities)
        if not last:
            meta = {"model": model, "date": date.today().isoformat(), "attempts": attempt + 1,
                    "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest()[:12], "source": "llm"}
            if save:
                ARCHETYPES_PATH.write_text(json.dumps([a.model_dump() for a in res.archetypes], indent=1), encoding="utf-8")
                write_json(ARCHETYPES_PATH.with_suffix(".meta.json"), meta)
            return res.archetypes, meta
    raise RuntimeError(f"archetype generation failed validation: {last[:10]}")


if __name__ == "__main__":
    arcs, meta = generate()
    print(f"saved {len(arcs)} archetypes from {meta['model']} in {meta['attempts']} attempt(s)")
