"""Persona-video relevance (D5), COMPUTED. A zero-shot proxy for interest, not a measurement.

CLIP: viewer vector = strength-weighted mean of its archetype's interest-phrase text embeddings;
video vector = mean of frame image embeddings; cos = cosine similarity. Then
    rel = w_rel * sigmoid(k * (cos - mean(cos)) / std(cos))
        + (1 - w_rel) * clip((cos - cos_lo) / (cos_hi - cos_lo), 0, 1)
The second term keeps an absolute floor so a video relevant to nobody is not made relevant
by normalisation. Fallback (no CLIP): the cosine is replaced by word overlap between the
video's text fields and the interest phrases, with the same formula on a [0, 1] scale.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.special import expit

from app.config import get_config
from app.personas.population import Population
from app.storage import cache_root, read_json, video_dir, write_json
from schemas import Archetype, RelevanceResult, VideoProfile

_model = None
_model_lock = threading.Lock()


class ClipUnavailable(RuntimeError):
    pass


def _load_clip():
    global _model
    with _model_lock:
        if _model is None:
            try:
                import open_clip
                import torch
                cfg = get_config()
                name, pre = str(cfg.get("relevance.clip_model")), str(cfg.get("relevance.clip_pretrained"))
                model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pre)
                model.eval()
                _model = (model, preprocess, open_clip.get_tokenizer(name), torch, f"{name}/{pre}")
            except Exception as e:  # noqa: BLE001 - any failure -> keyword fallback
                raise ClipUnavailable(f"{type(e).__name__}: {str(e)[:120]}") from e
        return _model


def _norm(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def clip_tag() -> str:
    cfg = get_config()
    return f"{cfg.get('relevance.clip_model')}/{cfg.get('relevance.clip_pretrained')}"


def archetype_text_vectors(arcs: list[Archetype]) -> np.ndarray:
    template = str(get_config().get("relevance.text_template"))
    key = hashlib.sha256(json.dumps([clip_tag(), template, [a.model_dump() for a in arcs]], sort_keys=True).encode()).hexdigest()[:12]
    path = cache_root() / f"clip_text_{key}.npy"
    if path.exists():
        return np.load(path)
    model, _, tok, torch, _ = _load_clip()
    phrases = [template.format(i.phrase) for a in arcs for i in a.interests]
    with torch.no_grad():
        emb = _norm(model.encode_text(tok(phrases)).float().numpy())
    vecs, k = [], 0
    for a in arcs:
        w = np.array([i.strength for i in a.interests])
        vecs.append((emb[k:k + len(w)] * w[:, None]).sum(0) / w.sum())
        k += len(w)
    out = _norm(np.array(vecs))
    np.save(path, out)
    return out


def image_vector(paths: list[str]) -> np.ndarray:
    model, preprocess, _, torch, _ = _load_clip()
    from PIL import Image
    with torch.no_grad():
        batch = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths])
        emb = _norm(model.encode_image(batch).float().numpy())
    return _norm(emb.mean(0))


def relevance_formula(score: np.ndarray, lo: float, hi: float) -> np.ndarray:
    cfg = get_config()
    w, k = float(cfg.get("relevance.w_rel")), float(cfg.get("relevance.k"))
    sd = float(score.std())
    z = (score - score.mean()) / sd if sd > 0 else np.zeros_like(score)
    return w * expit(k * z) + (1 - w) * np.clip((score - lo) / (hi - lo), 0, 1)


_WORD = re.compile(r"[a-z]+")
_STOP = {"a", "an", "and", "the", "of", "to", "in", "on", "for", "with", "video", "videos", "short", "about", "how", "your"}


def _words(text: str) -> set[str]:
    out = set()
    for w in _WORD.findall(text.lower()):
        if w in _STOP or len(w) < 3:
            continue
        out.add(w[:-1] if w.endswith("s") and len(w) > 3 else w)  # crude singular
    return out


def keyword_scores(profile: VideoProfile, arcs: list[Archetype]) -> np.ndarray:
    vid = _words(" ".join([profile.topic, profile.category, profile.summary, profile.transcript,
                           " ".join(profile.subtopics), " ".join(profile.audience_interest_tags)]))
    out = []
    for a in arcs:
        total = sum(i.strength for i in a.interests)
        hit = sum(i.strength * (len(_words(i.phrase) & vid) / max(len(_words(i.phrase)), 1)) for i in a.interests)
        cat_bonus = 1.0 if profile.category and profile.category.lower() == a.community_name.lower() else 0.0
        out.append(max(float(get_config().get("relevance.keyword_floor")), 0.5 * hit / total + 0.5 * cat_bonus))
    return np.array(out)


def compute_relevance(profile: VideoProfile, pop: Population, method: Optional[str] = None) -> tuple[np.ndarray, RelevanceResult]:
    """Per-viewer relevance in [0, 1] plus a summary. method: None (CLIP, else keyword) | 'keyword'."""
    cfg = get_config()
    arcs = pop.archetypes
    note = ""
    used = "keyword"
    if method != "keyword":
        try:
            frames = list(profile.frame_paths) + list(profile.opening_frame_paths)
            if not frames:
                raise ClipUnavailable("no frames")
            fkey = hashlib.sha256(json.dumps([clip_tag(), frames]).encode()).hexdigest()[:12]
            vpath = video_dir(profile.video_id) / f"clip_img_{fkey}.npy"
            v = np.load(vpath) if vpath.exists() else image_vector(frames)
            np.save(vpath, v)
            arc_score = archetype_text_vectors(arcs) @ v
            lo, hi = float(cfg.get("relevance.cos_lo")), float(cfg.get("relevance.cos_hi"))
            used = "clip"
            note = f"CLIP {clip_tag()} zero-shot similarity; a proxy for interest, not a measurement."
        except ClipUnavailable as e:
            note = f"CLIP unavailable ({e}); keyword-overlap fallback used."
    if used == "keyword":
        arc_score = keyword_scores(profile, arcs)
        lo, hi = 0.0, 1.0
        note = note or "Keyword-overlap relevance; a rough proxy for interest, not a measurement."
    score = arc_score[pop.archetype_idx]
    rel = relevance_formula(score, lo, hi)
    by_comm = {pop.community_names[c]: float(rel[pop.community == c].mean()) for c in range(pop.n_communities)}
    res = RelevanceResult(method=used, by_community=by_comm, mean=float(rel.mean()), population_version=pop.version, note=note)
    return rel, res


def archetype_cosines(profile: VideoProfile, pop: Population) -> dict[str, float]:
    """Raw CLIP cosine per archetype, for inspection/tuning of cos_lo/cos_hi."""
    v = image_vector(list(profile.frame_paths) + list(profile.opening_frame_paths))
    return {a.name: float(s) for a, s in zip(pop.archetypes, archetype_text_vectors(pop.archetypes) @ v)}


def save_relevance(video_id: str, rel: np.ndarray, res: RelevanceResult) -> None:
    d = video_dir(video_id)
    np.save(d / "relevance.npy", rel)
    write_json(d / "relevance.json", res.model_dump(mode="json"))


def load_relevance(video_id: str) -> Optional[tuple[np.ndarray, RelevanceResult]]:
    d = video_dir(video_id)
    meta = read_json(d / "relevance.json")
    if meta is None or not Path(d / "relevance.npy").exists():
        return None
    return np.load(d / "relevance.npy"), RelevanceResult.model_validate(meta)
