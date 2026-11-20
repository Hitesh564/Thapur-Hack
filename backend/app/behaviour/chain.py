"""Behaviour chain (A3): stay -> watch fraction -> like/comment/save/share. No LLM.

All randomness comes from pre-drawn uniforms passed in by the engine.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import betaincinv, expit

from app.config import Params

_EPS = 1e-9  # numerical guard for logit / Beta quantile, not a model parameter
REASONS = ("prior", "relevance", "attention", "duration")
ACTIONS = ("like", "comment", "save", "share")


def logit(x: np.ndarray | float) -> np.ndarray:
    x = np.clip(x, _EPS, 1 - _EPS)
    return np.log(x / (1 - x))


@dataclass
class BehaviourOut:
    p_stay: np.ndarray
    stay: np.ndarray
    watch_fraction: np.ndarray        # 0 for skippers
    complete: np.ndarray
    probs: dict[str, np.ndarray]      # action -> probability (0 for skippers)
    acts: dict[str, np.ndarray]       # action -> bool
    reason: np.ndarray                # int index into REASONS


def evaluate(P: Params, p: float, rel: np.ndarray, mean_rel: float, attn: np.ndarray,
             tendency: dict[str, np.ndarray], duration_s: float, u: dict[str, np.ndarray]) -> BehaviourOut:
    """Evaluate the chain for a batch of exposed viewers.

    ``u`` maps draw names (stay, watch, like, comment, save, share) to uniforms for these viewers.
    """
    attn_c = attn - P["behaviour.attn_centre"]
    rel_c = rel - mean_rel

    q0 = P["behaviour.q_min"] + (P["behaviour.q_max"] - P["behaviour.q_min"]) * p
    t_prior = np.full(rel.shape, float(logit(q0)))
    t_rel = P["behaviour.a_rel"] * rel_c
    t_attn = P["behaviour.c_attn"] * attn_c
    p_stay = expit(t_prior + t_rel + t_attn)
    stay = u["stay"] < p_stay

    m0 = P["behaviour.m_min"] + (P["behaviour.m_max"] - P["behaviour.m_min"]) * p
    t_dur = np.full(rel.shape, -P["behaviour.d_dur"] * np.log(duration_s / P["behaviour.ref_duration_s"]))
    m = expit(logit(m0) + P["behaviour.d_rel"] * rel_c + P["behaviour.d_attn"] * attn_c + t_dur)
    kf = P["behaviour.kappa_f"]
    wf = betaincinv(m * kf, (1 - m) * kf, np.clip(u["watch"], _EPS, 1 - _EPS))
    wf = np.where(stay, wf, 0.0)
    complete = stay & (wf >= P["behaviour.completion_threshold"])

    off = P["behaviour.action_offset"]
    scale = (off + rel) * (off + wf)
    probs, acts = {}, {}
    for a in ACTIONS:
        pr = np.clip(P[f"behaviour.base_{a}"] * tendency[a] * scale, 0.0, 1.0)
        pr = np.where(stay, pr, 0.0)
        probs[a] = pr
        acts[a] = stay & (u[a] < pr)

    reason = np.argmax(np.abs(np.stack([t_prior, t_rel, t_attn, t_dur])), axis=0)
    return BehaviourOut(p_stay, stay, wf, complete, probs, acts, reason)
