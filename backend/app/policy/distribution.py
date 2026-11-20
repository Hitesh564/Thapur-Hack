"""Our documented distribution policy (A4) -- not any platform's.

S_t     = alpha*stay_t + beta*watchfrac_t + gamma*like_t + delta*share_t
B_{t+1} = clip(B_t * exp(kappa * (S_t - S0)), 0, B_max)

Selection uses only viewer-indexed pre-drawn uniforms (common random numbers).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from app.config import Params

_EPS = 1e-12  # numerical guard for Gumbel keys

SEEDING_STRATEGIES = ("random", "relevance_biased", "mixed", "community_targeted")
POLICY_VARIANTS = ("default", "conservative", "exploratory", "interest_tight")


@dataclass
class Policy:
    alpha: float
    beta: float
    gamma: float
    delta: float
    S0: float
    kappa: float
    epsilon: float
    B0: int
    B_max: int

    def score(self, stay_rate: float, watch_frac: float, like_rate: float, share_rate: float) -> float:
        return (self.alpha * stay_rate + self.beta * watch_frac
                + self.gamma * like_rate + self.delta * share_rate)

    def next_batch(self, B_prev: float, S: float) -> float:
        return float(np.clip(B_prev * np.exp(self.kappa * (S - self.S0)), 0, self.B_max))


def resolve_policy(P: Params, variant: str = "default") -> Policy:
    if variant not in POLICY_VARIANTS:
        raise ValueError(f"unknown policy variant {variant!r}")
    pol = Policy(
        alpha=P["policy.alpha"], beta=P["policy.beta"], gamma=P["policy.gamma"], delta=P["policy.delta"],
        S0=P["policy.S0"], kappa=P["policy.kappa"], epsilon=P["policy.epsilon"],
        B0=int(P["policy.B0"]), B_max=int(P["policy.B_max"]),
    )
    if variant != "default":
        spec = P.cfg.get(f"policy.variants.{variant}")
        if "S0_add" in spec:
            pol.S0 += float(spec["S0_add"])
        if "epsilon_set" in spec:
            pol.epsilon = float(spec["epsilon_set"])
    return pol


def random_k(u: np.ndarray, k: int, mask: np.ndarray) -> np.ndarray:
    """k eligible viewers with the smallest uniforms (uniform random choice)."""
    k = int(min(k, mask.sum()))
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    keys = np.where(mask, u, np.inf)
    return np.sort(np.argsort(keys, kind="stable")[:k])


def gumbel_top_k(weights: np.ndarray, u: np.ndarray, k: int, mask: np.ndarray) -> np.ndarray:
    """Weighted sampling without replacement via Gumbel-top-k on pre-drawn uniforms."""
    k = int(min(k, mask.sum()))
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    uu = np.clip(u, _EPS, 1 - _EPS)
    keys = np.log(np.maximum(weights, _EPS)) - np.log(-np.log(uu))
    keys = np.where(mask, keys, -np.inf)
    return np.sort(np.argsort(-keys, kind="stable")[:k])


def select_rec(B: int, epsilon: float, rel: np.ndarray, eligible: np.ndarray,
               u_tiebreak: np.ndarray, u_select: np.ndarray) -> np.ndarray:
    """Recommendation batch: (1-epsilon)*B by relevance-weighted sampling, epsilon*B uniformly."""
    n_rand = int(round(epsilon * B))
    weighted = gumbel_top_k(rel, u_select, B - n_rand, eligible)
    rest = eligible.copy()
    rest[weighted] = False
    rand = random_k(u_tiebreak, n_rand, rest)
    return np.sort(np.concatenate([weighted, rand]))


def select_seeds(P: Params, strategy: str, n_seed: int, rel: np.ndarray, community: np.ndarray,
                 u_tiebreak: np.ndarray, u_select: np.ndarray, target_community: int | None = None) -> np.ndarray:
    n = rel.shape[0]
    every = np.ones(n, dtype=bool)
    if strategy == "random":
        return random_k(u_tiebreak, n_seed, every)
    if strategy == "relevance_biased":
        return gumbel_top_k(rel, u_select, n_seed, every)
    if strategy == "mixed":
        n_rel = int(round(float(P["seeding.mixed_relevance_share"]) * n_seed))
        a = gumbel_top_k(rel, u_select, n_rel, every)
        rest = every.copy()
        rest[a] = False
        return np.sort(np.concatenate([a, random_k(u_tiebreak, n_seed - a.size, rest)]))
    if strategy == "community_targeted":
        if target_community is None:
            means = np.bincount(community, weights=rel) / np.maximum(np.bincount(community), 1)
            target_community = int(np.argmax(means))
        return gumbel_top_k(rel, u_select, n_seed, community == target_community)
    raise ValueError(f"unknown seeding strategy {strategy!r}")


def share_exposures(sharers: np.ndarray, adj: sp.csr_matrix, unseen: np.ndarray, u_accept: np.ndarray,
                    u_tiebreak: np.ndarray, accept_prob: float, fanout_cap: int) -> tuple[np.ndarray, np.ndarray]:
    """Followers newly exposed by this round's sharers.

    A follower is eligible if unseen and its acceptance uniform < accept_prob. Each sharer
    exposes at most ``fanout_cap`` eligible followers, in tie-break order. A follower reached
    by several sharers is credited to the lowest sharer id. Returns (targets, sources).
    """
    if sharers.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    sharers = np.sort(sharers)
    sub = adj[sharers]
    src = np.repeat(sharers, np.diff(sub.indptr))
    fol = sub.indices.astype(np.int64)
    ok = unseen[fol] & (u_accept[fol] < accept_prob)
    src, fol = src[ok], fol[ok]
    if fol.size == 0:
        return fol, src
    order = np.lexsort((u_tiebreak[fol], src))
    src, fol = src[order], fol[order]
    group_start = np.r_[0, np.flatnonzero(np.diff(src)) + 1]
    rank = np.arange(src.size) - np.repeat(group_start, np.diff(np.r_[group_start, src.size]))
    keep = rank < fanout_cap
    src, fol = src[keep], fol[keep]
    # dedupe followers, credit to lowest sharer id (src is sorted ascending)
    _, first = np.unique(fol, return_index=True)
    return fol[first], src[first]
