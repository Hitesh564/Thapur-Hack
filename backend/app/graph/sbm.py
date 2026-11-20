"""Directed degree-corrected stochastic block model (A2).

Edge convention: ``adj[u, v] = 1`` means *v follows u*, so u's shares can reach v.
Row u of the CSR matrix is u's follower list; row nnz = follower count.
Roles are computed from the generated graph, never assigned.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from app.config import Config
from app.personas.population import Population

ROLE_OTHER, ROLE_HUB, ROLE_BRIDGE = 0, 1, 2
ROLE_NAMES = ("other", "hub", "bridge")


@dataclass
class Graph:
    version: str
    seed: int
    adj: sp.csr_matrix          # (n, n); row = who sees this viewer's shares
    community: np.ndarray       # (n,)
    role: np.ndarray            # (n,) int8 codes, see ROLE_NAMES
    followers: np.ndarray       # (n,) int
    cross_fraction: np.ndarray  # (n,) share of follower edges that cross communities

    @property
    def n(self) -> int:
        return int(self.adj.shape[0])

    def intra_edge_ratio(self) -> float:
        coo = self.adj.tocoo()
        if coo.nnz == 0:
            return 0.0
        return float(np.mean(self.community[coo.row] == self.community[coo.col]))

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sp.save_npz(path, self.adj)
        return path


def _weighted_pick(cdf: np.ndarray, members: np.ndarray, u: np.ndarray) -> np.ndarray:
    return members[np.minimum(np.searchsorted(cdf, u, side="right"), len(members) - 1)]


def build_graph(pop: Population, cfg: Config, seed: int | None = None) -> Graph:
    seed = int(pop.seed if seed is None else seed)
    rng = np.random.default_rng([seed, 1])  # stream distinct from the population's
    n = pop.n
    comm = pop.community
    w = pop.influence
    n_comm = pop.n_communities

    # each viewer follows ~Poisson(mean_out_degree) accounts; targets weighted by influence
    k = rng.poisson(float(cfg.get("graph.mean_out_degree")), n)
    follower = np.repeat(np.arange(n), k)
    m = follower.shape[0]
    intra = rng.random(m) < float(cfg.get("graph.intra_community_ratio"))
    target = np.empty(m, dtype=np.int64)

    members = [np.flatnonzero(comm == c) for c in range(n_comm)]
    cdfs = [np.cumsum(w[mem]) / w[mem].sum() for mem in members]
    for c in range(n_comm):
        sel = np.flatnonzero(intra & (comm[follower] == c))
        target[sel] = _weighted_pick(cdfs[c], members[c], rng.random(sel.size))

    all_ids = np.arange(n)
    global_cdf = np.cumsum(w) / w.sum()
    pending = np.flatnonzero(~intra)
    while pending.size:  # resample cross-community picks that landed in the own community
        target[pending] = _weighted_pick(global_cdf, all_ids, rng.random(pending.size))
        pending = pending[comm[target[pending]] == comm[follower[pending]]]

    # bridges: a small share of viewers get extra followers in other communities
    n_b = int(round(float(cfg.get("graph.bridge_pct")) * n))
    extra = int(cfg.get("graph.bridge_extra_edges"))
    b_src = np.repeat(rng.choice(n, size=n_b, replace=False), extra)
    b_dst = rng.integers(0, n, b_src.size)
    redo = np.flatnonzero(comm[b_dst] == comm[b_src])
    while redo.size:
        b_dst[redo] = rng.integers(0, n, redo.size)
        redo = redo[comm[b_dst[redo]] == comm[b_src[redo]]]

    src = np.concatenate([target, b_src])      # the followed account
    dst = np.concatenate([follower, b_dst])    # the follower
    keep = src != dst
    adj = sp.csr_matrix((np.ones(keep.sum(), dtype=np.int8), (src[keep], dst[keep])), shape=(n, n))
    adj.sum_duplicates()
    adj.data[:] = 1
    adj.sort_indices()

    followers = np.diff(adj.indptr).astype(np.int64)
    coo = adj.tocoo()
    cross = np.bincount(coo.row, weights=(comm[coo.row] != comm[coo.col]).astype(float), minlength=n)
    cross_fraction = np.divide(cross, followers, out=np.zeros(n), where=followers > 0)

    role = np.full(n, ROLE_OTHER, dtype=np.int8)
    n_hub = int(round(float(cfg.get("graph.hub_pct")) * n))
    hub_idx = np.argsort(-followers, kind="stable")[:n_hub]
    role[hub_idx] = ROLE_HUB
    bridge_key = np.where((role == ROLE_OTHER) & (followers > 0), cross_fraction, -1.0)
    n_bridge = int(round(float(cfg.get("graph.bridge_pct")) * n))
    bridge_idx = np.argsort(-bridge_key, kind="stable")[:n_bridge]
    role[bridge_idx[bridge_key[bridge_idx] > 0]] = ROLE_BRIDGE

    version = hashlib.sha256(f"{pop.version}|{cfg.section_hash('graph')}|{seed}".encode()).hexdigest()[:12]
    return Graph(version=version, seed=seed, adj=adj, community=comm.copy(), role=role,
                 followers=followers, cross_fraction=cross_fraction)
