"""Pathway classification (B7), computed from the event log with rules in config ``pathways``.

These describe simulated outcomes; they are not validated viral categories.
Rules (a run may match several; the primary is the first match in ``pathways.priority``):
* early_cascade_failure : ended within ``early_failure_rounds`` rounds by low signal / no exposures
* hub_driven            : share of share-channel exposures sent by hubs >= ``hub_driven_share``
* bridge_driven         : share of share-channel exposures sent by bridges >= ``bridge_driven_share``
* cross_community       : >= ``cross_community_min_fraction`` of share exposures crossed communities and they
                          landed in >= ``cross_community_min_communities`` communities
* niche_community_led   : the top community holds >= ``niche_top_community_share`` of all exposures
* deep_but_narrow       : share depth >= ``deep_narrow_depth`` and the milestone was not reached
* broad_but_shallow     : milestone reached and share depth <= ``broad_shallow_depth``
"""
from __future__ import annotations

import numpy as np

from app.config import Params
from app.eventlog.log import CH, ET, EventLog
from app.graph.sbm import ROLE_BRIDGE, ROLE_HUB

DESCRIPTIONS = {
    "early_cascade_failure": "Stalled in the first rounds",
    "hub_driven": "Spread mainly through shares by hubs",
    "bridge_driven": "Spread mainly through shares by bridge viewers",
    "cross_community": "Shares crossed into several communities",
    "niche_community_led": "Stayed mostly within one community",
    "deep_but_narrow": "Long share chains but limited reach",
    "broad_but_shallow": "Wide reach with short share chains",
}


def classify(log: EventLog, P: Params, stop_reason: str, n_rounds: int, depth: int,
             reached_milestone: bool) -> dict:
    c = log.cols
    exp = c["event_type"] == ET["exposure"]
    share = exp & (c["channel"] == CH["share"])
    role_of = dict(zip(c["viewer_id"][exp].tolist(), c["role"][exp].tolist()))
    comm_of = dict(zip(c["viewer_id"][exp].tolist(), c["community"][exp].tolist()))
    src = c["source_viewer"][share]
    n_share = int(share.sum())
    src_role = np.array([role_of[int(s)] for s in src]) if n_share else np.empty(0)
    src_comm = np.array([comm_of[int(s)] for s in src]) if n_share else np.empty(0)
    cross_targets = set(c["community"][share][src_comm != c["community"][share]].tolist()) if n_share else set()
    counts = np.bincount(c["community"][exp]) if exp.any() else np.zeros(1)

    ev = {
        "rounds": float(n_rounds), "depth": float(depth), "share_exposures": float(n_share),
        "hub_share_of_share_exposures": float(np.mean(src_role == ROLE_HUB)) if n_share else 0.0,
        "bridge_share_of_share_exposures": float(np.mean(src_role == ROLE_BRIDGE)) if n_share else 0.0,
        "cross_community_target_communities": float(len(cross_targets)),
        "cross_community_fraction": float(np.mean(src_comm != c["community"][share])) if n_share else 0.0,
        "top_community_share": float(counts.max() / max(counts.sum(), 1)),
        "reached_milestone": float(reached_milestone),
    }
    R = lambda k: float(P[f"pathways.{k}"])  # noqa: E731
    match = {
        "early_cascade_failure": n_rounds <= R("early_failure_rounds") and stop_reason in ("low_signal", "no_exposures"),
        "hub_driven": n_share > 0 and ev["hub_share_of_share_exposures"] >= R("hub_driven_share"),
        "bridge_driven": n_share > 0 and ev["bridge_share_of_share_exposures"] >= R("bridge_driven_share"),
        "cross_community": (len(cross_targets) >= R("cross_community_min_communities")
                            and ev["cross_community_fraction"] >= R("cross_community_min_fraction")),
        "niche_community_led": ev["top_community_share"] >= R("niche_top_community_share"),
        "deep_but_narrow": depth >= R("deep_narrow_depth") and not reached_milestone,
        "broad_but_shallow": reached_milestone and depth <= R("broad_shallow_depth"),
    }
    order = [k for k in P["pathways.priority"] if match.get(k)]
    return {"primary": order[0] if order else None, "matches": order, "evidence": ev,
            "descriptions": {k: DESCRIPTIONS[k] for k in order},
            "note": "Descriptions of this simulated outcome, not validated viral categories.",
            "evidence_tag": "SIMULATED"}
