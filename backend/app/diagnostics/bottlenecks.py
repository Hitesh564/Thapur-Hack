"""Bottleneck rules (B2). Each rule cites its numbers and says "in this simulated run".

Thresholds live in config ``diagnosis.thresholds``. Severity in [0, 1] = how far the measured
value is past its threshold (0 = just past it, 1 = as far as possible).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from app.config import Params
from app.eventlog.log import CH, ET, EventLog
from app.graph.sbm import ROLE_BRIDGE
from schemas import DiagnosisItem

SUFFIX = " in this simulated run"


def _th(P: Params, key: str) -> float:
    return float(P[f"diagnosis.thresholds.{key}"])


def _below(measured: float, threshold: float) -> float:
    return float(np.clip((threshold - measured) / threshold, 0, 1)) if threshold else 0.0


def _above(measured: float, threshold: float) -> float:
    return float(np.clip((measured - threshold) / max(1 - threshold, 1e-9), 0, 1))


def find_bottlenecks(log: EventLog, P: Params, relevance: np.ndarray, rounds_metrics: list,
                     stop_reason: str, community_sizes: np.ndarray, S0: float,
                     other_seed_milestone_rate: Optional[float] = None,
                     reached_milestone: Optional[bool] = None) -> list[DiagnosisItem]:
    c = log.cols
    et, rnd, ch, comm, vid = c["event_type"], c["round"], c["channel"], c["community"], c["viewer_id"]
    exp_m = et == ET["exposure"]
    items: list[DiagnosisItem] = []

    def add(code: str, msg: str, sev: float, **ev: float) -> None:
        items.append(DiagnosisItem(code=code, message=msg + SUFFIX + ".", severity=sev,
                                   evidence={k: float(v) for k, v in ev.items()}))

    n_exp = int(exp_m.sum())
    if n_exp == 0:
        return items

    # 1 weak initial attention (seed round stay rate)
    seed_E = int((exp_m & (rnd == 0)).sum())
    seed_stay = int(((et == ET["watch"]) & (rnd == 0)).sum()) / max(seed_E, 1)
    th = _th(P, "weak_attention_stay")
    if seed_stay < th:
        add("weak_initial_attention", f"Only {seed_stay:.0%} of seed viewers kept watching past the opening",
            _below(seed_stay, th), seed_stay_rate=seed_stay, threshold=th)

    # 2 low relevance in a newly reached community (first reached after round 0)
    th = _th(P, "low_relevance_new_community")
    seed_comms = set(np.unique(comm[exp_m & (rnd == 0)]).tolist())
    for cc in sorted(set(np.unique(comm[exp_m]).tolist()) - seed_comms):
        ids = vid[exp_m & (comm == cc)]
        r = float(relevance[ids].mean())
        if r < th:
            add("low_relevance_new_community", f"Community {cc} was reached later but its mean relevance was {r:.2f}",
                _below(r, th), community=cc, mean_relevance=r, threshold=th, viewers=ids.size)

    # 3 retention weakness
    stay_ids = vid[et == ET["watch"]]
    if stay_ids.size:
        wf = float(np.mean([m.mean_watch_fraction * m.exposed * m.stay_rate for m in rounds_metrics])
                   / max(np.mean([m.exposed * m.stay_rate for m in rounds_metrics]), 1e-9))
        th = _th(P, "retention_watchfrac")
        if wf < th:
            add("retention_weakness", f"Viewers who stayed watched {wf:.0%} of the video on average",
                _below(wf, th), mean_watch_fraction=wf, threshold=th)

    # 4 insufficient sharing
    share_rate = int((et == ET["share"]).sum()) / n_exp
    th = _th(P, "low_share_rate")
    if share_rate < th:
        add("insufficient_sharing", f"Share rate was {share_rate:.2%} of exposed viewers",
            _below(share_rate, th), share_rate=share_rate, threshold=th, shares=int((et == ET["share"]).sum()))

    # 5 distribution threshold not reached
    if stop_reason == "low_signal":
        S = [m.expansion_score for m in rounds_metrics]
        add("distribution_threshold_not_reached",
            f"The expansion score stayed below the policy threshold S0={S0:.2f} (last {S[-1]:.2f}), so recommendation stopped",
            _below(S[-1], S0), last_expansion_score=S[-1], S0=S0, rounds=len(S))

    # 6 community saturation
    th = _th(P, "community_saturation")
    counts = np.bincount(comm[exp_m], minlength=community_sizes.size)
    frac = counts / np.maximum(community_sizes, 1)
    for cc in np.flatnonzero(frac >= th):
        add("community_saturation", f"Community {cc} is {frac[cc]:.0%} exposed, leaving few new viewers there",
            _above(frac[cc], th), community=cc, exposed_fraction=frac[cc], threshold=th)

    # 7 failure to cross communities (share channel)
    share_exp = exp_m & (ch == CH["share"])
    if share_exp.any():
        cmap = dict(zip(vid[exp_m].tolist(), comm[exp_m].tolist()))
        src_c = np.array([cmap[int(s)] for s in c["source_viewer"][share_exp]])
        cross = float(np.mean(src_c != comm[share_exp]))
        th = _th(P, "cross_community_share")
        if cross < th:
            add("failure_to_cross_communities", f"Only {cross:.0%} of share exposures crossed into another community",
                _below(cross, th), cross_share=cross, threshold=th, share_exposures=int(share_exp.sum()))

    # 8 lack of effective bridges
    bridges_exposed = int((exp_m & (c["role"] == ROLE_BRIDGE)).sum())
    bridge_shares = int(((et == ET["share"]) & (c["role"] == ROLE_BRIDGE)).sum())
    if bridges_exposed > 0 and bridge_shares == 0:
        add("lack_of_effective_bridges", f"{bridges_exposed} bridge viewers saw the video but none shared it",
            1.0, bridges_exposed=bridges_exposed, bridge_shares=0)

    # 9 limited candidate audience
    rel_th = _th(P, "low_relevance_new_community")
    cand = float(np.mean(relevance >= rel_th))
    th = _th(P, "limited_audience_share")
    if cand < th:
        add("limited_candidate_audience", f"Only {cand:.0%} of the population has relevance of at least {rel_th:.2f}",
            _below(cand, th), candidate_share=cand, threshold=th)

    # 10 stochastic failure: this seed missed the milestone but most other seeds reach it
    th = _th(P, "stochastic_failure_other_seeds")
    if other_seed_milestone_rate is not None and reached_milestone is False and other_seed_milestone_rate >= th:
        add("stochastic_failure", f"This run missed the milestone, but {other_seed_milestone_rate:.0%} of other seeds reached it",
            _above(other_seed_milestone_rate, th), other_seed_milestone_rate=other_seed_milestone_rate, threshold=th)

    return sorted(items, key=lambda d: -d.severity)
