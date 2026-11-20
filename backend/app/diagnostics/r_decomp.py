"""R diagnosis (B1), computed from the event log only.

Per round t with E_t viewers exposed in round t:
    R_rec_t   = rec_exposures_{t+1}   / E_t
    R_share_t = share_exposures_{t+1} / E_t
    R_share_t = stay_rate_t * share_given_stay_t * exposures_per_sharer_t   (exact identity)
Binding constraint = factor with the lowest measured/reference ratio (references in config).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from app.config import Params
from app.eventlog.log import CH, ET, EventLog
from schemas import BindingConstraint

FACTORS = ("stay_rate", "share_given_stay", "exposures_per_sharer", "rec_batch_ratio")
REF_KEYS = {
    "stay_rate": "diagnosis.ref_stay_rate",
    "share_given_stay": "diagnosis.ref_share_given_stay",
    "exposures_per_sharer": "diagnosis.ref_exposures_per_sharer",
    "rec_batch_ratio": "diagnosis.ref_rec_batch_ratio",
}


def _div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def _factors(E: int, stayers: int, sharers: int, share_next: int, rec_next: Optional[int]) -> dict:
    d = {
        "exposed": int(E), "stayers": int(stayers), "sharers": int(sharers),
        "share_exposures_next": int(share_next),
        "stay_rate": _div(stayers, E),
        "share_given_stay": _div(sharers, stayers),
        "exposures_per_sharer": _div(share_next, sharers),
        "R_share": _div(share_next, E),
    }
    if rec_next is not None:
        d["rec_exposures_next"] = int(rec_next)
        d["R_rec"] = d["rec_batch_ratio"] = _div(rec_next, E)
        d["R"] = d["R_rec"] + d["R_share"]
    return d


def viewer_community_map(log: EventLog) -> dict[int, int]:
    m = log.mask("exposure")
    return dict(zip(log.cols["viewer_id"][m].tolist(), log.cols["community"][m].tolist()))


def round_table(log: EventLog, community: Optional[int] = None) -> list[dict]:
    """Per-round R factors. With ``community``, viewers (and sharers) are restricted to it
    and recommendation is reported only at whole-run level."""
    c = log.cols
    rnd, et, ch, comm = c["round"], c["event_type"], c["channel"], c["community"]
    last = int(rnd.max()) if len(log) else -1
    sel = np.ones(len(log), bool) if community is None else (comm == community)
    exp_m = et == ET["exposure"]
    share_exp = exp_m & (ch == CH["share"])
    if community is not None:
        cmap = viewer_community_map(log)
        src_comm = np.array([cmap.get(int(s), -1) for s in c["source_viewer"][share_exp]])
        share_src_ok = np.zeros(len(log), bool)
        share_src_ok[np.flatnonzero(share_exp)[src_comm == community]] = True
    else:
        share_src_ok = share_exp
    rows = []
    for t in range(last + 1):
        at, nxt = rnd == t, rnd == t + 1
        E = int((at & exp_m & sel).sum())
        stayers = int((at & (et == ET["watch"]) & sel).sum())
        sharers = int((at & (et == ET["share"]) & sel).sum())
        share_next = int((nxt & share_src_ok).sum())
        rec_next = None if community is not None else int((nxt & exp_m & (ch == CH["rec"])).sum())
        row = {"round": t, "has_next": t < last, **_factors(E, stayers, sharers, share_next, rec_next)}
        rows.append(row)
    return rows


def pooled(rows: list[dict]) -> dict:
    """Pool rounds that have a following round (the final round's onward spread is unobserved)."""
    use = [r for r in rows if r["has_next"]] or rows
    tot = lambda k: sum(r[k] for r in use)  # noqa: E731
    rec = tot("rec_exposures_next") if use and "rec_exposures_next" in use[0] else None
    return _factors(tot("exposed"), tot("stayers"), tot("sharers"), tot("share_exposures_next"), rec)


def binding_constraint(pool: dict, P: Params) -> tuple[BindingConstraint, list[BindingConstraint]]:
    R_share, R_rec = pool["R_share"], pool.get("R_rec", 0.0)
    R = R_share + R_rec
    m = (1.0 - R_rec) / R_share if (R_share > 0 and R < 1) else None
    items = []
    for f in FACTORS:
        if f not in pool:
            continue
        measured = float(pool[f])
        if f == "exposures_per_sharer" and pool["sharers"] == 0:
            continue  # undefined without sharers; share_given_stay (=0) is then binding
        ref = float(P[REF_KEYS[f]])
        if f == "rec_batch_ratio":
            needed = max(0.0, 1.0 - R_share) if R < 1 else None
            mult = None
        else:
            needed = measured * m if m is not None else None
            mult = m
        items.append(BindingConstraint(factor=f, measured=measured, reference=ref,
                                       ratio=measured / ref if ref else 0.0,
                                       needed_multiplier_for_R1=mult, needed_value_for_R1=needed))
    return min(items, key=lambda b: b.ratio), items


def diagnose_r(log: EventLog, P: Params, n_communities: int) -> dict:
    rows = round_table(log)
    pool = pooled(rows)
    binding, factors = binding_constraint(pool, P)
    by_comm = {}
    for cidx in range(n_communities):
        crow = round_table(log, community=cidx)
        cpool = pooled(crow)
        if cpool["exposed"] == 0:
            continue
        cb, cf = binding_constraint(cpool, P)
        by_comm[cidx] = {"rounds": crow, "pooled": cpool, "binding": cb, "factors": cf}
    return {"rounds": rows, "pooled": pool, "R": pool.get("R", pool["R_share"]),
            "binding": binding, "factors": factors, "by_community": by_comm}
