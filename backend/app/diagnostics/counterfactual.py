"""Counterfactual Trace (USP): why the variant wins, viewer by viewer.

Baseline and variant run on the same seed, the same assumption sample and the same random tensor
U (common random numbers), and every viewer reads the same personal dice in both runs. So for every
viewer we can say exactly what changed:

* decision flips: a viewer exposed in both runs whose stay / share / completion decision differs.
  With the same uniform u, a stay flip means the threshold moved across u: P_base < u <= P_var.
* exposure changes: viewers exposed in only one of the runs, by channel and, for shares, by
  whether the sharer is a "new sharer" (did not share in the other run) or shared in both.

Exact decompositions (identities, tested):
* delta reach = sum over channels (seed, rec, share) of delta exposures.
* delta share exposures = (from new sharers) + (from sharers in both runs) - (lost from sharers who
  only shared in the baseline).
* recommendation growth: while the batch is not capped, log B_t = log B0 + kappa * sum_{u<t}(S_u - S0),
  and S_u = alpha*stay + beta*watch + gamma*like + delta*share, so the difference in log batch size
  splits exactly into stay, watch, like and share contributions.
Everything is a property of this simulated pair of runs.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from app.config import Params
from app.engine.simulate import D, RunResult, SimInputs, draw_uniforms, max_rounds, simulate
from app.eventlog.log import CH
from app.graph.sbm import ROLE_NAMES
from app.policy.distribution import resolve_policy

SIGNALS = ("stay", "watch", "like", "share")


def _signal_components(run: RunResult, P: Params, variant: str) -> np.ndarray:
    """(rounds, 4) weighted contributions to the expansion score S_t."""
    pol = resolve_policy(P, variant)
    w = np.array([pol.alpha, pol.beta, pol.gamma, pol.delta])
    x = np.array([[m.stay_rate, m.mean_watch_fraction, m.like_rate, m.share_rate] for m in run.rounds])
    return x * w if len(x) else np.zeros((0, 4))


def trace_pair(base_inp: SimInputs, var_inp: SimInputs, seed: int, community_names: list[str],
               max_examples: int = 12) -> dict:
    P = base_inp.params
    if base_inp.policy_variant != var_inp.policy_variant:
        raise ValueError("Counterfactual Trace needs the same distribution policy in both runs.")
    pop = base_inp.world.population
    graph = base_inp.world.graph
    U = draw_uniforms(seed, max_rounds(P), pop.n)
    rb, rv = simulate(base_inp, seed, U), simulate(var_inp, seed, U)
    sb, sv = rb.state, rv.state
    eb, ev = sb["exposed"], sv["exposed"]
    both, only_v, only_b = eb & ev, ev & ~eb, eb & ~ev

    # ---- channel decomposition of the reach difference (exact) ----
    def by_ch(st, mask):
        return {c: int((mask & (st["channel"] == CH[c])).sum()) for c in ("seed", "rec", "share")}
    ch_b, ch_v = by_ch(sb, eb), by_ch(sv, ev)
    delta_reach = int(ev.sum() - eb.sum())

    # ---- share exposures: new sharers vs sharers in both runs ----
    shared_b, shared_v = sb["act_share"] & eb, sv["act_share"] & ev
    new_sharer = shared_v & ~shared_b          # shared only in the variant (flip or newly exposed)
    lost_sharer = shared_b & ~shared_v
    share_v = np.flatnonzero(ev & (sv["channel"] == CH["share"]))
    share_b = np.flatnonzero(eb & (sb["channel"] == CH["share"]))
    from_new = int(new_sharer[sv["source"][share_v]].sum())
    from_common_v = len(share_v) - from_new
    from_lost = int(lost_sharer[sb["source"][share_b]].sum())
    from_common_b = len(share_b) - from_lost
    share_parts = {"from_new_sharers": from_new, "lost_from_baseline_only_sharers": -from_lost,
                   "change_from_sharers_in_both": from_common_v - from_common_b}

    # ---- recommendation growth: exact split of the difference in log batch size ----
    comp_b, comp_v = _signal_components(rb, P, base_inp.policy_variant), _signal_components(rv, P, var_inp.policy_variant)
    pol = resolve_policy(P, base_inp.policy_variant)
    common = min(len(comp_b), len(comp_v))
    # rounds whose score feeds a recommendation batch observed in both runs: u < common - 1
    feed = max(common - 1, 0)
    dS_rounds = (comp_v[:common] - comp_b[:common]) if common else np.zeros((0, 4))
    dlog = pol.kappa * dS_rounds[:feed].sum(0) if feed else np.zeros(4)
    B_unc = lambda comps: pol.B0 * np.exp(pol.kappa * np.cumsum(comps.sum(1) - pol.S0))  # noqa: E731
    capped = bool((B_unc(comp_b) > pol.B_max).any() or (B_unc(comp_v) > pol.B_max).any())
    rec_rounds_b = [m.round for m in rb.rounds]
    rec_rounds_v = [m.round for m in rv.rounds]
    rec_by_round_b = np.bincount(sb["round"][eb & (sb["channel"] == CH["rec"])], minlength=max_rounds(P))
    rec_by_round_v = np.bincount(sv["round"][ev & (sv["channel"] == CH["rec"])], minlength=max_rounds(P))
    common_rounds = set(rec_rounds_b) & set(rec_rounds_v)
    d_rec_common = int(sum(rec_by_round_v[t] - rec_by_round_b[t] for t in common_rounds))
    d_rec_extra = int(sum(rec_by_round_v[t] for t in rec_rounds_v if t not in common_rounds)
                      - sum(rec_by_round_b[t] for t in rec_rounds_b if t not in common_rounds))

    # ---- decision flips among viewers exposed in both runs ----
    flips = {}
    for key, name in (("stay", "stay"), ("act_share", "share"), ("complete", "complete"), ("act_like", "like")):
        gained = both & sv[key] & ~sb[key]
        lost = both & sb[key] & ~sv[key]
        flips[name] = {"gained": int(gained.sum()), "lost": int(lost.sum())}
    u_stay = U[0, :, D["stay"]]
    stay_gain = np.flatnonzero(both & sv["stay"] & ~sb["stay"])
    stay_lost = np.flatnonzero(both & sb["stay"] & ~sv["stay"])
    examples = []
    # diverse examples: biggest threshold moves first, at most two per community
    picked, per_comm = [], {}
    for v in stay_gain[np.argsort(-(sv["p_stay"][stay_gain] - sb["p_stay"][stay_gain]), kind="stable")]:
        c = int(pop.community[v])
        if per_comm.get(c, 0) < 2:
            picked.append(v)
            per_comm[c] = per_comm.get(c, 0) + 1
        if len(picked) >= max_examples // 2:
            break
    for v in picked + list(stay_lost[: max_examples // 4]):
        examples.append({
            "viewer": int(v), "community": community_names[int(pop.community[v])],
            "persona": pop.archetypes[int(pop.archetype_idx[v])].name, "role": ROLE_NAMES[int(graph.role[v])],
            "dice": float(u_stay[v]), "p_stay_baseline": float(sb["p_stay"][v]), "p_stay_variant": float(sv["p_stay"][v]),
            "stayed_baseline": bool(sb["stay"][v]), "stayed_variant": bool(sv["stay"][v]),
            "shared_variant": bool(sv["act_share"][v]), "round_baseline": int(sb["round"][v]), "round_variant": int(sv["round"][v]),
        })

    # ---- where the difference landed ----
    k = pop.n_communities
    by_comm = (np.bincount(pop.community[ev], minlength=k) - np.bincount(pop.community[eb], minlength=k)).tolist()

    # ---- per-round expansion score for both runs ----
    rounds = []
    for t in range(max(len(rb.rounds), len(rv.rounds))):
        row = {"round": t}
        for tag, run, comps in (("baseline", rb, comp_b), ("variant", rv, comp_v)):
            if t < len(run.rounds):
                row[f"S_{tag}"] = float(run.rounds[t].expansion_score)
                row[f"exposed_{tag}"] = int(run.rounds[t].exposed)
                row[f"components_{tag}"] = dict(zip(SIGNALS, map(float, comps[t])))
        rounds.append(row)

    top_signal = SIGNALS[int(np.argmax(dlog))] if feed and np.any(dlog > 0) else None
    parts_total = ch_v["seed"] - ch_b["seed"] + d_rec_common + d_rec_extra + (len(share_v) - len(share_b))
    msg = _message(delta_reach, flips, share_parts, d_rec_common, d_rec_extra, dlog, top_signal, rb, rv)

    def cascade(run: RunResult, st: dict) -> dict:
        ids = np.flatnonzero(st["exposed"])
        return {"viewer": ids.tolist(), "round": st["round"][ids].tolist(), "channel": st["channel"][ids].tolist(),
                "source": st["source"][ids].tolist(), "stay": st["stay"][ids].astype(int).tolist(),
                "share": st["act_share"][ids].astype(int).tolist()}

    return {
        "seed": int(seed),
        "reach_baseline": int(eb.sum()), "reach_variant": int(ev.sum()), "delta_reach": delta_reach,
        "milestone_baseline": bool(rb.summary.reached_milestone), "milestone_variant": bool(rv.summary.reached_milestone),
        "stop_baseline": rb.stop_reason, "stop_variant": rv.stop_reason,
        "rounds_baseline": len(rb.rounds), "rounds_variant": len(rv.rounds),
        "exposure": {"both": int(both.sum()), "only_variant": int(only_v.sum()), "only_baseline": int(only_b.sum()),
                     "only_variant_by_channel": by_ch(sv, only_v), "only_baseline_by_channel": by_ch(sb, only_b)},
        "decomposition": {
            "seed": ch_v["seed"] - ch_b["seed"],
            "rec_common_rounds": d_rec_common,
            "rec_extra_rounds": d_rec_extra,
            "share": len(share_v) - len(share_b),
            "share_parts": share_parts,
            "check": parts_total == delta_reach,
        },
        "rec_growth": {
            "log_batch_difference_by_signal": dict(zip(SIGNALS, map(float, dlog))),
            "log_batch_difference": float(dlog.sum()),
            "batch_ratio": float(np.exp(dlog.sum())) if feed else 1.0,
            "rounds_used": feed, "capped": capped, "kappa": pol.kappa, "S0": pol.S0,
            "top_signal": top_signal,
        },
        "flips": flips, "examples": examples,
        "by_community": dict(zip(community_names, by_comm)),
        "rounds": rounds,
        "cascade_baseline": cascade(rb, sb), "cascade_variant": cascade(rv, sv),
        "message": msg,
        "note": ("Same seed, same assumptions, same per-viewer dice in both runs; every difference is traced to a "
                 "changed decision or a changed exposure. A property of this simulated pair, not of real viewers."),
        "evidence_tag": "SIMULATED",
    }


def _message(dr, flips, share_parts, d_common, d_extra, dlog, top_signal, rb, rv) -> str:
    if dr == 0 and flips["stay"]["gained"] == 0 and flips["stay"]["lost"] == 0:
        return "Both runs are identical on this seed: nothing changed for any viewer, in this simulation."
    parts = [f"The variant reached {dr:+d} viewers on this seed."]
    sg, sl = flips["stay"]["gained"], flips["stay"]["lost"]
    if sg or sl:
        parts.append(f"Among viewers who saw both versions, {sg} stayed only with the variant and {sl} only with the original "
                     f"(same dice, different threshold).")
    if rb.stop_reason == "low_signal" and rv.stop_reason != "low_signal":
        parts.append(f"The original stalled after {len(rb.rounds)} rounds while the variant kept growing: "
                     f"{d_extra:+d} viewers came from rounds the original never reached.")
    if top_signal:
        parts.append(f"Recommendation batches grew faster mainly through the {top_signal} signal "
                     f"(log-batch gap {dlog.sum():+.2f}); same-round recommendation changed by {d_common:+d}.")
    if share_parts["from_new_sharers"]:
        parts.append(f"{share_parts['from_new_sharers']} share exposures came from viewers who shared only in the variant.")
    parts.append("In this simulation.")
    return " ".join(parts)


def representative_seed(base_inp: SimInputs, var_inp: SimInputs, seeds: list[int]) -> tuple[int, list[int]]:
    """Seed whose reach difference is closest to the median across ``seeds`` (not cherry-picked)."""
    P = base_inp.params
    deltas = []
    for s in seeds:
        U = draw_uniforms(s, max_rounds(P), base_inp.world.population.n)
        deltas.append(simulate(var_inp, s, U).summary.unique_reach - simulate(base_inp, s, U).summary.unique_reach)
    med = float(np.median(deltas))
    idx = int(np.argmin([abs(d - med) + 1e-6 * i for i, d in enumerate(deltas)]))
    return seeds[idx], [int(d) for d in deltas]


__all__ = ["trace_pair", "representative_seed", "Optional"]
