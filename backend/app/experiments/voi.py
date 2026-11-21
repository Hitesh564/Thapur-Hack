"""Value of information (USP): which uncertain assumption, if measured, would most change the decision.

Decision: adopt the variant or keep the baseline. Per assumption sample a, the expected gain of
the variant is D_a = mean over seeds of (variant - baseline). Without more information we pick
the option with the best average: value = max(E[D], 0).

* EVPI (perfect information about every assumption) = E[max(D_a, 0)] - max(E[D], 0)
* EVPPI for one parameter: split samples into ``voi.bins`` groups by that parameter's value,
  EVPPI = sum_b w_b * max(mean D in b, 0) - max(E[D], 0)
* decision risk = share of samples where the other option would have been better.

With few assumption samples these are indicative, and the UI says so.
"""
from __future__ import annotations

import numpy as np

from app.config import Config

# How a creator could actually learn each parameter (product copy, not model numbers).
MEASURE_HINTS = {
    "behaviour.q_min": "your 3-second hold rate (share of viewers still watching after the opening)",
    "behaviour.q_max": "your 3-second hold rate on your best-performing posts",
    "behaviour.a_rel": "how much more your niche audience holds than non-followers (compare follower vs non-follower retention)",
    "behaviour.c_attn": "retention differences between audiences (hard to measure directly)",
    "behaviour.m_min": "average percentage watched on weaker posts",
    "behaviour.m_max": "average percentage watched on your best posts",
    "behaviour.d_rel": "average watch time from your niche vs. other viewers",
    "behaviour.d_attn": "average watch time across audience segments",
    "behaviour.d_dur": "how average percentage watched drops for your longer videos",
    "behaviour.kappa_f": "how spread out watch time is (retention curve shape)",
    "behaviour.base_like": "likes per view on recent posts",
    "behaviour.base_comment": "comments per view on recent posts",
    "behaviour.base_save": "saves per view on recent posts",
    "behaviour.base_share": "shares (sends) per view on a test post",
    "sharing.accept_prob": "how often shared links get opened (reach from shares vs. number of shares)",
    "policy.alpha": None, "policy.beta": None, "policy.gamma": None, "policy.delta": None,
    "policy.S0": None, "policy.kappa": None, "policy.epsilon": None,
    "prior_p": "how viewers rate the opening (a quick poll or A/B test of the first seconds)",
}


def outcome_drivers(outcome: np.ndarray, cols: dict[str, np.ndarray]) -> list[dict]:
    """Which assumptions explain the variation in the outcome (e.g. milestone chance) across samples.

    share = rho^2 / sum(rho^2), rho = Spearman rank correlation. Indicative with few samples."""
    from scipy.stats import spearmanr
    if np.ptp(outcome) <= 0:
        return []
    rows = []
    for k, x in cols.items():
        if np.ptp(x) > 0:
            rho = spearmanr(x, outcome).statistic
            if np.isfinite(rho):
                rows.append({"parameter": k, "rho": float(rho), "hint": MEASURE_HINTS.get(k),
                             "measurable": MEASURE_HINTS.get(k) is not None})
    tot = sum(r["rho"] ** 2 for r in rows) or 1.0
    for r in rows:
        r["share"] = r["rho"] ** 2 / tot
    return sorted(rows, key=lambda r: -r["share"])


def value_of_information(diff: np.ndarray, sample_params: list[dict], prior_p: np.ndarray, cfg: Config,
                         baseline_mean: float, unit: str = "viewers", outcome: np.ndarray | None = None) -> dict:
    D = diff.mean(axis=1)
    E = float(D.mean())
    current = max(E, 0.0)
    choose_variant = E > 0
    evpi = float(np.mean(np.maximum(D, 0)) - current)
    risk = float(np.mean(D < 0)) if choose_variant else float(np.mean(D > 0))  # other option strictly better
    bins = int(cfg.get("voi.bins"))
    cols = {k: np.array([p[k] for p in sample_params]) for k in sample_params[0]}
    cols["prior_p"] = prior_p
    per = []
    for k, x in cols.items():
        if np.ptp(x) <= 0 or len(x) < 2 * bins:
            continue
        order = np.argsort(x, kind="stable")
        groups = np.array_split(order, bins)
        val = sum(len(g) / len(x) * max(float(D[g].mean()), 0.0) for g in groups)
        per.append({"parameter": k, "evppi": float(val - current),
                    "hint": MEASURE_HINTS.get(k), "measurable": MEASURE_HINTS.get(k) is not None})
    per.sort(key=lambda r: -r["evppi"])
    floor = float(cfg.get("voi.min_relative")) * max(abs(baseline_mean), 1.0)
    top = next((r for r in per if r["measurable"] and r["evppi"] > floor), None)
    if evpi <= floor:
        msg = ("No single assumption would change this decision in a meaningful way: "
               "it holds across the assumption ranges, in this simulation.")
    elif top:
        msg = (f"The decision is most sensitive to {top['parameter'].split('.')[-1].replace('_', ' ')}. "
               f"Measuring {top['hint']} would be worth about {top['evppi']:.0f} {unit} on average, in this simulation.")
    else:
        msg = ("What remains uncertain is mostly the distribution policy, which a creator cannot measure; "
               "prefer the option that is robust across policies.")
    drivers = outcome_drivers(outcome, cols) if outcome is not None else []
    top_driver = next((r for r in drivers if r["measurable"]), None)
    if top_driver:
        direction = "higher" if top_driver["rho"] > 0 else "lower"
        driver_msg = (f"Whether the video reaches the milestone depends most on "
                      f"{top_driver['parameter'].split('.')[-1].replace('_', ' ')} ({direction} values help; about "
                      f"{top_driver['share']:.0%} of the explained variation). To narrow it down, measure {top_driver['hint']}.")
    else:
        driver_msg = ""
    return {"decision": "variant" if choose_variant else "baseline",
            "outcome_drivers": drivers[:8], "driver_recommendation": driver_msg, "expected_gain": E, "evpi": evpi,
            "decision_risk": risk, "per_parameter": per, "recommendation": msg, "unit": unit, "floor": floor,
            "note": f"From {len(D)} assumption samples; indicative, not exact.", "evidence_tag": "SIMULATED"}


def outcome_driver_sweep(world, cfg: Config, relevance: np.ndarray, prior_ensemble: list[float], duration_s: float,
                         n_samples: int | None = None, n_seeds: int | None = None, progress=None) -> dict:
    """What decides whether THIS video tips: many assumption samples x a few seeds on the video alone.

    Outcome per sample = mean reach share across seeds. Drivers ranked by share of explained
    rank variation (Spearman rho^2). Uses its own larger sample so the ranking is not noise."""
    from app.config import Params
    from app.engine.simulate import SimInputs, draw_uniforms, max_rounds, simulate
    from app.experiments.matched import assumption_samples, p_from_ensemble
    A = int(n_samples or cfg.get("voi.driver_samples"))
    S = int(n_seeds or cfg.get("voi.driver_seeds"))
    base_seed = int(cfg.get("experiments.base_seed"))
    samples = assumption_samples(cfg, A, base_seed + 7)
    T, n = max_rounds(Params(cfg)), world.population.n
    reach = np.zeros((A, S))
    hit = np.zeros((A, S))
    p_used = np.zeros(A)
    for si in range(S):
        if progress:
            progress(f"seed {si + 1} of {S}")
        U = draw_uniforms(base_seed + si, T, n)
        for a, params, u_p in samples:
            p_used[a] = p_from_ensemble(prior_ensemble, u_p, 0.0, cfg)
            r = simulate(SimInputs(world=world, relevance=relevance, prior_p=p_used[a], duration_s=duration_s,
                                   params=Params(cfg, params)), base_seed + si, U)
            reach[a, si], hit[a, si] = r.summary.reach_pct, r.summary.reached_milestone
    cols = {k: np.array([s[1][k] for s in samples]) for k in samples[0][1]}
    cols["prior_p"] = p_used
    drivers = outcome_drivers(reach.mean(1), cols)
    top = next((r for r in drivers if r["measurable"]), None)
    platform = [r for r in drivers if not r["measurable"]]
    platform_share = sum(r["share"] for r in platform)
    nm = lambda r: r["parameter"].split(".")[-1].replace("_", " ")  # noqa: E731
    parts = []
    if platform and platform_share >= 0.5:
        parts.append(f"Most of the uncertainty ({platform_share:.0%}) is platform-side, led by {nm(platform[0])}, "
                     f"which a creator cannot measure.")
    if top:
        direction = "higher helps" if top["rho"] > 0 else "lower helps"
        parts.append(f"On your side, the biggest lever is {nm(top)} ({direction}; {top['share']:.0%}): "
                     f"measure {top['hint']} to narrow it down.")
    msg = " ".join(parts) + " In this simulation." if parts else "No assumption stands out, in this simulation."
    return {"drivers": drivers[:10], "recommendation": msg, "milestone_rate": float(hit.mean()),
            "milestone_rate_by_sample": hit.mean(1).tolist(), "n_samples": A, "n_seeds": S,
            "note": f"{A} assumption samples x {S} seeds; rank correlations, indicative.", "evidence_tag": "SIMULATED"}
