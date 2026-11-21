"""Robust seeding plan (USP): who to show the video to first, chosen to hold up across assumptions.

Options: every seeding strategy, plus community-targeted seeding for each community. Every option
runs on the same seeds, random tensors and assumption samples (common random numbers). Per
assumption sample a, value_a(o) = share of seeds reaching the milestone (ties broken by mean reach).
regret_a(o) = best_a - value_a(o). The recommended option minimises the worst-case regret
(minimax regret, as in robust decision making); we also report the average.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from app.config import Config, Params
from app.engine.simulate import SimInputs, World, draw_uniforms, max_rounds, simulate
from app.experiments.matched import assumption_samples


def _pct(x: float) -> str:
    return f"{int(x * 100 + 0.5)}%"  # round half up, same as the UI


def _options(world: World, cfg: Config) -> list[dict]:
    opts = [{"key": s, "strategy": s, "community": None} for s in ("mixed", "relevance_biased", "random")]
    if bool(cfg.get("seeding_plan.include_communities")):
        for c, name in enumerate(world.population.community_names):
            opts.append({"key": f"community:{c}", "strategy": "community_targeted", "community": c, "community_name": name})
    return opts


def robust_seeding_plan(world: World, cfg: Config, relevance: np.ndarray, prior_ensemble: list[float],
                        duration_s: float, n_seeds: Optional[int] = None, n_samples: Optional[int] = None,
                        progress: Optional[Callable[[str], None]] = None) -> dict:
    from app.experiments.matched import p_from_ensemble
    n_seeds = int(n_seeds or cfg.get("seeding_plan.n_seeds"))
    n_samples = int(n_samples or cfg.get("seeding_plan.n_samples"))
    base_seed = int(cfg.get("experiments.base_seed"))
    samples = assumption_samples(cfg, n_samples, base_seed)
    opts = _options(world, cfg)
    T, n = max_rounds(Params(cfg)), world.population.n
    reach = np.zeros((len(opts), n_samples, n_seeds))
    hit = np.zeros((len(opts), n_samples, n_seeds))
    for si in range(n_seeds):
        if progress:
            progress(f"seed {si + 1} of {n_seeds}")
        U = draw_uniforms(base_seed + si, T, n)
        for a, params, u_p in samples:
            P = Params(cfg, params)
            p = p_from_ensemble(prior_ensemble, u_p, 0.0, cfg)
            for oi, o in enumerate(opts):
                r = simulate(SimInputs(world=world, relevance=relevance, prior_p=p, duration_s=duration_s, params=P,
                                       seeding_strategy=o["strategy"], target_community=o["community"]), base_seed + si, U)
                reach[oi, a, si], hit[oi, a, si] = r.summary.unique_reach, r.summary.reached_milestone
    # value per (option, sample): milestone rate, tie-broken by a tiny weight on mean reach
    value = hit.mean(2) + 1e-6 * reach.mean(2) / n
    regret = value.max(0, keepdims=True) - value
    worst = regret.max(1)
    best_i = int(np.argmin(worst + 1e-9 * -value.mean(1)))
    default_i = 0  # "mixed" is the config default
    rows = []
    for oi, o in enumerate(opts):
        rows.append({
            **o, "label": o.get("community_name", o["strategy"]).replace("_", " "),
            "milestone_rate": float(hit[oi].mean()), "mean_reach": float(reach[oi].mean()),
            "worst_case_milestone_rate": float(hit[oi].mean(1).min()),
            "max_regret": float(worst[oi]), "mean_regret": float(regret[oi].mean()),
            "best_in_samples": int((regret[oi] <= 1e-9).sum()),
        })
    order = sorted(range(len(opts)), key=lambda i: (worst[i], -value[i].mean()))
    best, default = rows[best_i], rows[default_i]
    gain = best["milestone_rate"] - default["milestone_rate"]
    if best_i == default_i:
        msg = ("The default mixed seeding is already the most robust choice across the assumption ranges, "
               "in this simulation.")
    else:
        robust = ("it was the best or tied-best option in every assumption setting" if best["max_regret"] <= 1e-9
                  else f"in no assumption setting is it more than {best['max_regret']:.0%} behind the best option")
        msg = (f"Seed with {best['label']} rather than the default: the chance of reaching the milestone goes "
               f"from {_pct(default['milestone_rate'])} to {_pct(best['milestone_rate'])}, and {robust}, in this simulation.")
        if abs(gain) < 0.05:
            msg += " The difference is small: early seeding matters little when recommendation drives most of the reach."
    return {"options": [rows[i] for i in order], "recommended": best["key"], "default": default["key"],
            "gain_vs_default": gain, "message": msg, "n_seeds": n_seeds, "n_samples": n_samples,
            "method": "minimax regret on milestone probability across assumption samples",
            "note": "Seed sets are simulated test audiences, not real people; a what-if under stated assumptions.",
            "evidence_tag": "SIMULATED"}
