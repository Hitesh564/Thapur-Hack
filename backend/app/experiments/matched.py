"""Matched A/B experiments with two-level robustness (B3, B4).

Outer loop: A assumption samples (ranged params + p drawn from the prior's ensemble range).
Inner loop: S seeds. Baseline and variant of each (sample, seed) pair read the same U tensor.
Labels are our own cut-offs, not statistical tests.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.stats import spearmanr

from app.config import Config, Params
from app.engine.simulate import SimInputs, World, draw_uniforms, max_rounds, simulate
from schemas import ExperimentConfig, ExperimentResult

VARIABLES = ("hook_clip", "hypothetical_hook", "seeding", "policy")


@dataclass
class Arm:
    """One side of an experiment. Only the experiment's variable should differ between arms."""
    prior_ensemble: list[float]
    relevance: np.ndarray
    duration_s: float
    seeding_strategy: Optional[str] = None
    policy_variant: str = "default"
    p_delta: float = 0.0              # hypothetical hook: added to p, labelled hypothetical
    prior_method: str = "vlm_rubric"
    video_id: str = "video"
    label: str = ""
    meta: dict = field(default_factory=dict)


def p_from_ensemble(ensemble: list[float], u: float, delta: float, cfg: Config) -> float:
    lo, hi = float(min(ensemble)), float(max(ensemble))
    p = lo + u * (hi - lo) + delta
    return float(np.clip(p, cfg.get("prior.clip_min"), cfg.get("prior.clip_max")))


def assumption_samples(cfg: Config, n_samples: int, base_seed: int) -> list[tuple[int, dict, float]]:
    out = []
    for a in range(n_samples):
        rng = np.random.default_rng([int(base_seed), a])
        params = cfg.sample(rng)
        out.append((a, params, float(rng.random())))
    return out


def stability_label(win_rate: float, stability: float, cfg: Config) -> str:
    robust, likely = float(cfg.get("robustness.robust_threshold")), float(cfg.get("robustness.likely_threshold"))
    if win_rate >= robust and stability >= robust:
        return "Robust"
    if win_rate >= likely and stability >= likely:
        return "Likely"
    return "Inconclusive"


def _metric(run, metric: str) -> float:
    return float(getattr(run.summary, metric))


def run_matched(world: World, cfg: Config, baseline: Arm, variant: Arm, variable: str,
                n_seeds: Optional[int] = None, n_samples: Optional[int] = None,
                base_seed: Optional[int] = None, metric: Optional[str] = None,
                experiment_id: Optional[str] = None) -> tuple[ExperimentResult, dict]:
    if variable not in VARIABLES:
        raise ValueError(f"unknown variable {variable!r}")
    n_seeds = int(n_seeds or cfg.get("robustness.n_seeds"))
    n_samples = int(n_samples or cfg.get("robustness.n_assumption_samples"))
    base_seed = int(cfg.get("experiments.base_seed") if base_seed is None else base_seed)
    metric = metric or str(cfg.get("experiments.default_metric"))
    samples = assumption_samples(cfg, n_samples, base_seed)
    n = world.population.n
    T = max_rounds(Params(cfg))

    base_vals = np.zeros((n_samples, n_seeds))
    var_vals = np.zeros((n_samples, n_seeds))
    p_used = np.zeros((n_samples, 2))
    seeds = [base_seed + s for s in range(n_seeds)]
    for si, seed in enumerate(seeds):
        U = draw_uniforms(seed, T, n)
        for a, params, u_p in samples:
            P = Params(cfg, params)
            for k, arm in enumerate((baseline, variant)):
                p = p_from_ensemble(arm.prior_ensemble, u_p, arm.p_delta, cfg)
                p_used[a, k] = p
                inp = SimInputs(world=world, relevance=arm.relevance, prior_p=p, duration_s=arm.duration_s,
                                params=P, seeding_strategy=arm.seeding_strategy, policy_variant=arm.policy_variant,
                                video_id=arm.video_id, prior_method=arm.prior_method, assumption_sample_id=a)
                val = _metric(simulate(inp, seed, U), metric)
                (base_vals if k == 0 else var_vals)[a, si] = val

    diff = var_vals - base_vals
    lo_q, hi_q = cfg.get("robustness.interval")
    win_rate = float(np.mean(diff > 0))
    stability = float(np.mean(np.median(var_vals, axis=1) > np.median(base_vals, axis=1)))
    label = stability_label(win_rate, stability, cfg)

    sensitivity: dict[str, float] = {}
    med_diff = np.median(diff, axis=1)
    if n_samples >= 3 and np.ptp(med_diff) > 0:
        cols = {k: np.array([s[1][k] for s in samples]) for k in samples[0][1]}
        cols["prior_p"] = p_used[:, 0]
        for k, x in cols.items():
            if np.ptp(x) > 0:
                rho = spearmanr(x, med_diff).statistic
                if np.isfinite(rho):
                    sensitivity[k] = float(rho)
        sensitivity = dict(sorted(sensitivity.items(), key=lambda kv: -abs(kv[1])))

    if experiment_id is None:
        key = json.dumps({"v": variable, "b": baseline.label, "x": variant.label, "s": seeds, "a": n_samples,
                          "pop": world.population.version, "g": world.graph.version}, sort_keys=True)
        experiment_id = "exp-" + hashlib.sha256(key.encode()).hexdigest()[:10]
    result = ExperimentResult(
        experiment_id=experiment_id, win_rate=win_rate, effect_median=float(np.median(diff)),
        effect_p10=float(np.quantile(diff, lo_q)), effect_p90=float(np.quantile(diff, hi_q)),
        assumption_stability=stability, label=label, sensitivity=sensitivity,
    )
    held = ["population", "graph", "prior_method", "exposure_budget", "random_tensors"]
    held += [h for h in ("seeding", "policy") if h != variable]
    exp_cfg = ExperimentConfig(
        experiment_id=experiment_id, variable=variable,  # type: ignore[arg-type]
        baseline={"label": baseline.label, "policy_variant": baseline.policy_variant,
                  "seeding_strategy": baseline.seeding_strategy, "p_delta": baseline.p_delta, **baseline.meta},
        variant={"label": variant.label, "policy_variant": variant.policy_variant,
                 "seeding_strategy": variant.seeding_strategy, "p_delta": variant.p_delta, **variant.meta},
        held_fixed=held, n_seeds=n_seeds, n_assumption_samples=n_samples,
    )
    record = {
        "config": exp_cfg.model_dump(), "result": result.model_dump(), "metric": metric,
        "versions": {"population": world.population.version, "graph": world.graph.version,
                     "config": cfg.section_hash(*cfg.raw.keys())},
        "seeds": seeds,
        "assumption_samples": [{"sample_id": a, "params": prm, "p_baseline": float(p_used[a, 0]),
                                "p_variant": float(p_used[a, 1])} for a, prm, _ in samples],
        "baseline_values": base_vals.tolist(), "variant_values": var_vals.tolist(),
    }
    return result, record
