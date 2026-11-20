"""Tipping-point curve (B6): share of runs reaching the growth milestone as p varies.

Every grid point uses the same seeds and random tensors (common random numbers), so differences
between points come from p alone. Two curves:
* ``points``: assumptions at their config defaults (PRD 11.6); the position message uses this one.
* ``points_uncertain`` (if ``tipping.sample_assumptions``): each seed also draws its own assumption
  sample, so the curve reflects parameter uncertainty as well as randomness.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from app.config import Config, Params
from app.engine.simulate import SimInputs, World, draw_uniforms, max_rounds, simulate


def _sweep(world: World, cfg: Config, grid: np.ndarray, relevance: np.ndarray, duration_s: float, n_seeds: int,
           base_seed: int, sample: bool, sim_kw: dict) -> tuple[np.ndarray, np.ndarray]:
    T, n = max_rounds(Params(cfg)), world.population.n
    reach = np.zeros((grid.size, n_seeds))
    hit = np.zeros((grid.size, n_seeds), bool)
    for si in range(n_seeds):
        U = draw_uniforms(base_seed + si, T, n)
        P = Params(cfg, cfg.sample(np.random.default_rng([base_seed, si]))) if sample else Params(cfg)
        for gi, p in enumerate(grid):
            r = simulate(SimInputs(world=world, relevance=relevance, prior_p=float(p), duration_s=duration_s,
                                   params=P, **sim_kw), base_seed + si, U)
            reach[gi, si], hit[gi, si] = r.summary.reach_pct, r.summary.reached_milestone
    return reach, hit


def _points(grid: np.ndarray, reach: np.ndarray, hit: np.ndarray) -> list[dict]:
    rate = hit.mean(1)
    return [{"p": float(g), "milestone_rate": float(rate[i]), "mean_reach_pct": float(reach[i].mean()),
             "reach_p10": float(np.quantile(reach[i], 0.1)), "reach_p90": float(np.quantile(reach[i], 0.9))}
            for i, g in enumerate(grid)]


def tipping_curve(world: World, cfg: Config, relevance: np.ndarray, duration_s: float, video_p: float,
                  n_seeds: Optional[int] = None, grid_points: Optional[int] = None, base_seed: Optional[int] = None,
                  **sim_kw) -> dict:
    n_seeds = int(n_seeds or cfg.get("tipping.n_seeds"))
    G = int(grid_points or cfg.get("tipping.grid_points"))
    base_seed = int(cfg.get("experiments.base_seed") if base_seed is None else base_seed)
    grid = np.linspace(float(cfg.get("tipping.p_min")), float(cfg.get("tipping.p_max")), G)
    reach, hit = _sweep(world, cfg, grid, relevance, duration_s, n_seeds, base_seed, False, sim_kw)
    rate = hit.mean(1)
    slope = np.gradient(rate, grid)
    frac = float(cfg.get("tipping.steep_fraction"))
    steep = np.flatnonzero(slope >= frac * slope.max()) if slope.max() > 0 else np.array([], int)
    if not steep.size:
        position, text = "flat", "In this simulation the milestone rate does not change across the prior range."
    elif grid[steep[0]] <= video_p <= grid[steep[-1]]:
        position = "steep"
        text = (f"The video's rating (p = {video_p:.2f}) sits on the steep part of the curve: small changes in the "
                f"opening move the share of runs reaching the milestone a lot, in this simulation.")
    elif video_p < grid[steep[0]]:
        position = "below"
        text = (f"The video's rating (p = {video_p:.2f}) sits below the steep part, which starts near p = "
                f"{grid[steep[0]]:.2f} (about +{grid[steep[0]] - video_p:.2f}), in this simulation.")
    else:
        position = "above"
        text = (f"The video's rating (p = {video_p:.2f}) is past the steep part (which ends near p = "
                f"{grid[steep[-1]]:.2f}); with default assumptions most runs reach the milestone, in this simulation.")
    milestone = float(cfg.get("tipping.milestone_pct"))
    out = {
        "points": _points(grid, reach, hit),
        "video_p": float(video_p), "milestone_rate_at_video": float(np.interp(video_p, grid, rate)),
        "steepest_p": float(grid[int(np.argmax(slope))]),
        "steep_range": [float(grid[steep[0]]), float(grid[steep[-1]])] if steep.size else None,
        "position": position, "message": text, "milestone_pct": milestone, "n_seeds": n_seeds,
        "note": (f"Share of {n_seeds} simulated runs per point reaching {milestone:.0%} of the population, "
                 f"with assumptions at their defaults; not a prediction."),
        "evidence_tag": "SIMULATED",
    }
    if bool(cfg.get("tipping.sample_assumptions")):
        r_u, h_u = _sweep(world, cfg, grid, relevance, duration_s, n_seeds, base_seed, True, sim_kw)
        out["points_uncertain"] = _points(grid, r_u, h_u)
        out["milestone_rate_at_video_uncertain"] = float(np.interp(video_p, grid, h_u.mean(1)))
        out["note_uncertain"] = "Same sweep with each seed drawing its own assumption sample (parameter uncertainty)."
    return out
