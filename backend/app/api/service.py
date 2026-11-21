"""Service layer behind the API: runs, experiments, payload shaping. Headless-safe."""
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Optional

import numpy as np
import pandas as pd

from app.config import Params, get_config
from app.diagnostics import diagnose_r, find_bottlenecks, share_breakdown, stop_explanation
from app.diagnostics.attribution import attribution, replay_aggregates
from app.diagnostics.pathways import classify
from app.engine import RunResult, SimInputs, simulate
from app.eventlog import EventLog
from app.experiments import Arm, run_matched
from app.pipeline import get_analysis, get_world
from app.policy import resolve_policy
from app.storage import cache_root, read_json, write_json

_RUNS: "OrderedDict[str, tuple[RunResult, np.ndarray]]" = OrderedDict()
_RUNS_MAX = 64
_lock = threading.Lock()


class NotAnalysed(KeyError):
    pass


def _analysis(video_id: str):
    a = get_analysis(video_id)
    if a is None:
        raise NotAnalysed(f"video {video_id} has not been analysed")
    return a


def run_simulation(video_id: str, seed: int = 0, seeding_strategy: Optional[str] = None,
                   policy_variant: str = "default", prior_override: Optional[float] = None,
                   stochastic_check_seeds: int = 0) -> dict:
    profile, pr, rel, _ = _analysis(video_id)
    cfg = get_config()
    P = Params(cfg)
    p = float(pr.percentile if prior_override is None else prior_override)
    method = pr.method if prior_override is None else "hypothetical"
    world = get_world()
    inp = SimInputs(world=world, relevance=rel, prior_p=p, duration_s=profile.duration_s, params=P,
                    seeding_strategy=seeding_strategy, policy_variant=policy_variant, video_id=video_id,
                    prior_method=method)
    run = simulate(inp, seed)
    other_rate = None
    if stochastic_check_seeds > 0:
        others = [simulate(inp, seed + 1 + k).summary.reached_milestone for k in range(stochastic_check_seeds)]
        other_rate = float(np.mean(others))
    with _lock:
        _RUNS[run.run_id] = (run, rel)
        while len(_RUNS) > _RUNS_MAX:
            _RUNS.popitem(last=False)
    payload = run_payload(run, rel, P, other_rate)
    write_json(cache_root() / "runs" / f"{run.run_id}.json", payload)
    run.log.to_parquet(cache_root() / "runs" / f"{run.run_id}.parquet")
    return payload


def run_payload(run: RunResult, rel: np.ndarray, P: Params, other_seed_milestone_rate: Optional[float] = None) -> dict:
    world = get_world()
    pop = world.population
    d = diagnose_r(run.log, P, pop.n_communities)
    names = pop.community_names
    S0 = resolve_policy(P, run.config.policy_variant).S0
    S_last = run.rounds[-1].expansion_score if run.rounds else 0.0
    bottlenecks = find_bottlenecks(run.log, P, rel, run.rounds, run.stop_reason, np.bincount(pop.community),
                                   resolve_policy(P, run.config.policy_variant).S0,
                                   other_seed_milestone_rate=other_seed_milestone_rate,
                                   reached_milestone=run.summary.reached_milestone, community_names=names)
    return {
        "run_id": run.run_id,
        "config": run.config.model_dump(mode="json"),
        "summary": run.summary.model_dump(mode="json"),
        "stop_reason": run.stop_reason,
        "repro": run.repro,
        "community_names": names,
        "diagnosis": {
            "R": d["R"], "R_tag": "SIMULATED",
            "pooled": d["pooled"],
            "binding": d["binding"].model_dump(mode="json"),
            "share_breakdown": share_breakdown(d["rounds"], run.rounds),
            "stop": stop_explanation(run.stop_reason, d["rounds"], S_last, S0),
            "factors": [f.model_dump(mode="json") for f in d["factors"]],
            "rounds": d["rounds"],
            "by_community": {names[c]: {"pooled": v["pooled"], "binding": v["binding"].model_dump(mode="json")}
                             for c, v in d["by_community"].items()},
            "bottlenecks": [b.model_dump(mode="json") for b in bottlenecks],
            "note": "Diagnosis of this simulated run; references are assumed 'healthy' settings from config.",
        },
        "pathways": classify(run.log, P, run.stop_reason, run.summary.rounds, run.summary.depth,
                             run.summary.reached_milestone),
        "attribution": attribution(run.log, names),
        "replay": replay_aggregates(run.log, pop.n_communities),
    }


def get_run(run_id: str) -> Optional[dict]:
    return read_json(cache_root() / "runs" / f"{run_id}.json")


def get_run_events(run_id: str, offset: int = 0, limit: int = 1000) -> Optional[dict]:
    with _lock:
        entry = _RUNS.get(run_id)
    if entry is not None:
        df = entry[0].log.to_frame()
    else:
        path = cache_root() / "runs" / f"{run_id}.parquet"
        if not path.exists():
            return None
        df = EventLog(run_id, {k: v.to_numpy() for k, v in pd.read_parquet(path).drop(columns="run_id").items()}).to_frame()
    page = df.iloc[offset:offset + limit]
    return {"run_id": run_id, "total": int(len(df)), "offset": offset,
            "events": page.astype(object).where(page.notna(), None).to_dict(orient="records")}


def _arm(video_id: str, label: str, **kw) -> Arm:
    profile, pr, rel, _ = _analysis(video_id)
    ens = pr.ensemble or [pr.percentile]
    return Arm(prior_ensemble=ens, relevance=rel, duration_s=profile.duration_s, prior_method=pr.method,
               video_id=video_id, label=label, meta={"video_id": video_id, "prior_method": pr.method,
                                                     "prior_percentile": pr.percentile}, **kw)


def run_experiment(variable: str, baseline_video_id: str, variant_video_id: Optional[str] = None,
                   baseline_seeding: Optional[str] = None, variant_seeding: Optional[str] = None,
                   baseline_policy: str = "default", variant_policy: str = "default",
                   hypothetical_delta: Optional[float] = None, n_seeds: Optional[int] = None,
                   n_samples: Optional[int] = None) -> dict:
    cfg = get_config()
    base = _arm(baseline_video_id, "baseline", seeding_strategy=baseline_seeding, policy_variant=baseline_policy)
    if variable == "hook_clip":
        if not variant_video_id:
            raise ValueError("hook_clip experiments need variant_video_id (the alternate opening)")
        var = _arm(variant_video_id, "alternate clip", seeding_strategy=baseline_seeding, policy_variant=baseline_policy)
        # same audience: the variant clip keeps its own relevance and prior; everything else is held fixed
    elif variable == "hypothetical_hook":
        delta = float(cfg.get("prior.hypothetical_delta") if hypothetical_delta is None else hypothetical_delta)
        var = _arm(baseline_video_id, f"hypothetical +{delta:.2f}", seeding_strategy=baseline_seeding,
                   policy_variant=baseline_policy, p_delta=delta)
        var.prior_method = "hypothetical"
        var.meta["hypothetical"] = True
    elif variable == "seeding":
        var = _arm(baseline_video_id, f"seeding={variant_seeding}", seeding_strategy=variant_seeding,
                   policy_variant=baseline_policy)
    elif variable == "policy":
        var = _arm(baseline_video_id, f"policy={variant_policy}", seeding_strategy=baseline_seeding,
                   policy_variant=variant_policy)
    else:
        raise ValueError(f"unknown variable {variable!r}")
    result, record = run_matched(get_world(), cfg, base, var, variable, n_seeds=n_seeds, n_samples=n_samples)
    record["result"]["evidence_tag"] = "SIMULATED"
    write_json(cache_root() / "experiments" / f"{result.experiment_id}.json", record)
    return record


def get_experiment(experiment_id: str) -> Optional[dict]:
    return read_json(cache_root() / "experiments" / f"{experiment_id}.json")


def run_tipping(video_id: str, n_seeds: Optional[int] = None, grid_points: Optional[int] = None,
                seeding_strategy: Optional[str] = None, policy_variant: str = "default") -> dict:
    from app.experiments import tipping_curve
    profile, pr, rel, _ = _analysis(video_id)
    out = tipping_curve(get_world(), get_config(), rel, profile.duration_s, pr.percentile, n_seeds=n_seeds,
                        grid_points=grid_points, seeding_strategy=seeding_strategy, policy_variant=policy_variant)
    out.update({"video_id": video_id, "prior_method": pr.method, "prior_tag": pr.evidence_tag.value})
    write_json(cache_root() / "tipping" / f"{video_id}.json", out)
    return out


def run_attention(video_id: str) -> dict:
    from app.video_analysis.attention import attention_timeline
    profile, pr, _, _ = _analysis(video_id)
    out = attention_timeline(profile, pr)
    return {"video_id": video_id, **out}


def _log_for(run_id: str):
    with _lock:
        entry = _RUNS.get(run_id)
    if entry is not None:
        return entry[0].log
    path = cache_root() / "runs" / f"{run_id}.parquet"
    if not path.exists():
        return None
    return EventLog(run_id, {k: v.to_numpy() for k, v in pd.read_parquet(path).drop(columns="run_id").items()})


def get_cascade(run_id: str) -> Optional[dict]:
    """Compact per-exposure arrays for the animated replay."""
    from app.eventlog import ET
    log = _log_for(run_id)
    if log is None:
        return None
    c = log.cols
    exp = c["event_type"] == ET["exposure"]
    vid = c["viewer_id"][exp]

    def flag(name: str) -> list[int]:
        s = set(c["viewer_id"][c["event_type"] == ET[name]].tolist())
        return [int(v in s) for v in vid.tolist()]

    return {"run_id": run_id, "viewer": vid.tolist(), "round": c["round"][exp].tolist(),
            "channel": c["channel"][exp].tolist(), "source": c["source_viewer"][exp].tolist(),
            "stay": flag("watch"), "complete": flag("completion"), "like": flag("like"), "share": flag("share"),
            "channels": ["seed", "rec", "share"], "evidence_tag": "SIMULATED"}


def world_layout() -> dict:
    """Per-viewer community, role, followers and a deterministic 2-D position (communities on a circle)."""
    w = get_world()
    pop, g = w.population, w.graph
    n, k = pop.n, pop.n_communities
    rng = np.random.default_rng(0)  # layout only, not simulation randomness
    ang = 2 * np.pi * pop.community / k
    cx, cy = np.cos(ang) * 0.62, np.sin(ang) * 0.62
    r = 0.22 * np.sqrt(rng.random(n))
    th = 2 * np.pi * rng.random(n)
    x, y = cx + r * np.cos(th), cy + r * np.sin(th)
    return {"population_version": pop.version, "graph_version": g.version,
            "community": pop.community.tolist(), "role": g.role.tolist(), "followers": g.followers.tolist(),
            "x": np.round(x, 4).tolist(), "y": np.round(y, 4).tolist(),
            "community_names": pop.community_names,
            "community_centres": [[float(np.cos(2 * np.pi * c / k) * 0.62), float(np.sin(2 * np.pi * c / k) * 0.62)]
                                  for c in range(k)]}
