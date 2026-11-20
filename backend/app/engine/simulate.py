"""Simulation engine (A5): vectorised round loop with common random numbers.

Randomness: one uniform tensor ``U[round, viewer, draw]`` per seed.
* Behaviour draws (stay, watch, like, comment, save, share) read ``U[0, v, d]``: a viewer's
  personal dice, identical whichever round they are exposed in.
* Distribution draws (accept, tiebreak, select) read ``U[t, v, d]``.
Nothing is indexed by an event counter. Headless: no UI, no network, no LLM.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from app.behaviour.chain import ACTIONS, evaluate
from app.config import Params
from app.eventlog.log import CH, EventLog, build_event_log
from app.graph.sbm import Graph
from app.personas.population import Population
from app.policy.distribution import resolve_policy, select_rec, select_seeds, share_exposures
from schemas import AssumptionSample, RoundMetrics, RunConfig, RunSummary

DRAWS = ("stay", "watch", "like", "comment", "save", "share", "accept", "tiebreak", "select")
D = {name: i for i, name in enumerate(DRAWS)}
PERSONAL_DRAWS = ("stay", "watch") + ACTIONS


@dataclass
class World:
    """Fixed across every scenario of an experiment."""
    population: Population
    graph: Graph


@dataclass
class SimInputs:
    world: World
    relevance: np.ndarray            # (n,) in [0, 1], COMPUTED
    prior_p: float                   # MODEL_RATED or ASSUMED
    duration_s: float
    params: Params                   # ASSUMED (one assumption sample)
    seeding_strategy: Optional[str] = None   # None -> config default
    policy_variant: str = "default"
    target_community: Optional[int] = None
    video_id: str = "video"
    prior_method: str = "vlm_rubric"
    assumption_sample_id: int = 0


@dataclass
class RunResult:
    run_id: str
    config: RunConfig
    summary: RunSummary
    rounds: list[RoundMetrics]
    log: EventLog
    state: dict[str, np.ndarray]
    stop_reason: str
    repro: dict = field(default_factory=dict)


def max_rounds(P: Params) -> int:
    return int(P["policy.max_rounds"])


def draw_uniforms(seed: int, n_rounds: int, n_viewers: int) -> np.ndarray:
    return np.random.default_rng(int(seed)).random((n_rounds, n_viewers, len(DRAWS)))


def simulate(inp: SimInputs, seed: int, U: Optional[np.ndarray] = None) -> RunResult:
    P = inp.params
    pop, graph = inp.world.population, inp.world.graph
    n = pop.n
    T = max_rounds(P)
    if U is None:
        U = draw_uniforms(seed, T, n)
    if U.shape != (T, n, len(DRAWS)):
        raise ValueError(f"U has shape {U.shape}, expected {(T, n, len(DRAWS))}")
    strategy = inp.seeding_strategy or str(P["seeding.strategy"])
    pol = resolve_policy(P, inp.policy_variant)
    rel = np.asarray(inp.relevance, dtype=float)
    mean_rel = float(rel.mean())
    p = float(inp.prior_p)
    tend = {a: pop.traits[a] for a in ACTIONS}
    attn = pop.traits["attention"]
    personal = {d: U[0, :, D[d]] for d in PERSONAL_DRAWS}
    accept_prob = float(P["sharing.accept_prob"])
    fanout = int(P["sharing.fanout_cap"])
    budget = int(np.floor(float(P["policy.exposure_budget_pct"]) * n))
    low_stop = int(P["policy.low_signal_stop_rounds"])

    st: dict[str, np.ndarray] = {
        "exposed": np.zeros(n, bool), "round": np.full(n, -1, np.int64),
        "channel": np.full(n, -1, np.int8), "source": np.full(n, -1, np.int64),
        "generation": np.full(n, -1, np.int64), "stay": np.zeros(n, bool),
        "p_stay": np.zeros(n), "watch_fraction": np.zeros(n), "complete": np.zeros(n, bool),
        "reason": np.zeros(n, np.int8),
    }
    for a in ACTIONS:
        st[f"act_{a}"] = np.zeros(n, bool)
        st[f"p_{a}"] = np.zeros(n)

    def expose(ids: np.ndarray, t: int, ch: int, src: Optional[np.ndarray] = None) -> None:
        st["exposed"][ids] = True
        st["round"][ids] = t
        st["channel"][ids] = ch
        if src is not None:
            st["source"][ids] = src
            st["generation"][ids] = st["generation"][src] + 1
        else:
            st["generation"][ids] = 0

    rounds: list[RoundMetrics] = []
    n_seed = max(1, int(round(float(P["population.seed_pct"]) * n)))
    new = select_seeds(P, strategy, min(n_seed, budget), rel, pop.community,
                       U[0, :, D["tiebreak"]], U[0, :, D["select"]], inp.target_community)
    expose(new, 0, CH["seed"])
    B = float(pol.B0)
    batch_used = int(new.size)
    S_prev = 0.0
    low_count = 0
    stop_reason = "max_rounds"

    for t in range(T):
        if t > 0:
            sharers = np.flatnonzero((st["round"] == t - 1) & st["act_share"])
            remaining = budget - int(st["exposed"].sum())
            sh_t, sh_s = share_exposures(sharers, graph.adj, ~st["exposed"], U[t, :, D["accept"]],
                                         U[t, :, D["tiebreak"]], accept_prob, fanout)
            sh_t, sh_s = sh_t[:max(remaining, 0)], sh_s[:max(remaining, 0)]
            expose(sh_t, t, CH["share"], sh_s)
            B = pol.next_batch(B, S_prev)
            remaining = budget - int(st["exposed"].sum())
            rec_n = max(0, min(int(round(B)), remaining))
            rec = select_rec(rec_n, pol.epsilon, rel, ~st["exposed"], U[t, :, D["tiebreak"]], U[t, :, D["select"]])
            expose(rec, t, CH["rec"])
            batch_used = int(round(B))
            prev = rounds[-1]
            E_prev = max(prev.exposed, 1)
            prev.rec_exposures_next = int(rec.size)
            prev.share_exposures_next = int(sh_t.size)
            prev.R_rec = rec.size / E_prev
            prev.R_share = sh_t.size / E_prev
            new = np.sort(np.concatenate([sh_t, rec]))
            if new.size == 0:
                stop_reason = "no_exposures"
                break

        out = evaluate(P, p, rel[new], mean_rel, attn[new], {a: tend[a][new] for a in ACTIONS},
                       inp.duration_s, {d: personal[d][new] for d in PERSONAL_DRAWS})
        st["stay"][new] = out.stay
        st["p_stay"][new] = out.p_stay
        st["watch_fraction"][new] = out.watch_fraction
        st["complete"][new] = out.complete
        st["reason"][new] = out.reason
        for a in ACTIONS:
            st[f"act_{a}"][new] = out.acts[a]
            st[f"p_{a}"][new] = out.probs[a]

        E = int(new.size)
        n_stay = int(out.stay.sum())
        stay_rate = n_stay / E
        wf = float(out.watch_fraction[out.stay].mean()) if n_stay else 0.0
        like_rate = float(out.acts["like"].mean())
        share_rate = float(out.acts["share"].mean())
        S = pol.score(stay_rate, wf, like_rate, share_rate)
        rounds.append(RoundMetrics(round=t, exposed=E, stay_rate=stay_rate, mean_watch_fraction=wf,
                                   like_rate=like_rate, share_rate=share_rate, expansion_score=S,
                                   batch_size=batch_used))
        S_prev = S
        low_count = low_count + 1 if S < pol.S0 else 0
        if low_count >= low_stop:
            stop_reason = "low_signal"
            break
        if int(st["exposed"].sum()) >= budget:
            stop_reason = "budget"
            break
        if not (~st["exposed"]).any():
            stop_reason = "no_eligible_viewers"
            break

    repro = {
        "seed": int(seed), "assumption_sample_id": inp.assumption_sample_id,
        "population_version": pop.version, "graph_version": graph.version,
        "prior_method": inp.prior_method, "prior_p": p, "policy_variant": inp.policy_variant,
        "seeding_strategy": strategy, "params": P.sampled_values(),
        "relevance_hash": hashlib.sha256(np.round(rel, 12).tobytes()).hexdigest()[:12],
        "duration_s": inp.duration_s, "video_id": inp.video_id,
    }
    run_id = "run-" + hashlib.sha256(json.dumps(repro, sort_keys=True).encode()).hexdigest()[:12]
    log = build_event_log(run_id, st, pop.community, graph.role)

    reach = int(st["exposed"].sum())
    reach_pct = reach / n
    summary = RunSummary(
        run_id=run_id, unique_reach=reach, reach_pct=reach_pct, rounds=len(rounds),
        depth=int(st["generation"].max()) if reach else 0,
        communities_reached=int(np.unique(pop.community[st["exposed"]]).size),
        total_shares=int(st["act_share"].sum()), rounds_metrics=rounds,
        reached_milestone=bool(reach_pct >= float(P["tipping.milestone_pct"])),
    )
    config = RunConfig(
        run_id=run_id, video_id=inp.video_id, seed=int(seed),
        assumption=AssumptionSample(sample_id=inp.assumption_sample_id, params=P.sampled_values(), prior_percentile=p),
        seeding_strategy=strategy, policy_variant=inp.policy_variant,
        population_version=pop.version, graph_version=graph.version, prior_method=inp.prior_method,
    )
    return RunResult(run_id, config, summary, rounds, log, st, stop_reason, repro)
