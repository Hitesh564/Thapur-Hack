"""Evidence ledger (B9): every user-visible metric with its evidence tag."""
from __future__ import annotations

from typing import Optional

from app.config import Config
from schemas import EvidenceTag, ExperimentResult, LedgerEntry, PriorResult, RunSummary

VALIDATION_LAYERS = [
    {"layer": "Engine correctness", "shows": "The simulator does what we say", "method": "Unit tests"},
    {"layer": "Input sanity", "shows": "Prior and relevance behave sensibly", "method": "Test videos"},
    {"layer": "Assumption robustness", "shows": "Conclusions do not hinge on guessed numbers",
     "method": "Two-level robustness and sensitivity ranking"},
    {"layer": "Mini backtest", "shows": "A rough real-world signal (anecdotal)", "method": "Public video pairs"},
]
NOT_VALIDATED = "Not validated against real outcomes."

# fixed (non-ranged) structural assumptions worth showing
FIXED_ASSUMPTIONS = (
    "population.n_viewers", "population.n_communities", "population.seed_pct",
    "graph.mean_out_degree", "graph.intra_community_ratio", "policy.B0", "policy.B_max",
    "policy.max_rounds", "sharing.fanout_cap", "behaviour.completion_threshold",
    "robustness.robust_threshold", "robustness.likely_threshold", "tipping.milestone_pct",
)


def build_ledger(cfg: Config, prior: Optional[PriorResult] = None,
                 relevance_by_community: Optional[dict[str, float]] = None,
                 run: Optional[RunSummary] = None, experiment: Optional[ExperimentResult] = None,
                 backtest: Optional[dict] = None) -> dict:
    entries: list[LedgerEntry] = []
    for path in cfg.ranged_paths():
        entries.append(LedgerEntry(metric=path, value=float(cfg.get(path)), tag=EvidenceTag.ASSUMED,
                                   range=cfg.range(path), note="Sampled uniformly over its range in robustness runs."))
    for path in FIXED_ASSUMPTIONS:
        entries.append(LedgerEntry(metric=path, value=cfg.get(path), tag=EvidenceTag.ASSUMED, note="Fixed assumption."))

    if prior is not None:
        rated = prior.method == "vlm_rubric"
        entries.append(LedgerEntry(
            metric="prior_result.percentile", value=prior.percentile, tag=prior.evidence_tag,
            range=(min(prior.ensemble), max(prior.ensemble)) if prior.ensemble else None,
            note="A model's judgement of the opening; not trained or validated." if rated
            else f"{prior.method}: not a model rating."))
        for dim, score in prior.rubric_scores.items():
            entries.append(LedgerEntry(metric=f"prior_result.rubric.{dim}", value=score, tag=prior.evidence_tag))
    for comm, val in (relevance_by_community or {}).items():
        entries.append(LedgerEntry(metric=f"relevance.mean.{comm}", value=val, tag=EvidenceTag.COMPUTED,
                                   note="Zero-shot embedding proxy for interest, not a measurement."))
    if run is not None:
        for k in ("unique_reach", "reach_pct", "rounds", "depth", "communities_reached", "total_shares"):
            entries.append(LedgerEntry(metric=f"run.{k}", value=float(getattr(run, k)), tag=EvidenceTag.SIMULATED,
                                       note="Output of this simulated run."))
    if experiment is not None:
        for k in ("win_rate", "effect_median", "effect_p10", "effect_p90", "assumption_stability"):
            entries.append(LedgerEntry(metric=f"experiment.{k}", value=float(getattr(experiment, k)),
                                       tag=EvidenceTag.SIMULATED, note=experiment.note))
        entries.append(LedgerEntry(metric="experiment.label", value=experiment.label, tag=EvidenceTag.SIMULATED,
                                   note=experiment.note))
    if backtest:
        entries.append(LedgerEntry(metric="backtest.pairwise_accuracy", value=backtest["accuracy"],
                                   tag=EvidenceTag.MEASURED, note=f"Anecdotal, n={backtest['n']} pairs."))
    return {
        "entries": [e.model_dump(mode="json") for e in entries],
        "validation": (f"Mini backtest: {backtest['accuracy']:.0%} pairwise accuracy on n={backtest['n']} pairs (anecdotal)."
                       if backtest else NOT_VALIDATED),
        "layers": VALIDATION_LAYERS,
    }
