"""Viralyst contracts (Pydantic v2). Change only together with tests and frontend types."""
from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class EvidenceTag(str, Enum):
    COMPUTED = "COMPUTED"
    MODEL_RATED = "MODEL_RATED"
    ASSUMED = "ASSUMED"
    SIMULATED = "SIMULATED"
    MEASURED = "MEASURED"


class HookAnalysis(BaseModel):
    categories: list[str] = []
    opening_window_s: float = 3.0
    strengths: list[str] = []
    weaknesses: list[str] = []
    evidence: list[str] = []          # e.g. "0.8s: text overlay asks a question"
    suggested_improvement: str = ""
    confidence: float = Field(0.5, ge=0, le=1)


class VideoProfile(BaseModel):
    video_id: str
    file_hash: str
    duration_s: float
    width: int = 0
    height: int = 0
    fps: float = 0
    has_audio: bool = False
    topic: str = ""
    category: str = ""
    subtopics: list[str] = []
    summary: str = ""
    format: str = ""
    narrative_structure: str = ""
    audience_interest_tags: list[str] = []
    transcript: str = ""
    cut_times_s: list[float] = []
    hook: HookAnalysis = HookAnalysis()
    frame_paths: list[str] = []
    frame_times_s: list[float] = []
    opening_frame_paths: list[str] = []
    opening_frame_times_s: list[float] = []
    features: dict[str, float] = {}       # COMPUTED: cut_rate, motion_energy, opening_motion, ...
    analysis_source: str = "none"         # "vlm:<model>" | "cache" | "none" (display-only fields empty)
    notes: list[str] = []                 # fallbacks and gaps, shown in the UI


class PriorResult(BaseModel):
    percentile: float = Field(ge=0, le=1)
    ensemble: list[float] = []
    method: Literal["vlm_rubric", "heuristic", "hypothetical"] = "vlm_rubric"
    rubric_scores: dict[str, float] = {}
    evidence_tag: EvidenceTag = EvidenceTag.MODEL_RATED
    model: str = ""
    rationale: str = ""
    note: str = ""


class RelevanceResult(BaseModel):
    method: Literal["clip", "keyword"]
    by_community: dict[str, float] = {}   # community name -> mean relevance
    mean: float = 0.0
    population_version: str = ""
    note: str = ""
    evidence_tag: EvidenceTag = EvidenceTag.COMPUTED


TRAIT_KEYS = ("attention", "like", "comment", "save", "share")


class Interest(BaseModel):
    phrase: str
    strength: float = Field(ge=0, le=1)


class Archetype(BaseModel):
    """A fictional audience archetype. No sensitive traits (ethnicity, religion, politics, health)."""
    name: str
    community: int = Field(ge=0)
    community_name: str
    interests: list[Interest] = Field(min_length=3, max_length=6)
    traits: dict[str, tuple[float, float]]   # trait -> (lo, hi); keys = TRAIT_KEYS
    influence_class: Literal["low", "medium", "high"] = "medium"


class Viewer(BaseModel):
    viewer_id: int
    archetype: str
    community: int
    role: Literal["hub", "bridge", "other"] = "other"
    followers: int = 0
    attention: float = 0.5
    like_tendency: float = 1.0
    comment_tendency: float = 1.0
    save_tendency: float = 1.0
    share_tendency: float = 1.0
    relevance: Optional[float] = None


class PopulationMeta(BaseModel):
    version: str
    seed: int
    n_viewers: int
    n_communities: int
    graph_version: str


class AssumptionSample(BaseModel):
    sample_id: int
    params: dict[str, float]
    prior_percentile: float


class RunConfig(BaseModel):
    run_id: str
    video_id: str
    seed: int
    assumption: AssumptionSample
    seeding_strategy: str = "mixed"
    policy_variant: str = "default"
    population_version: str
    graph_version: str
    prior_method: str


class Event(BaseModel):
    run_id: str
    round: int
    viewer_id: int
    event_type: Literal["exposure", "skip", "watch", "partial_view", "completion",
                        "like", "comment", "save", "share"]
    channel: Literal["seed", "rec", "share"]
    source_viewer: Optional[int] = None
    community: int
    role: str = "other"
    reason_code: str = ""
    probability: Optional[float] = None


class RoundMetrics(BaseModel):
    round: int
    exposed: int
    stay_rate: float
    mean_watch_fraction: float
    like_rate: float
    share_rate: float
    expansion_score: float
    batch_size: int
    rec_exposures_next: int = 0
    share_exposures_next: int = 0
    R_rec: float = 0
    R_share: float = 0
    # exposures_per_sharer breakdown (display): edges from this round's sharers to followers
    sharer_follower_edges: int = 0
    sharer_unseen_edges: int = 0
    sharer_accepted_edges: int = 0
    evidence_tag: EvidenceTag = EvidenceTag.SIMULATED


class RunSummary(BaseModel):
    run_id: str
    unique_reach: int
    reach_pct: float
    rounds: int
    depth: int
    communities_reached: int
    total_shares: int
    rounds_metrics: list[RoundMetrics] = []
    pathways: list[str] = []
    reached_milestone: bool = False
    evidence_tag: EvidenceTag = EvidenceTag.SIMULATED


class DiagnosisItem(BaseModel):
    code: str                      # e.g. "retention_weakness"
    message: str                   # must say "in this simulated run"
    evidence: dict[str, float] = {}
    severity: float = Field(0, ge=0, le=1)
    evidence_tag: EvidenceTag = EvidenceTag.SIMULATED


class BindingConstraint(BaseModel):
    factor: str
    measured: float
    reference: float
    ratio: float
    needed_multiplier_for_R1: Optional[float] = None
    needed_value_for_R1: Optional[float] = None    # value this factor needs, others fixed
    evidence_tag: EvidenceTag = EvidenceTag.SIMULATED


class ExperimentConfig(BaseModel):
    experiment_id: str
    variable: Literal["hook_clip", "hypothetical_hook", "seeding", "policy"]
    baseline: dict
    variant: dict
    held_fixed: list[str] = []
    n_seeds: int = 30
    n_assumption_samples: int = 10


class ExperimentResult(BaseModel):
    experiment_id: str
    win_rate: float
    effect_median: float
    effect_p10: float
    effect_p90: float
    assumption_stability: float
    label: Literal["Robust", "Likely", "Inconclusive"]
    sensitivity: dict[str, float] = {}         # parameter -> rank correlation
    note: str = "Labels are our own cut-offs, not statistical tests."
    evidence_tag: EvidenceTag = EvidenceTag.SIMULATED


class LedgerEntry(BaseModel):
    metric: str
    value: Optional[float | str] = None
    tag: EvidenceTag
    range: Optional[tuple[float, float]] = None
    note: str = ""
