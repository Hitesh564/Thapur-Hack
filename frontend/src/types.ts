// Mirrors backend/schemas.py (contracts) plus API payload shapes. Change together with schemas.py.

export type EvidenceTag = 'COMPUTED' | 'MODEL_RATED' | 'ASSUMED' | 'SIMULATED' | 'MEASURED'

export interface HookAnalysis {
  categories: string[]
  opening_window_s: number
  strengths: string[]
  weaknesses: string[]
  evidence: string[]
  suggested_improvement: string
  confidence: number
}

export interface VideoProfile {
  video_id: string
  file_hash: string
  duration_s: number
  width: number
  height: number
  fps: number
  has_audio: boolean
  topic: string
  category: string
  subtopics: string[]
  summary: string
  format: string
  narrative_structure: string
  audience_interest_tags: string[]
  transcript: string
  cut_times_s: number[]
  hook: HookAnalysis
  frame_paths: string[]
  frame_times_s: number[]
  opening_frame_paths: string[]
  opening_frame_times_s: number[]
  features: Record<string, number>
  analysis_source: string
  notes: string[]
}

export interface PriorResult {
  percentile: number
  ensemble: number[]
  method: 'vlm_rubric' | 'heuristic' | 'hypothetical'
  rubric_scores: Record<string, number>
  evidence_tag: EvidenceTag
  model: string
  rationale: string
  note: string
}

export interface RelevanceResult {
  method: 'clip' | 'keyword'
  by_community: Record<string, number>
  mean: number
  population_version: string
  note: string
  evidence_tag: EvidenceTag
}

export interface Interest { phrase: string; strength: number }
export interface Archetype {
  name: string
  community: number
  community_name: string
  interests: Interest[]
  traits: Record<string, [number, number]>
  influence_class: 'low' | 'medium' | 'high'
}

export interface VideoMeta {
  video_id: string
  file_hash: string
  filename: string
  duration_s: number
  width: number
  height: number
  fps: number
  has_audio: boolean
  vertical: boolean
  analysed?: boolean
  thumb?: string | null
  topic?: string
  prior?: number | null
}

export interface AnalysisBundle {
  profile: VideoProfile
  prior: PriorResult
  relevance: RelevanceResult
}

export interface RoundMetrics {
  round: number
  exposed: number
  stay_rate: number
  mean_watch_fraction: number
  like_rate: number
  share_rate: number
  expansion_score: number
  batch_size: number
  rec_exposures_next: number
  share_exposures_next: number
  R_rec: number
  R_share: number
  sharer_follower_edges: number
  sharer_unseen_edges: number
  sharer_accepted_edges: number
  evidence_tag: EvidenceTag
}

export interface RunSummary {
  run_id: string
  unique_reach: number
  reach_pct: number
  rounds: number
  depth: number
  communities_reached: number
  total_shares: number
  rounds_metrics: RoundMetrics[]
  pathways: string[]
  reached_milestone: boolean
  evidence_tag: EvidenceTag
}

export interface DiagnosisItem {
  code: string
  message: string
  evidence: Record<string, number>
  severity: number
  evidence_tag: EvidenceTag
}

export interface BindingConstraint {
  factor: string
  measured: number
  reference: number
  ratio: number
  needed_multiplier_for_R1: number | null
  needed_value_for_R1: number | null
  evidence_tag: EvidenceTag
}

export interface RoundRow {
  round: number
  has_next: boolean
  exposed: number
  stayers: number
  sharers: number
  share_exposures_next: number
  stay_rate: number
  share_given_stay: number
  exposures_per_sharer: number
  R_share: number
  rec_exposures_next?: number
  R_rec?: number
  R?: number
}

export interface Pooled extends Omit<RoundRow, 'round' | 'has_next'> { rec_batch_ratio?: number }

export interface RunPayload {
  run_id: string
  config: { seed: number; seeding_strategy: string; policy_variant: string; prior_method: string; video_id: string;
    assumption: { prior_percentile: number } }
  summary: RunSummary
  stop_reason: string
  community_names: string[]
  diagnosis: {
    R: number
    pooled: Pooled
    binding: BindingConstraint
    factors: BindingConstraint[]
    share_breakdown: { followers_per_sharer: number; unseen_fraction: number; acceptance: number;
      cap_and_overlap: number; exposures_per_sharer: number }
    stop: { stop_reason: string; message: string; R_last_observed_round: number }
    rounds: RoundRow[]
    by_community: Record<string, { pooled: Pooled; binding: BindingConstraint }>
    bottlenecks: DiagnosisItem[]
    note: string
  }
  pathways: { primary: string | null; matches: string[]; evidence: Record<string, number>;
    descriptions: Record<string, string>; note: string }
  attribution: {
    table: { channel: string; source_community: string; source_role: string; target_community: string;
      exposures: number; share_of_reach: number }[]
    by_channel: Record<string, number>
    share_cross_community: number
  }
  replay: { round: number; exposures: number; by_channel: Record<string, number>; by_community: number[];
    stays: number; completions: number; likes: number; shares: number }[]
}

export interface Cascade {
  run_id: string
  viewer: number[]
  round: number[]
  channel: number[]
  source: number[]
  stay: number[]
  complete: number[]
  like: number[]
  share: number[]
  channels: string[]
}

export interface WorldLayout {
  community: number[]
  role: number[]
  followers: number[]
  x: number[]
  y: number[]
  community_names: string[]
  community_centres: [number, number][]
}

export interface ExperimentResult {
  experiment_id: string
  win_rate: number
  effect_median: number
  effect_p10: number
  effect_p90: number
  assumption_stability: number
  label: 'Robust' | 'Likely' | 'Inconclusive'
  sensitivity: Record<string, number>
  note: string
  evidence_tag: EvidenceTag
}

export interface ExperimentRecord {
  config: { experiment_id: string; variable: string; baseline: Record<string, unknown>; variant: Record<string, unknown>;
    held_fixed: string[]; n_seeds: number; n_assumption_samples: number }
  result: ExperimentResult
  metric: string
  seeds: number[]
  assumption_samples: { sample_id: number; params: Record<string, number>; p_baseline: number; p_variant: number }[]
  baseline_values: number[][]
  variant_values: number[][]
}

export interface TippingPoint { p: number; milestone_rate: number; mean_reach_pct: number; reach_p10: number; reach_p90: number }
export interface Tipping {
  points: TippingPoint[]
  points_uncertain?: TippingPoint[]
  video_p: number
  milestone_rate_at_video: number
  milestone_rate_at_video_uncertain?: number
  steep_range: [number, number] | null
  position: string
  message: string
  milestone_pct: number
  n_seeds: number
  note: string
  note_uncertain?: string
}

export interface Attention {
  windows: { window: number; start_s: number; end_s: number; frames_removed: number; p_without: number; change: number; within_noise: boolean }[]
  baseline_p: number
  noise: number
  message: string
  label: string
  evidence_tag: EvidenceTag
}

export interface LedgerEntry {
  metric: string
  value: number | string | null
  tag: EvidenceTag
  range: [number, number] | null
  note: string
}
export interface Ledger {
  entries: LedgerEntry[]
  validation: string
  layers: { layer: string; shows: string; method: string }[]
}

export interface Audience {
  population_version: string
  graph_version: string
  n_viewers: number
  communities: { community: number; name: string; viewers: number; hubs: number; bridges: number; archetypes: string[];
    mean_relevance?: number }[]
  archetypes: Archetype[]
  note: string
}
