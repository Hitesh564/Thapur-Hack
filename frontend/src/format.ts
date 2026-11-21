export const pct = (x: number | null | undefined, d = 0) =>
  x == null || Number.isNaN(x) ? '—' : `${(x * 100).toFixed(d)}%`
export const num = (x: number | null | undefined, d = 2) =>
  x == null || Number.isNaN(x) ? '—' : x.toFixed(d)
export const int = (x: number | null | undefined) =>
  x == null || Number.isNaN(x) ? '—' : Math.round(x).toLocaleString()
export const signed = (x: number, d = 0) => `${x > 0 ? '+' : x < 0 ? '−' : ''}${Math.abs(x).toFixed(d)}`

const NICE: Record<string, string> = {
  stay_rate: 'Stay rate',
  share_given_stay: 'Share given stay',
  exposures_per_sharer: 'Exposures per sharer',
  rec_batch_ratio: 'Recommendation batch ratio',
  early_cascade_failure: 'Early cascade failure',
  niche_community_led: 'Niche / community-led',
  hub_driven: 'Hub-driven',
  bridge_driven: 'Bridge-driven',
  cross_community: 'Cross-community',
  broad_but_shallow: 'Broad but shallow',
  deep_but_narrow: 'Deep but narrow',
  low_signal: 'Low signal',
  max_rounds: 'Max rounds',
  relevance_biased: 'Relevance-biased',
  community_targeted: 'Community-targeted',
  interest_tight: 'Interest-tight',
  'behaviour.q_min': 'Stay chance, weakest hook (q min)',
  'behaviour.q_max': 'Stay chance, strongest hook (q max)',
  'behaviour.a_rel': 'Relevance effect on staying',
  'behaviour.c_attn': 'Attention effect on staying',
  'behaviour.m_min': 'Watch fraction, weakest hook',
  'behaviour.m_max': 'Watch fraction, strongest hook',
  'behaviour.d_rel': 'Relevance effect on watch time',
  'behaviour.d_attn': 'Attention effect on watch time',
  'behaviour.d_dur': 'Duration penalty on watch time',
  'behaviour.kappa_f': 'Watch-time spread (κ)',
  'behaviour.base_like': 'Base like rate',
  'behaviour.base_comment': 'Base comment rate',
  'behaviour.base_save': 'Base save rate',
  'behaviour.base_share': 'Base share rate',
  'policy.alpha': 'Policy weight on stays (α)',
  'policy.beta': 'Policy weight on watch time (β)',
  'policy.gamma': 'Policy weight on likes (γ)',
  'policy.delta': 'Policy weight on shares (δ)',
  'policy.S0': 'Growth threshold (S0)',
  'policy.kappa': 'Batch growth speed (κ)',
  'policy.epsilon': 'Random exploration share (ε)',
  'sharing.accept_prob': 'Follower acceptance chance',
  prior_p: 'Prior p (from rating samples)',
  hook_clip: 'Real alternate opening',
  hypothetical_hook: 'Hypothetical stronger hook',
}

export const nice = (s: string) =>
  NICE[s] ?? NICE[s.replace(/^.*\.(?=[^.]+\.[^.]+$)/, '')] ?? s.replace(/^(behaviour|policy|sharing|prior)\./, '').replace(/_/g, ' ').replace(/^./, (c) => c.toUpperCase())
