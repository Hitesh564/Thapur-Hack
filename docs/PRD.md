# Viralyst — Product Requirements Document (v1.2, dataset-free)

Status: locked for hackathon build. Time budget: **20 to 24 hours**, team of about four.
Audience of this file: Claude Code and the four team members. Read it fully before writing code.

---

## 0. How to use this document

- **Build in priority order:** MUST first, then SHOULD, then COULD. Never start a SHOULD while a MUST is broken.
- **Everything numeric that is not data is a placeholder.** Every default, range and threshold lives in `config.yaml`. Never hard-code a number from this document in source code. Read it from config.
- **No dataset is used.** Nothing is trained by us. Do not add a training pipeline. See section 5 for the "prior" and section 12 for how we validate honestly.
- **Honesty rules are product requirements,** not polish (section 2.3 and section 12). Every number shown to a user carries an evidence tag.
- When this document and `CLAUDE.md` disagree, `CLAUDE.md` wins on process and this document wins on product behaviour. If something is unspecified, choose the simplest option, write the choice into `docs/DECISIONS.md`, and continue.

---

## 1. Product summary

Viralyst is a **pre-publication experiment lab for short-form video**. A creator uploads a draft. We analyse the video, simulate how a synthetic audience watches it and spreads it through a modelled social network, and let the creator run **controlled experiments** on that simulation, such as swapping the opening of the video. Every result states how stable it is and which parts are evidence and which are assumption.

**The claim we pitch:**
> From viral scores to propagation experiments you can trust. Every verdict comes with its seed-robustness, its assumption-robustness and an evidence label.

**What it is not:** a view predictor, a copy of any platform's algorithm, or a claim about real people. It is a *what-if* tool under stated assumptions.

### 1.1 How it relates to simple audience simulators

Simple simulators (for example a hackathon project that gives each AI persona a personality and lets it skip, watch, like and share) have the same basic loop: expose a viewer, decide skip or watch, decide like or share, show the video to more viewers. Viralyst uses the same loop, but:

1. Viewers are numeric agents driven by probabilities, not LLM calls, so thousands run in milliseconds and every run is repeatable.
2. There are two spread channels (recommendation and social shares) with separate accounting.
3. The signature feature is the **Propagation Lab**: matched A/B experiments, repeated over random seeds *and* over uncertain assumptions, with diagnosis and attribution.

---

## 2. Scope

### 2.1 Users
Primary: short-form creators with a draft. Secondary: small brand teams, student creators, hackathon judges.

Core question: "I have this video ready. Which audiences might respond, where could it stall, and what change gives the biggest, most reliable improvement?"

### 2.2 Input
Any category of short-form video. MP4 preferred, 5 to 60 seconds, vertical preferred, audio optional. Content-agnostic (not only AI or tech videos).

### 2.3 Non-goals and claim rules

We never:
- guarantee virality or predict views;
- claim to reproduce Instagram, TikTok, Snapchat or YouTube ranking;
- claim personas are real people or represent real demographics;
- claim causal effects of a hook change;
- call an LLM per viewer or per round;
- require account access or scrape creator analytics;
- build a video editor or train a model.

UI and generated text must use wording such as "in this simulation", "the model suggests", "assumed", never "will", "predicts views", or "Instagram's algorithm".

---

## 3. USPs and how each is implemented

| # | USP | Implementation | Demo moment |
|---|---|---|---|
| 1 | **Two-level robustness** (seeds and assumptions) | Outer loop: A samples of uncertain parameters drawn from ranges in `config.yaml`. Inner loop: S seeds. Report win rate over all A×S pairs and the share of assumption samples where the variant's median beats the baseline's | "Variant B wins in 87% of runs and in 9 of 10 assumption settings" |
| 2 | **R diagnosis** | Reproduction number per round and community, decomposed into exactly measured factors; the factor furthest below its reference value is the binding constraint (section 11.3) | "R is 0.62. The weak factor is share rate in the Food community" |
| 3 | **Reach attribution** | Every exposure records channel, source community and source role; a flow table and diagram | "62% of reach came from recommendation; shares never left the seed community" |
| 4 | **Tipping-point curve** | Sweep the prior over a grid, run S seeds per point, plot the share of runs reaching the growth milestone, mark the video | "You sit on the steep part of the curve" |
| 5 | **Matched counterfactual with real alternate clip** | The creator uploads an alternate opening. Both clips run through the same pipeline. Paired runs share one random tensor | "Same audience, same graph, same dice; only the opening changed" |
| 6 | **Evidence ledger** | Every metric carries a tag (section 12.1) and one screen lists them | The assumed-vs-computed table, live |

---

## 4. Architecture

```
Upload video
   │
   ▼
Preprocess (frames, audio, transcript, cuts)  ──►  Video profile (cached by file hash)
   │                                                    │
   ├──► Prior (VLM rubric, pluggable)  ── percentile p + ensemble spread  [MODEL_RATED]
   ├──► Relevance (CLIP-style embeddings: persona interests vs frames)     [COMPUTED]
   └──► Behaviour + policy parameters (sampled from ranges)                [ASSUMED]
                          │
                          ▼
              Simulation engine (vectorised, no LLM per viewer)           [SIMULATED]
   population ─ graph ─ behaviour chain ─ policy ─ rounds ─ paired random tensor ─ event log
                          │
                          ▼
              Propagation Lab: experiments, robustness, diagnosis, attribution,
              tipping-point curve, evidence ledger
                          │
                          ▼
                   React UI (replay, diagnosis, Lab, ledger)
```

### 4.1 Backend modules (`backend/app/`)

| Module | Responsibility |
|---|---|
| `ingestion` | Validate upload (type, decode, duration, dimensions, fps, audio), store |
| `preprocessing` | Frames (uniform plus dense opening), audio, transcript, scene cuts; cache by hash |
| `video_analysis` | One VLM call per video for qualitative analysis and hook breakdown (validated JSON) |
| `prior` | `prior(video_profile) -> PriorResult`; pluggable implementations |
| `personas` | Load or generate archetypes, sample the population, persist with version and seed |
| `relevance` | Persona-video relevance from embeddings |
| `graph` | Community-structured directed graph, roles, sparse adjacency |
| `behaviour` | Probability chain: stay, watch fraction, like, comment, save, share |
| `policy` | Expansion score, batch size, recommendation selection |
| `engine` | Round loop, common random numbers, stop conditions |
| `eventlog` | Columnar event log, run summary |
| `diagnostics` | R decomposition, bottlenecks, pathways, attribution |
| `experiments` | Matched A/B, two-level robustness, tipping-point sweep, experiment records |
| `ledger` | Evidence tags and the ledger endpoint |
| `api` | FastAPI orchestration |

Stack: Python 3.11+, FastAPI, NumPy and SciPy (sparse), NetworkX for graph generation, Pydantic for contracts, pytest. Video: FFmpeg and OpenCV, PySceneDetect. Frontend: React with Vite (Streamlit only if nobody on the team can do React). Storage: SQLite and Parquet or JSON files. The engine must run with no UI and no network.

---

## 5. The prior (no dataset)

The **prior** is a number `p` in [0, 1] meaning "how strong this video's opening and content look as a short-form video". It is **not** trained and **not** validated against real outcomes. It is a structured model judgement.

### 5.1 Interface (never bypass)
```
prior(video_profile) -> PriorResult { percentile: float, ensemble: list[float], method: str, evidence_tag: str }
```
The engine only ever calls this function and uses `percentile` and `ensemble`.

### 5.2 Default implementation: VLM rubric (`method = "vlm_rubric"`, tag `MODEL_RATED`)
1. Send the sampled frames and the transcript to a vision-language model, with a fixed rubric prompt stored in `config.yaml` or `backend/app/prompts/`.
2. Rubric dimensions, each scored 1 to 10: immediate clarity, curiosity, visual impact, pacing, payoff clarity, audience fit. Return JSON only; validate with Pydantic.
3. Repeat the call `prior.vlm_samples` times (default 3) with sampling enabled. Each sample gives `p_i = mean(dimension scores) / 10`.
4. `percentile = clip(mean(p_i), 0.05, 0.95)`; `ensemble = [p_i ...]`. The spread is the model uncertainty, and the robustness loop samples from it.

### 5.3 Fallback (`method = "heuristic"`, tag `ASSUMED`)
If the VLM is unavailable, compute `p` from basic features (duration band, cut rate, motion energy, audio present) with a simple documented formula, centred near 0.5, and mark the result as a heuristic in the UI.

### 5.4 Alternate clip
The alternate opening runs through the identical pipeline and the identical prompt. Never compare scores produced with different prompts.

### 5.5 Hypothetical profile experiment
A labelled "hypothetical" experiment adds `delta` (config, default +0.1) to `percentile` to answer "what if the hook were stronger". It is never presented as a real clip result.

---

## 6. Video understanding and hook analysis

Preprocessing: validate; extract metadata; sample frames (uniform N, plus extra frames from the first seconds, counts in config); extract audio; transcribe speech if present; detect cuts; cache everything by file hash.

Qualitative analysis (one VLM call, validated JSON, cached): topic, category, subtopics, summary, format, narrative structure, audience-interest tags, **hook** (categories, opening window, strengths, weaknesses, evidence from frames or speech with timestamps, suggested improvement, confidence).

**Rule:** the VLM's qualitative hook fields are for display and explanation only. They never feed the engine. The engine is driven by the prior, embedding relevance and assumed parameters.

Attention-risk timeline (SHOULD): split the video into 8 windows; for each window re-run the prior with that window's frames removed and report the change. Label it "where the rating is sensitive", not "where viewers drop off".

If any external call fails, fall back to a cached analysis or basic features and say so in the UI.

---

## 7. Audience

### 7.1 Archetypes
50 archetypes across 6 to 10 interest communities, covering many content categories (not only tech). Each has: name, 3 to 6 interest phrases with strengths, trait ranges (attention, novelty preference, watch, like, comment, save and share tendencies), community, influence class. Generate once with an LLM and save to `backend/data/archetypes.json`; **also ship a hand-written fallback file** so the app never depends on the network. No sensitive traits (ethnicity, religion, politics, health).

### 7.2 Population
Sample `population.n_viewers` (default 4000) viewers from archetypes with bounded noise on traits and heavy-tailed influence. Persist `population.parquet` (or JSON) with a version and seed. Identities never change between videos.

### 7.3 Relevance (COMPUTED)
1. Embed each viewer's interest phrases with the text encoder of a CLIP-style model; the viewer vector is the strength-weighted mean, normalised.
2. Embed the video's frames with the image encoder; mean-pool and normalise.
3. `cos_v` = cosine similarity. Then
   `rel_v = w_rel * sigmoid(k * (cos_v - mean(cos)) / std(cos)) + (1 - w_rel) * clip((cos_v - cos_lo) / (cos_hi - cos_lo), 0, 1)`
   with `k`, `w_rel`, `cos_lo`, `cos_hi` from config (placeholders; tune by inspection). The first term spreads viewers within the population; the second term keeps an absolute floor so a video relevant to nobody is not made relevant by normalisation.
4. Document the limitation: this is a zero-shot proxy for interest, not a measurement.

---

## 8. Behaviour engine

No LLM call. Each exposed viewer follows this chain, with all randomness read from the pre-drawn uniform tensor (section 10).

1. **Stay** (continues past the opening):
   `q0 = q_min + (q_max - q_min) * p`
   `P(stay) = sigmoid( logit(q0) + a * (rel - mean_rel) + c * (attn - 0.5) )`
2. **Watch fraction** given stay: drawn from a Beta distribution via the inverse CDF of a pre-drawn uniform. Mean `m = sigmoid( logit(m0) + d_rel * (rel - mean_rel) + d_attn * (attn - 0.5) - d_dur * ln(duration / 30) )` with `m0 = m_min + (m_max - m_min) * p`; concentration `kappa_f` from config. Completion means watch fraction at or above `completion_threshold`.
3. **Like, comment, save** given stay:
   `P(action) = clip( base_action * tendency_action * (0.5 + rel) * (0.5 + watch_fraction), 0, 1 )`
4. **Share** given stay: the same form with `base_share` and `tendency_share`.
5. Record the dominant term as a **reason code** (prior, relevance, attention, duration).

A viewer who skips cannot like or share. All parameters in config have `default`, `min`, `max`. The robustness loop samples them uniformly between `min` and `max`.

---

## 9. Network and distribution

### 9.1 Graph
Directed, degree-corrected stochastic block model with `n_communities` communities. Edges are denser inside a community. Influence sets follower counts, so a few hubs exist. A small share of viewers get extra cross-community edges (bridges). Roles are **computed**, not assigned: hub = top `hub_pct` by follower count; bridge = top `bridge_pct` by fraction of edges that cross communities. Generate with NetworkX, convert to a SciPy CSR matrix, save with a version and seed. The graph is synthetic and separate from personas.

### 9.2 Two channels
| Channel | Mechanism | Logged as |
|---|---|---|
| Recommendation | After each round the policy gives a batch size `B`. Select `B` unseen viewers: a share `epsilon` uniformly at random, the rest by weighted sampling on relevance (Gumbel-top-k using each viewer's pre-drawn uniform) | `channel=rec` |
| Social share | Each sharer exposes unseen followers: a follower is eligible if its own pre-drawn acceptance uniform is below `accept_prob`; at most `fanout_cap` per sharer, taken in tie-break order | `channel=share`, with `source_viewer` |

### 9.3 Distribution policy (ours, not any platform's)
```
S_t = alpha*stay_t + beta*watchfrac_t + gamma*like_t + delta*share_t
B_{t+1} = clip( B_t * exp(kappa * (S_t - S_0)), 0, B_max )
```
The signal choice is inspired by publicly stated ranking signals (watch time, likes, sends) but is our own documented approximation. Policy variants for experiments: `conservative` (higher S_0), `exploratory` (higher epsilon), `interest_tight` (epsilon = 0).

### 9.4 Seeding
Strategies: `random`, `relevance_biased`, `mixed`, `community_targeted`. Default seed size is `seed_pct` of the population. The strategy is logged and is a Lab variable.

### 9.5 Stop conditions
No eligible unseen viewer; expansion score below `S_0` for two consecutive rounds; `max_rounds` reached; exposure budget used. Re-exposure is off.

---

## 10. Simulation engine

- Vectorised NumPy arrays per viewer: exposed, round_exposed, channel, source, stay, watch_fraction, like, comment, save, share.
- **Common random numbers (required).** For each seed, draw one tensor `U` of shape `(max_rounds, n_viewers, n_draws)` of uniform numbers, indexed by round, viewer and decision type: stay, watch_fraction, like, comment, save, share, accept, tiebreak. Both scenarios of a pair read the same `U`. **Only viewer-indexed uniforms are allowed** (never an event counter), so a viewer faces the same dice in both scenarios.
- Event log: columnar arrays flushed to Parquet or SQLite. Fields: run_id, round, viewer_id, event_type, channel, source_viewer, community, role, reason_code, probability. Event types: exposure, skip, watch, partial_view, completion, like, comment, save, share. The whole cascade must be reconstructable from the log.
- Reproducibility record per run: seed, assumption_sample_id, population/graph/prior/policy versions, parameters.
- Performance targets (estimates to confirm): one run on 4000 viewers well under a second; a matched experiment with 30 seeds × 10 assumption samples × 2 scenarios in under two minutes (use multiprocessing if needed). If missed, reduce seeds and samples first.

---

## 11. Propagation Lab (signature feature, MUST)

### 11.1 Experiment protocol
1. Choose one variable: hook clip (real alternate clip), hypothetical hook profile, seed strategy, or policy variant.
2. Hold everything else fixed: population, graph, prior method, policy (unless it is the variable), exposure budget, random tensors.
3. Draw A assumption samples (parameters from their ranges, `p` from the prior's ensemble).
4. For each sample and each of S seeds, run baseline and variant on the same tensor.
5. Compute paired differences per metric.
6. Save an experiment record: id, baseline, variant, changed and held-fixed variables, versions, seeds, assumption samples, repetitions, parameters, results.

### 11.2 Verdict
| Output | Definition |
|---|---|
| Win rate | Share of paired runs where the variant has higher unique reach |
| Effect | Median paired difference with 10th to 90th percentile interval |
| Assumption stability | Share of assumption samples where the variant's median beats the baseline's |
| Label | **Robust** if both rates are at least `robust_threshold` (default 0.8); **Likely** if both at least `likely_threshold` (0.6); else **Inconclusive** |
| Sensitivity | Rank correlation between each sampled parameter and the outcome |

These labels are our own cut-offs, not statistical tests; the UI says so. Never present a single run as a reliable result.

### 11.3 Diagnosis with the reproduction number R
Per round t, with `E_t` viewers exposed in round t:
```
R_t^rec   = rec_exposures_{t+1}   / E_t
R_t^share = share_exposures_{t+1} / E_t
R_t       = R_t^rec + R_t^share
R_t^share = stay_rate_t * share_given_stay_t * exposures_per_sharer_t     (exact identity)
```
`exposures_per_sharer` can be broken down further into mean eligible followers × unseen fraction × acceptance, for display. A test must confirm the identity holds on a real log.

**Binding constraint.** For each factor (stay rate, share given stay, exposures per sharer, recommendation batch ratio), compute `ratio = measured / reference`, where the reference values are "healthy" settings in `config.yaml`. The factor with the lowest ratio is the binding constraint. Also report the single multiplier needed to bring R to 1, `m = (1 - R_rec) / R_share`, and what value each factor would need to reach with the others fixed. (All multiplicative share factors need the same multiplier, so the multiplier alone does not identify the culprit; the reference gap does.) Compute per round and per community.

Bottlenecks, each citing its numbers: weak initial attention; low relevance in a newly reached community; retention weakness; insufficient sharing; distribution threshold not reached; community saturation; failure to cross communities; lack of effective bridges; limited candidate audience; stochastic failure (same scenario reaches the milestone in most other seeds). Thresholds in config. Diagnostic text must say "in this simulated run".

### 11.4 Pathway classification (computed from the log, rules in config)
Early cascade failure; niche or community-led; hub-driven; bridge-driven; cross-community; broad but shallow; deep but narrow. A run may match several; show the primary by priority order and list the rest. These are descriptions of simulated outcomes, not validated viral categories.

### 11.5 Reach attribution
Table and diagram from the log: channel → source community → source role (hub, bridge, other).

### 11.6 Tipping-point curve (SHOULD)
Sweep `p` over `tipping.grid_points` values; for each run S seeds; plot the share of runs reaching `milestone_pct` of the population. Mark the video's `p`. Report in words how close the video is to the steep part.

---

## 12. Validation and the Evidence ledger

We have no outcome dataset, so we do **not** claim predictive accuracy. We validate in layers and say what each layer shows.

| Layer | What it shows | Method |
|---|---|---|
| 1. Engine correctness | The simulator does what we say | Unit tests (section 17) |
| 2. Input sanity | Prior and relevance behave sensibly | Test videos: a cooking video is more relevant to food than gaming; a clearly weak opening rates below a clearly strong one |
| 3. Assumption robustness | Conclusions do not hinge on guessed numbers | Two-level robustness and sensitivity ranking |
| 4. Mini backtest (SHOULD, optional) | A rough real-world signal | 15 to 20 pairs of public videos from the same creator where one clearly outperformed; collect public view counts by hand into `backend/data/backtest_pairs.csv`; run both through the prior; report pairwise accuracy **with its sample size, labelled anecdotal** |

### 12.1 Evidence tags (shown as a badge on every metric)
| Tag | Meaning |
|---|---|
| `COMPUTED` | Deterministic function of the video or inputs (for example embedding relevance) |
| `MODEL_RATED` | A model's judgement (the VLM rubric prior) |
| `ASSUMED` | A parameter we chose; shown with its range |
| `SIMULATED` | An output of the simulation |
| `MEASURED` | Checked against real data (only the mini backtest, if done) |

The ledger screen lists every metric with its tag, and shows the mini-backtest result or "not validated against real outcomes".

---

## 13. UI and demo

### 13.1 Screens
1. **Upload and analysis:** preview, metadata, hook breakdown with evidence, prior with its tag and caveat, attention-risk timeline.
2. **Audience:** communities, sample persona cards, relevance by community.
3. **Replay:** animated cascade by round, channel colouring, hubs and bridges, event inspector. Fallback: round-by-round aggregate charts and community bars.
4. **Diagnosis:** R by round and community, factor ratios, binding constraint, pathway class, reach attribution.
5. **Lab:** scenario setup, matched A/B, verdict with stability label, sensitivity ranking, tipping-point curve.
6. **Evidence ledger.**

UX rules: every number has an evidence badge; uncertainty sits next to every verdict; advanced controls hidden by default; demo path is a few clicks.

### 13.2 Demo script (about 4 minutes)
0:00 problem. 0:25 upload and hook analysis, prior with its tag. 1:00 ledger: what is computed, rated and assumed, and the validation layers. 1:30 baseline replay, stall, diagnosis names the binding constraint with numbers, attribution. 2:10 Lab: upload the alternate opening, run the matched experiment, show win rate, interval, stability label. 3:05 tipping-point curve. 3:30 limits and roadmap.

Prepare before the day: a demo video and an alternate opening of the same video; cached analysis and cached experiment results; a recorded backup; fixed seeds; answers to "Is this Instagram's algorithm?" (no, our documented policy), "How do you validate?" (the layers in section 12), "Are the parameters guesses?" (yes, assumed, and the verdict is tested across their ranges).

---

## 14. API (may be simplified)

`POST /videos`; `POST /videos/{id}/analyze`; `GET /videos/{id}/analysis`; `POST /simulations`; `GET /simulations/{run_id}` with `/events` and `/metrics`; `POST /experiments`; `GET /experiments/{id}`; `POST /tipping`; `GET /ledger`; `GET /health`. Contracts are in `backend/schemas.py`. Do not change a schema without updating `schemas.py`, the affected tests and the frontend types in the same change.

---

## 15. Priorities

| Priority | Features |
|---|---|
| **MUST** | Upload, preprocessing, VLM analysis, prior (rubric with fallback), archetypes and population, embedding relevance, behaviour chain, graph, two channels, policy, engine with common random numbers, event log, replay, R diagnosis with bottleneck rules, matched A/B (real clip and hypothetical), evidence ledger |
| **SHOULD** | Two-level robustness verdict with stability label and sensitivity, reach attribution, attention-risk timeline, mini backtest |
| **COULD** | Tipping-point curve, pathway classes, seed and policy counterfactuals, report export |
| **WON'T** | Training any model, platform replication, real follower graphs, account analytics, video editing, LLM-driven personas |

Note: two-level robustness is the top USP. Treat it as the first SHOULD and start it as soon as the MVP works.

---

## 16. Build plan (20 to 24 hours, four people)

Roles: **A** engine and graph; **B** Lab and diagnostics; **C** frontend; **D** video analysis, prior, personas, relevance, demo assets and pitch. With fewer people, merge B into A and D into C only as a last resort.

| Phase | 24 h plan | 20 h plan | Goal |
|---|---|---|---|
| 0. Setup | 0 to 1 | 0 to 1 | Repo, contracts (`schemas.py`), `config.yaml`, stubs and fixtures, roles |
| 1. Parallel build on stubs | 1 to 8 | 1 to 7 | Each role builds against fake inputs |
| Checkpoint | 8 | 7 | Stubs replaced by real modules where possible |
| 2. Integration | 8 to 14 | 7 to 12 | **MVP:** upload → analysis → prior → simulation → replay → R diagnosis → one matched A/B |
| 3. Lab signatures | 14 to 18 | 12 to 15 | Two-level robustness, then attribution, then tipping-point curve |
| **Feature freeze** | 18 | 15 | No new features |
| 4. Polish | 18 to 24 | 15 to 20 | Rehearse three times, record backup, submit with at least 1.5 hours spare |

Cut lines: behind at hour 10 → reduce to 2000 viewers and 15 seeds; behind at hour 14 → drop the tipping-point curve, then the attribution diagram (keep the table); graph animation fails → aggregate charts; real alternate-clip run fails → hypothetical experiment, labelled; anything unstable after the freeze → show the cached run.

---

## 17. Testing and acceptance

Tests that must pass (pytest):
- **Determinism:** same seed, assumption sample and versions give an identical event log.
- **Common random numbers:** baseline and variant with identical inputs give identical outputs.
- **Monotonicity:** raising `p` never lowers mean reach across seeds.
- **Share channel:** with zero share probability there are no share-channel exposures.
- **Bounds:** no viewer exposed twice; exposures never exceed the population.
- **R identity:** `R_share = stay_rate * share_given_stay * exposures_per_sharer` holds on a real log, and counts match the log.
- **Relevance sanity:** cooking video scores higher for the food community than for gaming.
- **Pathway rules:** synthetic logs trigger the expected class for each rule.
- **Fallbacks:** with the VLM or embeddings unavailable the app completes a run and labels the gap.
- **Config:** every parameter used in code exists in `config.yaml`.

Product acceptance: a judge can upload or pick a video; see the hook analysis and the prior with its tag; watch the replay; read a diagnosis naming the binding constraint with numbers; run the matched A/B and read the verdict with its stability label; see which assumptions the verdict depends on; open the ledger.

---

## 18. Risks and ethics

| Risk | Mitigation |
|---|---|
| Parameters are guesses | Ranges, sampling, stability label, sensitivity ranking, ledger |
| Prior is an LLM judgement | `MODEL_RATED` tag, several samples, fallback, mini backtest |
| Relevance proxy is weak | Sanity tests, labelled approximation |
| Synthetic graph is arbitrary | Configurable, versioned, robust over seeds |
| Counterfactual read as causal proof | Always labelled a simulated scenario |
| Live demo fails | Cached runs and recorded backup |
| Scope creep | Priority order, feature freeze, cut lines |
| Rules on pre-existing code or APIs | Check the hackathon rules before relying on any |

Ethics: personas are fictional, no sensitive traits, uploaded videos stored only as needed for the demo, and no claim of real-world accuracy.

---

## 19. Open decisions (fill in `docs/DECISIONS.md`)

- Team size and role owners.
- Which VLM API and key; budget.
- Embedding model (CLIP-style) and whether a GPU exists (CPU is fine for a few frames per video).
- React or Streamlit.
- Whether to do the mini backtest.
