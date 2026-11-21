# Viralyst: progress report (check-in)

**Date:** 21 Nov 2026 · **Build mode:** solo, 24-hour plan · **Repo:** `Viralyst/` (own git repo, branch `main`, 6 commits)

> Viralyst is a pre-publication **propagation experiment lab** for short-form video. It analyses a draft, simulates a
> synthetic audience spreading it over a community graph, runs matched A/B experiments that are robust to random
> seeds and assumptions, diagnoses where spread stalls, and labels every number with an evidence tag.
> It does **not** predict views and does **not** reproduce any platform's algorithm.

---

## 1. Status at a glance

| Area | Status |
|---|---|
| Phase 0 (repo, config, contracts, fixtures) | ✅ Done |
| Workstream A: engine and graph (A1–A7) | ✅ Done |
| Workstream B: Lab and diagnostics (B1–B9) | ✅ Done (tipping curve and pathways included) |
| Workstream C: frontend (C1–C6) | ✅ Done, 6 screens, light + dark |
| Workstream D: video, prior, personas, relevance (D1–D6) | ✅ Done |
| D7: mini backtest | 🟡 Harness built; **needs hand-collected data** |
| D8: demo assets (real video + alternate opening, backup recording) | 🔴 **Not done** (test videos are stand-ins) |
| D9: pitch, Q&A sheet, rehearsals | 🔴 **Not done** |
| Extra USPs beyond the PRD | ✅ Hook Surgeon, value of information, outcome drivers, robust seeding plan, **Counterfactual Trace** |
| Tests | ✅ **69 passing** (`pytest -q` in `backend/`), ~2 min |
| Frontend production build | ✅ `npm run build` passes (~86 kB gzip JS) |

**Size:** ~4,100 lines of backend Python (32 modules), ~800 lines of tests, ~2,300 lines of TypeScript/React, 29 API endpoints.

**Commit history**

| Commit | What |
|---|---|
| `b2e9c6f` | engine: phase 0 + core engine, diagnostics, matched A/B, ledger |
| `19f0899` | video: preprocessing, OpenAI VLM prior + analysis, CLIP relevance, full API |
| `bf1a35e` | engine: LLM personas, pathways, tipping curve, attention timeline, backtest harness |
| `97e50f2` | ui: React frontend with six screens and evidence badges |
| `92bff23` | lab: Hook Surgeon, value of information, robust seeding plan, milestone headline |
| `db95eca` | lab: Counterfactual Trace, why a variant wins viewer by viewer |

---

## 2. Models and external services used

**Nothing is trained by us.** No dataset is used for training. Models are used only in preprocessing (never inside the
simulation loop), always cached by file hash, always validated against a Pydantic schema, always with an offline fallback.

| Model | Provider / where it runs | Used for | Fallback |
|---|---|---|---|
| **`gpt-6.2-sol`** | OpenAI API (vision + JSON) | Prior rubric (6 dimensions, 3 samples per video); hook analysis; generating the 50 archetypes; attention-risk timeline (16 re-ratings) | `gpt-5.5`, then heuristic prior (tagged ASSUMED) |
| `gpt-5.5` | OpenAI API | Automatic fallback if `gpt-6.2-sol` fails | heuristic |
| **`gpt-4o-transcribe`** | OpenAI API | Speech transcript of the video's audio | "no transcript", labelled in UI |
| **CLIP ViT-L-14, `datacomp_xl_s13b_b90k`** | open_clip, **local CPU** (~1.7 GB weights) | Persona–video relevance (text embedding of interests vs image embedding of frames) | keyword-overlap relevance |
| `gpt-image-2`, `gpt-4o-mini-tts` | OpenAI API | **Test fixtures only**: generated the cooking/gaming test images and voiceover | n/a |
| ffmpeg (via `imageio-ffmpeg`) | local | Audio extraction, Hook Surgeon re-cuts | n/a |

**How the model was chosen:** listed every model the API key can access, then probed the top candidates with an image +
JSON request. `gpt-6.2-sol` was newest and correct; `gpt-6.2-astra` misread the test image and was rejected.

The API key is read only from the `OPENAI_API_KEY` environment variable (or the Windows user environment). It is never
written to any file. `VIRALYST_OFFLINE=1` forces every fallback (used by the tests, so tests cost nothing).

**Stack:** Python 3.12, FastAPI, NumPy/SciPy (sparse), Pydantic v2, pandas/pyarrow, OpenCV, PyTorch (CPU) + open_clip,
pytest · React 19 + Vite 8 + TypeScript (no UI or chart libraries; hand-built SVG charts and a canvas replay).

---

## 3. How it works, end to end

```
upload ─► preprocess (frames, cuts, motion, audio, transcript)        cached by file hash
       ├─► GPT hook analysis (display only, never drives the engine)  [MODEL_RATED]
       ├─► GPT rubric prior p (3 samples → ensemble)                  [MODEL_RATED]
       └─► CLIP relevance per viewer                                   [COMPUTED]
personas (50 archetypes) ─► 4,000-viewer population ─► follower graph  (fixed "world", versioned)
                          ─► simulation engine (common random numbers) [SIMULATED]
                          ─► diagnosis · Lab experiments · Counterfactual Trace · evidence ledger
```

### 3.1 Video ingestion and preprocessing (`ingestion/`, `preprocessing/`)
- Upload validated (type, decodes, 3–90 s); stored under its SHA-256 hash, so the same file is never processed twice.
- **Frames:** 8 evenly spaced + 4 extra from the first 3 s, resized to 768 px.
- **Cuts:** colour-histogram change scanned at 10 fps (no PySceneDetect needed). **Motion:** mean frame difference, overall and in the opening.
- **Audio** extracted with ffmpeg → transcribed with `gpt-4o-transcribe`.
- Second call on the same video reads the cache (0.02 s).

### 3.2 The prior `p`: "how strong the opening looks" (`prior/`)
- Prompt in `app/prompts/rubric.md`: 6 dimensions scored 1–10 (immediate clarity, curiosity, visual impact, pacing,
  payoff clarity, audience fit), with timestamped frames + transcript.
- Called **3 times in parallel**; `p_i = mean score / 10`; `p = clip(mean, 0.05, 0.95)`; the 3 values are the
  **ensemble** (model uncertainty) that the Lab samples from.
- **Heuristic fallback** (documented formula: duration band, opening cut rate, opening motion, audio) → tagged ASSUMED.
- The **alternate clip goes through the identical pipeline and prompt**.

### 3.3 Hook analysis (`video_analysis/analysis.py`)
- One GPT call: topic, category, summary, format, interest tags, hook categories, strengths, weaknesses, timestamped
  evidence, suggested improvement, confidence. **Display only.**

### 3.4 Attention-risk timeline (`video_analysis/attention.py`)
- Video split into 8 windows; rubric re-run with each window's frames removed (2 samples each = 16 calls).
- Reports where **the rating** is sensitive (not where viewers drop off), the "drag" window, and the sampling noise.

### 3.5 Personas and population (`personas/`)
- **50 archetypes in 8 communities** (Food & Cooking, Gaming, Fitness & Sport, Beauty & Fashion, Tech & Gadgets,
  Comedy & Entertainment, Travel & Outdoors, Learning & DIY).
- Generated once by `gpt-6.2-sol` with a validator: counts per community, 3–6 interest phrases, trait-range bounds,
  **sensitive-word blocklist** (no ethnicity, religion, politics, health…). Took 2 attempts; validation errors are fed back to the model.
- A **hand-written fallback** (`data/archetypes_fallback.json`) keeps the app offline-safe.
- **Population:** 4,000 viewers sampled from archetypes, bounded noise on traits, heavy-tailed (Pareto) influence. Seeded and versioned; persisted as Parquet.

### 3.6 Social graph (`graph/sbm.py`)
- Directed, degree-corrected stochastic block model, built with NumPy (~30 ms): each viewer follows ~12 accounts
  (Poisson), **80 % inside their community**, targets weighted by influence.
- Current world: **47,655 edges**, median 8 followers, max 423.
- **Roles computed, not assigned:** hubs = top 2 % by followers (80); bridges = top 3 % by cross-community follower share (120).

### 3.7 Relevance (`relevance/`)
- CLIP text embedding of each archetype's interest phrases ("a short video about {phrase}", strength-weighted) vs
  mean CLIP image embedding of the frames → cosine → PRD §7.3 formula (relative + absolute term).
- `cos_lo/cos_hi` tuned by inspection for ViT-L (0.05 / 0.22).

### 3.8 Behaviour engine (`behaviour/`): no LLM, pure probability
For every exposed viewer:
1. **Stay:** `P = sigmoid(logit(q0) + a·(rel − mean rel) + c·(attention − 0.5))`, `q0` between 0.30 and 0.80 by `p`.
2. **Watch fraction:** Beta distribution via inverse CDF, higher with `p`, relevance, attention; penalised by duration.
3. **Like / comment / save / share:** `base × tendency × (0.5 + rel) × (0.5 + watch)`; skippers cannot act.
4. Reason code = dominant term (prior, relevance, attention, duration).

### 3.9 Distribution (`policy/`): our documented policy, not a platform's
- **Seeding:** 1.5 % of viewers (mixed / random / relevance-biased / community-targeted).
- **Recommendation:** score `S = 0.4·stay + 0.3·watch + 0.1·like + 0.2·share`; batch `B ← B·exp(2·(S − S0))`, S0 = 0.40,
  cap 1,500; 80 % of each batch by relevance (Gumbel-top-k), 20 % random.
- **Shares:** each sharer exposes up to 10 unseen followers who accept (50 %).
- **Stops:** nobody left, 10 rounds, budget used, or S < S0 two rounds running.

### 3.10 Simulation engine (`engine/`)
- **Common random numbers:** one table `U[round, viewer, decision]` per seed. Behaviour reads each viewer's
  **personal dice** (`U[0, v, ·]`), so baseline and variant face identical dice. Never indexed by an event counter.
- **Deterministic:** same seed + inputs → identical event log (tested).
- **Event log** (Parquet): exposure, skip, watch, partial view, completion, like, comment, save, share, with round,
  channel, source viewer, community, role, reason code, probability. The whole cascade can be rebuilt from it.
- **Speed:** one 4,000-viewer run 20–400 ms; 600-run matched experiment ≈ 10–30 s (single process).

### 3.11 Diagnosis (`diagnostics/`)
- **R per round and per community:** `R = R_rec + R_share`, with the exact identity
  `R_share = stay rate × share-given-stay × exposures-per-sharer` (tested on real logs); exposures-per-sharer further
  split into followers × unseen × acceptance × cap/overlap.
- **Binding constraint:** each factor ÷ its "healthy" reference (config); lowest ratio = weak link, plus the value needed for R = 1.
- **10 bottleneck rules**, each citing its numbers and ending "in this simulated run".
- **7 pathway classes** (early failure, niche, hub-driven, bridge-driven, cross-community, broad-but-shallow, deep-but-narrow).
- **Reach attribution:** channel → source community → source role.

### 3.12 Propagation Lab (`experiments/`)
- **Matched A/B:** change one variable (real alternate clip, hypothetical stronger hook, seeding, policy).
  **10 assumption samples** (every ranged parameter drawn from its range; `p` from the prior ensemble) × **30 seeds**,
  both arms on the same random table.
- **Verdict:** milestone chance (share of runs reaching 25 % of the audience) per arm, win rate, median effect with
  10–90 % interval, assumption stability, label **Robust** (both ≥ 80 %) / **Likely** (≥ 60 %) / **Inconclusive**
  (our cut-offs, not statistical tests), sensitivity ranking (Spearman).
- **Tipping-point curve:** milestone chance as `p` sweeps 0.05–0.95 (15 points × 20 seeds), with default assumptions
  and across assumption ranges.

### 3.13 Extra USPs (beyond the PRD, logged as deliberate extensions in `docs/DECISIONS.md`)
| Feature | What it does | How |
|---|---|---|
| **Hook Surgeon** | Suggests real edits and tests each | Splits the video at detected cuts; recipes: payoff first, 1 s teaser, cut the drag (from attention timeline), strongest shot first, trim slow start, swap first two; renders MP4s with ffmpeg; re-rates with the same prompt; matched experiment per edit (15 seeds × 8 samples); ranks by robustness. ~2 min, background job with progress |
| **Is this decision robust?** | Value of information per experiment | EVPI / EVPPI by binning assumption samples per parameter; decision risk |
| **What decides whether it takes off** | Which unknowns drive reach | 64 assumption samples × 4 seeds; share of explained rank variation; separates platform-side vs measurable by the creator; tells the creator what to measure |
| **Robust seeding plan** | Who to show it to first | 11 options (3 strategies + each community), minimax regret on milestone chance across assumption samples |
| **Counterfactual Trace** ⭐ | **Why** a variant wins, viewer by viewer | Same seed, assumptions and per-viewer dice for both runs; exact decomposition of the reach gain by channel and share source; exact split of recommendation growth into stay / watch / like / share; table of viewers who flipped (same roll, higher threshold); difference replay; seed = median of 21 so examples are representative |

The Counterfactual Trace is only possible because of the per-viewer common-random-numbers design: LLM-persona
simulators are not repeatable, AI editors don't simulate spread, and view predictors only output a number.

### 3.14 Evidence ledger (`ledger/`)
Every metric carries a tag: **COMPUTED** (deterministic from inputs), **MODEL_RATED** (GPT rubric), **ASSUMED**
(parameter with range), **SIMULATED** (simulation output), **MEASURED** (mini backtest only, anecdotal). The ledger
screen lists all of them (62 for a typical video + run + experiment) and states **"Not validated against real outcomes."**

### 3.15 Frontend (`frontend/`)
Six screens: **1 Video & hook** (player, prior + rubric, hook breakdown, attention timeline) · **2 Audience**
(relevance by community, population, persona cards) · **3 Replay** (animated canvas of 4,000 viewers by community,
channel colours, hubs/bridges, click-to-inspect) · **4 Diagnosis** · **5 Lab** (experiment, verdict, value of
information, Hook Surgeon, seeding plan, outcome drivers, Counterfactual Trace, tipping curve) · **6 Evidence ledger**.
- Colours from a validated colour-blind-safe palette (checked with the dataviz validator); light + dark mode.
- Every chart has hover tooltips and a table view; every number has an evidence badge.
- **Demo path: 6 clicks:** pick video → Replay (simulation auto-runs) → Diagnosis → Lab → Run experiment → Ledger.

---

## 4. Results so far (on generated test videos)

| Check | Result |
|---|---|
| Prior sanity: weak opening (static empty bowl) vs strong (cheese pull) | **0.59 vs 0.75** |
| Relevance sanity: cooking video | Food **0.74** vs Gaming 0.30 |
| Relevance sanity: gaming video | Gaming 0.66, Tech 0.62, Food 0.12 |
| Real-clip A/B (cheese pull vs empty bowl), 30 × 10 | **Likely**, wins 72 % of paired runs, 7 of 10 assumption settings |
| Attention timeline (weak video) | Flags 2.3–3.4 s (static bowl) as the drag |
| Hook Surgeon (weak video) | Best edit **"Cut the drag"**; full test: **Likely, milestone chance 37 % → 46 %** |
| Hook Surgeon (strong video) | Every edit worse or equal (it does not rubber-stamp) |
| Counterfactual Trace: "Cut the drag" | +613 viewers; wins mainly via **watch time** (batches 1.64× larger) |
| Counterfactual Trace: cheese-pull opening | +1,209 viewers; wins via **staying past the opening** (120 flipped viewers, batches 2.3× larger) |
| Outcome drivers | ~74 % of uncertainty platform-side (growth threshold S0); creator's biggest lever = hook rating |
| Robust seeding plan | Food & Cooking first; small gain (+2 pts) because recommendation drives most reach |

---

## 5. Testing and quality

- **69 pytest tests**, all passing, including every PRD §17 acceptance test: determinism, common random numbers
  (identical inputs → identical log), monotone reach in `p`, zero share prob → no share exposures, no double exposure,
  R identity on real logs, relevance sanity, pathway rules on synthetic logs, fallbacks offline, every config key used
  in code exists in `config.yaml`. Plus exactness tests for the Counterfactual Trace and value-of-information maths.
- Tests run **offline** in a temp cache (no API cost).
- **No hard-coded model numbers:** every probability, range, threshold and count is in `config.yaml` with
  default/min/max where uncertain (test enforces keys exist).
- Every design choice is logged in **`docs/DECISIONS.md`**; the pipeline is explained in **`docs/ENGINE.md`**.
- Frontend checked manually in the browser (both themes, full demo path); no automated UI tests.

---

## 6. What is left

### Must do before the demo
1. **Real demo video + an alternate opening of it** (D8). Ideally 15–40 s, 5+ distinct shots (Hook Surgeon needs cuts).
   Then analyse both, run attention timeline, Hook Surgeon, experiment and trace once so everything is **cached**.
2. **Recorded backup** of the full demo path (in case the live demo or API fails).
3. **Pitch deck + Q&A sheet + 3 rehearsals** (D9). Prepared answers: "Is this Instagram's algorithm?" (no, our
   documented policy) · "How do you validate?" (layers below) · "Are the parameters guesses?" (yes, ASSUMED, and every
   verdict is tested across their ranges).

### Validation (the main weakness: no ground truth yet)
4. **Mini backtest** (harness ready): fill `backend/data/backtest_pairs.csv` with 20–40 same-creator public video pairs
   (local files + public view counts), run `POST /backtest` → appears in the ledger as MEASURED, anecdotal, with n.
5. **Pattern checks** (planned, not built): show the simulator reproduces known patterns (heavy-tailed cascade sizes,
   mostly broadcast rather than deep share chains, as in Goel et al. 2016).
6. **Human panel** (planned): ~15 people pick the stronger opening in ~15 pairs; report GPT agreement with the majority.
7. **Optional:** evaluate the prior on a slice of a public short-video engagement dataset (evaluation only, no
   training; licence to be checked).

### Nice to have / known gaps
- Ledger does not yet list outputs of the new USPs (they carry SIMULATED badges on screen, but no ledger rows).
- Report export (PRD COULD) not built.
- Hook Surgeon reorders audio with video, so narration in "payoff first" edits can sound out of order.
- Background jobs are in memory (lost on backend restart); results themselves are saved to the cache.
- Timings to plan the demo around: new video analysis ~30–90 s (CLIP cold start ~20 s), experiment ~20–30 s,
  tipping curve ~40 s, Hook Surgeon ~2 min first time; everything is instant once cached.
- At default assumptions recommendation drives ~90 % of reach and share chains are 1–2 hops, so hub-/deep-pathways
  are rare (consistent with "broadcast-dominated" findings, but worth knowing).
- Value of information is ~0 for hook decisions because the engine is monotonic in `p` (a better-rated opening never
  loses); it is informative for trade-off decisions and in the outcome-drivers view.
- `docs/PRD (1).md` is a duplicate and `docs/Viralyst PRD v1.1 — Propagation Lab.md` is superseded (v1.1 trained on
  SnapUGC; v1.2 is dataset-free). Suggest deleting or archiving them.

---

## 7. How to run

```powershell
# backend (terminal 1)
cd backend
pip install -r requirements.txt
uvicorn app.api.main:app --port 8000

# frontend (terminal 2)
cd frontend
npm install
npm run dev          # http://localhost:5173  (/api is proxied to :8000)

# tests
cd backend
pytest -q
```

Requires `OPENAI_API_KEY` in the environment for live GPT calls (everything falls back offline without it).
CLIP weights download automatically on first use.

## 8. Key files

| Path | What |
|---|---|
| `docs/PRD.md`, `TASKS.md`, `CLAUDE.md` | Spec, task list with acceptance tests, project rules |
| `docs/ENGINE.md` | Plain-language walk-through of the pipeline |
| `docs/DECISIONS.md` | Every design decision with its reason |
| `config.yaml` | All parameters (ranges, thresholds, models) |
| `backend/schemas.py` ↔ `frontend/src/types.ts` | Data contracts (kept in sync) |
| `backend/app/engine/simulate.py` | Simulation engine |
| `backend/app/experiments/` | Matched A/B, tipping, Hook Surgeon, seeding plan, value of information, backtest |
| `backend/app/diagnostics/counterfactual.py` | Counterfactual Trace |
| `backend/data/archetypes*.json` | GPT-generated and hand-written personas |
| `backend/tests/fixtures/media/` | Generated test videos (stand-ins for the demo video) |
