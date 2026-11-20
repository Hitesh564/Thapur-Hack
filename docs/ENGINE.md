# How the Viralyst backend works

A plain-language walk-through of the pipeline, from upload to verdict. Every number below is an
**assumed** placeholder from `config.yaml` unless marked otherwise; the code never hard-codes them.
Nothing here predicts views — it is a *what-if* simulation under stated assumptions.

```
upload ─► preprocess ─► GPT analysis (display only)
                    ├─► GPT rubric prior p           [MODEL_RATED]
                    └─► CLIP relevance per viewer    [COMPUTED]
                                     │
   personas ─► population ─► social graph   (fixed "world", same for every video)
                                     │
                            simulation engine        [SIMULATED]
                                     │
        diagnosis (R, binding constraint, bottlenecks, pathways, attribution)
        Lab (matched A/B × seeds × assumptions, tipping curve) ─► evidence ledger
```

## 1. Video in (`ingestion`, `preprocessing`)
* Upload is validated (type, decodes, 3–90 s) and stored under its content hash; the same file is
  never processed twice.
* We sample **8 evenly spaced frames + 4 extra frames from the first 3 s**, detect scene cuts
  (colour-histogram jumps), measure motion, extract audio and transcribe it (`gpt-4o-transcribe`).

## 2. What the models contribute (preprocessing only, never inside the simulation)
* **Prior `p`** (`prior/`): GPT (`gpt-6.2-sol`) scores the opening 1–10 on six rubric points, **3 times**.
  `p = mean score / 10`, the three values form the *ensemble* (the model's uncertainty). Offline,
  a documented heuristic (duration, cut rate, motion, audio) replaces it and is tagged `ASSUMED`.
* **Analysis** (`video_analysis/`): topic, category, hook strengths/weaknesses with timestamps and a
  suggested fix. **Display only** — it never drives the engine.
* **Attention-risk timeline**: the rubric is re-run with each of 8 time windows removed; the change
  shows *where the rating is sensitive* (not where viewers drop off).
* **Relevance** (`relevance/`): CLIP ViT-L-14 embeds every archetype's interest phrases and the video
  frames; cosine similarity → per-viewer relevance in [0, 1] via the PRD §7.3 formula. Fallback:
  keyword overlap.

## 3. The synthetic audience (`personas/`)
* **50 archetypes in 8 communities** (Food, Gaming, Fitness, Beauty, Tech, Comedy, Travel, Learning),
  generated once by GPT and validated (counts, ranges, **no sensitive traits**), saved to
  `data/archetypes.json`. A hand-written `archetypes_fallback.json` keeps the app offline-safe.
* Each archetype has 3–6 interest phrases with strengths, trait **ranges** (attention; like, comment,
  save, share tendencies) and an influence class.
* **Population**: 4,000 viewers sampled from archetypes with bounded noise on traits and a
  heavy-tailed (Pareto) influence. Seeded and versioned; identities never change between videos.

## 4. The social graph (`graph/`)
Directed degree-corrected stochastic block model (current world: 47,655 edges).
* Each viewer follows ~12 accounts (Poisson); **80%** of follows stay inside the viewer's community;
  who gets followed is weighted by influence → a few big accounts (median 8 followers, max 423).
* A few viewers get extra cross-community followers (**bridges**).
* **Roles are computed, not assigned**: hubs = top 2% by followers (80 viewers);
  bridges = top 3% by share of followers in other communities (120 viewers).

## 5. One simulated run (`engine/`, `behaviour/`, `policy/`)
**Round 0 — seeding.** 1.5% of viewers (60) see the video: half chosen by relevance, half at random
(`mixed`; also `random`, `relevance_biased`, `community_targeted`).

**Every exposed viewer** goes through the behaviour chain (no LLM, pure probability):
1. **Stay** past the opening: `P = sigmoid(logit(q0) + a·(rel − mean rel) + c·(attention − 0.5))`,
   with `q0` between 0.30 and 0.80 depending on `p`.
2. **Watch fraction** (Beta distribution): higher with `p`, relevance and attention, lower for long videos.
3. **Like / comment / save / share**: `base rate × tendency × (0.5 + rel) × (0.5 + watch fraction)`.
   A viewer who skips cannot like or share. The dominant term is logged as a *reason code*.

**Next round — two channels.**
* **Share**: each sharer exposes up to 10 unseen followers who "accept" (50%).
* **Recommendation**: our documented policy (not any platform's) scores the round
  `S = 0.4·stay + 0.3·watch + 0.1·like + 0.2·share`; the batch grows if `S > S0 (0.40)`, shrinks if not:
  `B_next = B·exp(2·(S − S0))`, capped at 1,500. 80% of the batch is picked by relevance, 20% at random.

**Stops** when nobody is left, after 10 rounds, when the budget is used, or when `S < S0` two rounds running.

**Fair comparisons (common random numbers).** Each seed draws one table of random numbers
`U[round, viewer, decision]`. Baseline and variant read the *same* table, so a viewer faces the same
dice in both — only the changed variable differs. Same seed + inputs ⇒ identical event log.

**Event log.** Every exposure, skip, watch, completion, like, comment, save and share, with round,
channel, source viewer, community, role, reason and probability (Parquet). The whole cascade can be
rebuilt from it. One run takes ~0.02–0.4 s.

## 6. Diagnosis (`diagnostics/`) — "why did it stall?"
* **Reproduction number** per round and community: `R = R_rec + R_share`, with the exact identity
  `R_share = stay rate × share-given-stay × exposures-per-sharer`; exposures-per-sharer splits further
  into followers × unseen fraction × acceptance × cap/overlap.
* **Binding constraint**: each factor ÷ its "healthy" reference in config; the lowest ratio is the
  weak link, with the value it would need for `R = 1`.
* **10 bottleneck rules** (weak attention, retention, sharing, threshold not reached, saturation,
  bridges, …), each citing its numbers and saying "in this simulated run".
* **Pathways**: early failure, niche, hub-driven, bridge-driven, cross-community, broad-but-shallow,
  deep-but-narrow. **Reach attribution**: channel → source community → source role.

## 7. The Propagation Lab (`experiments/`)
* **Matched A/B**: change one variable (real alternate opening, hypothetical +0.1 hook, seeding,
  policy). Outer loop: **10 assumption samples** (every ranged parameter drawn from its range, `p`
  from the prior ensemble). Inner loop: **30 seeds**. 600 paired runs ≈ 10 s.
* **Verdict**: win rate, median effect with 10–90% interval, assumption stability, label
  **Robust** (both ≥ 0.8) / **Likely** (≥ 0.6) / **Inconclusive** — our cut-offs, not statistical tests —
  plus which assumptions the result depends on (rank correlation).
* **Tipping curve**: share of runs reaching 25% of the population as `p` sweeps 0.05–0.95, with
  default assumptions and with assumption uncertainty; marks where the video sits.

## 8. Honesty (`ledger/`)
Every number carries a tag: `COMPUTED`, `MODEL_RATED`, `ASSUMED` (with its range), `SIMULATED`,
`MEASURED` (only the optional mini backtest, labelled anecdotal with its n). Without a backtest the
ledger says **"Not validated against real outcomes."**
