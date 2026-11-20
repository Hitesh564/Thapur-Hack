# TASKS.md — Viralyst (20–24 h)

Legend: [M]=MUST [S]=SHOULD [C]=COULD. Each task has an acceptance test (AT). Gates: **H1** setup done · **H8 (H7 in 20h)** stubs→real checkpoint · **H14 (H12)** MVP end-to-end · **H18 (H15)** feature freeze · submit with ≥1.5 h spare.

## Phase 0 — Setup (H0–1) — all
- [M] Repo, `requirements.txt`, FastAPI skeleton, `schemas.py` import ok, `config.yaml` loader, fixture JSONs (fake VideoProfile, PriorResult, population, graph). AT: `pytest -q` runs; `GET /health` ok.
- [M] `docs/DECISIONS.md`: team roles, VLM provider, embedding model, React/Streamlit.

## Workstream A — Engine & graph
- [M] A1 Population sampler from archetypes (versioned, seeded). AT: same seed → identical parquet; n matches config.
- [M] A2 Directed degree-corrected SBM graph → CSR; hub/bridge roles computed. AT: intra-community edge ratio ≈ config; hubs = top 2% followers.
- [M] A3 Behaviour chain (stay, watch fraction, like/comment/save/share, reason code). AT: skip ⇒ no like/share; share<like on average; p↑ ⇒ stay↑.
- [M] A4 Policy + two channels + seeding + stop conditions. AT: no viewer exposed twice; exposures ≤ n; zero share prob ⇒ no share-channel exposures.
- [M] A5 Engine round loop with CRN tensor `U[round,viewer,draw]`. AT: determinism; identical inputs baseline=variant identical; monotone reach in p.
- [M] A6 Columnar event log + run summary + reproducibility record. AT: cascade reconstructable from log.
- [S] A7 Perf: 4000 viewers <1 s/run; multiprocessing for experiments. AT: 30×10×2 matched runs < 2 min.

## Workstream B — Lab & diagnostics
- [M] B1 R decomposition per round/community; binding constraint vs reference. AT: `R_share = stay*share|stay*exposures_per_sharer` on real log.
- [M] B2 Bottleneck rules (~10) with evidence numbers. AT: synthetic logs trigger each rule.
- [M] B3 Matched A/B runner (paired differences, shared U). AT: variant==baseline ⇒ difference exactly 0.
- [S] B4 Two-level robustness (A samples × S seeds), win rate, interval, stability, label, sensitivity (rank corr). AT: label thresholds correct on synthetic cases.
- [S] B5 Reach attribution table (channel→community→role).
- [C] B6 Tipping-point curve. [C] B7 Pathway classes. [C] B8 Seed/policy counterfactual variants.
- [M] B9 Evidence ledger endpoint. AT: every metric returned has a tag.

## Workstream C — Frontend
- [M] C1 Upload/analysis screen, prior with tag. [M] C2 Replay (aggregate charts first, graph animation if time). [M] C3 Diagnosis screen. [M] C4 Lab screen (setup, run, verdict + badge). [M] C5 Ledger screen. [S] C6 Audience screen, attribution flow, tipping chart.
- Rule: evidence badge on every number; uncertainty beside verdicts. AT: demo path ≤ 6 clicks on cached data.

## Workstream D — Video, prior, personas, demo
- [M] D1 Preprocessing (frames, audio, transcript, cuts, hash cache). AT: second run hits cache.
- [M] D2 VLM analysis (JSON validated, fallback to cached/basic). [M] D3 Prior `vlm_rubric` ×3 + heuristic fallback. AT: PriorResult in [0.05,0.95]; fallback works offline.
- [M] D4 Archetypes: generate 50 + hand-written fallback file. AT: no sensitive traits; loads offline.
- [M] D5 Relevance via embeddings. AT: cooking video > food vs gaming.
- [S] D6 Attention-risk timeline. [S] D7 Mini backtest (15–20 pairs, `backtest_pairs.csv`, pairwise accuracy with n, labelled anecdotal).
- [M] D8 Demo assets: demo video + alternate opening, cached analysis and experiment results, recorded backup, fixed seeds. [M] D9 Pitch + Q&A sheet; 3 rehearsals.

## Cut lines
Behind at H10 → 2000 viewers, 15 seeds. Behind at H14 → drop tipping curve, then attribution diagram (keep table). Graph animation fails → aggregate charts. Alt-clip fails → hypothetical experiment labelled. Unstable after freeze → cached run.
