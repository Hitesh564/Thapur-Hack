# CLAUDE.md — Viralyst

Read `docs/PRD.md` fully first. Then `TASKS.md`. Parameters live in `config.yaml`; contracts in `backend/schemas.py`.

**Open task:** switching the VLM from OpenAI to a local Hugging Face model. If you are picking this up, read `docs/HANDOFF_LOCAL_VLM.md` first; it has the full context, plan and acceptance checklist.

## What we are building
A short-form video **propagation experiment lab**: analyse a video, simulate a synthetic audience spreading it over a community graph, run matched A/B experiments with seed- and assumption-robustness, diagnose stalls, and label every number with an evidence tag. 20–24 hour hackathon build. No dataset, no training.

## Hard rules
1. **No hard-coded model numbers.** Every probability, range, threshold, weight and count comes from `config.yaml`. Add new parameters there with `default`, `min`, `max`.
2. **No LLM call inside the simulation loop.** LLMs/VLMs are used only in preprocessing (analysis, prior, archetype generation), always cached by file hash, always validated against a Pydantic schema, always with a non-network fallback.
3. **Common random numbers.** Per seed, one uniform tensor `U[round, viewer, draw_type]`. Baseline and variant read the same `U`. Never index randomness by event counter, only by viewer.
4. **Determinism.** Same seed + assumption sample + versions ⇒ identical event log. Use `numpy.random.Generator` with explicit seeds; no global RNG.
5. **Evidence tags** (`COMPUTED`, `MODEL_RATED`, `ASSUMED`, `SIMULATED`, `MEASURED`) attach to every user-visible metric. Never show a number without one.
6. **Wording.** Never write "predicts views", "will go viral", "Instagram's algorithm". Use "in this simulation", "assumed", "the model suggests".
7. VLM hook fields are display-only; they never drive the engine.
8. The engine must run headless: no UI, no network.
9. Do not change a schema without updating `schemas.py`, tests and frontend types in the same change.
10. Don't add a training pipeline, extra datasets or features outside the PRD. Prioritise MUST → SHOULD → COULD.

## Layout
```
backend/app/{ingestion,preprocessing,video_analysis,prior,personas,relevance,graph,
             behaviour,policy,engine,eventlog,diagnostics,experiments,ledger,api}/
backend/data/{archetypes.json,archetypes_fallback.json,backtest_pairs.csv}
backend/tests/
frontend/            # React + Vite
config.yaml  TASKS.md  docs/{PRD.md,DECISIONS.md,FIRST_PROMPT.md}
```

## Working style
- Build against stubs/fixtures first, replace with real modules at checkpoints (see TASKS.md gates).
- Write the test from the acceptance line in TASKS.md before or with the code; run `pytest -q` before every commit.
- Small commits, message prefix by area (`engine:`, `lab:`, `ui:`, `video:`). If an unspecified choice appears, pick the simplest, log it in `docs/DECISIONS.md`, move on.
- Performance: one 4,000-viewer run well under 1 s; if a matched experiment is too slow, cut seeds/samples before cutting features.
- Cut lines and gates are in TASKS.md; respect the feature freeze.

## Commands
```
cd backend && pip install -r requirements.txt && pytest -q
uvicorn app.api.main:app --reload
cd frontend && npm install && npm run dev
```
