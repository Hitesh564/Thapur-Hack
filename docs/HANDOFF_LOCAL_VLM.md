# Handoff: run Viralyst on a local VLM (GPU laptop)

Written 21 Nov 2026 for a teammate's Claude Code session that starts with **no context**.
Everything you need is in this repo. Read this file top to bottom before touching code.

---

## 0. Paste this as your first message to Claude Code

```
Read CLAUDE.md, then docs/HANDOFF_LOCAL_VLM.md fully, then docs/PRD.md, PROGRESS.md,
docs/ENGINE.md and docs/DECISIONS.md. Your task is the local-VLM migration in the handoff.
Start with section 4 (environment check). Before downloading any model, tell me the GPU,
its memory and which model tier you picked, with the download size. Work on branch
feat/local-vlm, follow the step order in section 8, run `pytest -q` before every commit,
and stop for my review after step 3.
```

---

## 1. What Viralyst is (60-second version)

A **propagation experiment lab** for short videos (hackathon build, PRD v1.2 in `docs/PRD.md`).

1. A creator uploads a 3–90 s video.
2. Preprocessing cuts it into 12 frames (8 evenly spaced + 4 from the first 3 s), detects scene cuts,
   measures motion and transcribes speech.
3. A vision-language model (VLM) rates the hook on 6 rubric points → a hook score **p** in [0, 1]
   (tag `MODEL_RATED`). CLIP matches the frames to 4,000 synthetic viewers' interests → a
   **relevance** per viewer (tag `COMPUTED`).
4. A **model-free** NumPy engine spreads the video through a synthetic follower graph
   (shares + a documented recommendation policy) using **common random numbers**: every viewer
   has fixed dice per seed, so two versions of a video can be compared fairly.
5. The Lab runs matched A/B experiments (30 seeds × 10 assumption samples), the Hook Surgeon
   re-cuts the creator's own shots and tests each edit, and the Counterfactual Trace explains
   why a variant wins, viewer by viewer. Every number carries an evidence tag.

The models only touch steps 2–3 (and the Hook Surgeon re-rating each edit). **The simulation
never calls a model** (CLAUDE.md rule 2).

Full plain-language walk-through: `docs/ENGINE.md`. What is built and how: `PROGRESS.md`.
Every design choice so far: `docs/DECISIONS.md`.

## 2. Current state (commit `666f7d4` on `main`)

* Backend and frontend complete for the PRD's MUST and most SHOULD items, plus four extra
  features (Hook Surgeon, value of information, robust seeding plan, Counterfactual Trace).
* `cd backend && pytest -q` → **69 passed** (they run offline; no model weights needed).
* Models today:

| Job | Model now | Where it runs |
|---|---|---|
| Hook rubric (p), 3 samples | `gpt-6.2-sol` (fallback `gpt-5.5`) | OpenAI API |
| Hook analysis text (display only) | same | OpenAI API |
| Attention-risk timeline (re-rating with windows removed) | same | OpenAI API |
| Speech transcript | `gpt-4o-transcribe` | OpenAI API |
| Viewer relevance | CLIP ViT-L-14 `datacomp_xl_s13b_b90k` | local (CPU) |
| Archetype generation (one-off, already done) | GPT, output saved in `backend/data/archetypes.json` | not re-run |

## 3. The task

**Replace every OpenAI call with local models downloaded from Hugging Face, using the biggest
VLM this laptop's GPU can hold.** After the change the app must run with no `OPENAI_API_KEY` and
no internet (once weights are downloaded).

In scope:
* A provider switch `vlm.provider: local | openai` (default `local`). Keep the OpenAI code
  path working; do not delete it.
* Local VLM for: rubric prior, hook analysis, attention timeline, Hook Surgeon re-rating
  (the Surgeon calls the same functions, so it follows automatically).
* Local Whisper for the transcript.
* Probability-based rubric scoring (section 6.3), which makes the rating deterministic.
* Model choice by GPU memory, pinned revisions, safe downloads.
* Tests with a fake model, docs, frontend label changes.

Out of scope (do **not** do these):
* Do not regenerate `archetypes.json` (it would change the whole synthetic world).
* Do not change the engine, policy, experiments, diagnostics or the dice.
* No training or fine-tuning, no new datasets (CLAUDE.md rule 10).
* Do not commit weights, caches or keys.

## 4. Environment check and setup (do this first, report back)

```bash
git clone https://github.com/Hitesh564/Thapur-Hack.git
cd Thapur-Hack
git checkout -b feat/local-vlm
python --version          # 3.12 expected
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r backend/requirements.txt
cd backend && pytest -q   # must show 69 passed before you change anything
```

**Detect the accelerator:**

```bash
nvidia-smi                                   # NVIDIA: name, memory, driver CUDA version
python -c "import torch;print(torch.__version__, torch.cuda.is_available(), getattr(torch.backends,'mps',None) and torch.backends.mps.is_available())"
```

* **NVIDIA:** `requirements.txt` may have installed a CPU-only torch. Reinstall the CUDA build that
  matches the driver; take the exact index URL from https://pytorch.org/get-started/locally/
  (for example `pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128`).
  Then `pip install -U transformers accelerate bitsandbytes huggingface_hub`.
  Confirm `torch.cuda.is_available()` is `True` and note `torch.cuda.get_device_properties(0).total_memory`.
* **Apple Silicon (MPS):** no bitsandbytes; use bf16/fp16 on `mps`. Use the unified-memory column in 5.1.
* **No usable GPU:** use the CPU tier (Qwen3.5-2B). Everything still works, just slower.

`transformers` must know the model types `qwen3_5` and `whisper`
(`from transformers.models.auto.configuration_auto import CONFIG_MAPPING; 'qwen3_5' in CONFIG_MAPPING`).
Version 5.5.4 already does; 5.24.0 is the latest on PyPI as of 21 Nov 2026.

Frontend: `cd frontend && npm install && npm run build` (must pass type-checking).

## 5. Model choice

### 5.1 VLM tiers

All are Qwen vision-language models with the **same architecture** in `transformers`
(`model_type: qwen3_5`, class from `AutoModelForImageTextToText`), Apache-2.0, not gated,
safetensors only, **no custom code** (`trust_remote_code` stays `False`). One code path covers all.

| Tier | When | Repo | Revision (pin this) | Weights on disk | Load as | Approx. VRAM in use |
|---|---|---|---|---|---|---|
| A | NVIDIA ≥ 22 GB | `Qwen/Qwen3.8-27B` (newest, Aug 2026) | `37561816e66b5b9153b1b64c10968c18c1e6f4a0` | 55.6 GB | 4-bit NF4 (bitsandbytes) | ~17 GB |
| B | NVIDIA 14–22 GB | `Qwen/Qwen3.5-9B` | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | 19.3 GB | 8-bit (bitsandbytes) | ~11 GB |
| C | NVIDIA 10–14 GB | `Qwen/Qwen3.5-9B` | same as B | 19.3 GB | 4-bit NF4 | ~7 GB |
| D | NVIDIA 6–10 GB | `Qwen/Qwen3.5-4B` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | 9.3 GB | 8-bit | ~6 GB |
| M | Apple, ≥ 32 GB unified | `Qwen/Qwen3.5-9B` | same as B | 19.3 GB | bf16 on `mps` | ~20 GB |
| M2 | Apple, 16–32 GB | `Qwen/Qwen3.5-4B` | same as D | 9.3 GB | bf16 on `mps` | ~10 GB |
| CPU | no GPU | `Qwen/Qwen3.5-2B` | `15852e8c16360a2fea060d615a32b45270f8a8fc` | 4.55 GB | bf16 on CPU | ~5 GB RAM |

VRAM figures are estimates. **Measure** peak memory on the first real rating
(`torch.cuda.max_memory_allocated()`) and step down a tier if it does not fit with ~1.5 GB headroom.
Bigger models also exist (`Qwen/Qwen3.6-35B-A3B` 72 GB, `google/gemma-4-31B-it` 62.5 GB) but do not
fit a laptop; ignore them.

Note: tier A downloads the full 55.6 GB bf16 checkpoint and quantises it while loading. Check free
disk first (`hf` cache lives in `~/.cache/huggingface` unless `HF_HOME` is set).

### 5.2 Whisper

| When | Repo | Revision | Size | Licence |
|---|---|---|---|---|
| Any GPU | `openai/whisper-large-v3-turbo` | `41f01f3fe87f28c78e2fbf8b568835947dd65ed9` | 1.6 GB | MIT |
| CPU | `openai/whisper-small` | `973afd24965f72e36ca33b3055d56a652f456b4d` | 0.97 GB | Apache-2.0 |

### 5.3 Download rules (security)

* Download with `huggingface_hub.snapshot_download(repo_id, revision=<sha>, allow_patterns=[...])`
  allowing only `*.json`, `*.safetensors`, `*.txt`, `*.model`, `*.jinja`, `*.tiktoken`, `merges.txt`, `vocab*`.
  **Never** download `*.bin`, `*.pt`, `*.pth`, `*.pkl` (pickle can execute code; `whisper-small`
  ships a `pytorch_model.bin` next to its safetensors, so the filter matters) or `*.py`.
* `trust_remote_code=False` always. Load with `use_safetensors=True`.
* Provide a command: `python -m app.video_analysis.local_vlm --download` that downloads the tier the
  config resolves to (VLM + Whisper) and prints sizes. The app itself may also download lazily on first
  use, but the explicit command is what the demo machine should run beforehand.
* No token is needed (models are not gated). Never write tokens or keys to files.

## 6. Design

### 6.1 Config (`config.yaml`) — every number lives here (CLAUDE.md rule 1)

Replace the `vlm:` block with something like this (names may change; keep them in config, with
`default/min/max` for anything the robustness loop should sample — none of these should be sampled):

```yaml
vlm:
  provider: local                 # local | openai
  max_frame_px: 768               # frames sent to the model (local may use local.max_frame_px)
  jpeg_quality: 85
  openai:                         # unchanged behaviour, only moved
    model: gpt-6.2-sol
    fallback_models: [gpt-5.5]
    transcribe_model: gpt-4o-transcribe
    timeout_s: 60
    client_retries: 1
    max_retries: 1
  local:
    model: auto                   # auto = first tier that fits; or a tier name (A, B, C, D, M, M2, CPU)
    device: auto                  # auto | cuda | mps | cpu
    headroom_gb: 1.5
    max_frame_px: 768             # lower (e.g. 448) on CPU to cut image tokens
    tiers:
      - {name: A,   device: cuda, min_mem_gb: 22, repo: Qwen/Qwen3.8-27B, revision: 37561816e66b5b9153b1b64c10968c18c1e6f4a0, quant: nf4}
      - {name: B,   device: cuda, min_mem_gb: 14, repo: Qwen/Qwen3.5-9B,  revision: c202236235762e1c871ad0ccb60c8ee5ba337b9a, quant: int8}
      - {name: C,   device: cuda, min_mem_gb: 10, repo: Qwen/Qwen3.5-9B,  revision: c202236235762e1c871ad0ccb60c8ee5ba337b9a, quant: nf4}
      - {name: D,   device: cuda, min_mem_gb: 6,  repo: Qwen/Qwen3.5-4B,  revision: 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a, quant: int8}
      - {name: M,   device: mps,  min_mem_gb: 32, repo: Qwen/Qwen3.5-9B,  revision: c202236235762e1c871ad0ccb60c8ee5ba337b9a, quant: none}
      - {name: M2,  device: mps,  min_mem_gb: 16, repo: Qwen/Qwen3.5-4B,  revision: 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a, quant: none}
      - {name: CPU, device: cpu,  min_mem_gb: 0,  repo: Qwen/Qwen3.5-2B,  revision: 15852e8c16360a2fea060d615a32b45270f8a8fc, quant: none}
    whisper:
      gpu: {repo: openai/whisper-large-v3-turbo, revision: 41f01f3fe87f28c78e2fbf8b568835947dd65ed9}
      cpu: {repo: openai/whisper-small,          revision: 973afd24965f72e36ca33b3055d56a652f456b4d}
    score_candidates: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]   # must match prior.rubric_min..rubric_max
    ensemble_quantiles: [0.1, 0.5, 0.9]
    ensemble_draws: 4000          # seeded Monte Carlo draws used to get the quantiles
    ensemble_seed: 11
    analysis_max_new_tokens: 900
    json_retries: 1
```

Also add `video.attention_min_change` (default `0.0167` = one rubric point on one of 6 dimensions,
divided by 10) — see 6.5. Update `backend/tests/test_config.py` expectations if it checks the vlm keys.

### 6.2 Code layout

| File | Change |
|---|---|
| `backend/app/video_analysis/local_vlm.py` (new) | Tier selection (pure function `pick_tier(device, mem_gb, tiers, headroom)`), safe download, lazy singleton load with a `threading.Lock`, `model_tag()` = `repo@rev[:8]:quant`, `score_rubric(...)`, `generate_json(...)`, `release()`, `__main__` with `--download` and `--info`. |
| `backend/app/video_analysis/vlm.py` | Keep `VLMUnavailable`, `load_prompt`. `call_json` dispatches on `vlm.provider`: `openai` → existing code (reading the moved `vlm.openai.*` keys), `local` → `local_vlm.generate_json`. Export `active_model_tag()` for cache keys. |
| `backend/app/prior/prior.py` | `vlm_rubric_prior` → if provider is local, use probability scoring (6.3); else current 3-sample path. Cache key must include `active_model_tag()` and a scoring version. `PriorResult.model` = the repo id + quant. |
| `backend/app/video_analysis/attention.py` | Re-rate with a shared helper `rate_items(profile, items) -> float` so local and OpenAI share the code. With the local (deterministic) scorer one sample per window is enough; use `video.attention_min_change` for `within_noise` (6.5). Cache key includes the model tag. |
| `backend/app/video_analysis/analysis.py` | Cache key uses `active_model_tag()` instead of `vlm.model`. No other change: `call_json` handles the provider. |
| `backend/app/preprocessing/frames.py` | `transcribe()`: provider local → Whisper via `transformers.pipeline("automatic-speech-recognition", ...)` on the chosen device, `chunk_length_s=30`; cache `transcript.json` with the model id. Same never-raise contract. |
| `backend/app/prompts/rubric_local.md` (new) | Rubric definitions written as one question per dimension (6.3). Keep `rubric.md` for OpenAI. |
| `backend/app/api/main.py` | `/health` reports provider, tier, model tag, device (no secrets). |
| `backend/requirements.txt` | add `transformers>=5.5`, `accelerate`, `huggingface_hub`, `safetensors`; `bitsandbytes` only on NVIDIA (document it; do not make CPU installs fail). Keep `openai`. |
| `frontend/src/screens/VideoScreen.tsx` lines ~27, 90, 122, 157 and `frontend/src/components/ui.tsx` line 6 | Replace the hard-coded word "GPT" with the model name the API returns (`prior.model`, `analysis_source`). Leave `AudienceScreen.tsx:45` ("Archetypes generated once by GPT") — that is still true. |
| `backend/app/experiments/surgeon.py` line 2 | Docstring says "GPT rubric prior" → "VLM rubric prior". |

**Concurrency gotcha:** the backend calls the VLM from thread pools (`prior.py` 3 parallel calls,
`attention.py` up to 12, `surgeon.py` analyses edits in parallel). With one local model on one GPU,
all local calls must go through the module lock (serialised). Do not load a second copy per thread.

**Memory gotcha:** CLIP (~1.7 GB) and Whisper also load. On small GPUs keep CLIP on CPU (as today);
optionally add `relevance.device: auto` to move it to the GPU when there is room. Load lazily,
and provide `release()` so the Surgeon can free memory between phases if needed.

### 6.3 Probability-based rubric scoring (the core change)

Today GPT writes JSON with six 1–10 scores, three times; p = mean/10 per sample; the three p values
form the ensemble. A local model should **not** be sampled for text. Instead read its probabilities:

1. **Shared prefix.** Build one chat with the rubric instructions, the 12 frames interleaved with
   their labels (reuse `frame_items(profile)` → `"Opening frame at 0.38s:"`, image, …), then duration,
   number of cuts and transcript. Use the processor's chat template with **thinking disabled**
   (`apply_chat_template(..., enable_thinking=False)` for Qwen3.x; otherwise the model opens with a
   `<think>` block and the score position is wrong). Run the prefix once with `use_cache=True`.
2. **One question per dimension** (`prior.rubric`: immediate_clarity, curiosity, visual_impact,
   pacing, payoff_clarity, audience_fit): append "Rate <dimension> (<definition>). Answer with one
   integer from 1 to 10." and the assistant-turn start.
3. **Score every candidate** `"1"`…`"10"` tokenizer-agnostically: log P(candidate tokens followed by
   the end-of-turn token | prefix + question) = sum of token log-probs. (Qwen splits `10` into `1`,`0`,
   so you cannot just read one logit.) Softmax over the 10 candidates → `P_d(s)`. Reuse the cached
   prefix (copy or crop the KV cache) or batch the candidates; correctness first, then speed.
4. **Expected score** `E_d = Σ s · P_d(s)`; `p = mean_d(E_d) / rubric_max`; `percentile = clip(p)`
   with `prior.clip_min/clip_max`.
5. **Ensemble (the model's own uncertainty), deterministic:** with
   `rng = np.random.default_rng([ensemble_seed, int(file_hash[:8], 16)])`, draw `ensemble_draws`
   score vectors (each dimension independently from `P_d`), compute p for each, and take the
   `ensemble_quantiles` → `ensemble = [p10, p50, p90]` (clipped). The engine uses the ensemble only as
   a range (`p_from_ensemble` draws between min and max), so this is a 10–90% band.
   Assumption "dimensions independent" → log it in DECISIONS.md.
6. `rubric_scores = {d: E_d}`; `method = "vlm_rubric"`; `evidence_tag = MODEL_RATED`;
   `note` says it is a local model's judgement read from its score probabilities, not trained or
   validated against real outcomes. `rationale` may be empty (or one short greedy generation, optional).
7. Optional (SHOULD, only if time): add `rubric_probs: dict[str, list[float]] | None` to `PriorResult`.
   That is a schema change → update `backend/schemas.py`, `frontend/src/types.ts` and tests in the
   same commit (CLAUDE.md rule 9), then show the distributions on the Video screen.

Why this is better: same video → same p every time (CLAUDE.md rule 4), no JSON parsing failures, and a
real measure of how sure the model is. Explain it in DECISIONS.md.

### 6.4 Hook analysis (display only, CLAUDE.md rule 7)

`generate_json(prompt, images, schema)`: greedy decoding (`do_sample=False`), thinking disabled,
`max_new_tokens = analysis_max_new_tokens`. Extract the first JSON object from the text (strip
```` ```json ```` fences), validate with the Pydantic `schema` (`AnalysisResponse`), retry once with a
"return only valid JSON" reminder, otherwise raise `VLMUnavailable` (the existing code then shows
"analysis unavailable"). Reuse `prompts/analysis.md`.

### 6.5 Attention-risk timeline

Same as today (8 windows, rerun the rubric with each window's frames removed), but through the local
scorer: deterministic, so `attention_samples` = 1 for local. The current `within_noise` rule uses the
std of the 3 GPT samples; with the quantile ensemble that std is a band, not sampling noise, so use
`video.attention_min_change` instead when the provider is local. Keep the label "where the rating is
sensitive (not where viewers drop off)".

### 6.6 Fallbacks (never crash the app)

provider local but torch/transformers missing, weights not downloaded and offline, or out of memory →
raise `VLMUnavailable` with a short reason → existing fallbacks: heuristic prior (tag `ASSUMED`),
"analysis unavailable", no transcript. `VIRALYST_OFFLINE=1` must still force the fallbacks (tests rely
on it). Log the reason in the profile notes.

### 6.7 Caching

All VLM outputs are cached by file hash (CLAUDE.md rule 2). Every cache key must include
`active_model_tag()` and a scoring version, so GPT-era and local results never mix and switching tiers
re-rates cleanly. Video IDs are content hashes, so the test videos keep their IDs.

## 7. Tests (no real weights in `pytest -q`)

Add `backend/tests/test_local_vlm.py` with a **fake model** (inject through a module-level hook or
monkeypatch `local_vlm._get_model`) that returns fixed candidate log-probs. Tests:

1. `pick_tier`: given (cuda, 24 GB) → A; (cuda, 16) → B; (cuda, 12) → C; (cuda, 8) → D; (mps, 36) → M;
   (cpu, 0) → CPU; headroom respected.
2. Scoring: each `P_d` sums to 1; `1 ≤ E_d ≤ 10`; `p` within clip bounds; `p10 ≤ p50 ≤ p90`.
3. Determinism: two calls on the same profile → identical `PriorResult`.
4. Cache key changes when the model tag changes.
5. JSON extraction: fenced/extra-text output → valid `AnalysisResponse`; garbage → `VLMUnavailable`.
6. Provider local with the fake "weights missing" → heuristic fallback with a note (no exception).
7. Provider `openai` still routes to the old path (monkeypatch the client).
8. Optional real-model smoke test marked `@pytest.mark.gpu`, skipped unless `VIRALYST_GPU_TESTS=1`:
   rate `tests/fixtures/media/cooking_weak.mp4` twice → identical p; print p for weak and strong.

All existing 69 tests must still pass. Frontend: `npm run build` must pass.

## 8. Step order (small commits, prefix by area, `pytest -q` before each)

1. **Environment** (section 4): report GPU, memory, chosen tier, download size. Wait for OK.
2. **Config + tier selection + safe download** (`local_vlm.py` skeleton, `--info`, `--download`),
   tests 1. Commit `video: local VLM config, tier selection and safe download`.
3. **Probability rubric prior** (6.3) + tests 2–4, 6. Run it on the three fixture videos and report
   p, the 10–90% band, time per video and peak VRAM. **Stop for review.**
4. **Hook analysis via `generate_json`** (6.4) + test 5. Commit.
5. **Attention timeline** on the local scorer (6.5). Commit.
6. **Whisper transcript** (`frames.py`). Commit.
7. **Provider switch tidy-up**: `/health`, OpenAI path still works (test 7), requirements. Commit.
8. **Frontend labels** (no "GPT" for ratings; show model name). `npm run build`. Commit `ui: ...`.
9. **End-to-end on the app** (`.claude/launch.json` has `backend` and `frontend` configs):
   upload the fixtures from `backend/tests/fixtures/media/`, run Analyse, a matched experiment
   (`cooking_weak` vs `cooking_strong`), the attention timeline and the Hook Surgeon on `cooking_weak`.
   Record timings.
10. **Docs**: DECISIONS.md (provider switch, tiers, probability scoring, independence assumption,
    quantile ensemble, attention threshold, timings), ENGINE.md section 2, PROGRESS.md "Models used",
    this file's checklist. Commit `docs: ...`.
11. Push the branch and open a PR into `main` with the before/after numbers. Do not push to `main`.

## 9. Acceptance checklist

- [ ] `vlm.provider: local` is the default; the app runs with `OPENAI_API_KEY` unset and Wi-Fi off.
- [ ] Rating the same video twice gives an identical p and ensemble.
- [ ] Video screen shows the local model's name with the `MODEL_RATED` tag; no rating says "GPT".
- [ ] Attention timeline and Hook Surgeon complete end to end on `cooking_weak`.
- [ ] Heuristic fallback still works (`VIRALYST_OFFLINE=1`).
- [ ] `pytest -q` green (69 old + new); `npm run build` green.
- [ ] No weights, caches or keys committed (`git status` clean of `cache/`, `*.safetensors`).
- [ ] DECISIONS.md, ENGINE.md, PROGRESS.md updated; timings recorded.

Targets on a decent laptop GPU (estimates; measure and record): model load < 90 s; one video
rating < 20 s after load; attention timeline < 3 min; Hook Surgeon (4 edits) < 5 min.

For reference, GPT-era ratings: `cooking_weak` p = 0.59 (0.58, 0.58, 0.60), `cooking_strong`
p = 0.75. The local numbers will differ; report them side by side, do not tune toward them.

## 10. Rules that bite (from CLAUDE.md — read it)

* No hard-coded numbers: every probability, threshold, size or count goes in `config.yaml`.
* No model call inside the simulation loop; models only in preprocessing, cached, schema-validated,
  with a non-network fallback.
* Determinism: explicit `numpy.random.Generator` seeds, no global RNG (also for the ensemble draws).
* Evidence tags on every user-visible number.
* Wording: never "predicts views", "will go viral" or "Instagram's algorithm"; say "in this simulation",
  "assumed", "the model suggests".
* Schema change → `schemas.py` + tests + `frontend/src/types.ts` in the same change.
* Windows: `storage.write_json` already retries the rename on `PermissionError`; keep using it.
* Do not edit or regenerate `backend/data/archetypes*.json`.

## 11. Useful facts

* Commands: `cd backend && pytest -q`; backend `python -m uvicorn app.api.main:app --app-dir backend --port 8000`
  (from the repo root); frontend `npm --prefix frontend run dev` (port 5173, proxies `/api` to 8000).
* Cache: `backend/cache/` (or `VIRALYST_CACHE_DIR`), git-ignored. A fresh clone starts empty; the
  synthetic world rebuilds deterministically from `population.default_seed`.
* Test media: `backend/tests/fixtures/media/` (`cooking_weak.mp4`, `cooking_strong.mp4`, `gaming.mp4`,
  images, `narration.mp3`). Known video IDs: cooking_weak `5cb41f49dac5c41c`, cooking_strong
  `ea835f18e0d8f735`, gaming `5ea1819ff4841295`.
* Frames: `preprocessing/frames.py` (`extract_frames`, `scan_video`); rubric dims in `prior.rubric`;
  frame list for prompts: `prior.frame_items(profile)`.
* Hook Surgeon calls `pipeline.analyze_video(edit_id, with_analysis=False)` for each edit, so the
  prior change applies to it automatically.
