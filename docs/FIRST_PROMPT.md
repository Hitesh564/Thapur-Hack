# First prompt for Claude Code

Paste this after placing the repo files (CLAUDE.md, TASKS.md, config.yaml, docs/PRD.md, backend/schemas.py).

---

Read CLAUDE.md, docs/PRD.md, TASKS.md, config.yaml and backend/schemas.py completely. We have 24 hours (adjust if I say 20). I am workstream [A/B/C/D].

Do Phase 0 now:
1. Create the repo layout from CLAUDE.md, `backend/requirements.txt` (fastapi, uvicorn, numpy, scipy, networkx, pydantic, pyyaml, pandas, pyarrow, pytest, opencv-python-headless, scenedetect), a config loader that returns typed access to config.yaml, a FastAPI app with `/health`, and fixtures (fake VideoProfile, PriorResult, population, graph) in `backend/tests/fixtures/`.
2. Start with the MUST tasks of my workstream in TASKS.md, test-first from each acceptance line. Run pytest after each task.
3. Do not add anything outside the PRD. If something is unspecified, pick the simplest option and log it in docs/DECISIONS.md.
4. Report back with what passes, what's stubbed, and the next task. Keep to the gates in TASKS.md.
