"""FastAPI orchestration (PRD section 14). Simulation, not prediction."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.api import service
from app.config import get_config
from app.ingestion.upload import IngestionError, get_meta, ingest_file
from app.ledger.evidence import build_ledger
from app.pipeline import analyze_video, get_analysis, get_world
from app.storage import read_json, video_dir
from schemas import ExperimentResult, RunSummary

app = FastAPI(title="Viralyst", description="Propagation experiment lab (simulation under stated assumptions, not prediction).")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "config_version": get_config().get("meta.version")}


# ---- videos -------------------------------------------------------------------
@app.post("/videos")
async def upload_video(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "upload.mp4").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)
    try:
        return ingest_file(tmp_path, file.filename)
    except IngestionError as e:
        raise HTTPException(400, str(e)) from e
    finally:
        tmp_path.unlink(missing_ok=True)


@app.get("/videos")
def list_videos() -> list[dict]:
    root = video_dir("_").parent
    out = []
    for d in sorted(root.iterdir()):
        meta = read_json(d / "meta.json")
        if meta:
            out.append({**meta, "analysed": (d / "analysis_bundle.json").exists()})
    return out


class AnalyzeRequest(BaseModel):
    prior_method: Optional[Literal["vlm_rubric", "heuristic"]] = None
    relevance_method: Optional[Literal["clip", "keyword"]] = None
    offline: bool = False


@app.post("/videos/{video_id}/analyze")
def analyze(video_id: str, req: AnalyzeRequest = AnalyzeRequest()) -> dict:
    try:
        get_meta(video_id)
    except KeyError as e:
        raise HTTPException(404, "unknown video") from e
    rel_method = "keyword" if req.relevance_method == "keyword" else None
    return analyze_video(video_id, prior_method=req.prior_method, relevance_method=rel_method, with_vlm=not req.offline)


@app.get("/videos/{video_id}/analysis")
def analysis(video_id: str) -> dict:
    bundle = read_json(video_dir(video_id) / "analysis_bundle.json")
    if not bundle:
        raise HTTPException(404, "not analysed yet")
    return bundle


@app.get("/videos/{video_id}/frames/{name}")
def frame(video_id: str, name: str):
    p = video_dir(video_id) / "frames" / Path(name).name
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


@app.get("/videos/{video_id}/source")
def source(video_id: str):
    try:
        return FileResponse(get_meta(video_id)["source_path"])
    except KeyError as e:
        raise HTTPException(404) from e


# ---- audience -----------------------------------------------------------------
@app.get("/audience")
def audience(video_id: Optional[str] = None) -> dict:
    import numpy as np
    w = get_world()
    pop, g = w.population, w.graph
    comms = []
    a = get_analysis(video_id) if video_id else None
    for c in range(pop.n_communities):
        m = pop.community == c
        entry = {"community": c, "name": pop.community_names[c], "viewers": int(m.sum()),
                 "hubs": int((g.role[m] == 1).sum()), "bridges": int((g.role[m] == 2).sum()),
                 "archetypes": sorted({pop.archetypes[i].name for i in np.unique(pop.archetype_idx[m])})}
        if a is not None:
            entry["mean_relevance"] = float(a[2][m].mean())
            entry["relevance_tag"] = "COMPUTED"
        comms.append(entry)
    personas = [a_.model_dump() for a_ in pop.archetypes]
    return {"population_version": pop.version, "graph_version": g.version, "n_viewers": pop.n,
            "communities": comms, "archetypes": personas,
            "note": "Personas are fictional archetypes, not real people or demographics."}


# ---- simulations --------------------------------------------------------------
class SimRequest(BaseModel):
    video_id: str
    seed: int = 0
    seeding_strategy: Optional[str] = None
    policy_variant: str = "default"
    prior_override: Optional[float] = None
    stochastic_check_seeds: int = 10


@app.post("/simulations")
def simulations(req: SimRequest) -> dict:
    try:
        return service.run_simulation(req.video_id, req.seed, req.seeding_strategy, req.policy_variant,
                                      req.prior_override, req.stochastic_check_seeds)
    except service.NotAnalysed as e:
        raise HTTPException(409, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.get("/simulations/{run_id}")
def get_simulation(run_id: str) -> dict:
    run = service.get_run(run_id)
    if not run:
        raise HTTPException(404)
    return run


@app.get("/simulations/{run_id}/metrics")
def get_simulation_metrics(run_id: str) -> dict:
    run = service.get_run(run_id)
    if not run:
        raise HTTPException(404)
    return {"summary": run["summary"], "rounds": run["summary"]["rounds_metrics"], "diagnosis": run["diagnosis"]}


@app.get("/simulations/{run_id}/events")
def get_simulation_events(run_id: str, offset: int = 0, limit: int = 1000) -> dict:
    ev = service.get_run_events(run_id, offset, min(limit, 20000))
    if ev is None:
        raise HTTPException(404, "events are kept in memory for recent runs only; re-run the simulation")
    return ev


# ---- experiments --------------------------------------------------------------
class ExperimentRequest(BaseModel):
    variable: Literal["hook_clip", "hypothetical_hook", "seeding", "policy"]
    baseline_video_id: str
    variant_video_id: Optional[str] = None
    baseline_seeding: Optional[str] = None
    variant_seeding: Optional[str] = None
    baseline_policy: str = "default"
    variant_policy: str = "default"
    hypothetical_delta: Optional[float] = None
    n_seeds: Optional[int] = None
    n_samples: Optional[int] = None


@app.post("/experiments")
def experiments(req: ExperimentRequest) -> dict:
    try:
        return service.run_experiment(**req.model_dump())
    except service.NotAnalysed as e:
        raise HTTPException(409, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.get("/experiments/{experiment_id}")
def get_experiment(experiment_id: str) -> dict:
    rec = service.get_experiment(experiment_id)
    if not rec:
        raise HTTPException(404)
    return rec


# ---- ledger -------------------------------------------------------------------
@app.get("/ledger")
def ledger(video_id: Optional[str] = None, run_id: Optional[str] = None, experiment_id: Optional[str] = None) -> dict:
    cfg = get_config()
    prior = rel = run = exp = None
    if video_id and (a := get_analysis(video_id)):
        prior, rel = a[1], a[3].by_community
    if run_id and (r := service.get_run(run_id)):
        run = RunSummary.model_validate(r["summary"])
    if experiment_id and (e := service.get_experiment(experiment_id)):
        exp = ExperimentResult.model_validate(e["result"])
    return build_ledger(cfg, prior=prior, relevance_by_community=rel, run=run, experiment=exp)
