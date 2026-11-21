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
            pre = read_json(d / "preprocess.json") or {}
            frames = pre.get("frame_paths") or []
            bundle = read_json(d / "analysis_bundle.json")
            out.append({**meta, "analysed": bool(bundle), "derived_from": meta.get("derived_from"), "thumb": Path(frames[0]).name if frames else None,
                        "topic": (bundle or {}).get("profile", {}).get("topic", ""),
                        "prior": (bundle or {}).get("prior", {}).get("percentile")})
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
    get_analysis(video_id)  # refreshes relevance if the population changed
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


@app.post("/videos/{video_id}/attention")
def attention(video_id: str) -> dict:
    from app.video_analysis import VLMUnavailable
    try:
        return service.run_attention(video_id)
    except service.NotAnalysed as e:
        raise HTTPException(409, str(e)) from e
    except VLMUnavailable as e:
        raise HTTPException(503, f"Attention-risk timeline unavailable: {e}") from e


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


@app.get("/simulations/{run_id}/cascade")
def get_simulation_cascade(run_id: str) -> dict:
    c = service.get_cascade(run_id)
    if c is None:
        raise HTTPException(404)
    return c


@app.get("/world/layout")
def world_layout() -> dict:
    return service.world_layout()


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


# ---- tipping-point curve -----------------------------------------------------
class TippingRequest(BaseModel):
    video_id: str
    n_seeds: Optional[int] = None
    grid_points: Optional[int] = None
    seeding_strategy: Optional[str] = None
    policy_variant: str = "default"


@app.post("/tipping")
def tipping(req: TippingRequest) -> dict:
    try:
        return service.run_tipping(**req.model_dump())
    except service.NotAnalysed as e:
        raise HTTPException(409, str(e)) from e


# ---- background jobs: Hook Surgeon, robust seeding plan ---------------------
@app.post("/videos/{video_id}/surgeon")
def surgeon_start(video_id: str) -> dict:
    from app.api import jobs
    from app.experiments.surgeon import run_surgeon
    if get_analysis(video_id) is None:
        raise HTTPException(409, "analyse the video first")
    return {"job_id": jobs.start("surgeon", run_surgeon, video_id=video_id)}


@app.get("/videos/{video_id}/surgeon")
def surgeon_latest(video_id: str) -> dict:
    from app.experiments.surgeon import latest_surgeon
    res = latest_surgeon(video_id)
    if not res:
        raise HTTPException(404, "no Hook Surgeon result yet")
    return res


class SeedingPlanRequest(BaseModel):
    video_id: str
    n_seeds: Optional[int] = None
    n_samples: Optional[int] = None


@app.post("/seeding-plan")
def seeding_plan_start(req: SeedingPlanRequest) -> dict:
    from app.api import jobs
    if get_analysis(req.video_id) is None:
        raise HTTPException(409, "analyse the video first")
    return {"job_id": jobs.start("seeding", service.run_seeding_plan, **req.model_dump())}


@app.get("/seeding-plan/{video_id}")
def seeding_plan_latest(video_id: str) -> dict:
    res = service.latest_seeding_plan(video_id)
    if not res:
        raise HTTPException(404, "no seeding plan yet")
    return res


@app.post("/videos/{video_id}/drivers")
def drivers_start(video_id: str) -> dict:
    from app.api import jobs
    if get_analysis(video_id) is None:
        raise HTTPException(409, "analyse the video first")
    return {"job_id": jobs.start("drivers", service.run_drivers, video_id=video_id)}


@app.get("/videos/{video_id}/drivers")
def drivers_latest(video_id: str) -> dict:
    res = service.latest_drivers(video_id)
    if not res:
        raise HTTPException(404, "no driver analysis yet")
    return res


class CounterfactualRequest(BaseModel):
    variable: Literal["hook_clip", "hypothetical_hook", "seeding"]
    baseline_video_id: str
    variant_video_id: Optional[str] = None
    hypothetical_delta: Optional[float] = None
    baseline_seeding: Optional[str] = None
    variant_seeding: Optional[str] = None
    seed: Optional[int] = None


@app.post("/counterfactual")
def counterfactual(req: CounterfactualRequest) -> dict:
    try:
        return service.run_counterfactual(**req.model_dump())
    except service.NotAnalysed as e:
        raise HTTPException(409, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    from app.api import jobs
    j = jobs.get(job_id)
    if j is None:
        raise HTTPException(404, "unknown job (jobs are kept in memory)")
    return j


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
    from app.experiments.backtest import latest_backtest
    return build_ledger(cfg, prior=prior, relevance_by_community=rel, run=run, experiment=exp, backtest=latest_backtest())


@app.get("/backtest")
def backtest() -> dict:
    from app.experiments.backtest import latest_backtest, load_pairs
    res = latest_backtest()
    return res or {"n": 0, "pairs_in_csv": len(load_pairs()), "label": "Not validated against real outcomes.",
                   "evidence_tag": "MEASURED"}


@app.post("/backtest")
def backtest_run() -> dict:
    from app.experiments.backtest import run_backtest
    return run_backtest()
