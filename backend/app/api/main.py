"""FastAPI orchestration. Phase 0: health and ledger; more endpoints at integration."""
from __future__ import annotations

from fastapi import FastAPI

from app.config import get_config
from app.ledger.evidence import build_ledger

app = FastAPI(title="Viralyst", description="Propagation experiment lab (simulation, not prediction).")


@app.get("/health")
def health() -> dict:
    cfg = get_config()
    return {"status": "ok", "config_version": cfg.get("meta.version")}


@app.get("/ledger")
def ledger() -> dict:
    return build_ledger(get_config())
