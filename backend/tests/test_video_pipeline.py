"""Workstream D + API tests. Offline (VIRALYST_OFFLINE=1) and in a temporary cache dir."""
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

MEDIA = Path(__file__).parent / "fixtures" / "media"
needs_media = pytest.mark.skipif(not (MEDIA / "cooking_weak.mp4").exists(), reason="run scripts/make_test_media.py")


@pytest.fixture(autouse=True)
def offline_tmp_cache(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("VIRALYST_OFFLINE", "1")
    monkeypatch.setenv("VIRALYST_CACHE_DIR", str(tmp_path_factory.getbasetemp() / "viralyst_cache"))


# ---- ingestion / preprocessing (D1) ----
def test_ingest_rejects_bad_type(tmp_path):
    from app.ingestion.upload import IngestionError, ingest_file
    f = tmp_path / "notes.txt"
    f.write_text("hello")
    with pytest.raises(IngestionError):
        ingest_file(f)


@needs_media
def test_preprocess_frames_cuts_and_cache():
    from app.ingestion.upload import ingest_file
    from app.preprocessing.frames import preprocess
    meta = ingest_file(MEDIA / "cooking_weak.mp4")
    assert meta["has_audio"] and meta["vertical"] and 5 < meta["duration_s"] < 15
    a = preprocess(meta["video_id"])
    assert len(a["frame_paths"]) == 8 and len(a["opening_frame_paths"]) == 4
    assert all(Path(p).exists() for p in a["frame_paths"])
    assert 1 <= len(a["cut_times_s"]) <= 4          # slideshow has 2 hard cuts
    assert a["features"]["opening_motion"] < a["features"]["motion_energy"]  # static weak opening
    assert any("Transcript unavailable" in n for n in a["notes"])          # offline -> labelled gap
    assert preprocess(meta["video_id"])["cache_hit"] is True               # AT: second run hits cache


# ---- prior (D3) ----
def test_heuristic_prior_bounds(video_profile, cfg):
    from app.prior import heuristic_prior
    pr = heuristic_prior(video_profile)
    assert cfg.get("prior.clip_min") <= pr.percentile <= cfg.get("prior.clip_max")
    assert pr.method == "heuristic" and pr.evidence_tag.value == "ASSUMED" and len(pr.ensemble) == 3


def test_prior_falls_back_offline(video_profile):
    from app.prior import prior
    pr = prior(video_profile, "vlm_rubric")
    assert pr.method == "heuristic" and "fallback" in pr.note


def test_rubric_schema_validates():
    from app.prior.prior import RubricResponse
    good = {d: 6 for d in ("immediate_clarity", "curiosity", "visual_impact", "pacing", "payoff_clarity", "audience_fit")}
    RubricResponse(scores=good)
    with pytest.raises(ValidationError):
        RubricResponse(scores={**good, "curiosity": 11})
    with pytest.raises(ValidationError):
        RubricResponse(scores={"vibes": 5})


# ---- relevance (D5) ----
def test_keyword_relevance_cooking_food_over_gaming(video_profile, world):
    from app.relevance import compute_relevance
    rel, res = compute_relevance(video_profile, world.population, method="keyword")
    assert res.method == "keyword" and res.evidence_tag.value == "COMPUTED"
    assert res.by_community["Food & Cooking"] > res.by_community["Gaming"]
    assert ((rel >= 0) & (rel <= 1)).all()


@needs_media
def test_clip_relevance_cooking_food_over_gaming(world):
    pytest.importorskip("open_clip")
    from app.ingestion.upload import ingest_file
    from app.pipeline import base_profile
    from app.relevance import compute_relevance
    out = {}
    for name in ("cooking_weak", "gaming"):
        vid = ingest_file(MEDIA / f"{name}.mp4")["video_id"]
        out[name] = compute_relevance(base_profile(vid, with_transcript=False), world.population)[1]
    if out["gaming"].method != "clip":
        pytest.skip("CLIP weights unavailable")
    assert out["cooking_weak"].by_community["Food & Cooking"] > out["cooking_weak"].by_community["Gaming"]
    assert out["gaming"].by_community["Gaming"] > out["gaming"].by_community["Food & Cooking"]


# ---- fallbacks + API end to end (offline) ----
@needs_media
def test_api_offline_end_to_end():
    from app.api.main import app
    c = TestClient(app)
    with open(MEDIA / "cooking_weak.mp4", "rb") as fh:
        vid = c.post("/videos", files={"file": ("cooking_weak.mp4", fh, "video/mp4")}).json()["video_id"]
    with open(MEDIA / "cooking_strong.mp4", "rb") as fh:
        alt = c.post("/videos", files={"file": ("cooking_strong.mp4", fh, "video/mp4")}).json()["video_id"]
    assert c.post("/simulations", json={"video_id": vid}).status_code == 409  # not analysed yet
    for v in (vid, alt):
        b = c.post(f"/videos/{v}/analyze", json={"relevance_method": "keyword"}).json()
        assert b["prior"]["method"] == "heuristic"                   # VLM unavailable -> labelled fallback
        assert any("unavailable" in n for n in b["profile"]["notes"])
    run = c.post("/simulations", json={"video_id": vid, "stochastic_check_seeds": 2}).json()
    assert run["summary"]["unique_reach"] > 0 and run["summary"]["evidence_tag"] == "SIMULATED"
    assert run["diagnosis"]["binding"]["factor"] and run["attribution"]["table"]
    assert all(b["message"].endswith("in this simulated run.") for b in run["diagnosis"]["bottlenecks"])
    ev = c.get(f"/simulations/{run['run_id']}/events?limit=5").json()
    assert ev["total"] > 0 and len(ev["events"]) == 5
    exp = c.post("/experiments", json={"variable": "hook_clip", "baseline_video_id": vid, "variant_video_id": alt,
                                       "n_seeds": 3, "n_samples": 2}).json()
    assert exp["result"]["label"] in ("Robust", "Likely", "Inconclusive")
    assert c.get(f"/experiments/{exp['result']['experiment_id']}").status_code == 200
    led = c.get(f"/ledger?video_id={vid}&run_id={run['run_id']}").json()
    assert all("tag" in e for e in led["entries"])
    assert np.isclose(sum(r["share_of_reach"] for r in run["attribution"]["table"]), 1.0)
