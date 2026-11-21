"""USP tests: value of information, outcome drivers, robust seeding plan, Hook Surgeon recipes and rendering."""
from pathlib import Path

import numpy as np
import pytest

from app.experiments.voi import outcome_drivers, value_of_information

MEDIA = Path(__file__).parent / "fixtures" / "media"


# ---- value of information ----
def _samples(n, rng):
    return [{"behaviour.base_share": float(x), "policy.S0": float(y)} for x, y in zip(rng.random(n), rng.random(n))]


def test_evpi_zero_when_variant_always_better(cfg):
    rng = np.random.default_rng(0)
    diff = np.abs(rng.normal(50, 10, (10, 5)))
    v = value_of_information(diff, _samples(10, rng), rng.random(10), cfg, 1000.0)
    assert v["decision"] == "variant" and v["evpi"] == pytest.approx(0) and v["decision_risk"] == 0
    assert "holds across" in v["recommendation"]


def test_evpi_identity_and_driver_found(cfg):
    rng = np.random.default_rng(1)
    params = _samples(20, rng)
    share = np.array([p["behaviour.base_share"] for p in params])
    D = 400 * (share - 0.4)  # the decision flips with the share rate
    diff = np.repeat(D[:, None], 4, axis=1)
    v = value_of_information(diff, params, rng.random(20), cfg, 1000.0)
    expected = np.mean(np.maximum(D, 0)) - max(D.mean(), 0)
    assert v["evpi"] == pytest.approx(expected)
    assert v["per_parameter"][0]["parameter"] == "behaviour.base_share"
    assert v["per_parameter"][0]["evppi"] > 0 and 0 < v["decision_risk"] < 1
    assert "share" in v["recommendation"]


def test_outcome_drivers_rank_true_driver():
    rng = np.random.default_rng(2)
    x, noise = rng.random(60), rng.random(60)
    rows = outcome_drivers(3 * x + 0.2 * rng.random(60), {"behaviour.base_share": x, "policy.S0": noise})
    assert rows[0]["parameter"] == "behaviour.base_share" and rows[0]["rho"] > 0.8
    assert sum(r["share"] for r in rows) == pytest.approx(1)


def test_driver_sweep_small(world, relevance, cfg):
    from app.experiments.voi import outcome_driver_sweep
    out = outcome_driver_sweep(world, cfg, relevance, [0.5, 0.6], 30.0, n_samples=8, n_seeds=2)
    assert out["drivers"] and "in this simulation" in out["recommendation"].lower()


# ---- robust seeding plan ----
def test_seeding_plan_minimax(world, relevance, cfg):
    from app.experiments.seeding_plan import robust_seeding_plan
    out = robust_seeding_plan(world, cfg, relevance, [0.55, 0.6], 30.0, n_seeds=2, n_samples=3)
    opts = out["options"]
    assert len(opts) == 3 + world.population.n_communities
    assert all(o["max_regret"] >= 0 for o in opts)
    rec = next(o for o in opts if o["key"] == out["recommended"])
    assert rec["max_regret"] == min(o["max_regret"] for o in opts)
    assert "in this simulation" in out["message"]


# ---- Hook Surgeon ----
def test_shots_merge_short_segments():
    from app.experiments.surgeon import shots
    segs = shots([1.0, 1.2, 5.0], 9.0, 0.4)
    assert segs[0][0] == 0 and segs[-1][1] == 9.0
    assert all(e - s >= 0.4 for s, e in segs)
    assert sum(e - s for s, e in segs) == pytest.approx(9.0)


def test_recipes_and_render_offline(tmp_path, monkeypatch):
    if not (MEDIA / "cooking_weak.mp4").exists():
        pytest.skip("no media")
    monkeypatch.setenv("VIRALYST_OFFLINE", "1")
    monkeypatch.setenv("VIRALYST_CACHE_DIR", str(tmp_path / "cache"))
    from app.experiments.surgeon import edit_recipes, render
    from app.ingestion.upload import get_meta, ingest_file, probe
    from app.preprocessing.frames import preprocess
    vid = ingest_file(MEDIA / "cooking_weak.mp4")["video_id"]
    pre = preprocess(vid, with_transcript=False)
    recipes = edit_recipes(vid, pre)
    names = [r["name"] for r in recipes]
    assert "payoff_first" in names and "teaser" in names
    pay = next(r for r in recipes if r["name"] == "payoff_first")
    assert pay["segments"][0][1] == pytest.approx(pre["duration_s"])  # final shot first
    meta = get_meta(vid)
    out = render(Path(meta["source_path"]), pay["segments"], tmp_path / "pay.mp4", meta["has_audio"])
    info = probe(out)
    assert abs(info["duration_s"] - pre["duration_s"]) < 0.3 and info["has_audio"]
