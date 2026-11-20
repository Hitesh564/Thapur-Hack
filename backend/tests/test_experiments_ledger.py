from fastapi.testclient import TestClient

from app.api.main import app
from app.experiments import Arm, run_matched, stability_label
from app.ledger import NOT_VALIDATED, build_ledger
from schemas import EvidenceTag


def test_identical_arms_give_exactly_zero_difference(world, relevance, cfg):
    arm = Arm([0.5, 0.55, 0.6], relevance, 30.0, label="same")
    res, rec = run_matched(world, cfg, arm, arm, "hook_clip", n_seeds=3, n_samples=2)
    assert rec["baseline_values"] == rec["variant_values"]
    assert res.effect_median == res.effect_p10 == res.effect_p90 == 0
    assert res.win_rate == 0 and res.label == "Inconclusive"


def test_stronger_hypothetical_hook_wins(world, relevance, cfg):
    base = Arm([0.5, 0.55, 0.6], relevance, 30.0, label="base")
    var = Arm([0.5, 0.55, 0.6], relevance, 30.0, p_delta=0.2, prior_method="hypothetical", label="hyp")
    res, rec = run_matched(world, cfg, base, var, "hypothetical_hook", n_seeds=4, n_samples=3)
    assert res.effect_median >= 0 and res.win_rate > 0.5
    assert len(rec["seeds"]) == 4 and len(rec["assumption_samples"]) == 3
    assert "population" in rec["config"]["held_fixed"]


def test_label_thresholds(cfg):
    assert stability_label(0.9, 0.85, cfg) == "Robust"
    assert stability_label(0.9, 0.7, cfg) == "Likely"
    assert stability_label(0.65, 0.95, cfg) == "Likely"
    assert stability_label(0.59, 0.95, cfg) == "Inconclusive"
    assert stability_label(0.8, 0.8, cfg) == "Robust"


def test_ledger_every_entry_tagged(cfg, prior_result, make_inputs):
    from app.engine import simulate
    run = simulate(make_inputs(0.6), seed=0).summary
    led = build_ledger(cfg, prior=prior_result, relevance_by_community={"Food & Cooking": 0.7}, run=run)
    tags = {t.value for t in EvidenceTag}
    assert led["entries"] and all(e["tag"] in tags for e in led["entries"])
    metrics = {e["metric"] for e in led["entries"]}
    assert set(cfg.ranged_paths()) <= metrics
    assert {"prior_result.percentile", "run.unique_reach"} <= metrics
    assert led["validation"] == NOT_VALIDATED


def test_api_health_and_ledger():
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    body = client.get("/ledger").json()
    assert all("tag" in e for e in body["entries"])
