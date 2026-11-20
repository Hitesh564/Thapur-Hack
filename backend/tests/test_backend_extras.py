"""Pathways (B7), share breakdown, stop explanation, tipping (B6), backtest (D7), archetype generator (D4)."""
import csv
from pathlib import Path

import numpy as np
import pytest

from app.config import Params
from app.diagnostics import round_table, share_breakdown, stop_explanation
from app.diagnostics.pathways import classify
from app.engine import simulate
from app.eventlog import CH, build_event_log
from app.graph import ROLE_BRIDGE, ROLE_HUB, ROLE_OTHER

MEDIA = Path(__file__).parent / "fixtures" / "media"


def synth_log(exposures, n_comm=8):
    """exposures: list of (viewer, round, channel, source, community, role, shared)."""
    n = max(e[0] for e in exposures) + 1
    st = {k: np.zeros(n, bool) for k in ("exposed", "stay", "complete", "act_like", "act_comment", "act_save", "act_share")}
    st.update({"round": np.full(n, -1), "channel": np.full(n, -1, np.int8), "source": np.full(n, -1),
               "p_stay": np.zeros(n), "reason": np.zeros(n, np.int8)})
    for a in ("like", "comment", "save", "share"):
        st[f"p_{a}"] = np.zeros(n)
    community, role = np.zeros(n, int), np.zeros(n, np.int8)
    for v, r, ch, src, cm, rl, shared in exposures:
        st["exposed"][v] = st["stay"][v] = True
        st["round"][v], st["channel"][v], st["source"][v] = r, CH[ch], src
        st["act_share"][v] = shared
        community[v], role[v] = cm, rl
    return build_event_log("synthetic", st, community, role)


def _cascade(src_role, cross, n_children=10, rounds=4):
    """One sharer of a given role in community 0; children land in community 0 or 1..k."""
    ex = [(0, 0, "seed", -1, 0, src_role, True)]
    for i in range(n_children):
        comm = (1 + i % 4) if cross else 0
        ex.append((i + 1, 1, "share", 0, comm, ROLE_OTHER, False))
    for j in range(rounds - 2):  # filler rec rounds so the run is not "early"
        ex.append((100 + j, 2 + j, "rec", -1, 0, ROLE_OTHER, False))
    return synth_log(ex)


@pytest.mark.parametrize("name,log_kw,run_kw", [
    ("early_cascade_failure", dict(src_role=ROLE_OTHER, cross=False, rounds=2), dict(stop="low_signal", rounds=2)),
    ("hub_driven", dict(src_role=ROLE_HUB, cross=False), {}),
    ("bridge_driven", dict(src_role=ROLE_BRIDGE, cross=True), {}),
    ("cross_community", dict(src_role=ROLE_OTHER, cross=True), {}),
    ("niche_community_led", dict(src_role=ROLE_OTHER, cross=False), {}),
    ("deep_but_narrow", dict(src_role=ROLE_OTHER, cross=False), dict(depth=7, milestone=False)),
    ("broad_but_shallow", dict(src_role=ROLE_OTHER, cross=True), dict(depth=1, milestone=True)),
])
def test_pathway_rules_trigger(cfg, name, log_kw, run_kw):
    log = _cascade(**log_kw)
    res = classify(log, Params(cfg), run_kw.get("stop", "max_rounds"), run_kw.get("rounds", 6),
                   run_kw.get("depth", 1), run_kw.get("milestone", False))
    assert name in res["matches"], (name, res["matches"], res["evidence"])
    assert res["primary"] == res["matches"][0]


def test_engine_fills_pathways(make_inputs):
    assert simulate(make_inputs(0.05), seed=0).summary.pathways[0] == "early_cascade_failure"


def test_share_breakdown_multiplies_to_exposures_per_sharer(make_inputs):
    for s in range(3):
        r = simulate(make_inputs(0.8), seed=s)
        b = share_breakdown(round_table(r.log), r.rounds)
        prod = b["followers_per_sharer"] * b["unseen_fraction"] * b["acceptance"] * b["cap_and_overlap"]
        assert prod == pytest.approx(b["exposures_per_sharer"])
        assert 0 <= b["unseen_fraction"] <= 1 and 0 <= b["acceptance"] <= 1 and 0 <= b["cap_and_overlap"] <= 1


def test_stop_explanation_wording(make_inputs):
    r = simulate(make_inputs(0.05), seed=0)
    e = stop_explanation(r.stop_reason, round_table(r.log), r.rounds[-1].expansion_score, 0.4)
    assert e["stop_reason"] == "low_signal" and "in this simulated run" in e["message"]


def test_tipping_curve_monotone_and_marks_video(world, relevance, cfg, monkeypatch):
    from app.experiments import tipping_curve
    out = tipping_curve(world, cfg, relevance, 30.0, 0.5, n_seeds=3, grid_points=5)
    rates = [pt["milestone_rate"] for pt in out["points"]]
    assert all(a <= b for a, b in zip(rates, rates[1:]))
    assert rates[0] < rates[-1]
    assert out["position"] in ("steep", "below", "above", "flat") and "in this simulation" in out["message"]
    assert "points_uncertain" in out and len(out["points_uncertain"]) == 5


def test_backtest_offline_counts_pairs(tmp_path, monkeypatch):
    if not (MEDIA / "cooking_weak.mp4").exists():
        pytest.skip("no media")
    monkeypatch.setenv("VIRALYST_OFFLINE", "1")
    monkeypatch.setenv("VIRALYST_CACHE_DIR", str(tmp_path / "cache"))
    from app.experiments.backtest import run_backtest
    p = tmp_path / "pairs.csv"
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pair_id", "creator", "video_a_path", "views_a", "video_b_path", "views_b"])
        w.writerow(["1", "test", MEDIA / "cooking_strong.mp4", 5000, MEDIA / "cooking_weak.mp4", 1000])
        w.writerow(["2", "test", "missing.mp4", 10, MEDIA / "gaming.mp4", 5])
    out = run_backtest(p)
    assert out["n"] == 1 and len(out["skipped"]) == 1
    assert 0 <= out["accuracy"] <= 1 and "Anecdotal" in out["label"] and out["evidence_tag"] == "MEASURED"


def test_archetype_validator(cfg):
    from app.personas import load_archetypes
    from app.personas.generate import validate_set
    from app.personas.population import FALLBACK_PATH
    arcs, _ = load_archetypes(FALLBACK_PATH)
    names = {a.community: a.community_name for a in arcs}
    comms = [names[i] for i in sorted(names)]
    assert validate_set(arcs, comms) == []
    bad = [a.model_copy(update={"name": "Politics Fan"}) for a in arcs[:1]] + arcs[1:]
    bad[0].interests[0].phrase = "political debates"
    assert any("sensitive" in p for p in validate_set(bad, comms))
    assert validate_set(arcs[:-5], comms)  # wrong count


def test_generated_archetypes_valid():
    from app.personas import load_archetypes
    from app.personas.generate import validate_set
    arcs, src = load_archetypes()
    names = {a.community: a.community_name for a in arcs}
    assert validate_set(arcs, [names[i] for i in sorted(names)]) == []
