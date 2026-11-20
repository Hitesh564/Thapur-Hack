import numpy as np
import pytest

from app.config import Params
from app.diagnostics import binding_constraint, diagnose_r, find_bottlenecks, round_table
from app.engine import simulate
from app.policy import resolve_policy


def test_r_identity_holds_on_real_log(make_inputs):
    for s in range(4):
        r = simulate(make_inputs(0.75), seed=s)
        for row in round_table(r.log):
            prod = row["stay_rate"] * row["share_given_stay"] * row["exposures_per_sharer"]
            assert row["R_share"] == pytest.approx(prod, abs=1e-12)


def test_r_counts_match_engine_metrics(make_inputs):
    r = simulate(make_inputs(0.75), seed=1)
    rows = round_table(r.log)
    assert len(rows) == len(r.rounds)
    for row, m in zip(rows, r.rounds):
        assert row["exposed"] == m.exposed
        assert row["rec_exposures_next"] == m.rec_exposures_next
        assert row["share_exposures_next"] == m.share_exposures_next
        assert row["R_share"] == pytest.approx(m.R_share)
        assert row["R_rec"] == pytest.approx(m.R_rec)
        assert row["stay_rate"] == pytest.approx(m.stay_rate)


def test_community_identity_and_binding(make_inputs, world, cfg):
    r = simulate(make_inputs(0.75), seed=0)
    d = diagnose_r(r.log, Params(cfg), world.population.n_communities)
    assert d["binding"].ratio == min(f.ratio for f in d["factors"])
    for c in d["by_community"].values():
        for row in c["rounds"]:
            assert row["R_share"] == pytest.approx(row["stay_rate"] * row["share_given_stay"] * row["exposures_per_sharer"])


def test_binding_constraint_picks_lowest_ratio(cfg):
    pool = {"exposed": 100, "stayers": 50, "sharers": 1, "share_exposures_next": 3,
            "stay_rate": 0.5, "share_given_stay": 0.02, "exposures_per_sharer": 3.0,
            "R_share": 0.03, "rec_exposures_next": 40, "R_rec": 0.4, "rec_batch_ratio": 0.4, "R": 0.43}
    b, items = binding_constraint(pool, Params(cfg))
    assert b.factor == "rec_batch_ratio"  # 0.4/1.0 is the lowest ratio
    m = (1 - 0.4) / 0.03
    share_item = next(i for i in items if i.factor == "share_given_stay")
    assert share_item.needed_multiplier_for_R1 == pytest.approx(m)
    assert share_item.needed_value_for_R1 == pytest.approx(0.02 * m)


def _bottlenecks(r, make_inputs_rel, cfg, world, **kw):
    P = Params(cfg, kw.pop("overrides", {}))
    sizes = np.bincount(world.population.community)
    return {d.code: d for d in find_bottlenecks(r.log, P, make_inputs_rel, r.rounds, r.stop_reason, sizes,
                                                resolve_policy(P).S0, **kw)}


def test_bottleneck_rules_trigger(make_inputs, relevance, cfg, world):
    weak = simulate(make_inputs(0.05), seed=0)
    codes = _bottlenecks(weak, relevance, cfg, world)
    assert "weak_initial_attention" in codes
    assert "distribution_threshold_not_reached" in codes
    for d in codes.values():
        assert d.message.endswith("in this simulated run.")
        assert d.evidence and 0 <= d.severity <= 1

    no_share = simulate(make_inputs(0.8, overrides={"behaviour.base_share": 0.0}), seed=0)
    assert "insufficient_sharing" in _bottlenecks(no_share, relevance, cfg, world)

    retention = simulate(make_inputs(0.5, overrides={"behaviour.m_min": 0.05, "behaviour.m_max": 0.1}), seed=0)
    assert "retention_weakness" in _bottlenecks(retention, relevance, cfg, world)

    big = simulate(make_inputs(0.95), seed=0)
    assert "community_saturation" in _bottlenecks(big, relevance, cfg, world)

    nobody = np.full(world.population.n, 0.05)
    small = simulate(make_inputs(0.5, rel=nobody), seed=0)
    assert "limited_candidate_audience" in _bottlenecks(small, nobody, cfg, world)

    codes = _bottlenecks(weak, relevance, cfg, world, other_seed_milestone_rate=0.8, reached_milestone=False)
    assert "stochastic_failure" in codes
