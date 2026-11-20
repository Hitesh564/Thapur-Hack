import numpy as np

from app.engine import draw_uniforms, simulate
from app.eventlog import CH, reconstruct_cascade


def _logs_equal(a, b):
    assert a.log.cols.keys() == b.log.cols.keys()
    for k in a.log.cols:
        np.testing.assert_array_equal(a.log.cols[k], b.log.cols[k])


def test_determinism_same_seed_identical_log(make_inputs):
    a = simulate(make_inputs(0.6), seed=5)
    b = simulate(make_inputs(0.6), seed=5)
    _logs_equal(a, b)
    assert a.run_id == b.run_id


def test_crn_identical_inputs_identical_outputs(make_inputs, world, cfg):
    U = draw_uniforms(9, cfg.get("policy.max_rounds"), world.population.n)
    a = simulate(make_inputs(0.55), seed=9, U=U)
    b = simulate(make_inputs(0.55), seed=9, U=U.copy())
    _logs_equal(a, b)


def test_monotone_mean_reach_in_p(make_inputs):
    means = [np.mean([simulate(make_inputs(p), seed=s).summary.unique_reach for s in range(8)])
             for p in (0.2, 0.4, 0.6, 0.8)]
    assert all(x <= y for x, y in zip(means, means[1:])), means


def test_zero_share_prob_no_share_exposures(make_inputs):
    r = simulate(make_inputs(0.8, overrides={"behaviour.base_share": 0.0}), seed=1)
    assert r.summary.total_shares == 0
    assert not (r.state["channel"] == CH["share"]).any()


def test_bounds_no_double_exposure(make_inputs, world):
    for s in range(3):
        r = simulate(make_inputs(0.9), seed=s)
        casc = reconstruct_cascade(r.log)
        assert np.unique(casc["viewer_id"]).size == casc["viewer_id"].size
        assert casc["viewer_id"].size <= world.population.n
        assert r.summary.unique_reach == casc["viewer_id"].size


def test_cascade_reconstructable_from_log(make_inputs):
    r = simulate(make_inputs(0.7), seed=2)
    casc = reconstruct_cascade(r.log)
    st = r.state
    ids = casc["viewer_id"]
    np.testing.assert_array_equal(np.sort(ids), np.flatnonzero(st["exposed"]))
    np.testing.assert_array_equal(casc["round"], st["round"][ids])
    np.testing.assert_array_equal(casc["channel"], st["channel"][ids])
    np.testing.assert_array_equal(casc["source_viewer"], st["source"][ids])
    # every share exposure has a sharer one round earlier
    sh = casc["channel"] == CH["share"]
    src = casc["source_viewer"][sh]
    assert st["act_share"][src].all()
    assert (st["round"][src] == casc["round"][sh] - 1).all()


def test_budget_respected(make_inputs, world):
    r = simulate(make_inputs(0.95, overrides={"policy.exposure_budget_pct": 0.2}), seed=0)
    assert r.summary.unique_reach <= int(0.2 * world.population.n)


def test_seeding_strategies_and_policy_variants(make_inputs, world):
    for strat in ("random", "relevance_biased", "mixed", "community_targeted"):
        r = simulate(make_inputs(0.6, seeding_strategy=strat), seed=0)
        assert r.summary.unique_reach > 0
    r = simulate(make_inputs(0.6, seeding_strategy="community_targeted", target_community=3), seed=0)
    seeds = r.state["round"] == 0
    assert (world.population.community[seeds] == 3).all()
    for v in ("conservative", "exploratory", "interest_tight"):
        assert simulate(make_inputs(0.6, policy_variant=v), seed=0).summary.unique_reach > 0
