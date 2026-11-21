"""Counterfactual Trace: every number must add up exactly."""
import numpy as np
import pytest

from app.diagnostics.counterfactual import representative_seed, trace_pair


def _names(world):
    return world.population.community_names


def test_identical_inputs_trace_to_nothing(make_inputs, world):
    out = trace_pair(make_inputs(0.6), make_inputs(0.6), 3, _names(world))
    assert out["delta_reach"] == 0
    assert all(v["gained"] == 0 and v["lost"] == 0 for v in out["flips"].values())
    assert out["exposure"]["only_variant"] == 0 and out["exposure"]["only_baseline"] == 0
    assert "identical" in out["message"]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_decomposition_is_exact(make_inputs, world, seed):
    out = trace_pair(make_inputs(0.5), make_inputs(0.65), seed, _names(world))
    d = out["decomposition"]
    assert d["check"] is True
    assert d["seed"] + d["rec_common_rounds"] + d["rec_extra_rounds"] + d["share"] == out["delta_reach"]
    sp = d["share_parts"]
    assert sp["from_new_sharers"] + sp["lost_from_baseline_only_sharers"] + sp["change_from_sharers_in_both"] == d["share"]
    assert sum(out["by_community"].values()) == out["delta_reach"]
    e = out["exposure"]
    assert e["both"] + e["only_variant"] == out["reach_variant"]
    assert e["both"] + e["only_baseline"] == out["reach_baseline"]


def test_stay_flips_cross_the_same_dice(make_inputs, world):
    out = trace_pair(make_inputs(0.4), make_inputs(0.7), 1, _names(world))
    assert out["flips"]["stay"]["gained"] > 0
    assert out["flips"]["stay"]["lost"] == 0  # higher p with same relevance never loses a stayer
    for ex in out["examples"]:
        if ex["stayed_variant"] and not ex["stayed_baseline"]:
            assert ex["p_stay_baseline"] <= ex["dice"] < ex["p_stay_variant"]


def test_rec_growth_split_sums(make_inputs, world):
    out = trace_pair(make_inputs(0.5), make_inputs(0.7), 2, _names(world))
    g = out["rec_growth"]
    assert sum(g["log_batch_difference_by_signal"].values()) == pytest.approx(g["log_batch_difference"])
    assert g["batch_ratio"] == pytest.approx(np.exp(g["log_batch_difference"]))


def test_representative_seed_is_median(make_inputs):
    seeds = list(range(7))
    s, deltas = representative_seed(make_inputs(0.5), make_inputs(0.65), seeds)
    assert s in seeds and len(deltas) == 7
    assert abs(deltas[seeds.index(s)] - np.median(deltas)) == min(abs(d - np.median(deltas)) for d in deltas)
