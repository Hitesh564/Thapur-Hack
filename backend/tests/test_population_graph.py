import json

import numpy as np
import pandas as pd

from app.config import DATA_DIR
from app.graph import ROLE_HUB, build_graph
from app.personas import load_archetypes, sample_population

SENSITIVE = ("ethnic", "race", "religio", "church", "politic", "party", "health", "disease",
             "disab", "sexual", "gender", "caste", "pregnan")


# ---- A1 population ----
def test_same_seed_identical_population(cfg, tmp_path):
    a, b = sample_population(cfg, seed=11), sample_population(cfg, seed=11)
    a.save(tmp_path / "a.parquet")
    b.save(tmp_path / "b.parquet")
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "a.parquet"), pd.read_parquet(tmp_path / "b.parquet"))
    assert a.version == b.version
    assert sample_population(cfg, seed=12).version != a.version


def test_population_size_matches_config(cfg):
    pop = sample_population(cfg)
    assert pop.n == cfg.get("population.n_viewers")
    assert set(np.unique(pop.community)) == set(range(cfg.get("population.n_communities")))
    assert (pop.traits["attention"] >= 0).all() and (pop.traits["attention"] <= 1).all()
    assert np.quantile(pop.influence, 0.99) > 3 * np.median(pop.influence)  # heavy tail


# ---- D4 archetypes ----
def test_archetypes_load_offline_without_sensitive_traits():
    arcs, src = load_archetypes()
    assert len(arcs) == 50
    text = json.dumps([a.model_dump() for a in arcs]).lower()
    assert not [w for w in SENSITIVE if w in text]


def test_archetypes_fall_back_on_bad_file(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("[{\"name\": \"x\"}]")
    arcs, src = load_archetypes(bad)
    assert src == "archetypes_fallback.json" and len(arcs) == 50
    assert (DATA_DIR / "archetypes_fallback.json").exists()


# ---- A2 graph ----
def test_intra_community_ratio_matches_config(world, cfg):
    assert abs(world.graph.intra_edge_ratio() - cfg.get("graph.intra_community_ratio")) < 0.03


def test_hubs_are_top_followers(world, cfg):
    g = world.graph
    hubs = g.role == ROLE_HUB
    assert hubs.sum() == round(cfg.get("graph.hub_pct") * g.n)
    assert g.followers[hubs].min() >= g.followers[~hubs].max()


def test_graph_deterministic(cfg):
    pop = sample_population(cfg, seed=3)
    a, b = build_graph(pop, cfg), build_graph(pop, cfg)
    assert a.version == b.version and (a.adj != b.adj).nnz == 0 and (a.role == b.role).all()
