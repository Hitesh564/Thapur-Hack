import json
from pathlib import Path

import numpy as np
import pytest

from app.config import Params, get_config
from app.engine import SimInputs, World
from app.graph import build_graph
from app.personas import sample_population
from schemas import PriorResult, VideoProfile

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def cfg():
    return get_config()


@pytest.fixture(scope="session")
def world(cfg):
    pop = sample_population(cfg)
    return World(pop, build_graph(pop, cfg))


@pytest.fixture(scope="session")
def relevance(world):
    """Fixture relevance: community 0 (food) high, others lower, plus viewer noise."""
    rng = np.random.default_rng(123)
    comm = world.population.community
    base = np.where(comm == 0, 0.7, 0.3)
    return np.clip(base + rng.normal(0, 0.1, comm.size), 0, 1)


@pytest.fixture
def make_inputs(world, relevance, cfg):
    def _make(p=0.6, overrides=None, **kw):
        return SimInputs(world=world, relevance=kw.pop("rel", relevance), prior_p=p, duration_s=kw.pop("duration_s", 30.0),
                         params=Params(cfg, overrides or {}), **kw)
    return _make


@pytest.fixture(scope="session")
def video_profile():
    return VideoProfile.model_validate(json.loads((FIXTURES / "video_profile_cooking.json").read_text()))


@pytest.fixture(scope="session")
def prior_result():
    return PriorResult.model_validate(json.loads((FIXTURES / "prior_result.json").read_text()))
