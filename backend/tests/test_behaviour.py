import numpy as np

from app.behaviour import ACTIONS, evaluate
from app.config import Params

N = 20000


def _run(cfg, p, seed=0):
    rng = np.random.default_rng(seed)
    rel = rng.random(N)
    attn = rng.random(N)
    tend = {a: np.ones(N) for a in ACTIONS}
    u = {d: rng.random(N) for d in ("stay", "watch") + ACTIONS}
    return evaluate(Params(cfg), p, rel, float(rel.mean()), attn, tend, 30.0, u)


def test_skip_implies_no_actions(cfg):
    out = _run(cfg, 0.5)
    for a in ACTIONS:
        assert not (out.acts[a] & ~out.stay).any()
    assert (out.watch_fraction[~out.stay] == 0).all()


def test_share_below_like_on_average(cfg):
    out = _run(cfg, 0.5)
    assert out.acts["share"].mean() < out.acts["like"].mean()


def test_higher_p_raises_stay(cfg):
    lo, hi = _run(cfg, 0.2), _run(cfg, 0.8)
    assert hi.stay.mean() > lo.stay.mean()
    # same dice: every viewer who stays at low p also stays at high p
    assert not (lo.stay & ~hi.stay).any()


def test_watch_fraction_bounds_and_reasons(cfg):
    out = _run(cfg, 0.5)
    assert (out.watch_fraction >= 0).all() and (out.watch_fraction <= 1).all()
    assert set(np.unique(out.reason)) <= {0, 1, 2, 3}
