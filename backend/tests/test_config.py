import re
from pathlib import Path

import numpy as np

from app.behaviour import ACTIONS
from app.config import Params
from app.diagnostics.r_decomp import REF_KEYS
from app.policy import POLICY_VARIANTS

APP = Path(__file__).resolve().parent.parent / "app"


def test_typed_access_and_ranges(cfg):
    assert cfg.get("behaviour.q_min") == 0.30
    assert cfg.range("behaviour.q_min") == (0.25, 0.40)
    assert cfg.range("population.n_viewers") is None
    assert "behaviour.base_share" in cfg.ranged_paths()


def test_sample_within_ranges(cfg):
    s = cfg.sample(np.random.default_rng(0))
    for k, v in s.items():
        lo, hi = cfg.range(k)
        assert lo <= v <= hi


def test_params_override_and_unknown_key(cfg):
    P = Params(cfg, {"behaviour.q_min": 0.33})
    assert P["behaviour.q_min"] == 0.33 and P["behaviour.q_max"] == cfg.get("behaviour.q_max")
    try:
        Params(cfg, {"behaviour.nope": 1})
        raise AssertionError("unknown key accepted")
    except KeyError:
        pass


def test_every_config_key_used_in_code_exists(cfg):
    sections = "|".join(map(re.escape, cfg.raw.keys()))
    pat = re.compile(rf'"((?:{sections})\.[A-Za-z0-9_.]+)"')
    used = set()
    for f in APP.rglob("*.py"):
        used |= {k for k in pat.findall(f.read_text(encoding="utf-8"))
                 if k.rsplit(".", 1)[-1] not in {"json", "npy", "parquet", "jpg", "mp3", "md"}}
    used |= {f"prior.heuristic.{k}" for k in cfg.get("prior.heuristic")}
    # keys built with f-strings
    used |= {f"behaviour.base_{a}" for a in ACTIONS}
    used |= {f"policy.variants.{v}" for v in POLICY_VARIANTS if v != "default"}
    used |= set(REF_KEYS.values())
    used |= {f"diagnosis.thresholds.{k}" for k in cfg.get("diagnosis.thresholds")}
    missing = sorted(k for k in used if not cfg.has(k))
    assert len(used) > 40
    assert not missing, missing
