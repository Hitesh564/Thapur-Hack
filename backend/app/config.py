"""Typed access to config.yaml. Every model number in the code comes from here.

Paths are dotted, e.g. ``cfg.get("behaviour.q_min")``. An entry written as
``{default, min, max}`` is a *ranged* (uncertain) parameter: ``get`` returns its
default and the robustness loop samples it uniformly in ``[min, max]``.
"""
from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

import numpy as np
import yaml

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = BACKEND_DIR.parent
DEFAULT_CONFIG_PATH = REPO_DIR / "config.yaml"
DATA_DIR = BACKEND_DIR / "data"


def is_range(node: Any) -> bool:
    return isinstance(node, dict) and {"default", "min", "max"} <= set(node.keys())


class Config:
    def __init__(self, raw: dict, path: Optional[Path] = None):
        self.raw = raw
        self.path = path

    # ---- lookup -------------------------------------------------------
    def node(self, path: str) -> Any:
        cur: Any = self.raw
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                raise KeyError(f"config key missing: {path}")
            cur = cur[part]
        return cur

    def get(self, path: str) -> Any:
        v = self.node(path)
        return v["default"] if is_range(v) else v

    def has(self, path: str) -> bool:
        try:
            self.node(path)
            return True
        except KeyError:
            return False

    def range(self, path: str) -> Optional[tuple[float, float]]:
        v = self.node(path)
        return (float(v["min"]), float(v["max"])) if is_range(v) else None

    # ---- ranged parameters ---------------------------------------------
    def ranged_paths(self) -> list[str]:
        out: list[str] = []

        def walk(node: Any, prefix: str) -> None:
            if is_range(node):
                out.append(prefix)
            elif isinstance(node, dict):
                for k, v in node.items():
                    walk(v, f"{prefix}.{k}" if prefix else str(k))

        walk(self.raw, "")
        return out

    def defaults(self) -> dict[str, float]:
        return {p: float(self.get(p)) for p in self.ranged_paths()}

    def sample(self, rng: np.random.Generator) -> dict[str, float]:
        """One assumption sample: every ranged parameter uniform in [min, max]."""
        out = {}
        for p in self.ranged_paths():
            lo, hi = self.range(p)  # type: ignore[misc]
            out[p] = float(lo + (hi - lo) * rng.random())
        return out

    def section_hash(self, *sections: str) -> str:
        payload = json.dumps({s: self.raw.get(s) for s in sections}, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


@lru_cache(maxsize=8)
def _load(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as fh:
        return Config(yaml.safe_load(fh), Path(path))


def get_config(path: Optional[str | Path] = None) -> Config:
    p = path or os.environ.get("VIRALYST_CONFIG") or DEFAULT_CONFIG_PATH
    return _load(str(Path(p).resolve()))


class Params:
    """Parameter view for one run: assumption-sample overrides on top of config.

    ``P["behaviour.q_min"]`` returns the sampled value if this run's assumption
    sample set it, else the config default. Unknown keys raise KeyError.
    """

    def __init__(self, cfg: Config, overrides: Optional[Mapping[str, float]] = None):
        self.cfg = cfg
        self.overrides = dict(overrides or {})
        unknown = [k for k in self.overrides if not cfg.has(k)]
        if unknown:
            raise KeyError(f"override keys not in config: {unknown}")

    def __getitem__(self, path: str) -> Any:
        if path in self.overrides:
            return self.overrides[path]
        return self.cfg.get(path)

    def with_overrides(self, extra: Mapping[str, float]) -> "Params":
        return Params(self.cfg, {**self.overrides, **extra})

    def sampled_values(self) -> dict[str, float]:
        return {p: float(self[p]) for p in self.cfg.ranged_paths()}

    def __iter__(self) -> Iterator[str]:
        return iter(self.cfg.ranged_paths())
