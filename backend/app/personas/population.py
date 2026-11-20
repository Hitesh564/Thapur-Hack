"""Archetype loading and population sampling (A1). Versioned and seeded."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from app.config import DATA_DIR, Config
from schemas import TRAIT_KEYS, Archetype

ARCHETYPES_PATH = DATA_DIR / "archetypes.json"
FALLBACK_PATH = DATA_DIR / "archetypes_fallback.json"


def load_archetypes(path: Optional[Path] = None) -> tuple[list[Archetype], str]:
    """Load and validate archetypes; fall back to the hand-written file on any problem.

    Returns (archetypes, source_label).
    """
    candidates = [path] if path else [ARCHETYPES_PATH, FALLBACK_PATH]
    if path and path != FALLBACK_PATH:
        candidates.append(FALLBACK_PATH)
    last_err: Exception | None = None
    for p in candidates:
        try:
            raw = json.loads(Path(p).read_text(encoding="utf-8"))
            arcs = [Archetype.model_validate(a) for a in raw]
            for a in arcs:
                missing = set(TRAIT_KEYS) - set(a.traits)
                if missing:
                    raise ValueError(f"{a.name}: missing traits {missing}")
            return arcs, Path(p).name
        except Exception as e:  # noqa: BLE001 - any failure -> try fallback
            last_err = e
    raise RuntimeError(f"no valid archetype file: {last_err}")


def _file_hash(arcs: list[Archetype]) -> str:
    payload = json.dumps([a.model_dump() for a in arcs], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


@dataclass
class Population:
    version: str
    seed: int
    archetypes: list[Archetype]
    archetype_idx: np.ndarray          # (n,) int
    community: np.ndarray              # (n,) int
    traits: dict[str, np.ndarray]      # trait -> (n,) float
    influence: np.ndarray              # (n,) float, heavy-tailed
    n_communities: int
    community_names: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return int(self.community.shape[0])

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "viewer_id": np.arange(self.n),
            "archetype": [self.archetypes[i].name for i in self.archetype_idx],
            "community": self.community,
            "influence": self.influence,
        })
        for k, v in self.traits.items():
            df[k] = v
        return df

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_frame()
        df.attrs = {}
        df.to_parquet(path, index=False)
        meta = {"version": self.version, "seed": self.seed, "n_viewers": self.n,
                "n_communities": self.n_communities}
        path.with_suffix(".meta.json").write_text(json.dumps(meta), encoding="utf-8")
        return path


def sample_population(cfg: Config, seed: Optional[int] = None,
                      archetypes: Optional[list[Archetype]] = None) -> Population:
    seed = int(cfg.get("population.default_seed") if seed is None else seed)
    arcs = archetypes if archetypes is not None else load_archetypes()[0]
    n = int(cfg.get("population.n_viewers"))
    n_comm = int(cfg.get("population.n_communities"))
    comm_of_arc = np.array([a.community for a in arcs])
    if comm_of_arc.max() + 1 != n_comm:
        raise ValueError(f"archetypes span {comm_of_arc.max() + 1} communities, config says {n_comm}")
    names = [""] * n_comm
    for a in arcs:
        names[a.community] = a.community_name

    rng = np.random.default_rng(seed)
    arc_idx = rng.integers(0, len(arcs), n)
    noise_sd = float(cfg.get("population.trait_noise"))
    noise_clip = float(cfg.get("population.trait_noise_clip")) * noise_sd
    traits: dict[str, np.ndarray] = {}
    for key in TRAIT_KEYS:
        lo = np.array([a.traits[key][0] for a in arcs])[arc_idx]
        hi = np.array([a.traits[key][1] for a in arcs])[arc_idx]
        base = lo + (hi - lo) * rng.random(n)
        noise = np.clip(rng.normal(0.0, noise_sd, n), -noise_clip, noise_clip)
        val = np.maximum(base * (1.0 + noise), 0.0)
        if key == "attention":
            val = np.clip(val, 0.0, 1.0)
        traits[key] = val

    mult_map = cfg.get("population.influence_class_mult")
    mult = np.array([float(mult_map[a.influence_class]) for a in arcs])[arc_idx]
    alpha = float(cfg.get("graph.influence_pareto_alpha"))
    influence = (rng.pareto(alpha, n) + 1.0) * mult

    version = hashlib.sha256(
        f"{_file_hash(arcs)}|{cfg.section_hash('population', 'graph')}|{seed}".encode()
    ).hexdigest()[:12]
    return Population(version=version, seed=seed, archetypes=arcs, archetype_idx=arc_idx,
                      community=comm_of_arc[arc_idx], traits=traits, influence=influence,
                      n_communities=n_comm, community_names=names)
