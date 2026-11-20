"""Columnar event log (A6). The whole cascade can be rebuilt from it."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from app.behaviour.chain import ACTIONS, REASONS
from app.graph.sbm import ROLE_NAMES

EVENT_TYPES = ("exposure", "skip", "watch", "partial_view", "completion", "like", "comment", "save", "share")
ET = {name: i for i, name in enumerate(EVENT_TYPES)}
CHANNELS = ("seed", "rec", "share")
CH = {name: i for i, name in enumerate(CHANNELS)}


@dataclass
class EventLog:
    run_id: str
    cols: dict[str, np.ndarray]   # round, viewer_id, event_type, channel, source_viewer,
                                  # community, role, reason_code, probability

    def __len__(self) -> int:
        return int(self.cols["viewer_id"].shape[0])

    def mask(self, event_type: str) -> np.ndarray:
        return self.cols["event_type"] == ET[event_type]

    def to_frame(self, decode: bool = True) -> pd.DataFrame:
        df = pd.DataFrame(self.cols)
        df.insert(0, "run_id", self.run_id)
        if decode:
            df["event_type"] = np.array(EVENT_TYPES)[df["event_type"]]
            df["channel"] = np.array(CHANNELS)[df["channel"]]
            df["role"] = np.array(ROLE_NAMES)[df["role"]]
            df["reason_code"] = np.array(REASONS)[df["reason_code"]]
            df["source_viewer"] = df["source_viewer"].astype("Int64").where(df["source_viewer"] >= 0)
        return df

    def to_parquet(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame(decode=False).to_parquet(path, index=False)
        return path


def build_event_log(run_id: str, state: dict[str, np.ndarray], community: np.ndarray, role: np.ndarray) -> EventLog:
    """Expand per-viewer engine state into one row per event, ordered by (round, viewer, type)."""
    ex = np.flatnonzero(state["exposed"])
    parts: list[tuple[np.ndarray, int, np.ndarray]] = []  # (viewer ids, event type, probability)
    nan = np.full(ex.size, np.nan)
    stay = state["stay"][ex]
    parts.append((ex, ET["exposure"], nan))
    parts.append((ex[~stay], ET["skip"], state["p_stay"][ex][~stay]))
    parts.append((ex[stay], ET["watch"], state["p_stay"][ex][stay]))
    comp = state["complete"][ex]
    parts.append((ex[stay & ~comp], ET["partial_view"], nan[stay & ~comp]))
    parts.append((ex[comp], ET["completion"], nan[comp]))
    for a in ACTIONS:
        did = state[f"act_{a}"][ex]
        parts.append((ex[did], ET[a], state[f"p_{a}"][ex][did]))

    vid = np.concatenate([p[0] for p in parts]).astype(np.int64)
    etype = np.concatenate([np.full(p[0].size, p[1], dtype=np.int8) for p in parts])
    prob = np.concatenate([p[2] for p in parts]).astype(float)
    rnd = state["round"][vid]
    order = np.lexsort((etype, vid, rnd))
    vid, etype, prob, rnd = vid[order], etype[order], prob[order], rnd[order]
    cols = {
        "round": rnd.astype(np.int16),
        "viewer_id": vid,
        "event_type": etype,
        "channel": state["channel"][vid].astype(np.int8),
        "source_viewer": state["source"][vid].astype(np.int64),
        "community": community[vid].astype(np.int16),
        "role": role[vid].astype(np.int8),
        "reason_code": state["reason"][vid].astype(np.int8),
        "probability": prob,
    }
    return EventLog(run_id=run_id, cols=cols)


def reconstruct_cascade(log: EventLog) -> dict[str, np.ndarray]:
    """Rebuild per-exposure (viewer, round, channel, source) from the log alone."""
    m = log.mask("exposure")
    return {k: log.cols[k][m] for k in ("viewer_id", "round", "channel", "source_viewer")}
