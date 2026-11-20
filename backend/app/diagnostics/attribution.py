"""Reach attribution (B5) and replay aggregates, computed from the event log."""
from __future__ import annotations

from collections import Counter

import numpy as np

from app.eventlog.log import CH, CHANNELS, ET, EventLog
from app.graph.sbm import ROLE_NAMES


def attribution(log: EventLog, community_names: list[str]) -> dict:
    """Exposures by channel -> source community -> source role (seed/rec have the platform as source)."""
    c = log.cols
    exp = c["event_type"] == ET["exposure"]
    vid, ch, comm, src = c["viewer_id"][exp], c["channel"][exp], c["community"][exp], c["source_viewer"][exp]
    info = {int(v): (int(cm), int(r)) for v, cm, r in zip(vid, comm, c["role"][exp])}
    rows: Counter = Counter()
    for v, chan, cm, s in zip(vid.tolist(), ch.tolist(), comm.tolist(), src.tolist()):
        if chan == CH["share"]:
            sc, sr = info[s]
            key = ("share", community_names[sc], ROLE_NAMES[sr], community_names[cm])
        else:
            key = (CHANNELS[chan], "platform", "platform", community_names[cm])
        rows[key] += 1
    total = max(int(exp.sum()), 1)
    table = [{"channel": k[0], "source_community": k[1], "source_role": k[2], "target_community": k[3],
              "exposures": n, "share_of_reach": n / total, "evidence_tag": "SIMULATED"}
             for k, n in sorted(rows.items(), key=lambda kv: -kv[1])]
    by_channel = {name: float(np.mean(ch == i)) if total else 0.0 for i, name in enumerate(CHANNELS)}
    share = ch == CH["share"]
    cross = float(np.mean([info[s][0] != cm for s, cm in zip(src[share].tolist(), comm[share].tolist())])) if share.any() else 0.0
    return {"table": table, "by_channel": by_channel, "share_cross_community": cross, "evidence_tag": "SIMULATED"}


def replay_aggregates(log: EventLog, n_communities: int) -> list[dict]:
    """Per round: exposures by channel and by community, stays and shares."""
    c = log.cols
    out = []
    for t in range(int(c["round"].max()) + 1 if len(log) else 0):
        at = c["round"] == t
        exp = at & (c["event_type"] == ET["exposure"])
        out.append({
            "round": t,
            "exposures": int(exp.sum()),
            "by_channel": {name: int((exp & (c["channel"] == i)).sum()) for i, name in enumerate(CHANNELS)},
            "by_community": np.bincount(c["community"][exp], minlength=n_communities).tolist(),
            "stays": int((at & (c["event_type"] == ET["watch"])).sum()),
            "completions": int((at & (c["event_type"] == ET["completion"])).sum()),
            "likes": int((at & (c["event_type"] == ET["like"])).sum()),
            "shares": int((at & (c["event_type"] == ET["share"])).sum()),
            "evidence_tag": "SIMULATED",
        })
    return out
