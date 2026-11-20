"""Mini backtest (D7, SHOULD, optional): a rough, anecdotal real-world signal for the prior.

Input: data/backtest_pairs.csv, filled by hand. Each row is two public videos from the same
creator with publicly visible view counts; paths point to local copies the team saved.
For each pair both videos go through the identical prior pipeline; the pair is "correct" if the
higher-rated video is the one with more views (a tie in rating counts as 0.5).
Reported as pairwise accuracy WITH its sample size, labelled anecdotal. Never a validation claim.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

from app.config import DATA_DIR
from app.storage import cache_root, read_json, write_json

PAIRS_CSV = DATA_DIR / "backtest_pairs.csv"
RESULT = "backtest.json"


def load_pairs(path: Path = PAIRS_CSV) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if (r.get("video_a_path") or "").strip()]


def run_backtest(path: Path = PAIRS_CSV, prior_method: Optional[str] = None) -> dict:
    from app.ingestion.upload import ingest_file
    from app.pipeline import base_profile
    from app.prior import prior

    rows, score, skipped = [], 0.0, []
    for r in load_pairs(path):
        try:
            va, vb = int(r["views_a"]), int(r["views_b"])
            ids = [ingest_file(Path(r[k].strip()))["video_id"] for k in ("video_a_path", "video_b_path")]
            pa, pb = (prior(base_profile(i), prior_method) for i in ids)
        except Exception as e:  # noqa: BLE001 - a bad row is skipped and reported
            skipped.append({"pair_id": r.get("pair_id"), "error": f"{type(e).__name__}: {str(e)[:120]}"})
            continue
        if va == vb:
            skipped.append({"pair_id": r.get("pair_id"), "error": "equal view counts"})
            continue
        if pa.percentile == pb.percentile:
            s = 0.5
        else:
            s = float((pa.percentile > pb.percentile) == (va > vb))
        score += s
        rows.append({"pair_id": r.get("pair_id"), "creator": r.get("creator"), "p_a": pa.percentile, "p_b": pb.percentile,
                     "views_a": va, "views_b": vb, "methods": [pa.method, pb.method], "score": s})
    n = len(rows)
    methods = {m for row in rows for m in row["methods"]}
    out = {"n": n, "accuracy": score / n if n else None, "pairs": rows, "skipped": skipped,
           "prior_methods": sorted(methods),
           "label": f"Anecdotal: pairwise accuracy on n={n} hand-collected pairs; not a validation of view prediction.",
           "evidence_tag": "MEASURED"}
    write_json(cache_root() / RESULT, out)
    return out


def latest_backtest() -> Optional[dict]:
    res = read_json(cache_root() / RESULT)
    return res if res and res.get("n") else None
