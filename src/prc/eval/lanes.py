"""Lane breakdown of a run's predictions (validation-protocol ADR 2026-09-17).

Lanes: with / without an NM off-block (``AOBT_3_flt``), the derived-AOBT spike
(``MVT − AOBT_3`` in [1194, 1206] s), target > 1 h, and "ordinary" (has AOBT, not on the
spike, ≤ 1 h). Rows, RMSE and the share of the run's squared error, pooled and per month.

Two entry points onto the same arithmetic:

* :func:`lane_table` / :func:`lane_metrics` take the rows in memory, so ``prc.eval.run``
  can store the breakdown in ``metrics.json`` at the end of a run without re-reading any
  parquet file;
* :func:`lanes` reads ``outputs/runs/<run>/oof.parquet`` and the movement table, which is
  what ``scripts/lane_report.py`` has always done.

The script owns everything above this (the reweighting and composition tables).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from prc import paths
from prc.data import schema as S

SPIKE = (1194, 1206)
# TODO(2026-09-18): SPIKE is the hardcoded LTFM "takeoff − 20 min" window. The arrival-regime
# doc shows the derived-AOBT constant is per
# airport and multi-modal (LTFM ~1200 and ~1500 s, LIRF ~595 and ~1500 s, EDDM ~300 s,
# EGLL ~600 s), so this window undercounts the derived lane and mixes derived rows into
# `with_aobt`. Both the lane table and the reweighting should move to the per-airport
# constants once they are a shared helper.

LANES = ("all", "with_aobt", "no_aobt", "spike", "gt1h", "ordinary")


def tag(d: pl.DataFrame) -> pl.DataFrame:
    """Add ``d_aobt`` / ``has_aobt`` / ``on_spike`` from ``MVT_TIME_UTC_mvt`` and ``AOBT_3_flt``."""
    return d.with_columns(
        (pl.col(S.MVT_TIME) - pl.col(S.AOBT_3)).dt.total_seconds().alias("d_aobt")
    ).with_columns(
        pl.col(S.AOBT_3).is_not_null().alias("has_aobt"),
        pl.col("d_aobt").is_between(*SPIKE).fill_null(False).alias("on_spike"),
    )


def lane_masks(y: np.ndarray, has_aobt: np.ndarray, on_spike: np.ndarray) -> dict[str, np.ndarray]:
    big = y > 3600
    return {
        "all": np.ones_like(has_aobt),
        "with_aobt": has_aobt,
        "no_aobt": ~has_aobt,
        "spike": on_spike,
        "gt1h": big,
        "ordinary": has_aobt & ~on_spike & ~big,
    }


def lane_rows(d: pl.DataFrame, run: str = "", pred: str = "pred") -> list[dict]:
    """One row per (month scope, lane). ``d`` needs ``month``, the target, ``pred`` and either
    ``has_aobt``/``on_spike`` or the columns :func:`tag` derives them from."""
    if "has_aobt" not in d.columns or "on_spike" not in d.columns:
        d = tag(d)
    y = d[S.TARGET].cast(pl.Float64).to_numpy()
    p = d[pred].cast(pl.Float64).to_numpy()
    month = d[S.SPLIT_MONTH].to_numpy()
    has = d["has_aobt"].to_numpy()
    spike = d["on_spike"].to_numpy()
    sse_total = float(np.sum((y - p) ** 2))
    masks = lane_masks(y, has, spike)
    rows = []
    for m_name in ["all", *sorted(set(month))]:
        mm = np.ones_like(has) if m_name == "all" else month == m_name
        for lane, mask in masks.items():
            k = mask & mm
            if k.sum() == 0:
                continue
            rows.append(
                {
                    "run": run,
                    "month": m_name,
                    "lane": lane,
                    "rows": int(k.sum()),
                    "rmse": round(float(np.sqrt(np.mean((y[k] - p[k]) ** 2))), 1),
                    "sse_share": round(float(np.sum((y[k] - p[k]) ** 2)) / sse_total, 3)
                    if sse_total > 0
                    else 0.0,
                }
            )
    return rows


def lane_table(d: pl.DataFrame, run: str = "", pred: str = "pred") -> pl.DataFrame:
    """:func:`lane_rows` as a frame — the table ``scripts/lane_report.py`` prints."""
    return pl.DataFrame(lane_rows(d, run, pred), schema=_TABLE_SCHEMA)


_TABLE_SCHEMA = {
    "run": pl.String,
    "month": pl.String,
    "lane": pl.String,
    "rows": pl.Int64,
    "rmse": pl.Float64,
    "sse_share": pl.Float64,
}


def lane_metrics(d: pl.DataFrame, pred: str = "pred") -> dict[str, dict[str, dict]]:
    """The same numbers nested for ``metrics.json``: ``{month scope: {lane: {rows, rmse,
    sse_share}}}``, with ``"all"`` as the pooled scope."""
    out: dict[str, dict[str, dict]] = {}
    for r in lane_rows(d, pred=pred):
        out.setdefault(str(r["month"]), {})[str(r["lane"])] = {
            "rows": r["rows"],
            "rmse": r["rmse"],
            "sse_share": r["sse_share"],
        }
    return out


def lane_rmse(metrics_lanes: dict | None, lane: str, scope: str = "all") -> float | None:
    """One number out of :func:`lane_metrics` (or ``None`` if that lane has no rows)."""
    try:
        return float((metrics_lanes or {})[scope][lane]["rmse"])
    except (KeyError, TypeError, ValueError):
        return None


def oof_with_lanes(run: str, runs_dir: Path | None = None) -> pl.DataFrame:
    """A run's OOF rows joined to the two movement columns the lanes are defined by."""
    oof = pl.read_parquet((runs_dir or paths.RUNS) / run / "oof.parquet").select(
        S.MVT_ID, S.SPLIT_MONTH, S.TARGET, "pred"
    )
    mv = pl.read_parquet(
        paths.INTERIM / "movements.parquet", columns=[S.MVT_ID, S.MVT_TIME, S.AOBT_3]
    )
    return tag(oof.join(mv, on=S.MVT_ID, how="left"))


def lanes(run: str) -> pl.DataFrame:
    """Lane table of a finished run, read from ``outputs/runs/<run>/oof.parquet``."""
    return lane_table(oof_with_lanes(run), run=run)
