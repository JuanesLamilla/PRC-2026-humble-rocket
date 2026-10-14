"""Timestamp inconsistency and combination rarity (backlog X6, Phase 1 item e ii / iii).

The > 1 h lane and the stamp-early rows are decided by *how the airport recorded* a departure,
not by how it taxied. A row
whose flight-plan stamps disagree with each other, or whose operator / stand / hour combination
is rare, is a candidate for a recording path the mixture's class head does not know. This block
scores that without a label:

Gap vector (seconds; every stamp test-visible):
``C = MVT − SCHED``, ``P = MVT − AOBT_3``, ``AOBT_3 − EOBT_1``, ``EOBT_1 − SCHED``,
``IOBT − SCHED``, ``LOBT − IOBT``, ``EOBT_1 − IOBT``, ``ARVT_3 − AOBT_3``, ``ARVT_3 − ARVT_1``,
``SCHED − in-block of the previous arrival at the stand`` (``turnaround_link``), and that
arrival's taxi-in.

* ``sa_z_rms`` / ``sa_z_max``: robust z-scores of each gap, (x − median) / (1.4826 · MAD) with
  the median / MAD per airport over the **2025 (training-month) departures**, clipped at ±20;
  root-mean-square and maximum |z| over the gaps present.
* ``sa_n_missing``: gaps missing (a no-NM row misses the NM ones).
* ``sa_iso``: an isolation forest (100 trees, 256 samples, seed 0) fitted on a fixed sample of
  ≤ 200k 2025 departures, input = the clipped z vector (missing → 0) plus missing indicators;
  the negated ``score_samples`` (higher = more isolated).
* ``sa_rare_op_stand`` / ``sa_rare_stand_hour`` / ``sa_rare_op_stand_hour``:
  −log((n_combo + 1) / (n_airport + 1)) with counts over **all** departures (train and test:
  every row counts itself, so train and test rows are treated alike); hour = local hour of the
  schedule.

No label is read: the z statistics and the forest are fitted on 2025 rows only because they are
the training months, not because a label is involved. Departure rows only.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from prc.data import schema as S
from prc.features.registry import build, feature
from prc.features.screen_common import deps_digest, local_hour, secs

DEPS = ("turnaround_link",)
GAPS = [
    "g_c",
    "g_p",
    "g_a_eobt",
    "g_eobt_sched",
    "g_iobt_sched",
    "g_lobt_iobt",
    "g_eobt_iobt",
    "g_arvt3_aobt",
    "g_arvt3_arvt1",
    "g_turn_slack",
    "g_turn_taxi_in",
]
OUT = [
    "sa_z_rms",
    "sa_z_max",
    "sa_n_missing",
    "sa_iso",
    "sa_rare_op_stand",
    "sa_rare_stand_hour",
    "sa_rare_op_stand_hour",
]
Z_CLIP = 20.0
MAD_FLOOR_S = 30.0
ISO_SAMPLE = 200_000
ISO_SEED = 0


def _sa_external() -> str:
    return deps_digest(__name__, DEPS)


def gap_frame(df: pl.DataFrame, deps: pl.DataFrame) -> pl.DataFrame:
    """Departures with ``airport``, ``is_test``, the gap vector and the rarity keys."""
    c = pl.col
    return (
        df.filter(c(S.PHASE) == "DEP")
        .select(
            S.MVT_ID,
            S.AIRPORT,
            S.IS_TEST,
            secs(S.MVT_TIME, S.SCHED_TIME).alias("g_c"),
            secs(S.MVT_TIME, S.AOBT_3).alias("g_p"),
            secs(S.AOBT_3, S.EOBT_1).alias("g_a_eobt"),
            secs(S.EOBT_1, S.SCHED_TIME).alias("g_eobt_sched"),
            secs(S.IOBT, S.SCHED_TIME).alias("g_iobt_sched"),
            secs(S.LOBT, S.IOBT).alias("g_lobt_iobt"),
            secs(S.EOBT_1, S.IOBT).alias("g_eobt_iobt"),
            secs(S.ARVT_3, S.AOBT_3).alias("g_arvt3_aobt"),
            secs(S.ARVT_3, S.ARVT_1).alias("g_arvt3_arvt1"),
            c(S.OPERATOR).cast(pl.Utf8).fill_null("NA").alias("_op"),
            c(S.STAND).cast(pl.Utf8).fill_null("NA").alias("_stand"),
            local_hour(S.SCHED_TIME).alias("_hour"),
        )
        .join(
            deps.select(S.MVT_ID, "turn_arr_inblock_gap_s", "turn_arr_taxi_in_s"),
            on=S.MVT_ID,
            how="left",
        )
        .with_columns(
            (c("turn_arr_inblock_gap_s").cast(pl.Float64) - c("g_c")).alias("g_turn_slack"),
            c("turn_arr_taxi_in_s").cast(pl.Float64).alias("g_turn_taxi_in"),
        )
        .drop("turn_arr_inblock_gap_s", "turn_arr_taxi_in_s")
    )


def robust_z(g: pl.DataFrame) -> pl.DataFrame:
    """Clipped robust z of every gap, with median / MAD per airport over non-test rows."""
    stats = (
        g.filter(~pl.col(S.IS_TEST))
        .group_by(S.AIRPORT)
        .agg(
            *[pl.col(x).median().alias(f"_med_{x}") for x in GAPS],
            *[(pl.col(x) - pl.col(x).median()).abs().median().alias(f"_mad_{x}") for x in GAPS],
        )
    )
    g = g.join(stats, on=S.AIRPORT, how="left")
    zs = [
        (
            (pl.col(x) - pl.col(f"_med_{x}"))
            / pl.max_horizontal(1.4826 * pl.col(f"_mad_{x}"), pl.lit(MAD_FLOOR_S))
        )
        .clip(-Z_CLIP, Z_CLIP)
        .alias(f"z_{x}")
        for x in GAPS
    ]
    return g.with_columns(zs).drop([c for c in g.columns if c.startswith(("_med_", "_mad_"))])


def isolation_scores(z: np.ndarray, missing: np.ndarray, fit_rows: np.ndarray) -> np.ndarray:
    from sklearn.ensemble import IsolationForest

    X = np.hstack([np.nan_to_num(z, nan=0.0), missing.astype(np.float32)]).astype(np.float32)
    idx = np.flatnonzero(fit_rows)
    if len(idx) == 0:
        return np.full(len(X), np.nan)
    rng = np.random.default_rng(ISO_SEED)
    if len(idx) > ISO_SAMPLE:
        idx = np.sort(rng.choice(idx, ISO_SAMPLE, replace=False))
    forest = IsolationForest(
        n_estimators=100, max_samples=min(256, len(idx)), random_state=ISO_SEED, n_jobs=1
    )
    forest.fit(X[idx])
    out = np.empty(len(X))
    step = 250_000
    for s in range(0, len(X), step):
        out[s : s + step] = -forest.score_samples(X[s : s + step])
    return out


def anomaly_from(df: pl.DataFrame, deps: pl.DataFrame) -> pl.DataFrame:
    g = robust_z(gap_frame(df, deps))
    zcols = [f"z_{x}" for x in GAPS]
    z = g.select(zcols).to_numpy().astype(np.float64)
    missing = np.isnan(z)
    absz = np.abs(z)
    n_present = (~missing).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        z_rms = np.sqrt(np.nansum(z**2, axis=1) / np.maximum(n_present, 1))
        z_max = np.where(n_present > 0, np.nanmax(np.where(missing, -1.0, absz), axis=1), np.nan)
    z_rms = np.where(n_present > 0, z_rms, np.nan)
    iso = isolation_scores(z, missing, ~g[S.IS_TEST].to_numpy())
    n_ap = pl.len().over(S.AIRPORT) + 1
    rare = g.with_columns(
        (-(((pl.len().over(S.AIRPORT, "_op", "_stand") + 1) / n_ap).log())).alias(
            "sa_rare_op_stand"
        ),
        (-(((pl.len().over(S.AIRPORT, "_stand", "_hour") + 1) / n_ap).log())).alias(
            "sa_rare_stand_hour"
        ),
        (-(((pl.len().over(S.AIRPORT, "_op", "_stand", "_hour") + 1) / n_ap).log())).alias(
            "sa_rare_op_stand_hour"
        ),
    )
    return rare.select(
        S.MVT_ID,
        pl.Series("sa_z_rms", z_rms).cast(pl.Float32),
        pl.Series("sa_z_max", z_max).cast(pl.Float32),
        pl.Series("sa_n_missing", missing.sum(1)).cast(pl.Float32),
        pl.Series("sa_iso", iso).cast(pl.Float32),
        pl.col("sa_rare_op_stand").cast(pl.Float32),
        pl.col("sa_rare_stand_hour").cast(pl.Float32),
        pl.col("sa_rare_op_stand_hour").cast(pl.Float32),
    )


@feature("stamp_anomaly", external=_sa_external)
def stamp_anomaly(df: pl.DataFrame) -> pl.DataFrame:
    """Departure rows only. Robust-z RMS / max and missing count over the flight-plan / NM /
    turnaround timestamp gaps, an isolation-forest score over them (fitted on 2025 rows), and
    rarity of (operator, stand), (stand, hour), (operator, stand, hour)."""
    return anomaly_from(df, build(df, list(DEPS)))
