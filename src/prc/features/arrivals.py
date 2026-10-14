"""Arrival-side congestion sensors (fully observed in the test months).

Arrivals keep landing time (MVT_TIME), in-block time (BLOCK_TIME) and therefore
taxi-in duration in ranking.parquet. Taxi-in of aircraft landing around a departure's
takeoff is an unbiased, unblanked measurement of surface congestion, de-icing days,
and low-visibility operations. Literature: Wu et al. 2026 (30-min mean taxi time is a
top-SHAP feature), Lee et al. 2016 (arrivals on the surface by concourse).
"""

from __future__ import annotations

import polars as pl

from prc.data import schema as S
from prc.features.registry import feature

WINDOWS = ("30m", "60m", "120m")


def _taxi_in(df: pl.DataFrame) -> pl.DataFrame:
    """Arrival rows with taxi-in seconds = in-block − landing (computed, never read)."""
    return (
        df.filter(pl.col(S.PHASE) == "ARR")
        .select(S.AIRPORT, S.MVT_TIME, S.BLOCK_TIME)
        .drop_nulls()
        .with_columns(
            (pl.col(S.BLOCK_TIME) - pl.col(S.MVT_TIME))
            .dt.total_seconds()
            .cast(pl.Float32)
            .alias("taxi_in")
        )
        .filter((pl.col("taxi_in") > 0) & (pl.col("taxi_in") < 7200))
        .sort([S.AIRPORT, S.MVT_TIME])
    )


@feature("arrival_taxi_in")
def arrival_taxi_in(df: pl.DataFrame) -> pl.DataFrame:
    """Rolling mean/median/p90 taxi-in of arrivals that landed in the window before t."""
    arr = _taxi_in(df)
    rows = df.select(S.MVT_ID, S.AIRPORT, S.MVT_TIME).sort([S.AIRPORT, S.MVT_TIME])
    out = rows
    for w in WINDOWS:
        stats = (
            arr.rolling(index_column=S.MVT_TIME, period=w, group_by=S.AIRPORT, closed="left")
            .agg(
                pl.col("taxi_in").mean().alias(f"arr_taxiin_mean_{w}"),
                pl.col("taxi_in").median().alias(f"arr_taxiin_med_{w}"),
                pl.col("taxi_in").quantile(0.9).alias(f"arr_taxiin_p90_{w}"),
                pl.len().alias(f"arr_taxiin_n_{w}"),
            )
            .unique(subset=[S.AIRPORT, S.MVT_TIME], keep="last")
            .sort([S.AIRPORT, S.MVT_TIME])
        )
        out = out.join_asof(stats, on=S.MVT_TIME, by=S.AIRPORT, strategy="backward")
    return out.drop(S.AIRPORT, S.MVT_TIME)


@feature("arrival_occupancy")
def arrival_occupancy(df: pl.DataFrame) -> pl.DataFrame:
    """Arrivals still taxiing in (landed, not yet in-block) at the row's takeoff time,
    plus landings in [t-15m, t+15m] (symmetric window: they contend for the surface)."""
    arr = _taxi_in(df)
    rows = df.select(S.MVT_ID, S.AIRPORT, S.MVT_TIME).sort([S.AIRPORT, S.MVT_TIME])
    # landed-before-t minus in-block-before-t == on the surface at t
    landed = (
        arr.select(S.AIRPORT, S.MVT_TIME)
        .sort([S.AIRPORT, S.MVT_TIME])
        .with_columns(pl.int_range(1, pl.len() + 1).over(S.AIRPORT).alias("n_landed"))
        .unique(subset=[S.AIRPORT, S.MVT_TIME], keep="last")
    )
    inblock = (
        arr.select(S.AIRPORT, pl.col(S.BLOCK_TIME).alias("_t"))
        .sort([S.AIRPORT, "_t"])
        .with_columns(pl.int_range(1, pl.len() + 1).over(S.AIRPORT).alias("n_inblock"))
        .unique(subset=[S.AIRPORT, "_t"], keep="last")
    )
    out = (
        rows.join_asof(landed, on=S.MVT_TIME, by=S.AIRPORT, strategy="backward")
        .join_asof(inblock, left_on=S.MVT_TIME, right_on="_t", by=S.AIRPORT, strategy="backward")
        .with_columns(
            (pl.col("n_landed").fill_null(0) - pl.col("n_inblock").fill_null(0))
            .clip(0, None)
            .cast(pl.Int32)
            .alias("arr_on_surface")
        )
        .drop("n_landed", "n_inblock", "_t")
    )
    # symmetric window: landings within ±15 min of t
    fwd = (
        arr.select(S.AIRPORT, S.MVT_TIME)
        .sort([S.AIRPORT, S.MVT_TIME])
        .rolling(
            index_column=S.MVT_TIME, period="30m", offset="-15m", group_by=S.AIRPORT, closed="both"
        )
        .agg(pl.len().alias("n_arr_ldg_pm15m"))
        .unique(subset=[S.AIRPORT, S.MVT_TIME], keep="last")
        .sort([S.AIRPORT, S.MVT_TIME])
    )
    out = out.join_asof(fwd, on=S.MVT_TIME, by=S.AIRPORT, strategy="nearest", tolerance="15m")
    return out.drop(S.AIRPORT, S.MVT_TIME).with_columns(
        pl.col("n_arr_ldg_pm15m").fill_null(0).cast(pl.Int32)
    )


@feature("arrival_occupancy_v2")
def arrival_occupancy_v2(df: pl.DataFrame) -> pl.DataFrame:
    """``arrival_occupancy`` with the two cumulative counters sorted before the as-of join.

    Code review 2026-09-19, finding #2: the ``landed`` / ``inblock`` frames are
    ``unique(..., keep="last")``-ed and were never re-sorted, and polars reads an unsorted
    right frame in ``join_asof`` without complaining. The shipped ``arr_on_surface`` was
    exactly 0 for all of Jan 2025 and a monotonically growing counter afterwards (max
    247,306 aircraft "on the surface"), i.e. a date detector; correct values are 0-27 with
    a mean near 5. Rebuilding the block twice does not even reproduce the same numbers — the
    2026-09-19 rebuild differs from the cached parquet on this column alone, which is what an
    unsorted as-of join looks like. ``exp0038``'s rejection of this block is therefore void.

    Every other line is identical to ``arrival_occupancy``; the two ``.sort()`` calls are
    the fix.
    """
    arr = _taxi_in(df)
    rows = df.select(S.MVT_ID, S.AIRPORT, S.MVT_TIME).sort([S.AIRPORT, S.MVT_TIME])
    landed = (
        arr.select(S.AIRPORT, S.MVT_TIME)
        .sort([S.AIRPORT, S.MVT_TIME])
        .with_columns(pl.int_range(1, pl.len() + 1).over(S.AIRPORT).alias("n_landed"))
        .unique(subset=[S.AIRPORT, S.MVT_TIME], keep="last")
        .sort([S.AIRPORT, S.MVT_TIME])
    )
    inblock = (
        arr.select(S.AIRPORT, pl.col(S.BLOCK_TIME).alias("_t"))
        .sort([S.AIRPORT, "_t"])
        .with_columns(pl.int_range(1, pl.len() + 1).over(S.AIRPORT).alias("n_inblock"))
        .unique(subset=[S.AIRPORT, "_t"], keep="last")
        .sort([S.AIRPORT, "_t"])
    )
    out = (
        rows.join_asof(landed, on=S.MVT_TIME, by=S.AIRPORT, strategy="backward")
        .join_asof(inblock, left_on=S.MVT_TIME, right_on="_t", by=S.AIRPORT, strategy="backward")
        .with_columns(
            (pl.col("n_landed").fill_null(0) - pl.col("n_inblock").fill_null(0))
            .clip(0, None)
            .cast(pl.Int32)
            .alias("arr_on_surface")
        )
        .drop("n_landed", "n_inblock", "_t")
    )
    fwd = (
        arr.select(S.AIRPORT, S.MVT_TIME)
        .sort([S.AIRPORT, S.MVT_TIME])
        .rolling(
            index_column=S.MVT_TIME, period="30m", offset="-15m", group_by=S.AIRPORT, closed="both"
        )
        .agg(pl.len().alias("n_arr_ldg_pm15m"))
        .unique(subset=[S.AIRPORT, S.MVT_TIME], keep="last")
        .sort([S.AIRPORT, S.MVT_TIME])
    )
    out = out.join_asof(fwd, on=S.MVT_TIME, by=S.AIRPORT, strategy="nearest", tolerance="15m")
    return out.drop(S.AIRPORT, S.MVT_TIME).with_columns(
        pl.col("n_arr_ldg_pm15m").fill_null(0).cast(pl.Int32)
    )
