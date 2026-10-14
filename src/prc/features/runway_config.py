"""Runway-configuration and discharge-process features.

The takeoff timestamps of every departure are known, so the runway queue discharge
process (Simaiakis & Balakrishnan) is observed directly: inter-departure gaps, wake
separation pairs, number of runways active, share of traffic on this runway, and
whether arrivals share it (mixed mode). Named a top causal factor in Idris 2002,
Diana 2018, Lee 2016.
"""

from __future__ import annotations

import polars as pl

from prc.data import schema as S
from prc.features.registry import feature


@feature("runway_config")
def runway_config(df: pl.DataFrame) -> pl.DataFrame:
    """Active runways, own-runway share, mixed-mode arrivals, in trailing 60 min."""
    base = df.select(S.MVT_ID, S.AIRPORT, S.RUNWAY, S.PHASE, S.MVT_TIME).sort(
        [S.AIRPORT, S.MVT_TIME]
    )
    rows = base.select(S.MVT_ID, S.AIRPORT, S.RUNWAY, S.MVT_TIME)
    deps = base.filter(pl.col(S.PHASE) == "DEP")
    arrs = base.filter(pl.col(S.PHASE) == "ARR")

    # distinct departure runways used in the last 60 min at the airport
    active = (
        deps.rolling(index_column=S.MVT_TIME, period="60m", group_by=S.AIRPORT, closed="left")
        .agg(
            pl.col(S.RUNWAY).n_unique().alias("n_dep_rwy_active_60m"), pl.len().alias("_n_dep_60m")
        )
        .unique(subset=[S.AIRPORT, S.MVT_TIME], keep="last")
        .sort([S.AIRPORT, S.MVT_TIME])
    )
    # takeoffs on own runway in the last 60 min (for the share)
    own = (
        deps.rolling(
            index_column=S.MVT_TIME, period="60m", group_by=[S.AIRPORT, S.RUNWAY], closed="left"
        )
        .agg(pl.len().alias("_n_own_60m"))
        .unique(subset=[S.AIRPORT, S.RUNWAY, S.MVT_TIME], keep="last")
        .sort([S.AIRPORT, S.RUNWAY, S.MVT_TIME])
    )
    # arrivals on the same runway in the last 30 min (mixed mode)
    mixed = (
        arrs.rolling(
            index_column=S.MVT_TIME, period="30m", group_by=[S.AIRPORT, S.RUNWAY], closed="left"
        )
        .agg(pl.len().alias("n_arr_same_rwy_30m"))
        .unique(subset=[S.AIRPORT, S.RUNWAY, S.MVT_TIME], keep="last")
        .sort([S.AIRPORT, S.RUNWAY, S.MVT_TIME])
    )
    out = (
        rows.join_asof(active, on=S.MVT_TIME, by=S.AIRPORT, strategy="backward")
        .sort([S.AIRPORT, S.RUNWAY, S.MVT_TIME])
        .join_asof(own, on=S.MVT_TIME, by=[S.AIRPORT, S.RUNWAY], strategy="backward")
        .join_asof(mixed, on=S.MVT_TIME, by=[S.AIRPORT, S.RUNWAY], strategy="backward")
        .with_columns(
            pl.col("n_dep_rwy_active_60m").fill_null(0).cast(pl.Int8),
            (pl.col("_n_own_60m").fill_null(0) / pl.col("_n_dep_60m").fill_null(0).clip(1, None))
            .cast(pl.Float32)
            .alias("own_rwy_share_60m"),
            pl.col("n_arr_same_rwy_30m").fill_null(0).cast(pl.Int16),
        )
    )
    return out.select(S.MVT_ID, "n_dep_rwy_active_60m", "own_rwy_share_60m", "n_arr_same_rwy_30m")


@feature("discharge")
def discharge(df: pl.DataFrame) -> pl.DataFrame:
    """Runway discharge cadence: rolling mean gap over previous 5/10 takeoffs on the same
    runway, wake category of the preceding takeoff, and queue position within the hour."""
    deps = (
        df.filter(pl.col(S.PHASE) == "DEP")
        .select(S.MVT_ID, S.AIRPORT, S.RUNWAY, S.MVT_TIME, S.WAKE_CAT)
        .sort([S.AIRPORT, S.RUNWAY, S.MVT_TIME])
    )
    grp = [S.AIRPORT, S.RUNWAY]
    t = pl.col(S.MVT_TIME)
    gap = (t - t.shift(1).over(grp)).dt.total_seconds().cast(pl.Float32)
    out = deps.with_columns(gap.alias("_gap")).with_columns(
        pl.col("_gap").rolling_mean(window_size=5, min_samples=1).over(grp).alias("rwy_gap_mean5"),
        pl.col("_gap")
        .rolling_mean(window_size=10, min_samples=1)
        .over(grp)
        .alias("rwy_gap_mean10"),
        pl.col(S.WAKE_CAT)
        .shift(1)
        .over(grp)
        .cast(pl.Utf8)
        .fill_null("NA")
        .cast(pl.Categorical)
        .alias("prev_wake"),
        pl.concat_str(
            [
                pl.col(S.WAKE_CAT).shift(1).over(grp).cast(pl.Utf8).fill_null("NA"),
                pl.lit(">"),
                pl.col(S.WAKE_CAT).cast(pl.Utf8).fill_null("NA"),
            ]
        )
        .cast(pl.Categorical)
        .alias("wake_pair"),
        t.dt.truncate("1h").alias("_hour"),
    )
    out = out.with_columns(
        pl.int_range(1, pl.len() + 1)
        .over([S.AIRPORT, "_hour"])
        .cast(pl.Int16)
        .alias("tko_rank_in_hour"),
        pl.int_range(1, pl.len() + 1)
        .over([S.AIRPORT, S.RUNWAY, "_hour"])
        .cast(pl.Int16)
        .alias("tko_rank_in_rwy_hour"),
    )
    return out.select(
        S.MVT_ID,
        "rwy_gap_mean5",
        "rwy_gap_mean10",
        "prev_wake",
        "wake_pair",
        "tko_rank_in_hour",
        "tko_rank_in_rwy_hour",
    )


#: See ``prc.features.congestion.MAX_GAP_S``; repeated so the module hash covers it.
MAX_GAP_S = 86_400.0


@feature("discharge_v2")
def discharge_v2(df: pl.DataFrame) -> pl.DataFrame:
    """``discharge`` with the two bugs the 2026-09-19 code review found fixed.

    #3 — ``tko_rank_in_hour`` ("queue position within the hour") was an ``int_range`` over a
    frame sorted by ``[airport, runway, MVT_TIME]``, so within an hour it counted *runways*,
    not time: it agreed with the time-ordered rank on 33 % of rows and correlated 0.52 with
    the minute in the hour instead of 0.81. Fixed by ranking on the takeoff time itself.
    (``tko_rank_in_rwy_hour`` was always correct — its sort matched its partition.)

    #5 — ``rwy_gap_mean5`` / ``rwy_gap_mean10`` average a ``shift(1)`` gap that, at the edge
    of a data island, spans 151 days. Gaps above 24 h are nulled before the rolling mean, so
    such a row averages the gaps it does have (``min_samples=1``) instead of one enormous one.

    ``exp0040``'s rejection of ``discharge`` is confounded by both and should be re-run.
    """
    deps = (
        df.filter(pl.col(S.PHASE) == "DEP")
        .select(S.MVT_ID, S.AIRPORT, S.RUNWAY, S.MVT_TIME, S.WAKE_CAT)
        .sort([S.AIRPORT, S.RUNWAY, S.MVT_TIME])
    )
    grp = [S.AIRPORT, S.RUNWAY]
    t = pl.col(S.MVT_TIME)
    gap = (t - t.shift(1).over(grp)).dt.total_seconds().cast(pl.Float32)
    capped = pl.when(gap > MAX_GAP_S).then(None).otherwise(gap)
    out = deps.with_columns(capped.alias("_gap")).with_columns(
        pl.col("_gap").rolling_mean(window_size=5, min_samples=1).over(grp).alias("rwy_gap_mean5"),
        pl.col("_gap")
        .rolling_mean(window_size=10, min_samples=1)
        .over(grp)
        .alias("rwy_gap_mean10"),
        pl.col(S.WAKE_CAT)
        .shift(1)
        .over(grp)
        .cast(pl.Utf8)
        .fill_null("NA")
        .cast(pl.Categorical)
        .alias("prev_wake"),
        pl.concat_str(
            [
                pl.col(S.WAKE_CAT).shift(1).over(grp).cast(pl.Utf8).fill_null("NA"),
                pl.lit(">"),
                pl.col(S.WAKE_CAT).cast(pl.Utf8).fill_null("NA"),
            ]
        )
        .cast(pl.Categorical)
        .alias("wake_pair"),
        t.dt.truncate("1h").alias("_hour"),
    )
    out = out.with_columns(
        # rank by takeoff time inside (airport, hour) and inside (airport, runway, hour)
        t.rank("ordinal").over([S.AIRPORT, "_hour"]).cast(pl.Int16).alias("tko_rank_in_hour"),
        t.rank("ordinal")
        .over([S.AIRPORT, S.RUNWAY, "_hour"])
        .cast(pl.Int16)
        .alias("tko_rank_in_rwy_hour"),
    )
    return out.select(
        S.MVT_ID,
        "rwy_gap_mean5",
        "rwy_gap_mean10",
        "prev_wake",
        "wake_pair",
        "tko_rank_in_hour",
        "tko_rank_in_rwy_hour",
    )
