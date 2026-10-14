"""Airport-state (queue / throughput) feature blocks.

The challenge is post-operational: for every test departure we know its takeoff
time, runway, and all other movements' times. So the strongest signals are
counts of other movements in windows around the takeoff time. These are computed
on the FULL table (train + test) per airport with sorted rolling windows.

Departure-queue proxies at time t (takeoff of the row):
  n_dep_tko_{w}  : takeoffs at the airport within [t-w, t)
  n_arr_ldg_{w}  : landings at the airport within [t-w, t)
  n_rwy_tko_{w}  : takeoffs on the SAME runway within [t-w, t)
  gap_prev_tko   : seconds since previous takeoff on the same runway
  gap_next_tko   : seconds until next takeoff on the same runway
  n_sched_dep_{w}: scheduled departures within [t-w, t+w)  (demand proxy, no leakage)
"""

from __future__ import annotations

import polars as pl

from prc.data import schema as S
from prc.features.registry import feature

WINDOWS = ("10m", "30m", "60m")


def _rolling_count(df: pl.DataFrame, by: list[str], window: str, name: str) -> pl.DataFrame:
    """Count rows in (t - window, t] per group, using rolling over MVT_TIME."""
    return (
        df.sort(by + [S.MVT_TIME])
        .rolling(index_column=S.MVT_TIME, period=window, group_by=by, closed="left")
        .agg(pl.len().alias(name))
        .select(by + [S.MVT_TIME, name])
    )


@feature("throughput")
def throughput(df: pl.DataFrame) -> pl.DataFrame:
    """Rolling counts of takeoffs/landings at the airport and on the runway before t."""
    base = df.select(S.MVT_ID, S.AIRPORT, S.RUNWAY, S.PHASE, S.MVT_TIME).sort(
        [S.AIRPORT, S.MVT_TIME]
    )
    out = base.select(S.MVT_ID, S.AIRPORT, S.RUNWAY, S.MVT_TIME)
    deps = base.filter(pl.col(S.PHASE) == "DEP")
    arrs = base.filter(pl.col(S.PHASE) == "ARR")
    for w in WINDOWS:
        for src, name, by in (
            (deps, f"n_dep_tko_{w}", [S.AIRPORT]),
            (arrs, f"n_arr_ldg_{w}", [S.AIRPORT]),
            (deps, f"n_rwy_tko_{w}", [S.AIRPORT, S.RUNWAY]),
        ):
            # counts at every event time of the source stream, then asof-join to all rows
            counts = _rolling_count(src, by, w, name).unique(subset=by + [S.MVT_TIME], keep="last")
            out = out.sort(by + [S.MVT_TIME]).join_asof(
                counts.sort(by + [S.MVT_TIME]),
                on=S.MVT_TIME,
                by=by,
                strategy="backward",
            )
    return out.drop(S.AIRPORT, S.RUNWAY, S.MVT_TIME).with_columns(
        pl.exclude(S.MVT_ID).fill_null(0).cast(pl.Int32)
    )


@feature("runway_gaps")
def runway_gaps(df: pl.DataFrame) -> pl.DataFrame:
    """Seconds to previous / next takeoff on the same runway (departure rows only)."""
    deps = (
        df.filter(pl.col(S.PHASE) == "DEP")
        .select(S.MVT_ID, S.AIRPORT, S.RUNWAY, S.MVT_TIME)
        .sort([S.AIRPORT, S.RUNWAY, S.MVT_TIME])
    )
    grp = [S.AIRPORT, S.RUNWAY]
    t = pl.col(S.MVT_TIME)
    return deps.with_columns(
        (t - t.shift(1).over(grp)).dt.total_seconds().cast(pl.Float32).alias("gap_prev_tko"),
        (t.shift(-1).over(grp) - t).dt.total_seconds().cast(pl.Float32).alias("gap_next_tko"),
    ).select(S.MVT_ID, "gap_prev_tko", "gap_next_tko")


@feature("sched_demand")
def sched_demand(df: pl.DataFrame) -> pl.DataFrame:
    """Scheduled departures per airport in a +/- window around the row's takeoff time."""
    sched = (
        df.filter(pl.col(S.PHASE) == "DEP")
        .select(S.AIRPORT, pl.col(S.SCHED_TIME).alias("_t"))
        .drop_nulls()
        .sort([S.AIRPORT, "_t"])
    )
    rows = df.select(S.MVT_ID, S.AIRPORT, S.MVT_TIME).sort([S.AIRPORT, S.MVT_TIME])
    out = rows
    for w in ("30m", "60m"):
        # count scheduled departures in [t-w, t) via asof on a rolling count
        c = (
            sched.rolling(index_column="_t", period=w, group_by=S.AIRPORT, closed="left")
            .agg(pl.len().alias(f"n_sched_dep_{w}"))
            .unique(subset=[S.AIRPORT, "_t"], keep="last")
            .sort([S.AIRPORT, "_t"])
        )
        out = out.join_asof(
            c, left_on=S.MVT_TIME, right_on="_t", by=S.AIRPORT, strategy="backward"
        ).drop("_t")
    return out.drop(S.AIRPORT, S.MVT_TIME).with_columns(
        pl.exclude(S.MVT_ID).fill_null(0).cast(pl.Int32)
    )


#: Gaps longer than this are nulled by the ``_v2`` blocks (code review 2026-09-19, #5).
#: The movement table is three disconnected islands — 2025-01→2025-12, 2026-01, 2026-07 —
#: so the first row of a group in July 2026 carries a ~151-day "gap" that never occurs in
#: training. Nulling every gap above a day removes the island edges and, in both years
#: alike, the gaps that are meaningless anyway (a runway unused overnight).
MAX_GAP_S = 86_400.0


@feature("runway_gaps_v2")
def runway_gaps_v2(df: pl.DataFrame) -> pl.DataFrame:
    """``runway_gaps`` with gaps above 24 h nulled (data-island edges; see ``MAX_GAP_S``)."""
    out = runway_gaps(df)
    return out.with_columns(
        [
            pl.when(pl.col(c) > MAX_GAP_S).then(None).otherwise(pl.col(c)).alias(c)
            for c in ("gap_prev_tko", "gap_next_tko")
        ]
    )
