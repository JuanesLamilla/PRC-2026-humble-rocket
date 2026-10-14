"""Stand-occupancy chain: sandwich the unknown off-block between known in-blocks (B1).

Arrivals keep their in-block time (``BLOCK_TIME_UTC_mvt``) in *both* the training and the
ranking set, so for every departure — including the test rows whose own off-block is blank —
the stand's occupancy history is fully observed. The aircraft that was in-block at stand *S*
most recently before a departure from *S* is, at most airports, the previous occupant of the
stand (usually the same airframe). That gives bounds on the target with no label involved:

* our aircraft cannot have pushed back earlier than ``prev_inblock + MIN_TURN_S``, so
  ``taxi <= (MVT_TIME - prev_inblock) - MIN_TURN_S`` — an **upper bound**;
* if the last in-block before the takeoff is *too recent to be our own turnaround*
  (gap < ``MIN_TURN_S``) it must be the **next** occupant, which means our aircraft had
  already left the stand by then, so ``taxi >= MVT_TIME - prev_inblock`` — a **lower bound**.
  (The literal "next in-block after the takeoff" version of that bound is vacuous: any
  in-block later than the takeoff constrains nothing, ``max(0, MVT - next_inblock) == 0``.
  ``stand_next_inblock_gap_s`` is kept anyway as a stand-pressure / dwell signal.)
  ``stand_prev2_inblock_gap_s`` carries the in-block before that one, which is our own
  arrival in exactly the short-turn case, so the model can recover the true turnaround.

Never reads ``BLOCK_TIME_UTC_mvt`` / ``TAXITIME_SEC_mvt`` of the departure it describes: the
in-block stream is built from ARR rows only, and the previous-departure gap uses takeoff
times (known for every row, train and test).

"""

from __future__ import annotations

import polars as pl

from prc.data import schema as S
from prc.features.registry import feature

MIN_TURN_S = 1200.0  # 20 min: the shortest plausible turnaround at a European hub

OUT = [
    "stand_prev_inblock_gap_s",
    "stand_prev2_inblock_gap_s",
    "stand_next_inblock_gap_s",
    "taxi_upper_bound_s",
    "taxi_lower_bound_s",
    "stand_prev_dep_gap_s",
]


def _inblocks(df: pl.DataFrame) -> pl.DataFrame:
    """One row per (airport, stand, in-block time) from ARR rows, with the preceding one."""
    return (
        df.filter(
            (pl.col(S.PHASE) == "ARR")
            & pl.col(S.STAND).is_not_null()
            & pl.col(S.BLOCK_TIME).is_not_null()
        )
        .select(S.AIRPORT, S.STAND, pl.col(S.BLOCK_TIME).alias("_ib"))
        .unique(subset=[S.AIRPORT, S.STAND, "_ib"])
        .sort([S.AIRPORT, S.STAND, "_ib"])
        .with_columns(pl.col("_ib").shift(1).over([S.AIRPORT, S.STAND]).alias("_ib_prev"))
    )


@feature("stand_bounds")
def stand_bounds(df: pl.DataFrame) -> pl.DataFrame:
    """Previous / next arrival in-block at the same (airport, stand) around the takeoff, the
    resulting upper and lower bounds on taxi-out, and the gap to the previous departure from
    the same stand. Null when the stand is unknown or the stand has no in-block history."""
    ib = _inblocks(df)
    deps = (
        df.filter(pl.col(S.PHASE) == "DEP")
        .select(S.MVT_ID, S.AIRPORT, S.STAND, S.MVT_TIME)
        .sort([S.AIRPORT, S.STAND, S.MVT_TIME])
    )
    prev_dep = (
        pl.col(S.MVT_TIME) - pl.col(S.MVT_TIME).shift(1).over([S.AIRPORT, S.STAND])
    ).dt.total_seconds()
    deps = deps.with_columns(
        pl.when(pl.col(S.STAND).is_not_null())
        .then(prev_dep)
        .otherwise(None)
        .cast(pl.Float32)
        .alias("stand_prev_dep_gap_s")
    ).sort([S.AIRPORT, S.STAND, S.MVT_TIME])

    out = deps.join_asof(
        ib,
        left_on=S.MVT_TIME,
        right_on="_ib",
        by=[S.AIRPORT, S.STAND],
        strategy="backward",
        allow_exact_matches=False,
        coalesce=False,
    ).join_asof(
        ib.select(S.AIRPORT, S.STAND, pl.col("_ib").alias("_ib_next")),
        left_on=S.MVT_TIME,
        right_on="_ib_next",
        by=[S.AIRPORT, S.STAND],
        strategy="forward",
        allow_exact_matches=False,
        coalesce=False,
    )

    t = pl.col(S.MVT_TIME)
    prev_gap = (t - pl.col("_ib")).dt.total_seconds().cast(pl.Float32)
    prev2_gap = (t - pl.col("_ib_prev")).dt.total_seconds().cast(pl.Float32)
    next_gap = (pl.col("_ib_next") - t).dt.total_seconds().cast(pl.Float32)
    short_turn = prev_gap < MIN_TURN_S
    return out.select(
        S.MVT_ID,
        prev_gap.alias("stand_prev_inblock_gap_s"),
        prev2_gap.alias("stand_prev2_inblock_gap_s"),
        next_gap.alias("stand_next_inblock_gap_s"),
        # upper bound: the in-block that plausibly *is* our own arrival, minus the min turn
        pl.when(short_turn)
        .then(prev2_gap - MIN_TURN_S)
        .otherwise(prev_gap - MIN_TURN_S)
        .cast(pl.Float32)
        .alias("taxi_upper_bound_s"),
        # lower bound: non-zero only when the stand was taken over before our takeoff
        pl.when(prev_gap.is_null())
        .then(None)
        .when(short_turn)
        .then(prev_gap.clip(0.0, None))
        .otherwise(pl.lit(0.0))
        .cast(pl.Float32)
        .alias("taxi_lower_bound_s"),
        pl.col("stand_prev_dep_gap_s"),
    )


#: See ``prc.features.congestion.MAX_GAP_S`` — repeated here rather than imported so that the
#: block's module hash covers it (a cross-module constant would not invalidate this cache).
MAX_GAP_S = 86_400.0
#: Every column of the block is a gap or a bound derived from one.
GAP_COLS = OUT


@feature("stand_bounds_v2")
def stand_bounds_v2(df: pl.DataFrame) -> pl.DataFrame:
    """``stand_bounds`` with every gap / bound above 24 h nulled.

    Code review 2026-09-19, finding #5: ``stand_prev_dep_gap_s`` averages 34,075 s in
    2025-07 and 190,239 s in 2026-07, and 1,674 of the 2,269 Jul-2026 departures with a
    gap over 140 days are on 2026-07-01 — the first day of the data island. A gap that
    large says "this is the start of the file", not "this stand was quiet".
    """
    out = stand_bounds(df)
    return out.with_columns(
        [pl.when(pl.col(c) > MAX_GAP_S).then(None).otherwise(pl.col(c)).alias(c) for c in GAP_COLS]
    )
