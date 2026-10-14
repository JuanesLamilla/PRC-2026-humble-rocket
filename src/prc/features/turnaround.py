"""Turnaround link: give a no-flight-plan departure the columns of the arrival it came from.

The lane that decides our score is 1.6 % of departures with **every** ``_flt`` column null: no operator, no market segment, no flight type,
no aircraft type from the flight plan, RMSE ~2,400 s. Those columns are not missing from the
*airport*, though — the aircraft that is about to push back from stand *S* arrived at stand *S*,
and that arrival is in the movement table with its own NM record, in the training data and in
the ranking set alike (arrivals are complete on both sides).

So for every departure this block finds the most recent arrival at the same
``(airport, stand)`` strictly before the takeoff and exposes that arrival's identity
(operator, aircraft type, market segment, flight type), how long ago it went in-block, how
long it taxied in, and whether it had a flight plan at all. On a regime-C departure those are
the only operator-shaped columns the model will ever see.

Never reads the departure's own ``BLOCK_TIME_UTC_mvt`` / ``TAXITIME_SEC_mvt``: the linked row
is an ARR row, and an arrival's in-block time and taxi-in are published for the ranking months.

**Do not put this block in ``mask_for_no_nm``.** Every other surface block is masked on no-NM
departures because it gave the tree ways to fit catastrophic labels; this one exists *for*
those rows and is useless without them. Keep it in the unmasked set.
"""

from __future__ import annotations

import polars as pl

from prc.data import schema as S
from prc.features.registry import feature

OUT = [
    "turn_arr_operator",
    "turn_arr_actype",
    "turn_arr_segment",
    "turn_arr_flight_type",
    "turn_arr_inblock_gap_s",
    "turn_arr_taxi_in_s",
    "turn_arr_has_fpl",
    "turn_arr_inblock_eq_sched",
    "turn_arr_day_sched_share",
]

MAX_TAXI_IN_S = 7200.0  # beyond this the arrival pair is itself corrupt; drop the value


def _cat(expr: pl.Expr) -> pl.Expr:
    return expr.cast(pl.Utf8).fill_null("NA").cast(pl.Categorical)


@feature("turnaround_link")
def turnaround_link(df: pl.DataFrame) -> pl.DataFrame:
    """The most recent arrival at the same (airport, stand) before the takeoff: its operator,
    aircraft type, market segment and flight type, the gap since it went in-block, its taxi-in,
    and whether it carried an NM flight plan. Null / "NA" where the stand is unknown or the
    stand has no arrival history."""
    arr = (
        df.filter(
            (pl.col(S.PHASE) == "ARR")
            & pl.col(S.STAND).is_not_null()
            & pl.col(S.BLOCK_TIME).is_not_null()
        )
        .select(
            S.AIRPORT,
            S.STAND,
            pl.col(S.BLOCK_TIME).alias("_ib"),
            pl.col(S.OPERATOR).alias("_op"),
            pl.col(S.AIRCRAFT_TYPE_FLT).alias("_actype"),
            pl.col(S.MARKET_SEGMENT).alias("_segment"),
            pl.col(S.FLIGHT_TYPE).alias("_ftype"),
            pl.col(S.FLIGHT_ID).is_not_null().cast(pl.Int8).alias("_has_fpl"),
            (pl.col(S.BLOCK_TIME) - pl.col(S.MVT_TIME))
            .dt.total_seconds()
            .cast(pl.Float32)
            .alias("_taxi_in"),
            # the ARRIVAL's in-block stamp equals its schedule: the airport substituted the
            # schedule for this rotation. Arrivals keep their in-block time in the ranking set,
            # so this is observable for test departures and never reads the departure's label.
            ((pl.col(S.BLOCK_TIME) - pl.col(S.SCHED_TIME)).dt.total_seconds().abs() <= 10)
            .cast(pl.Int8)
            .alias("_ib_eq_sched"),
            pl.col(S.BLOCK_TIME).dt.date().alias("_ib_date"),
        )
        .with_columns(
            pl.col("_ib_eq_sched")
            .cast(pl.Float32)
            .mean()
            .over([S.AIRPORT, "_ib_date"])
            .alias("_day_sched_share")
        )
        .unique(subset=[S.AIRPORT, S.STAND, "_ib"], keep="first")
        .sort([S.AIRPORT, S.STAND, "_ib"])
    )
    deps = (
        df.filter(pl.col(S.PHASE) == "DEP")
        .select(S.MVT_ID, S.AIRPORT, S.STAND, S.MVT_TIME)
        .sort([S.AIRPORT, S.STAND, S.MVT_TIME])
    )
    out = deps.join_asof(
        arr,
        left_on=S.MVT_TIME,
        right_on="_ib",
        by=[S.AIRPORT, S.STAND],
        strategy="backward",
        allow_exact_matches=False,
        coalesce=False,
    )
    taxi_in = pl.col("_taxi_in")
    return out.select(
        S.MVT_ID,
        _cat(pl.col("_op")).alias("turn_arr_operator"),
        _cat(pl.col("_actype")).alias("turn_arr_actype"),
        _cat(pl.col("_segment")).alias("turn_arr_segment"),
        _cat(pl.col("_ftype")).alias("turn_arr_flight_type"),
        (pl.col(S.MVT_TIME) - pl.col("_ib"))
        .dt.total_seconds()
        .cast(pl.Float32)
        .alias("turn_arr_inblock_gap_s"),
        pl.when((taxi_in > 0) & (taxi_in < MAX_TAXI_IN_S))
        .then(taxi_in)
        .cast(pl.Float32)
        .alias("turn_arr_taxi_in_s"),
        pl.col("_has_fpl").cast(pl.Int8).alias("turn_arr_has_fpl"),
        pl.col("_ib_eq_sched").cast(pl.Int8).alias("turn_arr_inblock_eq_sched"),
        pl.col("_day_sched_share").cast(pl.Float32).alias("turn_arr_day_sched_share"),
    )
