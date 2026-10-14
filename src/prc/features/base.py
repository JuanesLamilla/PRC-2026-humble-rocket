"""Row-local feature blocks: calendar, categorical ids, flight-plan deltas."""

from __future__ import annotations

import polars as pl

from prc.data import schema as S
from prc.features.registry import feature


def _cat(col: str) -> pl.Expr:
    return pl.col(col).cast(pl.Utf8).fill_null("NA").cast(pl.Categorical)


@feature("calendar")
def calendar(df: pl.DataFrame) -> pl.DataFrame:
    """Time-of-day / day-of-week / month / minute-of-day from takeoff time."""
    t = pl.col(S.MVT_TIME)
    return df.select(
        S.MVT_ID,
        t.dt.hour().alias("hour"),
        (t.dt.hour() * 60 + t.dt.minute()).alias("minute_of_day"),
        t.dt.weekday().alias("dow"),
        t.dt.month().alias("month_num"),
        t.dt.ordinal_day().alias("doy"),
        (t.dt.weekday() >= 6).cast(pl.Int8).alias("is_weekend"),
    )


@feature("cats")
def cats(df: pl.DataFrame) -> pl.DataFrame:
    """Categorical identifiers as pl.Categorical (GBDTs consume them natively)."""
    return df.select(
        S.MVT_ID,
        _cat(S.AIRPORT).alias("airport"),
        _cat(S.RUNWAY).alias("runway"),
        _cat(S.STAND).alias("stand"),
        _cat(S.AIRCRAFT_TYPE).alias("actype"),
        _cat(S.WAKE_CAT).alias("wake"),
        _cat(S.OPERATOR).alias("operator"),
        _cat(S.MARKET_SEGMENT).alias("segment"),
        _cat(S.FLIGHT_TYPE).alias("flight_type"),
        _cat(S.ADES).alias("dest"),
        pl.concat_str([pl.col(S.AIRPORT), pl.lit("|"), pl.col(S.RUNWAY).cast(pl.Utf8)])
        .cast(pl.Categorical)
        .alias("airport_runway"),
        pl.concat_str([pl.col(S.AIRPORT), pl.lit("|"), pl.col(S.STAND).cast(pl.Utf8)])
        .cast(pl.Categorical)
        .alias("airport_stand"),
        # the tail lives in (LIRF x no NM match): make the interaction explicit
        pl.concat_str(
            [
                pl.col(S.AIRPORT),
                pl.lit("|nm="),
                pl.col(S.FLIGHT_ID).is_null().cast(pl.Int8).cast(pl.Utf8),
            ]
        )
        .cast(pl.Categorical)
        .alias("airport_nm"),
    )


@feature("fpl_deltas")
def fpl_deltas(df: pl.DataFrame) -> pl.DataFrame:
    """Deltas (seconds) between takeoff and the flight-plan off-block estimates.

    NOTE: these are the first thing to audit in EDA — if AOBT_3_flt is populated in the
    ranking set it is essentially the actual off-block time and (MVT_TIME - AOBT_3)
    is the target. Keep the block, but confirm what is actually available at test time.
    """
    t = pl.col(S.MVT_TIME)

    def secs(col: str) -> pl.Expr:
        return (t - pl.col(col)).dt.total_seconds().cast(pl.Float32)

    return df.select(
        S.MVT_ID,
        secs(S.SCHED_TIME).alias("tko_minus_sched"),
        secs(S.LOBT).alias("tko_minus_lobt"),
        secs(S.IOBT).alias("tko_minus_iobt"),
        secs(S.EOBT_1).alias("tko_minus_eobt1"),
        secs(S.AOBT_3).alias("tko_minus_aobt3"),
        (pl.col(S.LOBT) - pl.col(S.IOBT))
        .dt.total_seconds()
        .cast(pl.Float32)
        .alias("lobt_minus_iobt"),
        pl.col(S.FLIGHT_ID).is_null().cast(pl.Int8).alias("nm_unmatched"),
        (pl.col(S.ADES_FILED) != pl.col(S.ADES_FLT)).cast(pl.Int8).alias("diverted"),
    )


@feature("calendar_v2")
def calendar_v2(df: pl.DataFrame) -> pl.DataFrame:
    """``calendar`` with ``minute_of_day`` computed correctly.

    Code review 2026-09-19, finding #1: ``dt.hour()`` and ``dt.minute()`` are both ``Int8``
    in polars, so ``hour * 60 + minute`` overflows and wraps — 04:02 became **-14**, 11:06
    **-102**. The shipped column is ``(hour*60 + minute) mod 256`` read as signed: a
    six-tooth sawtooth, not a time of day, in every experiment since ``exp0001``. Casting
    before the multiplication is the whole fix; every other column is bit-identical to
    ``calendar``'s, so a run of this block against R5 isolates exactly that one column.
    """
    t = pl.col(S.MVT_TIME)
    return df.select(
        S.MVT_ID,
        t.dt.hour().alias("hour"),
        (t.dt.hour().cast(pl.Int32) * 60 + t.dt.minute().cast(pl.Int32))
        .cast(pl.Int16)
        .alias("minute_of_day"),
        t.dt.weekday().alias("dow"),
        t.dt.month().alias("month_num"),
        t.dt.ordinal_day().alias("doy"),
        (t.dt.weekday() >= 6).cast(pl.Int8).alias("is_weekend"),
    )
