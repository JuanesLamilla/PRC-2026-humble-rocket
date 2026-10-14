"""How the airport records its takeoff stamp: off-grid takeoffs at minute-rounded airports.

Each
airport either records a stamp to the second or minute-rounds it and adds a jitter of
about ±6 s. The class is fixed per airport and island, identical in 2025, 2026-01 and
2026-07. At the airports that minute-round their takeoff stamps (EDDM, EGLL, LFPG, LIRF, LTFM),
every takeoff is within 6 s of a whole minute except a handful per month. Those off-grid rows
came through another recording path. At LFPG (97 departures in 2025, 8 in 2026-01, 11 in
2026-07) 60 % of them have a label above 1 h (0.14 % for the rest of the airport), and
the ``exp0185`` residual on them is +2,644 s on average. The takeoff stamp is visible in the
test months, so the flag is too.

Columns (departure rows only):

* ``sp_mvt_off_grid`` Int8: 1 when the row's (airport, data island) is *minute-rounded*
  (≥ ``GRID_SHARE`` of all its takeoff and landing stamps, over at least ``GRID_MIN_N`` rows,
  lie within ±``NEAR_S`` s of a whole minute) and this departure's takeoff is more than
  ``NEAR_S`` s from a whole minute; 0 otherwise (also at second-precision airports).

Reads only ``MVT_TIME`` (every row, both phases), the airport and the month. It never reads
a target, an off-block stamp or ``AOBT_3``.
"""

from __future__ import annotations

import polars as pl

from prc.data import schema as S
from prc.features.raw_extra import island_expr
from prc.features.registry import feature

#: |offset from the whole minute| ≤ this = on the grid, seconds (the jitter spans −6…+6).
NEAR_S = 6
#: An (airport, island) is minute-rounded when at least this share of its stamps is on the
#: grid. Uniform seconds give 13/60 = 0.217; the rounded airports give 0.999–1.000.
GRID_SHARE = 0.99
#: Fewer stamps than this in an (airport, island) → never classed as minute-rounded.
GRID_MIN_N = 200

COLS: tuple[str, ...] = ("sp_mvt_off_grid",)


def _offset_s(col: str) -> pl.Expr:
    """Signed offset of the stamp from the nearest whole minute, seconds in [−30, 29]."""
    return (pl.col(col).dt.second().cast(pl.Int16) + 30) % 60 - 30


@feature("stamp_process")
def stamp_process(df: pl.DataFrame) -> pl.DataFrame:
    """Departure rows only. ``sp_mvt_off_grid`` = the takeoff stamp is off the airport's
    minute grid (more than 6 s from a whole minute at an (airport, island) whose takeoff and
    landing stamps are ≥ 99 % on the grid). Int8, never null."""
    c = pl.col
    base = df.select(
        S.MVT_ID,
        S.PHASE,
        S.AIRPORT,
        island_expr().alias("_island"),
        (_offset_s(S.MVT_TIME).abs() > NEAR_S).alias("_off"),
        c(S.MVT_TIME).is_not_null().alias("_has"),
    )
    grid = (
        base.filter(c("_has"))
        .group_by(S.AIRPORT, "_island")
        .agg(_n=pl.len(), _on=(~c("_off")).mean())
        .with_columns(_rounded=(c("_n") >= GRID_MIN_N) & (c("_on") >= GRID_SHARE))
        .select(S.AIRPORT, "_island", "_rounded")
    )
    return (
        base.filter(c(S.PHASE) == "DEP")
        .join(grid, on=[S.AIRPORT, "_island"], how="left")
        .select(
            S.MVT_ID,
            (c("_rounded").fill_null(False) & c("_off").fill_null(False))
            .cast(pl.Int8)
            .alias("sp_mvt_off_grid"),
        )
        .sort(S.MVT_ID)
    )
