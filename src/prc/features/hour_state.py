"""Airport-hour state from observable departure stamps (Phase 0 lead 1, re-screened in Phase 1).

internal note 2026-09-26_ordinary-residuals-ltfm-lfpg-lirf §5: v15's residuals share an
airport-hour component (label-using ceiling −2.4 to −7.9 s at LIRF / LFPG / LTFM), and at LIRF
an out-of-month model on hour aggregates recovered part of it (R² 0.12), led by the hour's
median ``AOBT_3 − EOBT_1``. This block exposes the four aggregates that study named, for all ten
airports, so a residual screen can test them pre-registered.

Key: airport × UTC hour of the takeoff (every airport has a whole-hour UTC offset, so this is
the airport × local-day × local-hour cell of the study). Aggregates over **all departures** of
the cell, train and test rows alike (label-free inputs only):

* ``hs_n_dep``        departures taking off in the hour;
* ``hs_a_eobt_med``   median ``AOBT_3 − EOBT_1`` (s) over the hour's departures with both;
* ``hs_c_med``        median ``MVT − SCHED`` (s);
* ``hs_temp``         Open-Meteo 2 m temperature (°C) at the hour (``weather_openmeteo``'s
  archive; null when the archive is absent).

Reads takeoff, schedule, ``EOBT_1`` and ``AOBT_3`` of departures. Never a target or an
airport off-block stamp.
"""

from __future__ import annotations

import polars as pl

from prc.data import schema as S
from prc.features.registry import archive_files, feature
from prc.features.screen_common import deps_digest

OUT = ["hs_n_dep", "hs_a_eobt_med", "hs_c_med", "hs_temp"]


def _secs(a: str, b: str) -> pl.Expr:
    return (pl.col(a) - pl.col(b)).dt.total_seconds().cast(pl.Float64)


def _hs_external() -> str:
    # weather_openmeteo's hash covers its archive loader, which this block calls
    return deps_digest(__name__, ["weather_openmeteo"], archive_files("openmeteo_*", "*.json"))


def hour_state_from(df: pl.DataFrame, weather: pl.DataFrame | None) -> pl.DataFrame:
    """``weather``: hourly frame with ``airport``, ``hour`` (UTC) and ``om_temp``, or ``None``."""
    dep = df.filter(pl.col(S.PHASE) == "DEP").select(
        S.MVT_ID,
        S.AIRPORT,
        pl.col(S.MVT_TIME).dt.truncate("1h").alias("hour"),
        _secs(S.AOBT_3, S.EOBT_1).alias("_a_eobt"),
        _secs(S.MVT_TIME, S.SCHED_TIME).alias("_c"),
    )
    agg = dep.group_by(S.AIRPORT, "hour").agg(
        pl.len().cast(pl.Float32).alias("hs_n_dep"),
        pl.col("_a_eobt").median().cast(pl.Float32).alias("hs_a_eobt_med"),
        pl.col("_c").median().cast(pl.Float32).alias("hs_c_med"),
    )
    if weather is not None:
        agg = agg.join(
            weather.select(S.AIRPORT, "hour", pl.col("om_temp").cast(pl.Float32).alias("hs_temp")),
            on=[S.AIRPORT, "hour"],
            how="left",
        )
    else:
        agg = agg.with_columns(pl.lit(None, dtype=pl.Float32).alias("hs_temp"))
    return dep.join(agg, on=[S.AIRPORT, "hour"], how="left").select(S.MVT_ID, *OUT)


@feature("hour_state", external=_hs_external)
def hour_state(df: pl.DataFrame) -> pl.DataFrame:
    """Departure rows only. Per airport × UTC hour of takeoff: departures, median
    AOBT_3 − EOBT_1, median MVT − SCHED (s), Open-Meteo temperature."""
    from prc.features.weather_openmeteo import _archive

    return hour_state_from(df, _archive())
