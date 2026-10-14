"""Weather as a process, at the expected off-block time (backlog X4).

The weather blocks in the members describe the moment of takeoff (``metar_visibility``) or of
the NM off-block (``winter_ops``, flags). The January 2026 de-icing days are driven by intensity and accumulation before
the departure leaves the stand: how much it snowed or rained in the last hours, how low the
visibility went, how cold and moist the air is, and whether the wind stands across the runway.

Reference time: the **expected off-block** ``coalesce(AOBT_3, EOBT_1, SCHED)`` of a departure
(landing time for arrivals, which the model never scores). Never ``BLOCK_TIME`` (the label) and
not the takeoff.

Columns:

* METAR (IEM archive, most recent report at or before the reference time, 90 min tolerance):
  ``wp_vis_min_1h`` (lowest visibility in the reports of the last hour, m), ``wp_temp_c``,
  ``wp_dewpoint_spread_c``, ``wp_wspd_kt``, ``wp_gust_kt`` (gust, else the mean wind),
  ``wp_headwind_kt`` / ``wp_crosswind_kt`` (wind resolved on the departure runway's designator
  heading; the runway is test-visible), ``wp_frozen_3h`` (frozen-precipitation reports in the
  last 3 h weighted light 1 / moderate 2 / heavy 3).
* Open-Meteo (hourly reanalysis, the hour containing the reference time):
  ``wp_prcp_1h`` / ``wp_prcp_3h`` (mm over the last 1 / 3 hours), ``wp_snow_3h`` (cm),
  ``wp_temp_change_3h`` (°C over 3 h).
* ``wp_icing_index`` in [0, 1] = coldness × moisture, coldness = clip((3 − T) / 6, 0, 1) with the
  METAR temperature (Open-Meteo when missing), moisture = 1 when frozen precipitation was
  reported or precipitation fell in the last 3 h, else clip((3 − dew-point spread) / 3, 0, 1).

Returns all-null columns when both archives are absent.
"""

from __future__ import annotations

import math
from pathlib import Path

import polars as pl

from prc.data import schema as S
from prc.features.registry import archive_files, feature
from prc.features.screen_common import deps_digest

TOLERANCE = "90m"
METAR_OUT = [
    "wp_vis_min_1h",
    "wp_temp_c",
    "wp_dewpoint_spread_c",
    "wp_wspd_kt",
    "wp_gust_kt",
    "wp_headwind_kt",
    "wp_crosswind_kt",
    "wp_frozen_3h",
]
#: columns taken from the report itself (the runway-relative wind is derived after the join)
OBS_COLS = [c for c in METAR_OUT if c not in ("wp_headwind_kt", "wp_crosswind_kt")]
OM_OUT = ["wp_prcp_1h", "wp_prcp_3h", "wp_snow_3h", "wp_temp_change_3h"]
OUT = [*METAR_OUT, *OM_OUT, "wp_icing_index"]
_MCOLS = ["station", "valid", "tmpf", "dwpf", "drct", "sknt", "gust", "vsby", "wxcodes", "metar"]


def _metar_files() -> list[Path]:
    return archive_files("metar_iem_*", "*.csv")


def _om_files() -> list[Path]:
    return archive_files("openmeteo_*", "*.json")


def _wp_external() -> str:
    return deps_digest(
        __name__,
        ["winter_ops", "metar_visibility", "weather_openmeteo"],
        [*_metar_files(), *_om_files()],
    )


def metar_obs(files: list[Path]) -> pl.DataFrame | None:
    """Per (airport, report time): visibility, temperature, dew point, wind, frozen weight and
    the 1 h minimum visibility / 3 h frozen weight over the report series."""
    if not files:
        return None
    from prc.features.metar_visibility import _vis_metres
    from prc.features.winter_ops import FROZEN_RE

    raw = pl.concat(
        [
            pl.read_csv(
                f, columns=_MCOLS, schema_overrides={c: pl.Utf8 for c in _MCOLS}, ignore_errors=True
            )
            for f in files
        ]
    )

    def num(col: str) -> pl.Expr:
        return pl.col(col).cast(pl.Float64, strict=False)

    wx = pl.col("wxcodes").cast(pl.Utf8).fill_null("").str.to_uppercase().replace("M", "")
    frozen = wx.str.contains(FROZEN_RE)
    weight = (
        pl.when(~frozen)
        .then(0.0)
        .when(wx.str.contains(r"\+"))
        .then(3.0)
        .when(wx.str.contains("-"))
        .then(1.0)
        .otherwise(2.0)
    )
    temp = (num("tmpf") - 32.0) * 5.0 / 9.0
    dew = (num("dwpf") - 32.0) * 5.0 / 9.0
    obs = (
        raw.select(
            pl.col("station").alias(S.AIRPORT),
            pl.col("valid").str.to_datetime("%Y-%m-%d %H:%M", time_zone="UTC").alias("_t"),
            _vis_metres().alias("_vis"),
            temp.alias("wp_temp_c"),
            (temp - dew).alias("wp_dewpoint_spread_c"),
            num("drct").alias("_drct"),
            num("sknt").alias("wp_wspd_kt"),
            pl.coalesce(num("gust"), num("sknt")).alias("wp_gust_kt"),
            weight.alias("_frozen_w"),
        )
        .drop_nulls("_t")
        .unique(subset=[S.AIRPORT, "_t"], keep="last")
        .sort([S.AIRPORT, "_t"])
    )
    r1 = obs.rolling(index_column="_t", period="1h", group_by=S.AIRPORT, closed="right").agg(
        pl.col("_vis").min().alias("wp_vis_min_1h")
    )
    r3 = obs.rolling(index_column="_t", period="3h", group_by=S.AIRPORT, closed="right").agg(
        pl.col("_frozen_w").sum().alias("wp_frozen_3h")
    )
    return (
        obs.join(r1, on=[S.AIRPORT, "_t"], how="left")
        .join(r3, on=[S.AIRPORT, "_t"], how="left")
        .sort([S.AIRPORT, "_t"])
    )


def om_hourly(wx: pl.DataFrame | None) -> pl.DataFrame | None:
    """Open-Meteo hourly frame → (airport, hour, wp_prcp_1h, wp_prcp_3h, wp_snow_3h,
    wp_temp_change_3h, _om_temp)."""
    if wx is None:
        return None
    return wx.sort(S.AIRPORT, "hour").select(
        S.AIRPORT,
        "hour",
        pl.col("om_prcp").alias("wp_prcp_1h"),
        pl.col("om_prcp")
        .rolling_sum(window_size=3, min_samples=1)
        .over(S.AIRPORT)
        .alias("wp_prcp_3h"),
        pl.col("om_snowfall")
        .rolling_sum(window_size=3, min_samples=1)
        .over(S.AIRPORT)
        .alias("wp_snow_3h"),
        (pl.col("om_temp") - pl.col("om_temp").shift(3).over(S.AIRPORT)).alias("wp_temp_change_3h"),
        pl.col("om_temp").alias("_om_temp"),
    )


def reference_time() -> pl.Expr:
    """Expected off-block for departures (AOBT_3, else EOBT_1, else SCHED); landing for
    arrivals. Never the label, never the takeoff of a departure."""
    return (
        pl.when(pl.col(S.PHASE) == "DEP")
        .then(pl.coalesce(pl.col(S.AOBT_3), pl.col(S.EOBT_1), pl.col(S.SCHED_TIME)))
        .otherwise(pl.col(S.MVT_TIME))
    )


def weather_process_from(
    df: pl.DataFrame, obs: pl.DataFrame | None, om: pl.DataFrame | None
) -> pl.DataFrame:
    base = df.select(
        S.MVT_ID,
        S.AIRPORT,
        S.RUNWAY,
        reference_time().alias("_ref"),
    ).with_columns(pl.col("_ref").dt.truncate("1h").alias("hour"))
    out = base
    if obs is not None:
        m = (
            base.drop_nulls("_ref")
            .sort([S.AIRPORT, "_ref"])
            .join_asof(
                obs.select(S.AIRPORT, "_t", "_drct", *OBS_COLS),
                left_on="_ref",
                right_on="_t",
                by=S.AIRPORT,
                strategy="backward",
                tolerance=TOLERANCE,
            )
            .select(S.MVT_ID, "_drct", *OBS_COLS)
        )
        out = out.join(m, on=S.MVT_ID, how="left")
        rwy_hdg = pl.col(S.RUNWAY).cast(pl.Utf8).str.extract(r"^(\d{2})", 1).cast(pl.Float64) * 10.0
        rel = (pl.col("_drct") - rwy_hdg) * (math.pi / 180.0)
        out = out.with_columns(
            (pl.col("wp_wspd_kt") * rel.cos()).alias("wp_headwind_kt"),
            (pl.col("wp_wspd_kt") * rel.sin().abs()).alias("wp_crosswind_kt"),
        )
    else:
        out = out.with_columns([pl.lit(None, dtype=pl.Float64).alias(c) for c in METAR_OUT])
    if om is not None:
        out = out.join(om, on=[S.AIRPORT, "hour"], how="left")
    else:
        out = out.with_columns(
            [pl.lit(None, dtype=pl.Float64).alias(c) for c in [*OM_OUT, "_om_temp"]]
        )
    t = pl.coalesce(pl.col("wp_temp_c"), pl.col("_om_temp"))
    cold = ((3.0 - t) / 6.0).clip(0.0, 1.0)
    wet = (pl.col("wp_frozen_3h").fill_null(0) > 0) | (pl.col("wp_prcp_3h").fill_null(0) > 0)
    moist = (
        pl.when(wet)
        .then(1.0)
        .otherwise(((3.0 - pl.col("wp_dewpoint_spread_c")) / 3.0).clip(0.0, 1.0).fill_null(0.0))
    )
    out = out.with_columns((cold * moist).alias("wp_icing_index"))
    return out.select(S.MVT_ID, *[pl.col(c).cast(pl.Float32) for c in OUT])


@feature("weather_process", external=_wp_external)
def weather_process(df: pl.DataFrame) -> pl.DataFrame:
    """Weather at the expected off-block time: 1 h minimum visibility, temperature, dew-point
    spread, wind / gust and head- / crosswind on the runway, 3 h frozen-precipitation weight
    (METAR); 1 h / 3 h precipitation, 3 h snowfall, 3 h temperature change (Open-Meteo); an
    icing index."""
    from prc.features.weather_openmeteo import _archive

    return weather_process_from(df, metar_obs(_metar_files()), om_hourly(_archive()))
