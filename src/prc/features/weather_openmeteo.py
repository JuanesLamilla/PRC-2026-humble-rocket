"""Open-Meteo hourly reanalysis at the airport (CC BY 4.0, fetched by
``scripts/fetch_external.py openmeteo``), joined on (airport, UTC hour of ``MVT_TIME``), with the
same derived quantities as the Meteostat block plus lagged precipitation / frost persistence.
Registry-path weather block: open licence, all 10 airports, no nulls. Agreement with station
data: internal note 2026-09-18_weather-sources. Returns all-null columns when the archive is
absent so tests and fresh clones still run.
"""

from __future__ import annotations

import json
import math

import polars as pl

from prc import paths
from prc.data import schema as S
from prc.features.registry import archive_files, external_digest, feature

VARS = {
    "temperature_2m": "om_temp",
    "dew_point_2m": "om_dewpoint",
    "relative_humidity_2m": "om_rhum",
    "precipitation": "om_prcp",
    "snowfall": "om_snowfall",
    "snow_depth": "om_snow_depth",
    "weather_code": "om_wcode",
    "pressure_msl": "om_pres",
    "cloud_cover": "om_cldc",
    "cloud_cover_low": "om_cldc_low",
    "wind_speed_10m": "om_wspd",
    "wind_direction_10m": "om_wdir",
    "wind_gusts_10m": "om_gust",
    "vapour_pressure_deficit": "om_vpd",
}
DERIVED = [
    "om_dewpoint_spread",
    "om_deicing",
    "om_frost",
    "om_fog",
    "om_crosswind",
    "om_headwind",
    "om_prcp_3h",
    "om_prcp_24h",
    "om_frost_hours_24h",
    "om_temp_change_6h",
]
OUT = [*VARS.values(), *DERIVED]
# WMO codes: 45/48 fog, 56/57 freezing drizzle, 66/67 freezing rain, 71-77 snow, 85/86 snow showers
FOG = [45, 48]
FROZEN = [56, 57, 66, 67, 71, 73, 75, 77, 85, 86]


def _archive() -> pl.DataFrame | None:
    dirs = sorted(paths.EXTERNAL.glob("openmeteo_*"))
    if not dirs:
        return None
    frames = []
    for f in sorted(dirs[-1].glob("*.json")):
        if f.stem == "manifest":
            continue
        with open(f) as fh:
            h = json.load(fh)["hourly"]
        d = pl.DataFrame({k: h[k] for k in ["time", *VARS]}).with_columns(
            pl.col("time").str.to_datetime("%Y-%m-%dT%H:%M", time_zone="UTC").alias("hour"),
            pl.lit(f.stem).alias(S.AIRPORT),
        )
        frames.append(d.drop("time").rename(VARS))
    if not frames:
        return None
    wx = pl.concat(frames).sort(S.AIRPORT, "hour")
    # rolling / lagged quantities on the hourly grid, per airport
    return wx.with_columns(
        pl.col("om_prcp")
        .rolling_sum(window_size=3, min_samples=1)
        .over(S.AIRPORT)
        .alias("om_prcp_3h"),
        pl.col("om_prcp")
        .rolling_sum(window_size=24, min_samples=1)
        .over(S.AIRPORT)
        .alias("om_prcp_24h"),
        (pl.col("om_temp") <= 0)
        .cast(pl.Int32)
        .rolling_sum(window_size=24, min_samples=1)
        .over(S.AIRPORT)
        .alias("om_frost_hours_24h"),
        (pl.col("om_temp") - pl.col("om_temp").shift(6).over(S.AIRPORT)).alias("om_temp_change_6h"),
    )


#: Code review 2026-09-19, finding #4: this block reads ``data/external/`` and must say so,
#: or the cache cannot tell a build made without the archive from one made with it — the
#: 2026-09-18 cache incident. ``__name__`` also folds the module's own source in.


def _openmeteo_external() -> str:
    return external_digest(archive_files("openmeteo_*", "*.json"), __name__)


@feature("weather_openmeteo", external=_openmeteo_external)
def weather_openmeteo(df: pl.DataFrame) -> pl.DataFrame:
    """Open-Meteo hourly weather at the airport hour + de-icing / fog / runway-relative wind +
    3 h / 24 h precipitation and 24 h frost-hour accumulations."""
    wx = _archive()
    base = df.select(
        S.MVT_ID, S.AIRPORT, S.RUNWAY, pl.col(S.MVT_TIME).dt.truncate("1h").alias("hour")
    )
    if wx is None:
        return base.select(S.MVT_ID).with_columns(
            [pl.lit(None, dtype=pl.Float32).alias(c) for c in OUT]
        )
    out = base.join(wx, on=[S.AIRPORT, "hour"], how="left")
    t = pl.col("om_temp")
    code = pl.col("om_wcode").cast(pl.Int32)
    rwy_hdg = pl.col(S.RUNWAY).cast(pl.Utf8).str.extract(r"^(\d{2})", 1).cast(pl.Float64) * 10
    rel = (pl.col("om_wdir") - rwy_hdg) * (math.pi / 180)
    return out.select(
        S.MVT_ID,
        *[pl.col(c).cast(pl.Float32) for c in VARS.values()],
        (t - pl.col("om_dewpoint")).cast(pl.Float32).alias("om_dewpoint_spread"),
        ((t <= 3) & ((pl.col("om_rhum") >= 80) | (pl.col("om_prcp") > 0) | code.is_in(FROZEN)))
        .cast(pl.Int8)
        .alias("om_deicing"),
        (t <= 0).cast(pl.Int8).alias("om_frost"),
        (code.is_in(FOG) | ((t - pl.col("om_dewpoint")) <= 1.0)).cast(pl.Int8).alias("om_fog"),
        (pl.col("om_wspd") * rel.sin().abs()).cast(pl.Float32).alias("om_crosswind"),
        (pl.col("om_wspd") * rel.cos()).cast(pl.Float32).alias("om_headwind"),
        pl.col("om_prcp_3h").cast(pl.Float32),
        pl.col("om_prcp_24h").cast(pl.Float32),
        pl.col("om_frost_hours_24h").cast(pl.Float32),
        pl.col("om_temp_change_6h").cast(pl.Float32),
    )
