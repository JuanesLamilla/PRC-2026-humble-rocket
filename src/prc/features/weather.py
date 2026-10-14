"""Hourly airport weather (Meteostat) joined by movement id, plus the derived quantities the
literature and other teams found to matter: dew-point spread (de-icing on clear frosty
mornings), a de-icing regime flag, a low-visibility proxy, and wind resolved onto the
runway heading (crosswind / headwind) using the runway designator as the heading.

Source: a teammate's Meteostat pull (hour of ``MVT_TIME``), archived at
``data/external/meteostat_teammate_2026-09-17/`` and registered in
``docs/data-sources.md``. LTFM has no station in that pull (all null); LightGBM
handles the nulls natively. If the archive is absent (tests, fresh clone) the block returns
all-null columns so the pipeline still runs.
"""

from __future__ import annotations

import math

import polars as pl

from prc import paths
from prc.data import schema as S
from prc.features.registry import external_digest, feature

ARCHIVE = paths.EXTERNAL / "meteostat_teammate_2026-09-17"
RAW = ["temp", "rhum", "prcp", "snwd", "wdir", "wspd", "wpgt", "pres", "cldc", "coco"]
# Meteostat condition codes: 5/6 fog, 10/11 freezing rain, 12/13 sleet, 14-16 snow, 19-22
# sleet/snow showers.
FOG = [5, 6]
FROZEN_PRECIP = [10, 11, 12, 13, 14, 15, 16, 19, 20, 21, 22]
OUT = [
    *[f"wx_{c}" for c in RAW],
    "wx_dewpoint_spread",
    "wx_deicing",
    "wx_frost",
    "wx_lvp",
    "wx_crosswind",
    "wx_headwind",
]


def _load_weather() -> pl.DataFrame | None:
    files = sorted(ARCHIVE.glob("*.parquet"))
    if not files:
        return None
    cols = [S.MVT_ID, *[f"weather_{c}" for c in RAW]]
    parts = [pl.read_parquet(f, columns=cols) for f in files]
    return pl.concat(parts).unique(subset=S.MVT_ID, keep="first")


#: Code review 2026-09-19, finding #4: this block reads ``data/external/`` and must say so,
#: or the cache cannot tell a build made without the archive from one made with it — the
#: 2026-09-18 cache incident. ``__name__`` also folds the module's own source in.


def _meteostat_external() -> str:
    return external_digest(sorted(ARCHIVE.glob("*.parquet")), __name__)


@feature("weather_meteostat", external=_meteostat_external)
def weather_meteostat(df: pl.DataFrame) -> pl.DataFrame:
    """Meteostat hourly weather at the airport at the movement hour + derived de-icing /
    LVP / runway-relative wind features."""
    wx = _load_weather()
    if wx is None:
        return df.select(S.MVT_ID).with_columns(
            [pl.lit(None, dtype=pl.Float32).alias(c) for c in OUT]
        )
    wx = wx.with_columns(pl.col(S.MVT_ID).cast(df.schema[S.MVT_ID]))
    out = df.select(S.MVT_ID, S.RUNWAY).join(wx, on=S.MVT_ID, how="left")
    t = pl.col("weather_temp")
    rh = pl.col("weather_rhum").clip(1, 100)
    gamma = (rh / 100).log() + 17.625 * t / (243.04 + t)
    dewpoint = 243.04 * gamma / (17.625 - gamma)
    coco = pl.col("weather_coco").cast(pl.Int32)
    # runway designator digits -> magnetic heading in degrees ("25L" -> 250)
    rwy_hdg = pl.col(S.RUNWAY).cast(pl.Utf8).str.extract(r"^(\d{2})", 1).cast(pl.Float64) * 10
    rel = (pl.col("weather_wdir") - rwy_hdg) * (math.pi / 180)
    return out.select(
        S.MVT_ID,
        *[pl.col(f"weather_{c}").cast(pl.Float32).alias(f"wx_{c}") for c in RAW],
        (t - dewpoint).cast(pl.Float32).alias("wx_dewpoint_spread"),
        ((t <= 3) & ((rh >= 80) | (pl.col("weather_prcp") > 0) | coco.is_in(FROZEN_PRECIP)))
        .cast(pl.Int8)
        .alias("wx_deicing"),
        (t <= 0).cast(pl.Int8).alias("wx_frost"),
        (coco.is_in(FOG) | ((rh >= 97) & (pl.col("weather_cldc") >= 90)))
        .cast(pl.Int8)
        .alias("wx_lvp"),
        (pl.col("weather_wspd") * rel.sin().abs()).cast(pl.Float32).alias("wx_crosswind"),
        (pl.col("weather_wspd") * rel.cos()).cast(pl.Float32).alias("wx_headwind"),
    )
