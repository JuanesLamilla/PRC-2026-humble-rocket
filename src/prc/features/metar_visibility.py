"""METAR visibility / ceiling / low-visibility-procedure regime (plan row P1-c).

Iowa Environmental Mesonet ASOS archive (upstream NOAA ISD), archived by
``scripts/fetch_external.py metar --years 2025 2026`` to
``data/external/metar_iem_<date>/<ICAO>_<year>.csv`` — 30-minute observations for the ten
airports. This is the mechanism the ERA5 grid in ``weather_openmeteo`` cannot see: when an
airport goes to low-visibility procedures the departure rate collapses, aircraft are held on
stand or in the queue, and taxi-out inflates by minutes. That is a January mechanism at
EHAM / LFPG / EDDM / LSZH.

The IEM ``vsby`` column is in statute miles and **saturates at 6.21 SM**, so it is useless
above ~10 km but exact below it; the raw ``metar`` column is parsed for the European
four-digit metre group instead, with ``CAVOK`` → 10000 m and ``vsby`` as the fallback. The
trend part of the report (``TEMPO`` / ``BECMG`` / ``NOSIG`` / ``RMK``) is cut off first so a
forecast visibility is never mistaken for the observed one. Ceiling comes from the decoded
``skyc*`` / ``skyl*`` columns (lowest BKN / OVC / VV layer). LVP = visibility < 600 m or
ceiling < 200 ft; ``mtr_lvp_frac_6h`` carries the persistence of the regime, which is what
actually drives the airport's declared LVP state.

Joined by ``join_asof`` backward on (airport, ``MVT_TIME``) with a 90-minute tolerance, so a
departure is described by the most recent observation at or before its takeoff and never by
a later one. Returns all-null columns when the archive is absent.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from prc import paths
from prc.data import schema as S
from prc.features.registry import archive_files, external_digest, feature

SM_TO_M = 1609.344
CEILING_CODES = ("BKN", "OVC", "VV")
LVP_VIS_M = 600.0
LVP_CEIL_FT = 200.0
TOLERANCE = "90m"

OUT = [
    "mtr_vis_m",
    "mtr_vis_min_3h",
    "mtr_ceiling_ft",
    "mtr_lvp",
    "mtr_lvp_frac_6h",
    "mtr_rvr",
    "mtr_fog",
    "mtr_freezing",
    "mtr_snow",
    "mtr_obs_age_s",
]

_COLS = [
    "station",
    "valid",
    "vsby",
    "wxcodes",
    "metar",
    *[f"skyc{i}" for i in range(1, 5)],
    *[f"skyl{i}" for i in range(1, 5)],
]


def _vis_metres() -> pl.Expr:
    """Observed visibility in metres from the raw report, ``vsby`` as the fallback."""
    body = (
        pl.lit(" ")
        + pl.col("metar")
        .cast(pl.Utf8)
        .fill_null("")
        .str.replace_all(r"(TEMPO|BECMG|NOSIG|RMK).*", "")
        + pl.lit(" ")
    )
    tok = body.str.extract(r" (\d{4})(?:NDV)? ", 1).cast(pl.Float64)
    cavok = body.str.contains("CAVOK")
    vsby_m = pl.col("vsby").cast(pl.Float64, strict=False) * SM_TO_M
    return (
        pl.when(tok.is_not_null())
        .then(pl.when(tok >= 9999).then(10000.0).otherwise(tok))
        .when(cavok)
        .then(10000.0)
        .otherwise(vsby_m)
    )


def _ceiling_layers() -> list[pl.Expr]:
    """One column per sky layer: its height in feet when it is a ceiling layer, else null.

    Materialised with ``with_columns`` rather than folded inline, so that a chunk in which
    no row has a ceiling still yields a full-length null column instead of a scalar."""
    return [
        pl.when(pl.col(f"skyc{i}").cast(pl.Utf8).str.strip_chars().is_in(list(CEILING_CODES)))
        .then(pl.col(f"skyl{i}").cast(pl.Float64, strict=False))
        .otherwise(None)
        .alias(f"_ceil{i}")
        for i in range(1, 5)
    ]


def _archive(d: Path) -> pl.DataFrame | None:
    files = sorted(f for f in d.glob("*.csv"))
    if not files:
        return None
    raw = pl.concat(
        [
            pl.read_csv(
                f,
                columns=_COLS,
                schema_overrides={c: pl.Utf8 for c in _COLS},
                ignore_errors=True,
            )
            for f in files
        ]
    )
    raw = raw.with_columns(_ceiling_layers())
    wx = pl.col("wxcodes").cast(pl.Utf8).fill_null("")
    obs = (
        raw.select(
            pl.col("station").alias(S.AIRPORT),
            pl.col("valid").str.to_datetime("%Y-%m-%d %H:%M", time_zone="UTC").alias("_t"),
            _vis_metres().cast(pl.Float32).alias("mtr_vis_m"),
            pl.min_horizontal([f"_ceil{i}" for i in range(1, 5)])
            .cast(pl.Float32)
            .alias("mtr_ceiling_ft"),
            pl.col("metar")
            .cast(pl.Utf8)
            .fill_null("")
            .str.contains(r" R\d{2}[LCR]?/")
            .cast(pl.Int8)
            .alias("mtr_rvr"),
            wx.str.contains("FG").cast(pl.Int8).alias("mtr_fog"),
            wx.str.contains("FZ").cast(pl.Int8).alias("mtr_freezing"),
            wx.str.contains("SN").cast(pl.Int8).alias("mtr_snow"),
        )
        .drop_nulls("_t")
        .unique(subset=[S.AIRPORT, "_t"], keep="last")
        .sort([S.AIRPORT, "_t"])
    )
    lvp = (
        (pl.col("mtr_vis_m") < LVP_VIS_M).fill_null(False)
        | (pl.col("mtr_ceiling_ft") < LVP_CEIL_FT).fill_null(False)
    ).cast(pl.Int8)
    obs = obs.with_columns(lvp.alias("mtr_lvp"))
    roll6 = obs.rolling(index_column="_t", period="6h", group_by=S.AIRPORT, closed="right").agg(
        pl.col("mtr_lvp").mean().cast(pl.Float32).alias("mtr_lvp_frac_6h")
    )
    roll3 = obs.rolling(index_column="_t", period="3h", group_by=S.AIRPORT, closed="right").agg(
        pl.col("mtr_vis_m").min().cast(pl.Float32).alias("mtr_vis_min_3h")
    )
    return (
        obs.join(roll6, on=[S.AIRPORT, "_t"], how="left")
        .join(roll3, on=[S.AIRPORT, "_t"], how="left")
        .sort([S.AIRPORT, "_t"])
    )


#: Code review 2026-09-19, finding #4: this block reads ``data/external/`` and must say so,
#: or the cache cannot tell a build made without the archive from one made with it — the
#: 2026-09-18 cache incident. ``__name__`` also folds the module's own source in.


def _metar_external() -> str:
    return external_digest(archive_files("metar_iem_*", "*.csv"), __name__)


@feature("metar_visibility", external=_metar_external)
def metar_visibility(df: pl.DataFrame) -> pl.DataFrame:
    """Visibility (m), ceiling (ft), LVP flag and its 6 h persistence, RVR presence and
    fog / freezing / snow present-weather flags from the most recent METAR at or before the
    takeoff (90-minute tolerance)."""
    dirs = sorted(paths.EXTERNAL.glob("metar_iem_*"))
    obs = _archive(dirs[-1]) if dirs else None
    base = df.select(S.MVT_ID, S.AIRPORT, S.MVT_TIME).sort([S.AIRPORT, S.MVT_TIME])
    if obs is None:
        return base.select(S.MVT_ID).with_columns(
            [pl.lit(None, dtype=pl.Float32).alias(c) for c in OUT]
        )
    out = base.join_asof(
        obs,
        left_on=S.MVT_TIME,
        right_on="_t",
        by=S.AIRPORT,
        strategy="backward",
        tolerance=TOLERANCE,
        coalesce=False,
    )
    return out.select(
        S.MVT_ID,
        pl.col("mtr_vis_m"),
        pl.col("mtr_vis_min_3h"),
        pl.col("mtr_ceiling_ft"),
        pl.col("mtr_lvp"),
        pl.col("mtr_lvp_frac_6h"),
        pl.col("mtr_rvr"),
        pl.col("mtr_fog"),
        pl.col("mtr_freezing"),
        pl.col("mtr_snow"),
        (pl.col(S.MVT_TIME) - pl.col("_t"))
        .dt.total_seconds()
        .cast(pl.Float32)
        .alias("mtr_obs_age_s"),
    )
