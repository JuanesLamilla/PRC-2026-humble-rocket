"""Winter operations from METAR present-weather codes (idea backlog B2, TODO 4b).

January is half the scored data, and the mechanism that inflates a January taxi-out is not
visibility — it is de-icing. An aircraft that goes through a de-icing pad, or that is sprayed
on stand, spends five to twenty extra minutes between off-block and takeoff, and the airport
runs a reduced departure rate while it happens. ``metar_visibility`` sees none of that: it
carries a single ``mtr_snow`` flag taken from the report at the moment of *takeoff*.

This block parses the same Iowa Environmental Mesonet archive
(``data/external/metar_iem_<date>/<ICAO>_<year>.csv``, fetched by
``scripts/fetch_external.py metar``) for what actually drives the decision to de-ice:

* **frozen precipitation** — ``SN`` snow (incl. ``SHSN`` / ``SNRA`` / ``RASN``), ``SG`` snow
  grains, ``PL`` ice pellets, ``GR`` hail, ``GS`` small hail / snow pellets, ``IC`` ice
  crystals, and the freezing precipitation types ``FZRA`` / ``FZDZ``;
* **intensity** — the ``+`` / ``-`` qualifier on the group (``+SN`` is a different day from
  ``-SN``);
* **freezing fog** ``FZFG``, the classic "everything ices up on stand" morning;
* **temperature and dew point** (the archive is in Fahrenheit; converted here), with the
  ``≤ 0 °C`` / ``≤ 3 °C`` thresholds the holdover tables are built around, mist / fog at
  sub-zero temperature, and rain at ``≤ 3 °C`` — the case where the airframe is cold-soaked
  and wet, which requires de-icing even though nothing frozen is falling;
* **history** — whether frozen precipitation fell in the last 1 / 3 / 6 h, how long ago the
  last frozen report was, and how many of the last 12 h had it. Contamination persists: a
  pad that ran all night is still slow at 07:00 in clear air.

TIMESTAMP (the audit's §c point). The row is described by the most recent report at or
before **the best pre-departure timestamp that is observable at test time** — ``AOBT_3_flt``
when the flight has an NM off-block estimate, else ``SCHED_TIME_UTC_mvt`` — never
``BLOCK_TIME_UTC_mvt`` (that is the label) and not ``MVT_TIME_UTC_mvt`` (the *end* of the
taxi being predicted). Arrivals, which the model never scores but which are in the table, are
described at their landing time. ``join_asof`` backward, 90-minute tolerance, exactly as
``metar_visibility``; ``wo_obs_age_s`` records the staleness.

Coverage: all ten airports including LTFM, 2025 and 2026 — see
internal note 2026-09-19_winter-ops-coverage.

Cache: the block declares its external inputs (``external=``), so a build against a
different METAR pull — or against no archive at all — cannot be served from another build's
parquet (the 2026-09-18 cache incident).

Returns all-null columns when the archive is absent, so tests and fresh clones still run.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from prc.data import schema as S
from prc.features.registry import archive_files, external_digest, feature

TOLERANCE = "90m"
#: hours since the last frozen report, capped (a whole summer of "no snow" is one value)
MAX_HOURS_SINCE = 240.0
FROZEN_RE = r"SN|SG|PL|GR|GS|IC|FZRA|FZDZ"

OUT = [
    "wo_temp_c",
    "wo_dewpoint_c",
    "wo_dewpoint_spread",
    "wo_temp_le0",
    "wo_temp_le3",
    "wo_frozen_now",
    "wo_frozen_intensity",
    "wo_frozen_1h",
    "wo_frozen_3h",
    "wo_frozen_6h",
    "wo_frozen_hours_12h",
    "wo_hours_since_frozen",
    "wo_freezing_fog",
    "wo_cold_fog",
    "wo_cold_rain",
    "wo_deicing_likely",
    "wo_obs_age_s",
]

_COLS = ["station", "valid", "tmpf", "dwpf", "wxcodes", "metar"]


def _metar_files() -> list[Path]:
    return archive_files("metar_iem_*", "*.csv")


def _f_to_c(col: str) -> pl.Expr:
    return (pl.col(col).cast(pl.Float64, strict=False) - 32.0) * 5.0 / 9.0


def _observations(files: list[Path]) -> pl.DataFrame | None:
    """One row per (airport, report time) with the winter-operations state at that report."""
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
    # The IEM writes "M" for a missing decoded field; the present-weather group of the raw
    # report is the fallback when `wxcodes` is missing but the report itself carries one.
    wx = (
        pl.col("wxcodes")
        .cast(pl.Utf8)
        .fill_null("")
        .str.strip_chars()
        .str.to_uppercase()
        .replace("M", "")
    )
    body = (
        pl.col("metar")
        .cast(pl.Utf8)
        .fill_null("")
        .str.to_uppercase()
        .str.replace_all(r"(TEMPO|BECMG|NOSIG|RMK).*", "")  # never read a forecast group
    )
    frozen = (wx.str.contains(FROZEN_RE)) | (body.str.contains(r" [+-]?(?:" + FROZEN_RE + r")"))
    temp_c = _f_to_c("tmpf")
    dew_c = _f_to_c("dwpf")
    obs = (
        raw.select(
            pl.col("station").alias(S.AIRPORT),
            pl.col("valid").str.to_datetime("%Y-%m-%d %H:%M", time_zone="UTC").alias("_t"),
            temp_c.cast(pl.Float32).alias("wo_temp_c"),
            dew_c.cast(pl.Float32).alias("wo_dewpoint_c"),
            (temp_c - dew_c).cast(pl.Float32).alias("wo_dewpoint_spread"),
            frozen.cast(pl.Int8).alias("wo_frozen_now"),
            pl.when(~frozen)
            .then(0)
            .when(wx.str.contains(r"\+") | body.str.contains(r" \+(?:" + FROZEN_RE + r")"))
            .then(1)
            .when(wx.str.contains("-") | body.str.contains(r" -(?:" + FROZEN_RE + r")"))
            .then(-1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("wo_frozen_intensity"),
            wx.str.contains("FZFG").cast(pl.Int8).alias("wo_freezing_fog"),
            (wx.str.contains("FG") | wx.str.contains("BR")).alias("_obscured"),
            (wx.str.contains("RA") & ~wx.str.contains("FZRA")).alias("_rain"),
        )
        .drop_nulls("_t")
        .unique(subset=[S.AIRPORT, "_t"], keep="last")
        .sort([S.AIRPORT, "_t"])
    )
    obs = obs.with_columns(
        (pl.col("wo_temp_c") <= 0.0).cast(pl.Int8).alias("wo_temp_le0"),
        (pl.col("wo_temp_c") <= 3.0).cast(pl.Int8).alias("wo_temp_le3"),
        (pl.col("_obscured") & (pl.col("wo_temp_c") <= 0.0)).cast(pl.Int8).alias("wo_cold_fog"),
        (pl.col("_rain") & (pl.col("wo_temp_c") <= 3.0)).cast(pl.Int8).alias("wo_cold_rain"),
    ).drop("_obscured", "_rain")

    # Rolling history over the report series itself (30-minute cadence at these stations).
    def roll(period: str, name: str, expr: pl.Expr) -> pl.DataFrame:
        return obs.rolling(
            index_column="_t", period=period, group_by=S.AIRPORT, closed="right"
        ).agg(expr.alias(name))

    obs = (
        obs.join(roll("1h", "wo_frozen_1h", pl.col("wo_frozen_now").max()), on=[S.AIRPORT, "_t"])
        .join(roll("3h", "wo_frozen_3h", pl.col("wo_frozen_now").max()), on=[S.AIRPORT, "_t"])
        .join(roll("6h", "wo_frozen_6h", pl.col("wo_frozen_now").max()), on=[S.AIRPORT, "_t"])
        .join(
            roll("12h", "wo_frozen_hours_12h", pl.col("wo_frozen_now").mean() * 12.0),
            on=[S.AIRPORT, "_t"],
        )
    )
    last_frozen = (
        pl.when(pl.col("wo_frozen_now") == 1)
        .then(pl.col("_t"))
        .otherwise(None)
        .forward_fill()
        .over(S.AIRPORT)
    )
    obs = obs.with_columns(
        ((pl.col("_t") - last_frozen).dt.total_seconds() / 3600.0)
        .clip(0.0, MAX_HOURS_SINCE)
        .cast(pl.Float32)
        .alias("wo_hours_since_frozen"),
    )
    return obs.with_columns(
        (
            (pl.col("wo_frozen_3h") == 1)
            | (pl.col("wo_freezing_fog") == 1)
            | (pl.col("wo_cold_fog") == 1)
            | (pl.col("wo_cold_rain") == 1)
        )
        .cast(pl.Int8)
        .alias("wo_deicing_likely"),
        pl.col("wo_frozen_hours_12h").cast(pl.Float32),
    ).sort([S.AIRPORT, "_t"])


def reference_time() -> pl.Expr:
    """The timestamp a departure is described at: the NM off-block estimate when there is one,
    else the scheduled off-block. Both survive into the ranking set; ``BLOCK_TIME_UTC_mvt`` is
    the label and ``MVT_TIME_UTC_mvt`` is the end of the taxi. Arrivals use their landing."""
    return (
        pl.when(pl.col(S.PHASE) == "DEP")
        .then(pl.coalesce(pl.col(S.AOBT_3), pl.col(S.SCHED_TIME), pl.col(S.MVT_TIME)))
        .otherwise(pl.col(S.MVT_TIME))
    )


def _winter_external() -> str:
    return external_digest(_metar_files(), __name__)


@feature("winter_ops", external=_winter_external)
def winter_ops(df: pl.DataFrame) -> pl.DataFrame:
    """De-icing-likely indicator, frozen precipitation now / 1 / 3 / 6 h / 12 h-hours and
    hours since the last of it, freezing fog, sub-zero fog and cold rain, temperature, dew
    point and spread — from the METAR at or before the departure's best observable
    pre-departure timestamp (``AOBT_3_flt``, else the schedule)."""
    obs = _observations(_metar_files())
    base = (
        df.select(S.MVT_ID, S.AIRPORT, reference_time().alias("_ref"))
        .drop_nulls("_ref")
        .sort([S.AIRPORT, "_ref"])
    )
    if obs is None:
        return df.select(S.MVT_ID).with_columns(
            [pl.lit(None, dtype=pl.Float32).alias(c) for c in OUT]
        )
    out = base.join_asof(
        obs,
        left_on="_ref",
        right_on="_t",
        by=S.AIRPORT,
        strategy="backward",
        tolerance=TOLERANCE,
        coalesce=False,
    ).select(
        S.MVT_ID,
        *[pl.col(c) for c in OUT if c != "wo_obs_age_s"],
        (pl.col("_ref") - pl.col("_t")).dt.total_seconds().cast(pl.Float32).alias("wo_obs_age_s"),
    )
    # rows whose reference timestamp was null (never a departure) still need a row
    return df.select(S.MVT_ID).join(out, on=S.MVT_ID, how="left")
