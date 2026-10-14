"""Physical aircraft characteristics per ICAO type from the FAA Aircraft Characteristics Database.

Source: FAA ACD (US federal government work, public domain), fetched by
``scripts/fetch_external.py faa_acd`` into ``data/external/faa_acd_<date>/faa_acd.csv`` (the
``ACD_Data`` sheet of the published xlsx) and registered in ``docs/data-sources.md``.

Why: the members see the aircraft type, wake category and operator only as categories. A rare
type, or a type the neural members embed from a few hundred rows, gets no size or performance
information from them. The ACD gives numbers that transfer across types: wingspan, length, MTOW,
the airplane design group (ADG, the taxiway / stand size class), the taxiway design group (TDG,
gear geometry), the approach category and the engine count / class.

Join key: ``AIRCRAFT_TYPE_mvt`` (airport feed), falling back to ``AIRCRAFT_TYPE_flt`` (NM) when
the airport's code is missing or unknown to the ACD. Both are test-visible. ``ac_matched`` = 2
(airport code matched), 1 (NM code matched), 0 (no match; every other column null).

Columns (departures and arrivals alike; static per type, no timestamps, nothing can leak):
``ac_wingspan_ft`` (with winglets when listed, else without), ``ac_length_ft``,
``ac_tail_height_ft``, ``ac_wheelbase_ft``, ``ac_main_gear_width_ft``, ``ac_mtow_klb``,
``ac_adg`` (I–VI → 1–6), ``ac_tdg`` (1A 1B 2A 2B 3 4 5 6 7 → 1…9), ``ac_aac`` (A–E → 1–5),
``ac_approach_kt``, ``ac_num_engines``, ``ac_engine_class`` (piston 0, turboprop 1, jet 2),
``ac_parking_area_kft2``, ``ac_matched``.

Returns all-null columns (``ac_matched`` = 0) when the archive is absent.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from prc.data import schema as S
from prc.features.registry import archive_files, external_digest, feature

ADG = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6}
TDG = {"1A": 1, "1B": 2, "2A": 3, "2B": 4, "3": 5, "4": 6, "5": 7, "6": 8, "7": 9}
AAC = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
ENGINE = {"piston": 0, "turboprop": 1, "turboshaft": 1, "jet": 2}

NUM_COLS = [
    "ac_wingspan_ft",
    "ac_length_ft",
    "ac_tail_height_ft",
    "ac_wheelbase_ft",
    "ac_main_gear_width_ft",
    "ac_mtow_klb",
    "ac_adg",
    "ac_tdg",
    "ac_aac",
    "ac_approach_kt",
    "ac_num_engines",
    "ac_engine_class",
    "ac_parking_area_kft2",
]
OUT = [*NUM_COLS, "ac_matched"]


def _acd_files() -> list[Path]:
    return archive_files("faa_acd_*", "faa_acd.csv")


def _acd_external() -> str:
    return external_digest(_acd_files(), __name__)


def _num(col: str) -> pl.Expr:
    return pl.col(col).cast(pl.Utf8).str.strip_chars().cast(pl.Float64, strict=False)


def _ordinal(col: str, mapping: dict[str, int]) -> pl.Expr:
    return (
        pl.col(col)
        .cast(pl.Utf8)
        .str.strip_chars()
        .str.to_uppercase()
        .replace_strict(mapping, default=None, return_dtype=pl.Float64)
    )


def acd_table(files: list[Path] | None = None) -> pl.DataFrame | None:
    """One row per ICAO type code with the numeric ``ac_*`` columns (first row wins on a
    duplicated code)."""
    files = _acd_files() if files is None else files
    if not files:
        return None
    raw = pl.read_csv(files[-1], infer_schema=False)
    wing = pl.coalesce(
        _num("Wingspan_ft_with_winglets_sharklets"), _num("Wingspan_ft_without_winglets_sharklets")
    )
    engine = (
        pl.col("Physical_Class_Engine")
        .cast(pl.Utf8)
        .str.strip_chars()
        .str.to_lowercase()
        .replace_strict(ENGINE, default=None, return_dtype=pl.Float64)
    )
    return (
        raw.select(
            pl.col("ICAO_Code").str.strip_chars().str.to_uppercase().alias("_type"),
            wing.alias("ac_wingspan_ft"),
            _num("Length_ft").alias("ac_length_ft"),
            _num("Tail_Height_at_OEW_ft").alias("ac_tail_height_ft"),
            _num("Wheelbase_ft").alias("ac_wheelbase_ft"),
            _num("Main_Gear_Width_ft").alias("ac_main_gear_width_ft"),
            (_num("MTOW_lb") / 1000.0).alias("ac_mtow_klb"),
            _ordinal("ADG", ADG).alias("ac_adg"),
            _ordinal("TDG", TDG).alias("ac_tdg"),
            _ordinal("AAC", AAC).alias("ac_aac"),
            _num("Approach_Speed_knot").alias("ac_approach_kt"),
            _num("Num_Engines").alias("ac_num_engines"),
            engine.alias("ac_engine_class"),
            (_num("Parking_Area_ft2") / 1000.0).alias("ac_parking_area_kft2"),
        )
        .filter(pl.col("_type").is_not_null() & (pl.col("_type") != ""))
        .unique(subset="_type", keep="first", maintain_order=True)
    )


def chars_from(df: pl.DataFrame, table: pl.DataFrame | None) -> pl.DataFrame:
    """The block's output for ``df`` given the ACD table (``None`` → all-null)."""
    if table is None:
        return df.select(S.MVT_ID).with_columns(
            *[pl.lit(None, dtype=pl.Float32).alias(c) for c in NUM_COLS],
            pl.lit(0, dtype=pl.Int8).alias("ac_matched"),
        )

    def key(col: str) -> pl.Expr:
        return pl.col(col).cast(pl.Utf8).str.strip_chars().str.to_uppercase()

    known = table.select("_type").with_columns(pl.lit(True).alias("_known"))
    base = (
        df.select(
            S.MVT_ID, key(S.AIRCRAFT_TYPE).alias("_mvt"), key(S.AIRCRAFT_TYPE_FLT).alias("_flt")
        )
        .join(known.rename({"_type": "_mvt", "_known": "_k_mvt"}), on="_mvt", how="left")
        .join(known.rename({"_type": "_flt", "_known": "_k_flt"}), on="_flt", how="left")
        .with_columns(
            pl.when(pl.col("_k_mvt").fill_null(False))
            .then(pl.col("_mvt"))
            .when(pl.col("_k_flt").fill_null(False))
            .then(pl.col("_flt"))
            .otherwise(None)
            .alias("_type"),
            pl.when(pl.col("_k_mvt").fill_null(False))
            .then(2)
            .when(pl.col("_k_flt").fill_null(False))
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("ac_matched"),
        )
    )
    return base.join(table, on="_type", how="left").select(
        S.MVT_ID, *[pl.col(c).cast(pl.Float32) for c in NUM_COLS], "ac_matched"
    )


@feature("aircraft_chars", external=_acd_external)
def aircraft_chars(df: pl.DataFrame) -> pl.DataFrame:
    """FAA ACD per aircraft type: wingspan, length, tail height, wheelbase, gear width, MTOW,
    ADG / TDG / approach category (ordinal), approach speed, engine count and class, parking
    area; ``ac_matched`` 2 = airport type code, 1 = NM type code, 0 = none."""
    return chars_from(df, acd_table())
