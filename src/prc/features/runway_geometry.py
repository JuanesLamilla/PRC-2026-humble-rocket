"""Runway geometry from OurAirports (plan row P1-f).

``runways.csv`` (public domain, archived by ``scripts/fetch_external.py ourairports`` to
``data/external/ourairports_<date>/``) gives, per physical runway, its length and width in
feet and the true heading of each of its two thresholds. Joined to the departure's runway
designator it is a proxy for two things a taxi-out model otherwise cannot see: how far down
the field the departure end sits (long runways mean long taxi at EDDF/LFPG/LTFM, and an
intersection departure shortens it), and the runway's true heading, which turns a wind
direction into a crosswind component without going through the two-digit designator.

Each record is melted into its two ends (``le_ident`` / ``he_ident``) so a designator joins
whichever end it names. Designators are normalised (upper case, numeric part zero-padded to
two digits) on both sides. ``rwy_heading_deg`` falls back to the designator's own numeric
heading (``25L`` -> 250°) when OurAirports has no matching end, and
``rwy_heading_matched`` says which of the two happened.

Returns all-null columns when the archive is absent, so tests and fresh clones still run.
"""

from __future__ import annotations

import polars as pl

from prc import paths
from prc.data import schema as S
from prc.features.registry import archive_files, external_digest, feature

OUT = [
    "rwy_length_ft",
    "rwy_width_ft",
    "rwy_heading_deg",
    "rwy_heading_matched",
    "rwy_displaced_thr_ft",
]


def _norm_desig(col: str) -> pl.Expr:
    """``7l`` / ``07L`` -> ``07L``; non-numeric designators (``HEL``, ``NA``) pass through."""
    c = pl.col(col).cast(pl.Utf8).str.strip_chars().str.to_uppercase()
    num = c.str.extract(r"^(\d{1,2})", 1)
    side = c.str.extract(r"^\d{1,2}([LCR])$", 1).fill_null("")
    return pl.when(num.is_null()).then(c).otherwise(num.str.zfill(2) + side)


def _archive() -> pl.DataFrame | None:
    """One row per (airport ident, runway end) with length / width / heading."""
    dirs = sorted(paths.EXTERNAL.glob("ourairports_*"))
    if not dirs:
        return None
    f = dirs[-1] / "runways.csv"
    if not f.is_file():
        return None
    rw = pl.read_csv(
        f,
        columns=[
            "airport_ident",
            "length_ft",
            "width_ft",
            "closed",
            "le_ident",
            "le_heading_degT",
            "le_displaced_threshold_ft",
            "he_ident",
            "he_heading_degT",
            "he_displaced_threshold_ft",
        ],
        schema_overrides={
            "length_ft": pl.Float64,
            "width_ft": pl.Float64,
            "closed": pl.Int8,
            "le_heading_degT": pl.Float64,
            "he_heading_degT": pl.Float64,
            "le_displaced_threshold_ft": pl.Float64,
            "he_displaced_threshold_ft": pl.Float64,
        },
        ignore_errors=True,
    ).filter(pl.col("airport_ident").is_in(list(S.AIRPORTS)))
    ends = []
    for side in ("le", "he"):
        ends.append(
            rw.select(
                pl.col("airport_ident").alias(S.AIRPORT),
                _norm_desig(f"{side}_ident").alias("_desig"),
                pl.col("length_ft").alias("rwy_length_ft"),
                pl.col("width_ft").alias("rwy_width_ft"),
                pl.col(f"{side}_heading_degT").alias("rwy_heading_deg"),
                pl.col(f"{side}_displaced_threshold_ft").alias("rwy_displaced_thr_ft"),
                pl.col("closed").alias("_closed"),
            )
        )
    return (
        pl.concat(ends)
        .drop_nulls("_desig")
        .sort([S.AIRPORT, "_desig", "_closed"])  # prefer open runways on a duplicate designator
        .unique(subset=[S.AIRPORT, "_desig"], keep="first", maintain_order=True)
        .drop("_closed")
    )


#: Code review 2026-09-19, finding #4: this block reads ``data/external/`` and must say so,
#: or the cache cannot tell a build made without the archive from one made with it — the
#: 2026-09-18 cache incident. ``__name__`` also folds the module's own source in.


def _ourairports_external() -> str:
    return external_digest(archive_files("ourairports_*", "*.csv"), __name__)


@feature("runway_geometry", external=_ourairports_external)
def runway_geometry(df: pl.DataFrame) -> pl.DataFrame:
    """Runway length / width / displaced threshold (ft) and true heading for the departure
    runway, with the designator's own numeric heading as the fallback."""
    rw = _archive()
    base = df.select(S.MVT_ID, S.AIRPORT, _norm_desig(S.RUNWAY).alias("_desig"))
    desig_hdg = (pl.col("_desig").str.extract(r"^(\d{2})", 1).cast(pl.Float64) * 10.0).replace(
        0.0, 360.0
    )
    if rw is None:
        return base.select(S.MVT_ID).with_columns(
            [pl.lit(None, dtype=pl.Float32).alias(c) for c in OUT]
        )
    out = base.join(rw, on=[S.AIRPORT, "_desig"], how="left")
    return out.select(
        S.MVT_ID,
        pl.col("rwy_length_ft").cast(pl.Float32),
        pl.col("rwy_width_ft").cast(pl.Float32),
        pl.coalesce(pl.col("rwy_heading_deg"), desig_hdg).cast(pl.Float32).alias("rwy_heading_deg"),
        pl.col("rwy_heading_deg").is_not_null().cast(pl.Int8).alias("rwy_heading_matched"),
        pl.col("rwy_displaced_thr_ft").cast(pl.Float32),
    )
