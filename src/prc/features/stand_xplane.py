"""Stand attributes and stand → runway-threshold distance from X-Plane ``apt.dat`` (2026-09-24).

The rank-3 team uses the X-Plane Scenery Gateway airport files for stand lists and positions.
They cover some of the stands OpenStreetMap misses (EDDM, LEBL) and carry per-stand
attributes OSM does not have. This block adds those as numeric columns; it is measured by one
twin, ``exp0250`` against ``exp0213``.

Data: the recommended gateway pack per airport (``uv run python scripts/fetch_external.py
xplane`` → ``data/external/xplane_gateway_<date>/<ICAO>.dat``). **Licence: GPL v2 or later**
(the pack's README / COPYING; ``docs/data-sources.md``). Runway thresholds come
from the OurAirports ``runways.csv`` pull (public domain) that the ADS-B build uses; the gap
filler reads the OSM ``taxi_distance`` table (ODbL).

``apt.dat`` rows read, per stand:

* ``1300 lat lon heading type aircraft_types name`` — ``type`` is ``gate`` / ``tie_down`` /
  ``hangar`` / ``misc``; ``heading`` is the aircraft's true heading when parked;
* ``1301 size operation_type airlines`` (the line right after its ``1300``) — ICAO size class
  ``A``–``F``, ``airline`` / ``cargo`` / ``general_aviation`` / ``military`` / ``none``, and
  a space-separated list of airline codes.

**Name parsing.** ``STAND_mvt`` meets the X-Plane name through :func:`xp_stand_key`: a
trailing parenthesised note is dropped (LTFM's size suffix ``A2-(F)`` / ``A2L-(C)``, EGLL's
``325(constr)``), a leading word ``Stand`` / ``Gate`` / ``Ramp`` / ``Parking`` / ``Position``
is dropped (LSZH ``Stand W 42`` / ``Gate B36``, LEBL ``Ramp 175``), and the rest goes through
``taxi_distance.norm_stand`` (spaces, punctuation and leading zeros removed). The sub-position
fallback is ``taxi_distance.base_stand`` (``115A`` → ``115``), applied to both sides, as in
``taxi_distance``. Where several X-Plane stands share one key, positions are averaged, the
size class is the largest and type / operation / heading are the first stand's.

Columns (all numeric; ``exp0197`` showed categorical apron columns hurt):

=========================  ==================================================================
``xp_stand_matched``       2 exact, 1 through the sub-position fallback, 0 no X-Plane stand
``xp_size_ord``            ICAO size class, A = 1 … F = 6
``xp_stand_type``          gate 1, tie_down 2, hangar 3, misc 4
``xp_op_type``             none 0, airline 1, cargo 2, general_aviation 3, military 4
``xp_n_airlines``          number of airline codes on the stand
``xp_straight_m``          great-circle stand → threshold of ``RUNWAY_mvt`` (OurAirports)
``xp_heading_to_rwy_deg``  |parked heading − bearing stand → threshold|, 0–180
``xp_fill_straight_m``     OSM ``td_straight_m`` where OSM matched the stand, else
                           ``xp_straight_m``: the X-Plane distance fills the OSM gaps
=========================  ==================================================================

Everything is static geometry and stand metadata: no timestamps, no labels. The block uses no
NM field (as ``taxi_distance``, it is not in ``mask_for_no_nm``). Unmatched stands give nulls,
never a guessed coordinate. Returns all-null columns when the X-Plane archive is absent.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import polars as pl

from prc.data import schema as S
from prc.features.registry import archive_files, external_digest, feature
from prc.features.taxi_distance import _td_external, base_stand, norm_runway, norm_stand

EARTH_R = 6_371_000.0

OUT = [
    "xp_stand_matched",
    "xp_size_ord",
    "xp_stand_type",
    "xp_op_type",
    "xp_n_airlines",
    "xp_straight_m",
    "xp_heading_to_rwy_deg",
    "xp_fill_straight_m",
]

SIZE_ORD = {c: i + 1 for i, c in enumerate("ABCDEF")}
STAND_TYPE = {"gate": 1, "tie_down": 2, "hangar": 3, "misc": 4}
OP_TYPE = {"none": 0, "airline": 1, "cargo": 2, "general_aviation": 3, "military": 4}

_SUFFIX = re.compile(r"\s*-?\s*\([^)]*\)\s*$")
_PREFIX = re.compile(r"^(stand|gate|ramp|parking|position|pos)\b[\s.:#-]*", re.IGNORECASE)


def _xp_files() -> list[Path]:
    return archive_files("xplane_gateway_*", "*.dat")


def _runways_file() -> list[Path]:
    return [f for f in archive_files("ourairports_*", "runways.csv") if f.is_file()]


def _xp_external() -> str:
    """X-Plane ``.dat`` files + OurAirports ``runways.csv`` + this module's source, and
    ``taxi_distance``'s own external key (OSM files + its module), which the gap filler reads."""
    own = external_digest(_xp_files() + _runways_file(), __name__)
    return hashlib.sha256(f"{own}|{_td_external()}".encode()).hexdigest()[:8]


# --------------------------------------------------------------------------- parsing


def xp_stand_key(name: str | None) -> str | None:
    """``A2-(F)`` -> ``A2``; ``Stand W 42`` -> ``W42``; ``Gate B36`` -> ``B36``;
    ``Ramp 157A`` -> ``157A``; ``K01-(E)`` -> ``K1``; ``325(constr)`` -> ``325``."""
    if name is None:
        return None
    s = _SUFFIX.sub("", str(name).strip())
    s = _PREFIX.sub("", s).strip() or s
    return norm_stand(s)


def parse_apt_dat(text: str) -> list[dict]:
    """The stands (``1300`` + following ``1301``) of one ``apt.dat`` airport record."""
    stands: list[dict] = []
    cur: dict | None = None
    for line in text.splitlines():
        p = line.split()
        if not p:
            continue
        if p[0] == "1300" and len(p) >= 7:
            try:
                lat, lon, hdg = float(p[1]), float(p[2]), float(p[3])
            except ValueError:
                cur = None
                continue
            cur = {
                "name": " ".join(p[6:]),
                "lat": lat,
                "lon": lon,
                "heading": hdg % 360.0,
                "stand_type": STAND_TYPE.get(p[4]),
                "size_ord": None,
                "op_type": None,
                "n_airlines": None,
            }
            stands.append(cur)
        elif p[0] == "1301" and cur is not None:
            cur["size_ord"] = SIZE_ORD.get(p[1].upper()) if len(p) > 1 else None
            cur["op_type"] = OP_TYPE.get(p[2]) if len(p) > 2 else None
            cur["n_airlines"] = len(p) - 3 if len(p) > 3 else 0
            cur = None
        else:
            cur = None
    return stands


def _stand_table() -> pl.DataFrame | None:
    """One row per (airport, X-Plane stand) with its match keys."""
    rows: list[dict] = []
    for f in _xp_files():
        for st in parse_apt_dat(f.read_text(encoding="utf-8", errors="replace")):
            key = xp_stand_key(st["name"])
            if key:
                rows.append({S.AIRPORT: f.stem, "_key": key, **st})
    if not rows:
        return None
    return pl.DataFrame(rows, infer_schema_length=None).with_columns(
        pl.col("_key").map_elements(base_stand, return_dtype=pl.Utf8).alias("_key_base")
    )


def _collapse(tbl: pl.DataFrame, key: str) -> pl.DataFrame:
    return tbl.group_by([S.AIRPORT, key], maintain_order=True).agg(
        pl.col("lat").mean(),
        pl.col("lon").mean(),
        pl.col("heading").first(),
        pl.col("size_ord").max(),
        pl.col("stand_type").first(),
        pl.col("op_type").first(),
        pl.col("n_airlines").first(),
    )


def _thresholds() -> pl.DataFrame:
    """(airport, normalised designator) -> threshold lat/lon from OurAirports ``runways.csv``."""
    schema = {S.AIRPORT: pl.Utf8, "_runway": pl.Utf8, "thr_lat": pl.Float64, "thr_lon": pl.Float64}
    files = _runways_file()
    if not files:
        return pl.DataFrame(schema=schema)
    rw = pl.read_csv(files[0], infer_schema_length=0).filter(
        pl.col("airport_ident").is_in(list(S.AIRPORTS))
    )
    rows: list[dict] = []
    for r in rw.sort(["airport_ident", "closed", "id"]).iter_rows(named=True):
        for side in ("le", "he"):
            d = norm_runway(r[f"{side}_ident"])
            try:
                lat = float(r[f"{side}_latitude_deg"])
                lon = float(r[f"{side}_longitude_deg"])
            except (TypeError, ValueError):
                continue
            if d:
                rows.append(
                    {S.AIRPORT: r["airport_ident"], "_runway": d, "thr_lat": lat, "thr_lon": lon}
                )
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema).unique(
        subset=[S.AIRPORT, "_runway"], keep="first", maintain_order=True
    )


def _haversine(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _bearing(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


def _null_frame(df: pl.DataFrame) -> pl.DataFrame:
    return df.select(S.MVT_ID).with_columns([pl.lit(None, dtype=pl.Float32).alias(c) for c in OUT])


def _osm_straight(df: pl.DataFrame) -> pl.DataFrame:
    """``td_straight_m`` / ``td_stand_matched`` per row, from the ``taxi_distance`` function
    (called, not edited; its cache is untouched)."""
    from prc.features.taxi_distance import taxi_distance

    return taxi_distance(df).select(S.MVT_ID, "td_straight_m", "td_stand_matched")


@feature("stand_xplane", external=_xp_external)
def stand_xplane(df: pl.DataFrame) -> pl.DataFrame:
    """X-Plane Scenery Gateway stand attributes (size class, stand type, operation type,
    airline count), straight-line stand → runway-threshold distance and the parked heading
    relative to it, and an OSM distance with its gaps filled by X-Plane (GPL v2+)."""
    stands = _stand_table()
    if stands is None:
        return _null_frame(df)
    exact = _collapse(stands, "_key").rename({"_key": "_stand"})
    base = _collapse(stands, "_key_base").rename({"_key_base": "_stand_base"})
    attrs = ["lat", "lon", "heading", "size_ord", "stand_type", "op_type", "n_airlines"]

    keys = (
        df.select(
            S.AIRPORT,
            pl.col(S.STAND).cast(pl.Utf8).fill_null("").alias("_stand_raw"),
            pl.col(S.RUNWAY).cast(pl.Utf8).fill_null("").alias("_runway_raw"),
        )
        .unique()
        .with_columns(
            pl.col("_stand_raw").map_elements(norm_stand, return_dtype=pl.Utf8).alias("_stand"),
            pl.col("_runway_raw").map_elements(norm_runway, return_dtype=pl.Utf8).alias("_runway"),
        )
        .with_columns(
            pl.col("_stand").map_elements(base_stand, return_dtype=pl.Utf8).alias("_stand_base")
        )
        .join(
            exact.with_columns(pl.lit(2, pl.Int8).alias("_m2")),
            on=[S.AIRPORT, "_stand"],
            how="left",
        )
        .join(
            base.with_columns(pl.lit(1, pl.Int8).alias("_m1")),
            on=[S.AIRPORT, "_stand_base"],
            how="left",
            suffix="_b",
        )
        .with_columns(
            [pl.coalesce(pl.col(c), pl.col(f"{c}_b")).alias(c) for c in attrs]
            + [pl.coalesce(pl.col("_m2"), pl.col("_m1"), pl.lit(0, pl.Int8)).alias("_matched")]
        )
        .join(_thresholds(), on=[S.AIRPORT, "_runway"], how="left")
    )
    lat, lon = keys["lat"].to_numpy(), keys["lon"].to_numpy()
    tlat, tlon = keys["thr_lat"].to_numpy(), keys["thr_lon"].to_numpy()
    with np.errstate(invalid="ignore"):
        straight = _haversine(lat, lon, tlat, tlon)
        diff = np.abs(
            ((keys["heading"].to_numpy() - _bearing(lat, lon, tlat, tlon) + 180.0) % 360.0) - 180.0
        )
    keys = keys.with_columns(
        pl.Series("xp_straight_m", straight, dtype=pl.Float64).fill_nan(None),
        pl.Series("xp_heading_to_rwy_deg", diff, dtype=pl.Float64).fill_nan(None),
    )

    out = (
        df.select(
            S.MVT_ID,
            S.AIRPORT,
            pl.col(S.STAND).cast(pl.Utf8).fill_null("").alias("_stand_raw"),
            pl.col(S.RUNWAY).cast(pl.Utf8).fill_null("").alias("_runway_raw"),
        )
        .join(keys, on=[S.AIRPORT, "_stand_raw", "_runway_raw"], how="left")
        .join(_osm_straight(df), on=S.MVT_ID, how="left")
    )
    osm_ok = pl.col("td_stand_matched").fill_null(0) > 0
    return out.select(
        S.MVT_ID,
        pl.col("_matched").fill_null(0).cast(pl.Float32).alias("xp_stand_matched"),
        pl.col("size_ord").cast(pl.Float32).alias("xp_size_ord"),
        pl.col("stand_type").cast(pl.Float32).alias("xp_stand_type"),
        pl.col("op_type").cast(pl.Float32).alias("xp_op_type"),
        pl.col("n_airlines").cast(pl.Float32).alias("xp_n_airlines"),
        pl.col("xp_straight_m").cast(pl.Float32),
        pl.col("xp_heading_to_rwy_deg").cast(pl.Float32),
        pl.when(osm_ok & pl.col("td_straight_m").is_not_null())
        .then(pl.col("td_straight_m"))
        .otherwise(pl.col("xp_straight_m"))
        .cast(pl.Float32)
        .alias("xp_fill_straight_m"),
    )
