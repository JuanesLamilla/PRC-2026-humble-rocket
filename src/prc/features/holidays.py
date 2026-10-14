"""Public-holiday regime per airport (plan row P1-e).

Nager.Date public holidays (MIT licence), archived by
``scripts/fetch_external.py holidays`` to ``data/external/nager_holidays_<date>/<CC>_<year>.json``.
Holidays move the traffic mix (leisure peaks, thinner business banks) and the surface
regime with it; ``calendar`` already carries day-of-week and day-of-year, so this block is
expected to be worth ~0 and exists to falsify that expectation cheaply.

Regional holidays matter at a single-airport granularity (Epiphany is a public holiday in
Bavaria but not in Hesse), so each airport is mapped to its ISO-3166-2 region and a
holiday applies to an airport when it is national (``counties == null``) or lists that
region. School holidays are not in this source; the day-before / day-after flags are the
cheap stand-in for the travel peaks around a break.

Returns all-null columns when the archive is absent, so tests and fresh clones still run.
"""

from __future__ import annotations

import json

import polars as pl

from prc import paths
from prc.data import schema as S
from prc.features.registry import archive_files, external_digest, feature

# (ISO country as pulled by scripts/fetch_external.py COUNTRY, ISO-3166-2 region of the airport)
AIRPORT_REGION: dict[str, tuple[str, str | None]] = {
    "EDDF": ("DE", "DE-HE"),
    "EDDM": ("DE", "DE-BY"),
    "EGLL": ("GB", "GB-ENG"),
    "EHAM": ("NL", None),
    "LEBL": ("ES", "ES-CT"),
    "LEMD": ("ES", "ES-MD"),
    "LFPG": ("FR", None),
    "LIRF": ("IT", None),
    "LSZH": ("CH", "CH-ZH"),
    "LTFM": ("TR", None),
}

OUT = [
    "hol_is_holiday",
    "hol_days_to_next",
    "hol_days_since",
    "hol_day_before",
    "hol_day_after",
]


def _archive() -> pl.DataFrame | None:
    """(airport, holiday date) for the 10 airports, from the newest Nager archive."""
    dirs = sorted(paths.EXTERNAL.glob("nager_holidays_*"))
    if not dirs:
        return None
    by_country: dict[str, list[dict]] = {}
    for f in sorted(dirs[-1].glob("*.json")):
        if f.stem == "manifest":
            continue
        cc = f.stem.split("_")[0]
        with open(f) as fh:
            by_country.setdefault(cc, []).extend(json.load(fh))
    rows: list[dict[str, str]] = []
    for icao, (cc, region) in AIRPORT_REGION.items():
        for h in by_country.get(cc, []):
            counties = h.get("counties")
            if counties and (region is None or region not in counties):
                continue
            rows.append({S.AIRPORT: icao, "hol_date": h["date"]})
    if not rows:
        return None
    return (
        pl.DataFrame(rows)
        .with_columns(pl.col("hol_date").str.to_date("%Y-%m-%d"))
        .unique()
        .sort([S.AIRPORT, "hol_date"])
    )


#: Code review 2026-09-19, finding #4: this block reads ``data/external/`` and must say so,
#: or the cache cannot tell a build made without the archive from one made with it — the
#: 2026-09-18 cache incident. ``__name__`` also folds the module's own source in.


def _holidays_external() -> str:
    return external_digest(archive_files("nager_holidays_*", "*.json"), __name__)


@feature("holidays", external=_holidays_external)
def holidays(df: pl.DataFrame) -> pl.DataFrame:
    """Is the takeoff date a public holiday at the airport's region, days to the next and
    since the previous one, and day-before / day-after flags."""
    hol = _archive()
    if hol is None:
        return df.select(S.MVT_ID).with_columns(
            [pl.lit(None, dtype=pl.Float32).alias(c) for c in OUT]
        )
    rows = df.select(S.MVT_ID, S.AIRPORT, pl.col(S.MVT_TIME).dt.date().alias("d")).sort(
        [S.AIRPORT, "d"]
    )
    prev = hol.select(S.AIRPORT, pl.col("hol_date").alias("_prev"))
    nxt = hol.select(S.AIRPORT, pl.col("hol_date").alias("_next"))
    out = rows.join_asof(
        prev, left_on="d", right_on="_prev", by=S.AIRPORT, strategy="backward", coalesce=False
    ).join_asof(
        nxt, left_on="d", right_on="_next", by=S.AIRPORT, strategy="forward", coalesce=False
    )
    days_since = (pl.col("d") - pl.col("_prev")).dt.total_days().cast(pl.Float32)
    days_to_next = (pl.col("_next") - pl.col("d")).dt.total_days().cast(pl.Float32)
    return out.select(
        S.MVT_ID,
        ((days_since == 0) | (days_to_next == 0)).cast(pl.Float32).alias("hol_is_holiday"),
        days_to_next.alias("hol_days_to_next"),
        days_since.alias("hol_days_since"),
        (days_to_next == 1).cast(pl.Float32).alias("hol_day_before"),
        (days_since == 1).cast(pl.Float32).alias("hol_day_after"),
    )
