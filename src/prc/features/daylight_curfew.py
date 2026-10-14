"""Daylight, local time and night-curfew structure (idea backlog B3, TODO 4c).

Computed, not downloaded: nothing under ``data/external/`` is read, so this block builds
identically on any machine and nothing has to be shipped to the cluster for it.

Two mechanisms, neither of which the UTC clock features (``calendar``) can express.

**Daylight.** Visibility of the manoeuvring area, the use of follow-me vehicles and low-sun
runway-crossing procedures all change at dawn and dusk, and dawn moves by three hours
between January and July — exactly the two months that are scored. Solar elevation, minutes
since sunrise and minutes to sunset are the direct quantities; ``hour`` is a poor proxy for
them because it means something different in each month and at each airport (LTFM and EGLL
are 28° of longitude apart).

**Curfew.** Four of the ten airports have a published night ban. A ban does not just remove
movements: it *creates a queue*. Everything that wants to leave before the ban starts
compresses into the last half hour, and everything that could not leave overnight pushes back
together at the end of the ban. Minutes-since-curfew-end and minutes-to-curfew-start name
those two banks directly.

Published curfew hours used (local time, departures; retrieved 2026-09-19):

======  =============  ==========================================================
ICAO    ban (local)    source
======  =============  ==========================================================
EDDF    23:00 – 05:00  Fraport, "Runway system and operating hours" — absolute
                       night-flight ban since 2011, no scheduled take-offs or
                       landings 23:00–05:00.
                       https://www.fraport.com/en/sustainability/dialog-with-neighbors/noise-and-air/flight-operations/runway-system-and-operating-hours.html
EDDM    00:00 – 05:00  Munich Airport, "Night flight — strict rules for a quiet
                       sleep": the *core night* runs from midnight to 05:00; the
                       22:00–24:00 and 05:00–06:00 margins are restricted but not
                       closed, so only the core is used here.
                       https://www.munich-airport.com/night-flight-264466
EGLL    23:00 – 06:00  Heathrow, "Night flights": no departures are *scheduled*
                       between 23:00 and 06:00 (the Night Quota Period itself is
                       23:30–06:00; the departure-relevant edge is 23:00).
                       https://www.heathrow.com/company/local-community/noise/operations/night-flights
LSZH    23:30 – 06:00  Zurich Airport, "Flight operations": operating hours
                       06:00–23:30, with 23:00–23:30 reserved for clearing
                       delayed departures.
                       https://www.flughafen-zuerich.ch/en/company/responsibility/noise-and-sound-insulation/flight-operations
======  =============  ==========================================================

**The other six airports are deliberately left out.** LEBL, LEMD, LFPG, LIRF, LTFM and EHAM
all have night *restrictions* (noise quotas, movement caps, aircraft-type bans) rather than a
clean published window, and guessing a window would put a wrong number in front of the model.
``dc_has_curfew`` says which airports carry the feature, so a tree can tell "no curfew" from
"curfew not known"; the curfew columns are null at the other six.

TIMESTAMP. Everything is evaluated at ``SCHED_TIME_UTC_mvt`` for departures (the curfew
constrains the *schedule*, and the schedule is the one pre-departure stamp that is present on
every row and observable at test time), falling back to ``MVT_TIME_UTC_mvt``; arrivals use
their landing time. ``BLOCK_TIME_UTC_mvt`` is never read. The block deliberately does not use
``AOBT_3_flt``, so it has no NM dependency and does not belong in ``mask_for_no_nm``.

Solar position is the NOAA Solar Calculator's low-precision algorithm (equation of time +
declination from the fractional year), accurate to well under a degree — far finer than the
half-hour structure it is used to describe.
"""

from __future__ import annotations

import math

import polars as pl

from prc.data import schema as S
from prc.features.registry import feature

#: Airport coordinates, from OurAirports (public domain), captured 2026-09-17 — the same
#: table ``scripts/fetch_external.py`` uses for the Open-Meteo pull.
AIRPORT_LATLON: dict[str, tuple[float, float]] = {
    "EDDF": (50.026706, 8.55835),
    "EDDM": (48.353802, 11.7861),
    "EGLL": (51.470748, -0.459909),
    "EHAM": (52.308601, 4.76389),
    "LEBL": (41.2971, 2.07846),
    "LEMD": (40.493407, -3.572249),
    "LFPG": (49.00896, 2.554117),
    "LIRF": (41.804532, 12.251998),
    "LSZH": (47.458056, 8.548056),
    "LTFM": (41.274874, 28.732136),
}

#: IANA zone per airport (all ten are unambiguous).
AIRPORT_TZ: dict[str, str] = {
    "EDDF": "Europe/Berlin",
    "EDDM": "Europe/Berlin",
    "EGLL": "Europe/London",
    "EHAM": "Europe/Amsterdam",
    "LEBL": "Europe/Madrid",
    "LEMD": "Europe/Madrid",
    "LFPG": "Europe/Paris",
    "LIRF": "Europe/Rome",
    "LSZH": "Europe/Zurich",
    "LTFM": "Europe/Istanbul",
}

#: (curfew start, curfew end) in local minutes of day, for the four airports with a
#: published window. See the module docstring for the sources.
CURFEW: dict[str, tuple[int, int]] = {
    "EDDF": (23 * 60, 5 * 60),
    "EDDM": (0, 5 * 60),  # the core night starts at midnight, so this window does not wrap
    "EGLL": (23 * 60, 6 * 60),
    "LSZH": (23 * 60 + 30, 6 * 60),
}

#: Minutes-since / minutes-to are capped: beyond this the row is simply "mid-day".
CAP_MIN = 720.0
#: Standard refraction-corrected solar zenith of sunrise / sunset.
SUNRISE_ZENITH_DEG = 90.833

OUT = [
    "dc_solar_elev_deg",
    "dc_is_day",
    "dc_min_since_sunrise",
    "dc_min_to_sunset",
    "dc_daylength_min",
    "dc_local_minute_of_day",
    "dc_local_hour",
    "dc_has_curfew",
    "dc_in_curfew",
    "dc_min_since_curfew_end",
    "dc_min_to_curfew_start",
]


def _reference_time() -> pl.Expr:
    """Scheduled off-block for departures, landing time for arrivals; never the label."""
    return (
        pl.when(pl.col(S.PHASE) == "DEP")
        .then(pl.coalesce(pl.col(S.SCHED_TIME), pl.col(S.MVT_TIME)))
        .otherwise(pl.col(S.MVT_TIME))
    )


def _solar(ref: pl.Expr, lat: float, lon: float) -> dict[str, pl.Expr]:
    """NOAA low-precision solar position / sunrise / sunset at one airport.

    ``ref`` is a UTC datetime expression. Returns elevation in degrees and sunrise / sunset
    as minutes-of-day UTC for the date of ``ref``.
    """
    lat_r = math.radians(lat)
    doy = ref.dt.ordinal_day().cast(pl.Float64)
    # NB: dt.hour() is Int8 — multiplying before the cast silently wraps (04:02 -> -14).
    minute_utc = ref.dt.hour().cast(pl.Float64) * 60.0 + ref.dt.minute().cast(pl.Float64)
    # fractional year (radians)
    g = 2.0 * math.pi / 365.0 * (doy - 1.0 + (minute_utc / 60.0 - 12.0) / 24.0)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * g.cos()
        - 0.032077 * g.sin()
        - 0.014615 * (2 * g).cos()
        - 0.040849 * (2 * g).sin()
    )
    decl = (
        0.006918
        - 0.399912 * g.cos()
        + 0.070257 * g.sin()
        - 0.006758 * (2 * g).cos()
        + 0.000907 * (2 * g).sin()
        - 0.002697 * (3 * g).cos()
        + 0.00148 * (3 * g).sin()
    )
    true_solar_min = minute_utc + eqtime + 4.0 * lon
    ha_deg = true_solar_min / 4.0 - 180.0
    ha_r = ha_deg * (math.pi / 180.0)
    cos_zen = math.sin(lat_r) * decl.sin() + math.cos(lat_r) * decl.cos() * ha_r.cos()
    elev = (math.pi / 2.0 - cos_zen.clip(-1.0, 1.0).arccos()) * (180.0 / math.pi)
    # sunrise / sunset hour angle for this date
    cos_ha0 = (
        math.cos(math.radians(SUNRISE_ZENITH_DEG)) / (math.cos(lat_r) * decl.cos())
        - math.tan(lat_r) * decl.tan()
    )
    ha0_deg = cos_ha0.clip(-1.0, 1.0).arccos() * (180.0 / math.pi)
    sunrise = 720.0 - 4.0 * (lon + ha0_deg) - eqtime
    sunset = 720.0 - 4.0 * (lon - ha0_deg) - eqtime
    return {
        "elev": elev,
        "sunrise": sunrise,
        "sunset": sunset,
        "minute_utc": minute_utc,
        "polar": cos_ha0.abs() > 1.0,
    }


def _one_airport(part: pl.DataFrame, icao: str) -> pl.DataFrame:
    lat, lon = AIRPORT_LATLON[icao]
    tz = AIRPORT_TZ[icao]
    ref = pl.col("_ref")
    sol = _solar(ref, lat, lon)
    local = ref.dt.convert_time_zone(tz)
    lmin = local.dt.hour().cast(pl.Float64) * 60.0 + local.dt.minute().cast(pl.Float64)
    out = part.with_columns(
        sol["elev"].cast(pl.Float32).alias("dc_solar_elev_deg"),
        (sol["elev"] > 0.0).cast(pl.Int8).alias("dc_is_day"),
        pl.when(sol["polar"])
        .then(None)
        .otherwise(sol["minute_utc"] - sol["sunrise"])
        .cast(pl.Float32)
        .alias("dc_min_since_sunrise"),
        pl.when(sol["polar"])
        .then(None)
        .otherwise(sol["sunset"] - sol["minute_utc"])
        .cast(pl.Float32)
        .alias("dc_min_to_sunset"),
        pl.when(sol["polar"])
        .then(None)
        .otherwise(sol["sunset"] - sol["sunrise"])
        .cast(pl.Float32)
        .alias("dc_daylength_min"),
        lmin.cast(pl.Float32).alias("dc_local_minute_of_day"),
        (lmin / 60.0).floor().cast(pl.Int8).alias("dc_local_hour"),
    )
    if icao not in CURFEW:
        return out.with_columns(
            pl.lit(0, dtype=pl.Int8).alias("dc_has_curfew"),
            pl.lit(None, dtype=pl.Int8).alias("dc_in_curfew"),
            pl.lit(None, dtype=pl.Float32).alias("dc_min_since_curfew_end"),
            pl.lit(None, dtype=pl.Float32).alias("dc_min_to_curfew_start"),
        )
    start, end = CURFEW[icao]
    m = pl.col("dc_local_minute_of_day").cast(pl.Float64)
    # EDDF / EGLL / LSZH close before midnight and reopen after it, so their window wraps;
    # EDDM's core night starts at midnight and does not.
    in_curfew = (
        ((m >= float(start)) | (m < float(end)))
        if start > end
        else ((m >= float(start)) & (m < float(end)))
    )
    return out.with_columns(
        pl.lit(1, dtype=pl.Int8).alias("dc_has_curfew"),
        in_curfew.cast(pl.Int8).alias("dc_in_curfew"),
        ((m - float(end)) % 1440.0)
        .clip(0.0, CAP_MIN)
        .cast(pl.Float32)
        .alias("dc_min_since_curfew_end"),
        ((float(start) - m) % 1440.0)
        .clip(0.0, CAP_MIN)
        .cast(pl.Float32)
        .alias("dc_min_to_curfew_start"),
    )


@feature("daylight_curfew")
def daylight_curfew(df: pl.DataFrame) -> pl.DataFrame:
    """Solar elevation, minutes since sunrise / to sunset, day length, timezone-aware local
    hour, and — at EDDF / EDDM / EGLL / LSZH — minutes since the night curfew ended and
    until it starts, evaluated at the scheduled off-block time."""
    base = df.select(S.MVT_ID, S.AIRPORT, _reference_time().alias("_ref"))
    parts = [
        _one_airport(part, part[S.AIRPORT][0])
        for part in base.partition_by(S.AIRPORT)
        if part[S.AIRPORT][0] in AIRPORT_LATLON
    ]
    known = pl.concat(parts).select(S.MVT_ID, *OUT) if parts else None
    out = df.select(S.MVT_ID)
    if known is None:
        return out.with_columns([pl.lit(None, dtype=pl.Float32).alias(c) for c in OUT])
    return out.join(known, on=S.MVT_ID, how="left")
