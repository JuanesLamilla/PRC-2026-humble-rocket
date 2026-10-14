"""Fetch open external datasets into data/external/<source>_<YYYY-MM-DD>/ with a
manifest of SHA-256 hashes, so every pull is archived and reproducible (prize rule:
external data must be open, documented, and the pull must be repeatable).

    uv run python scripts/fetch_external.py metar --years 2025 2026
    uv run python scripts/fetch_external.py eurocontrol --years 2025 2026
    uv run python scripts/fetch_external.py ourairports
    uv run python scripts/fetch_external.py holidays --years 2025 2026
    uv run python scripts/fetch_external.py openmeteo   # hourly reanalysis at the 10 airports

Sources (all verified reachable on 2026-09-04; licences in docs/data-sources.md):
  metar        Iowa Environmental Mesonet ASOS/METAR archive (30-min METARs per ICAO station)
  eurocontrol  ansperformance.eu performance CSVs: monthly additional taxi-out time, daily traffic,
               pre-departure delays, ATFM slot adherence, ASMA
  ourairports  runways.csv / airports.csv (public domain)
  holidays     Nager.Date public holidays per country (MIT)
  openmeteo    Open-Meteo historical weather API (ERA5/ICON reanalysis, hourly, CC BY 4.0)
  osm          OpenStreetMap aeroway geometry via the Overpass API (ODbL 1.0 — attribution
               required: "(c) OpenStreetMap contributors"): parking positions / gates,
               runways and taxiways at the ten airports, one polite query per airport
  bts          US BTS Reporting Carrier On-Time Performance, monthly PREZIP files (US public
               domain), streamed one month at a time into column-subset parquet for the
               US_AIRPORTS below; ZIP deleted after conversion (external-transfer test)
  metar_us     IEM ASOS/METAR for the US_AIRPORTS, same archive and CSV format as `metar`
               (routine + special reports only; the US default adds 5-minute HFMETAR)
  prc2024      PRC Data Challenge 2024 `flight_list.csv` from 4TU.ResearchData (CC BY 4.0);
               that one file only, never the daily trajectory parquets
  xplane       X-Plane Scenery Gateway `apt.dat`, the recommended pack per airport (GPL v2+);
               stand rows 1300/1301 feed the `stand_xplane` block
  faa_acd      FAA Aircraft Characteristics Database (one xlsx, US public domain); the
               `ACD_Data` sheet is also written as `faa_acd.csv` (stdlib parser, no new
               dependency) for the `aircraft_chars` block

    uv run python scripts/fetch_external.py bts          # the approved 34 months
    uv run python scripts/fetch_external.py metar_us --years 2019 2022 2023 2024 2025 2026
    uv run python scripts/fetch_external.py prc2024
    uv run python scripts/fetch_external.py xplane
    uv run python scripts/fetch_external.py faa_acd
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from prc import paths
from prc.data.schema import AIRPORTS

COUNTRY = {
    "EDDF": "DE",
    "EDDM": "DE",
    "EGLL": "GB",
    "EHAM": "NL",
    "LEBL": "ES",
    "LEMD": "ES",
    "LFPG": "FR",
    "LIRF": "IT",
    "LTFM": "TR",
    "LSZH": "CH",
}

IEM = (
    "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station={st}&data=all"
    "&year1={y}&month1=1&day1=1&year2={y2}&month2=1&day2=1&tz=Etc/UTC&format=onlycomma"
    "&latlon=no&missing=M&direct=no"
)  # NOTE: do not add report_type= — it drops EHAM/LEBL/LFPG/LIRF
EC = "https://www.eurocontrol.int/performance/data/download/csv/{ds}_{y}.csv"
EC_DATASETS = (
    "taxi_out_additional_time",
    "airport_traffic",
    "atc_pre_departure_delays",
    "all_pre_departure_delays",
    "atfm_slot_adherence",
    "asma_additional_time",
)
OURAIRPORTS = {
    "runways.csv": "https://davidmegginson.github.io/ourairports-data/runways.csv",
    "airports.csv": "https://davidmegginson.github.io/ourairports-data/airports.csv",
}
NAGER = "https://date.nager.at/api/v3/PublicHolidays/{y}/{cc}"
# Coordinates from OurAirports (public domain), captured 2026-09-17.
AIRPORT_LATLON = {
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
OPENMETEO_VARS = (
    "temperature_2m,relative_humidity_2m,dew_point_2m,apparent_temperature,precipitation,rain,"
    "snowfall,snow_depth,weather_code,pressure_msl,surface_pressure,cloud_cover,cloud_cover_low,"
    "wind_speed_10m,wind_direction_10m,wind_gusts_10m,vapour_pressure_deficit,is_day"
)
OPENMETEO = (
    "https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
    "&start_date={d1}&end_date={d2}&hourly={vars}&timezone=UTC&wind_speed_unit=kmh&format=json"
)


OVERPASS = "https://overpass-api.de/api/interpreter"
#: Half-width of the bounding box around each airport reference point, in degrees. LTFM and
#: LFPG are the widest fields (~6 km across); 0.09 deg of latitude is ~10 km, and the
#: longitude half-width is widened by 1/cos(lat) so the box is square on the ground.
OSM_BOX_DEG = 0.09
#: Overpass is a shared volunteer service: one query per airport, a pause between them, and
#: an identifying User-Agent (their usage policy asks for all three).
OSM_PAUSE_S = 12.0
OSM_QUERY = """[out:json][timeout:180];
(
  node["aeroway"="parking_position"]({bbox});
  way["aeroway"="parking_position"]({bbox});
  node["aeroway"="gate"]({bbox});
  way["aeroway"="runway"]({bbox});
  way["aeroway"="taxiway"]({bbox});
  way["aeroway"="apron"]({bbox});
);
out body geom;
"""


def _post(url: str, data: str, dest: Path) -> dict:
    body = urllib.parse.urlencode({"data": data}).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "User-Agent": "prc-challenge/0.1 (PRC Data Challenge 2026; prc-data-challenge-2026-humble-rocket)",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        raw = r.read()
    dest.write_bytes(raw)
    return {
        "file": dest.name,
        "url": url,
        "query": data,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _get(url: str, dest: Path) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "prc-challenge-fetch/0.1"})
    with urllib.request.urlopen(req, timeout=300) as r:
        data = r.read()
    dest.write_bytes(data)
    return {
        "file": dest.name,
        "url": url,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _outdir(source: str) -> Path:
    d = paths.EXTERNAL / f"{source}_{datetime.now(UTC):%Y-%m-%d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _manifest(outdir: Path, entries: list[dict], note: str) -> None:
    (outdir / "manifest.json").write_text(
        json.dumps(
            {
                "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "note": note,
                "files": entries,
            },
            indent=2,
        )
    )
    print(f"{len(entries)} files -> {outdir}")


def metar(years: list[int]) -> None:
    out = _outdir("metar_iem")
    entries = []
    for st in AIRPORTS:
        for y in years:
            entries.append(_get(IEM.format(st=st, y=y, y2=y + 1), out / f"{st}_{y}.csv"))
            print("  ", entries[-1]["file"], entries[-1]["bytes"])
    _manifest(
        out,
        entries,
        "IEM ASOS/METAR archive; vsby saturates at 6.21 SM, parse the raw `metar` column",
    )


def eurocontrol(years: list[int]) -> None:
    out = _outdir("eurocontrol_perf")
    entries = []
    for ds in EC_DATASETS:
        for y in years:
            try:
                entries.append(_get(EC.format(ds=ds, y=y), out / f"{ds}_{y}.csv"))
                print("  ", entries[-1]["file"], entries[-1]["bytes"])
            except Exception as e:  # noqa: BLE001 — a missing year must not abort the others
                print("   skip", ds, y, e)
    _manifest(out, entries, "ansperformance.eu open performance CSVs (monthly/daily per airport)")


def ourairports() -> None:
    out = _outdir("ourairports")
    entries = [_get(u, out / n) for n, u in OURAIRPORTS.items()]
    _manifest(out, entries, "OurAirports, public domain")


def holidays(years: list[int]) -> None:
    out = _outdir("nager_holidays")
    entries = []
    for cc in sorted(set(COUNTRY.values())):
        for y in years:
            entries.append(_get(NAGER.format(y=y, cc=cc), out / f"{cc}_{y}.json"))
    _manifest(out, entries, "Nager.Date public holidays (MIT); includes regional `counties`")


def openmeteo(start: str = "2024-12-31", end: str = "2026-07-31") -> None:
    """One JSON per airport (hourly, UTC). ERA5 reanalysis is ~10 km grid model data, not the
    airport's own METAR — good for temperature / wind / precipitation, weak for fog."""
    out = _outdir("openmeteo")
    entries = []
    for icao, (lat, lon) in AIRPORT_LATLON.items():
        url = OPENMETEO.format(lat=lat, lon=lon, d1=start, d2=end, vars=OPENMETEO_VARS)
        entries.append(_get(url, out / f"{icao}.json"))
    _manifest(
        out,
        entries,
        "Open-Meteo historical archive API, CC BY 4.0 (https://open-meteo.com/en/terms)",
    )


def osm() -> None:
    """One Overpass query per airport into ``data/external/osm_<date>/<ICAO>.json``.

    ``out body geom`` inlines every way's node coordinates, so the block needs no second
    pass to resolve node references. Licence: ODbL 1.0; the attribution required by the
    licence is recorded in the archive's README and in docs/data-sources.md.
    """
    import math
    import time

    out = _outdir("osm")
    entries = []
    for i, (icao, (lat, lon)) in enumerate(AIRPORT_LATLON.items()):
        dest = out / f"{icao}.json"
        if dest.is_file() and dest.stat().st_size > 1000:  # resume a partially-fetched pull
            raw = dest.read_bytes()
            entries.append(
                {
                    "file": dest.name,
                    "url": OVERPASS,
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
            print("   have", dest.name, len(raw))
            continue
        dlon = OSM_BOX_DEG / max(math.cos(math.radians(lat)), 0.1)
        bbox = f"{lat - OSM_BOX_DEG:.5f},{lon - dlon:.5f},{lat + OSM_BOX_DEG:.5f},{lon + dlon:.5f}"
        # Overpass answers 429 / 504 when it is loaded; back off rather than hammering it.
        for attempt in range(5):
            if i or attempt:
                time.sleep(OSM_PAUSE_S * (attempt + 1))
            try:
                entries.append(_post(OVERPASS, OSM_QUERY.format(bbox=bbox), dest))
                break
            except Exception as e:  # noqa: BLE001 — retryable by design
                print(f"   {icao} attempt {attempt + 1}: {e}")
        else:
            raise SystemExit(f"Overpass refused {icao} five times; re-run later to resume")
        print("  ", entries[-1]["file"], entries[-1]["bytes"])
    (out / "README.md").write_text(OSM_README)
    _manifest(
        out,
        entries,
        "OpenStreetMap via Overpass API. (c) OpenStreetMap contributors, ODbL 1.0 "
        "(https://www.openstreetmap.org/copyright). Attribution required on any published use.",
    )


OSM_README = """# OpenStreetMap aeroway extract (Overpass API)

**Source:** https://overpass-api.de/api/interpreter — one query per airport, bounding box
+/- 0.09 deg latitude around the OurAirports reference point (longitude widened by
1/cos(lat)). Query and per-file SHA-256 in `manifest.json`.

**Contents:** `aeroway=parking_position` (nodes and ways), `aeroway=gate`,
`aeroway=runway`, `aeroway=taxiway`, `aeroway=apron`, with inline geometry (`out body geom`).

**Licence:** Open Database License (ODbL) v1.0.
**Attribution (required):** "(c) OpenStreetMap contributors", data available under the
Open Database License, https://www.openstreetmap.org/copyright.

Any published model or report using `taxi_distance` must carry that attribution, and any
redistributed derived database must be offered under ODbL. Registered in
`docs/data-sources.md`.

**Consumed by:** `src/prc/features/taxi_distance.py` (block `taxi_distance`). The block
hashes every file in this directory into its feature-cache key, so rebuilding against a
different pull cannot silently reuse this one's parquet.
"""


# --------------------------------------------------------------------------- X-Plane gateway

XPLANE_API = "https://gateway.x-plane.com/apiv1"


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "prc-challenge-fetch/0.1"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def xplane_extract(blob_b64: str, icao: str) -> tuple[bytes, bytes | None, bytes | None]:
    """``masterZipBlob`` (base64 zip) -> (``<ICAO>.dat``, pack ``README.txt``, ``COPYING``).

    The outer zip holds ``<ICAO>.dat`` (the airport's ``apt.dat`` record), ``<ICAO>.txt``
    (the DSF as text) and ``<ICAO>_Scenery_Pack.zip``; the README and the GPL text that WED
    writes into every upload live in that inner pack. Only the ``.dat`` is read by a block.
    """
    import base64
    import zipfile

    outer = zipfile.ZipFile(io.BytesIO(base64.b64decode(blob_b64)))
    dat = outer.read(f"{icao}.dat")
    readme = copying = None
    pack = f"{icao}_Scenery_Pack.zip"
    if pack in outer.namelist():
        inner = zipfile.ZipFile(io.BytesIO(outer.read(pack)))
        for n in inner.namelist():
            if n.endswith("/README.txt"):
                readme = inner.read(n)
            elif n.endswith("/COPYING"):
                copying = inner.read(n)
    return dat, readme, copying


def xplane() -> None:
    """The recommended X-Plane Scenery Gateway pack per airport ->
    ``data/external/xplane_gateway_<date>/<ICAO>.dat`` (block ``stand_xplane``).

    ``GET /apiv1/airport/<ICAO>`` names the ``recommendedSceneryId``; ``GET
    /apiv1/scenery/<id>`` returns that upload with its zip as base64. Scenery ids are
    immutable uploads, so the ids in the manifest re-fetch the same bytes later even after
    the recommended pack moves on. Licence: GPL v2 or later (the pack's README and COPYING,
    kept here as ``<ICAO>_README.txt`` and ``COPYING``).
    """
    out = _outdir("xplane_gateway")
    entries = []
    copying = None
    for icao in AIRPORTS:
        ap = _get_json(f"{XPLANE_API}/airport/{icao}")
        ap = ap.get("airport", ap)
        sid = ap["recommendedSceneryId"]
        url = f"{XPLANE_API}/scenery/{sid}"
        sc = _get_json(url)
        sc = sc.get("scenery", sc)
        dat, readme, cp = xplane_extract(sc["masterZipBlob"], icao)
        (out / f"{icao}.dat").write_bytes(dat)
        if readme is not None:
            (out / f"{icao}_README.txt").write_bytes(readme)
        copying = copying or cp
        entries.append(
            {
                "file": f"{icao}.dat",
                "url": url,
                "icao": icao,
                "scenery_id": sid,
                "parent_id": sc.get("parentId"),
                "uploader": sc.get("userName"),
                "date_uploaded": sc.get("dateUploaded"),
                "date_approved": sc.get("dateApproved"),
                "zip_sha256": hashlib.sha256(sc["masterZipBlob"].encode()).hexdigest(),
                "bytes": len(dat),
                "sha256": hashlib.sha256(dat).hexdigest(),
            }
        )
        print(f"   {icao} scenery {sid} ({sc.get('userName')}, {sc.get('dateApproved')})")
        time.sleep(1.0)  # polite
    if copying is not None:
        (out / "COPYING").write_bytes(copying)
    _manifest(
        out,
        entries,
        "X-Plane Scenery Gateway, recommended scenery per airport. GPL v2 or later: 'The "
        "scenery packs shared via the X-Plane Scenery Gateway are free software; you can "
        "redistribute it and/or modify it under the terms of the GNU General Public License as "
        "published by the Free Software Foundation; either version 2 of the License, or (at "
        "your option) any later version.' (pack README.txt). zip_sha256 is over the base64 blob.",
    )


# --------------------------------------------------------------------------- external transfer

BTS = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{y}_{m}.zip"
)
#: Months approved for download: Dec, Jan, Feb, Jun, Jul, Aug of 2019 and 2022-2025, plus
#: Jan, Feb, Jun, Jul 2026 (BTS publishes with a ~2-month lag; July 2026 is the latest).
BTS_MONTHS = tuple(
    [f"{y}-{m:02d}" for y in (2019, 2022, 2023, 2024, 2025) for m in (1, 2, 6, 7, 8, 12)]
    + ["2026-01", "2026-02", "2026-06", "2026-07"]
)
#: Kept columns under their names in the PREZIP header (verified on the 2025-01 file,
#: 2026-09-24). The brief's ``Flight_Number`` is ``Flight_Number_Reporting_Airline``.
BTS_COLS = (
    "FlightDate",
    "Reporting_Airline",
    "Tail_Number",
    "Flight_Number_Reporting_Airline",
    "Origin",
    "Dest",
    "CRSDepTime",
    "DepTime",
    "DepDelay",
    "TaxiOut",
    "WheelsOff",
    "WheelsOn",
    "TaxiIn",
    "Cancelled",
    "Diverted",
    "CarrierDelay",
    "WeatherDelay",
    "NASDelay",
    "LateAircraftDelay",
    "Distance",
)
#: The 30 largest origin airports by reporting-carrier departures in 2025-01, the first month
#: pulled (MIA, #15, replaced a guessed IAD, #36). Every month's manifest entry records the
#: departures of all origins, so the choice can be re-checked. Values are the IANA time zones
#: for the local-hhmm -> UTC conversion.
US_AIRPORTS = {
    "ATL": "America/New_York",
    "DFW": "America/Chicago",
    "DEN": "America/Denver",
    "ORD": "America/Chicago",
    "CLT": "America/New_York",
    "LAS": "America/Los_Angeles",
    "PHX": "America/Phoenix",
    "LAX": "America/Los_Angeles",
    "SEA": "America/Los_Angeles",
    "MCO": "America/New_York",
    "IAH": "America/Chicago",
    "DTW": "America/Detroit",
    "MSP": "America/Chicago",
    "SFO": "America/Los_Angeles",
    "EWR": "America/New_York",
    "LGA": "America/New_York",
    "BOS": "America/New_York",
    "SLC": "America/Denver",
    "JFK": "America/New_York",
    "FLL": "America/New_York",
    "BWI": "America/New_York",
    "DCA": "America/New_York",
    "MDW": "America/Chicago",
    "SAN": "America/Los_Angeles",
    "PHL": "America/New_York",
    "BNA": "America/Chicago",
    "TPA": "America/New_York",
    "AUS": "America/Chicago",
    "MIA": "America/New_York",
    "DAL": "America/Chicago",
}
IEM_US = (
    "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station={st}&data=all"
    "&year1={y}&month1=1&day1=1&year2={y2}&month2=1&day2=1&tz=Etc/UTC&format=onlycomma"
    "&latlon=no&missing=M&direct=no&report_type=3&report_type=4"
)  # US stations: without report_type= the archive returns 5-minute HFMETAR rows too
PRC2024 = {
    "flight_list.csv": (
        (
            "https://data.4tu.nl/file/8cb8484b-dbe7-4750-8b87-a5b1dbc621b4/"
            "2bfbef1a-a75b-4045-a46e-3325bf363c27"
        ),
        "2da9db49afcae255eab0828f1e2f8575",  # md5 published by 4TU (computed_md5)
    )
}


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, tries: int = 4) -> None:
    """Stream ``url`` to ``dest`` in 1 MB chunks (never the whole body in memory)."""
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "prc-challenge-fetch/0.1"})
            with urllib.request.urlopen(req, timeout=600) as r, dest.open("wb") as f:
                for chunk in iter(lambda: r.read(1 << 20), b""):
                    f.write(chunk)
            return
        except Exception as e:  # noqa: BLE001 — transient network errors are retried
            print(f"   attempt {attempt + 1} failed: {e}")
            time.sleep(10 * (attempt + 1))
    raise SystemExit(f"download failed: {url}")


def bts_convert(csv_stream: io.BufferedIOBase, airports: set[str]) -> tuple:
    """Column-subset one BTS month and keep rows that leave from or land at ``airports``.

    Returns ``(frame, departures_by_origin)``; the second is over **all** origins, so the
    manifest shows the airport choice. Column names are checked against ``BTS_COLS``.
    """
    import polars as pl

    df = pl.read_csv(
        csv_stream,
        columns=list(BTS_COLS),
        schema_overrides={
            "FlightDate": pl.Utf8,
            "Tail_Number": pl.Utf8,
            "CRSDepTime": pl.Utf8,
            "DepTime": pl.Utf8,
            "WheelsOff": pl.Utf8,
            "WheelsOn": pl.Utf8,
        },
        infer_schema_length=100_000,
    )
    counts = (
        df.filter(pl.col("Cancelled") == 0).group_by("Origin").len().sort("len", descending=True)
    )
    keep = df.filter(pl.col("Origin").is_in(airports) | pl.col("Dest").is_in(airports))
    return keep, dict(zip(counts["Origin"].to_list(), counts["len"].to_list(), strict=True))


def bts(months: list[str]) -> None:
    import zipfile

    # resume into the newest pull (a 34-month pull can cross a UTC midnight)
    prev = sorted(paths.EXTERNAL.glob("bts_ontime_*"))
    out = prev[-1] if prev else _outdir("bts_ontime")
    man = out / "manifest.json"
    entries = json.loads(man.read_text())["files"] if man.is_file() else []
    done = {e["month"] for e in entries}
    for ym in months:
        if ym in done:
            print("   have", ym)
            continue
        y, m = ym.split("-")
        url = BTS.format(y=int(y), m=int(m))
        z = out / f"_tmp_{ym}.zip"
        _download(url, z)
        zsha, zbytes = _sha256_file(z), z.stat().st_size
        with zipfile.ZipFile(z) as zf:
            name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
            with zf.open(name) as fh:  # streamed out of the archive, never extracted
                frame, counts = bts_convert(fh, set(US_AIRPORTS))
        pq = out / f"bts_{ym}.parquet"
        frame.write_parquet(pq, compression="zstd")
        z.unlink()
        entries.append(
            {
                "month": ym,
                "url": url,
                "zip_bytes": zbytes,
                "zip_sha256": zsha,
                "csv_member": name,
                "file": pq.name,
                "rows": frame.height,
                "bytes": pq.stat().st_size,
                "sha256": _sha256_file(pq),
                "departures_by_origin_all": counts,
            }
        )
        print(
            f"   {ym}: zip {zbytes / 1e6:.1f} MB -> {frame.height} rows, {pq.stat().st_size / 1e6:.1f} MB"
        )
        _manifest(
            out,
            entries,
            "BTS Reporting Carrier On-Time Performance (US DOT, public domain). Rows whose "
            f"Origin or Dest is one of {len(US_AIRPORTS)} airports; columns {list(BTS_COLS)}. "
            "Raw ZIP/CSV deleted after conversion; zip_sha256 identifies the upstream file.",
        )


def metar_us(years: list[int], top: int | None = None) -> None:
    """Same IEM archive and CSV layout as ``metar`` (so ``metar_visibility`` / ``winter_ops``
    parse it unchanged), for the BTS airports (the ``top`` largest when given). The directory
    is ``usmetar_iem_<date>`` so the ``metar_iem_*`` globs of the European blocks can never pick
    it up. Resumes into the newest pull; IEM throttles (HTTP 429 or an empty body), so every
    request is paced and retried with a growing pause."""
    prev = sorted(paths.EXTERNAL.glob("usmetar_iem_*"))
    out = prev[-1] if prev else _outdir("usmetar_iem")
    stations = list(US_AIRPORTS)[:top] if top else list(US_AIRPORTS)
    for st in stations:
        for y in years:
            dest = out / f"{st}_{y}.csv"
            if dest.is_file() and dest.stat().st_size > 1000:
                continue
            url = IEM_US.format(st=st, y=y, y2=y + 1)
            for attempt in range(8):
                try:
                    e = _get(url, dest)
                    if e["bytes"] > 1000:
                        break
                except Exception as err:  # noqa: BLE001 — 429 / timeouts are retried
                    print(f"   {st} {y} attempt {attempt + 1}: {err}", flush=True)
                time.sleep(30 * (attempt + 1))
            else:
                raise SystemExit(f"IEM returned no data for {st} {y}; re-run later to resume")
            print("  ", e["file"], e["bytes"], flush=True)
            time.sleep(5.0)
    entries = []
    for f in sorted(out.glob("*.csv")):
        st, y = f.stem.split("_")
        entries.append(
            {
                "file": f.name,
                "url": IEM_US.format(st=st, y=int(y), y2=int(y) + 1),
                "bytes": f.stat().st_size,
                "sha256": _sha256_file(f),
            }
        )
    _manifest(
        out,
        entries,
        "IEM ASOS/METAR archive (Iowa State; upstream NOAA), US stations, routine + special "
        "reports (report_type=3,4). vsby is statute miles (US reports carry no metre group).",
    )


def prc2024() -> None:
    out = _outdir("prc2024")
    entries = []
    for name, (url, md5) in PRC2024.items():
        dest = out / name
        _download(url, dest)
        got = hashlib.md5(dest.read_bytes()).hexdigest()  # upstream publishes md5
        if got != md5:
            raise SystemExit(f"{name}: md5 {got} != published {md5}")
        entries.append(
            {
                "file": name,
                "url": url,
                "bytes": dest.stat().st_size,
                "md5": got,
                "sha256": _sha256_file(dest),
            }
        )
    _manifest(
        out,
        entries,
        "PRC Data Challenge 2024 dataset, 4TU.ResearchData doi:10.4121/8cb8484b-dbe7-4750-8b87-"
        "a5b1dbc621b4 (v2), CC BY 4.0. flight_list.csv only (no trajectory parquets).",
    )


#: FAA Aircraft Characteristics Database: the download link on the landing page
#: https://www.faa.gov/airports/engineering/aircraft_char_database serves the xlsx directly.
FAA_ACD_URL = "https://www.faa.gov/airports/engineering/aircraft_char_database/aircraft_data"
FAA_ACD_SHEET = "ACD_Data"


def xlsx_sheet_rows(blob: bytes, sheet: str) -> list[list[str]]:
    """Rows of one worksheet of an .xlsx as strings (shared strings resolved, empty cells ``""``).

    Stdlib only (``zipfile`` + ``ElementTree``): the project has no Excel reader and the FAA
    file is a plain table. Cell positions come from the ``r`` reference so gaps stay aligned.
    """
    import re
    import xml.etree.ElementTree as ET
    import zipfile

    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rel_ns = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
    z = zipfile.ZipFile(io.BytesIO(blob))
    shared: list[str] = []
    if "xl/sharedStrings.xml" in z.namelist():
        root = ET.fromstring(z.read("xl/sharedStrings.xml"))
        for si in root.findall("m:si", ns):
            shared.append("".join(t.text or "" for t in si.iter(f"{{{ns['m']}}}t")))
    wb = ET.fromstring(z.read("xl/workbook.xml"))
    rid = next(s.get(rel_ns) for s in wb.find("m:sheets", ns) if s.get("name") == sheet)
    rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
    target = next(r.get("Target") for r in rels if r.get("Id") == rid).lstrip("/")
    target = target if target.startswith("xl/") else f"xl/{target}"
    data = ET.fromstring(z.read(target)).find("m:sheetData", ns)

    def col_index(ref: str) -> int:
        letters = re.match(r"[A-Z]+", ref).group(0)  # type: ignore[union-attr]
        n = 0
        for ch in letters:
            n = n * 26 + ord(ch) - 64
        return n - 1

    rows: list[list[str]] = []
    for row in data.findall("m:row", ns):
        cells: dict[int, str] = {}
        for c in row.findall("m:c", ns):
            v = c.find("m:v", ns)
            if c.get("t") == "inlineStr":
                val = "".join(t.text or "" for t in c.iter(f"{{{ns['m']}}}t"))
            elif v is None or v.text is None:
                val = ""
            elif c.get("t") == "s":
                val = shared[int(v.text)]
            else:
                val = v.text
            cells[col_index(c.get("r", "A1"))] = val.strip()
        if cells:
            rows.append([cells.get(i, "") for i in range(max(cells) + 1)])
    width = len(rows[0]) if rows else 0
    return [(r + [""] * width)[:width] for r in rows]


def faa_acd() -> None:
    """The FAA Aircraft Characteristics Database (US government work, public domain)."""
    import csv

    out = _outdir("faa_acd")
    xlsx = out / "faa_acd.xlsx"
    # faa.gov answers 403 to the fetch user agent (and to "Mozilla/5.0 (compatible; ...)");
    # the bare "Mozilla/5.0" string is served (checked 2026-09-25).
    req = urllib.request.Request(
        FAA_ACD_URL, headers={"User-Agent": "Mozilla/5.0", "Accept": "*/*"}
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        xlsx.write_bytes(r.read())
    rows = xlsx_sheet_rows(xlsx.read_bytes(), FAA_ACD_SHEET)
    if not rows or rows[0][0] != "ICAO_Code":
        raise SystemExit(f"unexpected FAA ACD layout: header {rows[0][:5] if rows else None}")
    csv_path = out / "faa_acd.csv"
    with csv_path.open("w", newline="") as f:
        csv.writer(f).writerows(rows)
    entries = [
        {
            "file": xlsx.name,
            "url": FAA_ACD_URL,
            "bytes": xlsx.stat().st_size,
            "sha256": _sha256_file(xlsx),
        },
        {
            "file": csv_path.name,
            "derived_from": f"{xlsx.name} sheet {FAA_ACD_SHEET}",
            "rows": len(rows) - 1,
            "bytes": csv_path.stat().st_size,
            "sha256": _sha256_file(csv_path),
        },
    ]
    _manifest(
        out,
        entries,
        "FAA Aircraft Characteristics Database, "
        "https://www.faa.gov/airports/engineering/aircraft_char_database (US federal government "
        "work, public domain, 17 U.S.C. 105). faa_acd.csv = the ACD_Data sheet, all columns, "
        "parsed with the stdlib reader in scripts/fetch_external.py.",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "source",
        choices=[
            "metar",
            "eurocontrol",
            "ourairports",
            "holidays",
            "openmeteo",
            "osm",
            "bts",
            "metar_us",
            "prc2024",
            "xplane",
            "faa_acd",
        ],
    )
    ap.add_argument("--years", nargs="+", type=int, default=[2025, 2026])
    ap.add_argument("--months", nargs="+", default=list(BTS_MONTHS), help="bts: YYYY-MM list")
    ap.add_argument("--top", type=int, default=None, help="metar_us: the N largest airports")
    a = ap.parse_args()
    {
        "metar": lambda: metar(a.years),
        "eurocontrol": lambda: eurocontrol(a.years),
        "ourairports": ourairports,
        "holidays": lambda: holidays(a.years),
        "openmeteo": openmeteo,
        "osm": osm,
        "bts": lambda: bts(a.months),
        "metar_us": lambda: metar_us(a.years, a.top),
        "prc2024": prc2024,
        "xplane": xplane,
        "faa_acd": faa_acd,
    }[a.source]()


if __name__ == "__main__":
    main()
