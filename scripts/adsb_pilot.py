"""ADSB.lol pilot: can open ADS-B ground traces measure off-block time?

Feasibility probe (2026-09-19), **not** a production pipeline. About half the squared
error of the model sits in the ~1.6 % of departures with no network-manager off-block
(`AOBT_3_flt` null, mostly LIRF). ADS-B surface positions would give a direct measurement
of the moment the aircraft starts moving off stand, i.e. of `BLOCK_TIME_UTC_mvt`, on
exactly those rows.

Source: https://github.com/adsblol/globe_history_2025 (and `_2026`) — one GitHub release
per day per replica, tag ``v<YYYY.MM.DD>-planes-readsb-<prod|staging|mlatonly>-0``, assets
``<tag>.tar.aa/.ab/.ac`` (2 GB split parts of one uncompressed tar). The tar holds
``LICENSE-ODbL.txt``, ``LICENSE-cc0.txt``, ``heatmap/*.bin.ttf`` and
``traces/<last 2 hex>/trace_full_<icao24>.json`` — each trace file is gzip-compressed JSON
in wiedehopf/readsb "trace json" format (README-json.md). **Licence: ODbL 1.0**
(repository LICENSE + README); attribution to ADSB.lol and share-alike on derived
databases.

Subcommands (each re-runnable; every step writes a parquet the next one reads):

    uv run python scripts/adsb_pilot.py days                     # which days to pull, from the labels
    uv run python scripts/adsb_pilot.py fetch --days 2025-07-10 ...  # download + stream-filter -> points/
    uv run python scripts/adsb_pilot.py match                    # traces -> movement rows
    uv run python scripts/adsb_pilot.py extract                  # per-departure ADS-B signals
    uv run python scripts/adsb_pilot.py evaluate                 # the decision tables

Compute rules for the pilot: desktop only, CPU only, one day at a time, tar parts are
deleted as soon as a day is filtered (peak transient ≈ 4 GB), kept output is the filtered
parquet only.

Note for a production pull (measured 2026-09-19):
the ``heatmap/`` directory sits at the *front* of the tar, is ~750 MB/day instead of
3.3 GB, needs no JSON parsing, and reproduces this script's first-movement time within
15 s on 97.4 % of visits. A production pipeline should read that, not the traces.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
import re
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prc import paths
from prc.data import schema as S

# --------------------------------------------------------------------------------------
# source constants

REPO = {2025: "adsblol/globe_history_2025", 2026: "adsblol/globe_history_2026"}
REPLICAS = ("prod", "staging")  # README: use prod unless it is down, then staging
API = "https://api.github.com/repos/{repo}/releases/tags/{tag}"
PREFERRED = "https://raw.githubusercontent.com/{repo}/main/PREFERRED_RELEASES.txt"
UA = "prc-challenge-adsb-pilot/0.1 (research; contact via github.com/adsblol issues)"
LICENCE = "ODbL-1.0 (https://opendatacommons.org/licenses/odbl/1.0/)"
ATTRIBUTION = (
    "Contains information from ADSB.lol (https://www.adsb.lol/, "
    "https://github.com/adsblol/globe_history_2025), which is made available under the "
    "Open Database License (ODbL) v1.0."
)

# OurAirports coordinates, same table as scripts/fetch_external.py (public domain).
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

KEEP_RADIUS_KM = 12.0  # points further out than this are dropped
KEEP_ALT_FT = 4000.0  # airborne points above this are dropped (takeoff roll + first climb)
PREFILTER_DEG = 0.15  # half-width of the cheap byte-level prefilter box (~17 km)
TRANSIENT = Path(os.environ.get("PRC_ADSB_TMP", "~/.cache/prc-adsb-pilot")).expanduser()

# movement-matching
MATCH_WINDOW_S = 300.0  # |ADS-B lift-off − MVT_TIME| for a time match
SESSION_GAP_S = 1800.0  # a gap this long splits one aircraft's points into two visits
MOVE_KT = 3.0  # ground speed that counts as "moving"
STOP_KT = 2.0  # ground speed that counts as "stationary"


def outdir() -> Path:
    """``data/external/adsblol_<date>/`` — the newest existing one, else today's."""
    existing = sorted(paths.EXTERNAL.glob("adsblol_*"))
    if existing:
        return existing[-1]
    d = paths.EXTERNAL / f"adsblol_{datetime.now(UTC):%Y-%m-%d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------------------
# 1. day selection


def _dep(columns: list[str] | None = None) -> pl.DataFrame:
    df = pl.read_parquet(paths.INTERIM / "movements.parquet", columns=columns)
    return df.filter(pl.col(S.PHASE) == "DEP")


def cmd_days(args: argparse.Namespace) -> None:
    """Rank candidate pilot days by the LIRF no-NM squared-error mass they carry."""
    d = _dep(
        [S.MVT_ID, S.PHASE, S.MVT_TIME, S.AOBT_3, S.TARGET, S.AIRPORT, S.SPLIT_MONTH, S.IS_TEST]
    ).with_columns(
        pl.col(S.MVT_TIME).dt.date().alias("day"),
        pl.col(S.AOBT_3).is_null().alias("no_nm"),
    )
    for month in ("2025-01", "2025-07"):
        t = (
            d.filter(
                (pl.col(S.SPLIT_MONTH) == month) & (pl.col(S.AIRPORT) == "LIRF") & pl.col("no_nm")
            )
            .group_by("day")
            .agg(
                pl.len().alias("lirf_no_nm"),
                (pl.col(S.TARGET) > 3600).sum().alias("gt1h"),
                pl.col(S.TARGET).max().alias("max_label_s"),
                pl.col(S.TARGET).cast(pl.Float64).pow(2).sum().alias("sse"),
            )
            .sort("sse", descending=True)
            .head(10)
        )
        print(f"\n=== {month}: LIRF no-NM days by squared-error mass ===")
        print(t)
    print("\n=== departures per day (all airports) ===")
    print(
        d.group_by(S.SPLIT_MONTH)
        .agg(pl.len().alias("rows"), pl.col("no_nm").sum().alias("no_nm"))
        .sort(S.SPLIT_MONTH)
    )


# --------------------------------------------------------------------------------------
# 2. fetch + stream filter


@dataclass(frozen=True)
class Asset:
    name: str
    url: str
    size: int


def _api(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def preferred_urls(day: date) -> tuple[str, list[str]] | None:
    """(tag, asset URLs) the publisher marks as the good release for that day.

    ``PREFERRED_RELEASES.txt`` in each repo lists one line of comma-separated asset URLs
    per day, newest first. prod is usually it, but on ~40 % of days the publisher points
    at ``staging`` instead (prod was down or truncated), and on a handful of days only
    ``mlatonly`` exists — those days carry MLAT positions only, no ADS-B. Reading this
    file instead of the releases API also keeps the unauthenticated 60 requests/hour API
    limit out of the way: a 426-day pull needs one file per year, not one call per day.
    """
    repo = REPO.get(day.year)
    if repo is None:
        return None
    cache = TRANSIENT / f"PREFERRED_RELEASES_{day.year}.txt"
    if not cache.is_file():
        TRANSIENT.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(PREFERRED.format(repo=repo), headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=60) as r:
            cache.write_bytes(r.read())
    want = f"/v{day:%Y.%m.%d}-planes-readsb-"
    for line in cache.read_text().splitlines():
        if want in line:
            urls = [u for u in line.strip().split(",") if u]
            # some tags end in "-0tmp" instead of "-0", so match to the path separator
            m = re.search(r"/(v\d{4}\.\d{2}\.\d{2}-planes-readsb-[^/]+)/", urls[0])
            if m:
                return m.group(1), sorted(urls)
    return None


def _asset_size(url: str) -> int:
    """Total size from a one-byte ranged GET (``Content-Range: bytes 0-0/<size>``)."""
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        cr = r.headers.get("Content-Range", "")
    m = re.search(r"/(\d+)$", cr)
    if not m:
        raise SystemExit(f"no Content-Range for {url}")
    return int(m.group(1))


def release_assets(day: date) -> tuple[str, list[Asset]]:
    """The tar parts of a day's release: the publisher's preferred replica, else prod."""
    repo = REPO.get(day.year)
    if repo is None:
        raise SystemExit(f"no globe_history repo for {day.year}")
    pref = preferred_urls(day)
    if pref:
        tag, urls = pref
        return tag, [Asset(u.rsplit("/", 1)[-1], u, _asset_size(u)) for u in urls]
    last: Exception | None = None
    for replica in REPLICAS:  # fallback: ask the releases API directly
        tag = f"v{day:%Y.%m.%d}-planes-readsb-{replica}-0"
        try:
            rel = _api(API.format(repo=repo, tag=tag))
        except urllib.error.HTTPError as e:
            last = e
            continue
        assets = sorted(
            (Asset(a["name"], a["browser_download_url"], int(a["size"])) for a in rel["assets"]),
            key=lambda a: a.name,
        )
        if assets:
            return tag, assets
    raise SystemExit(f"no release assets for {day} ({last})")


def download(asset: Asset, dest: Path, retries: int = 4) -> str:
    """Resumable single-stream download. Returns the sha256 of the finished file."""
    for attempt in range(retries):
        have = dest.stat().st_size if dest.exists() else 0
        if have == asset.size:
            break
        if have > asset.size:
            dest.unlink()
            have = 0
        headers = {"User-Agent": UA}
        if have:
            headers["Range"] = f"bytes={have}-"
        req = urllib.request.Request(asset.url, headers=headers)
        try:
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=120) as r, open(dest, "ab") as f:
                shutil.copyfileobj(r, f, length=1 << 22)
            print(
                f"    {dest.name}: {dest.stat().st_size / 1e9:.2f} GB in {time.time() - t0:.0f}s",
                flush=True,
            )
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            print(f"    retry {attempt + 1}/{retries} after {e}", flush=True)
            time.sleep(5 * (attempt + 1))
    if not dest.exists() or dest.stat().st_size != asset.size:
        raise SystemExit(
            f"{dest.name}: got {dest.stat().st_size if dest.exists() else 0} bytes, "
            f"expected {asset.size}"
        )
    h = hashlib.sha256()
    with open(dest, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


class ChainReader(io.RawIOBase):
    """The split ``.tar.aa/.ab/.ac`` parts read as one non-seekable stream."""

    def __init__(self, files: list[Path]) -> None:
        self._files = files
        self._i = 0
        self._f = open(files[0], "rb")  # noqa: SIM115 — closed in close()

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:  # type: ignore[override]
        while True:
            n = self._f.readinto(b)
            if n:
                return n
            if self._i + 1 >= len(self._files):
                return 0
            self._f.close()
            self._i += 1
            self._f = open(self._files[self._i], "rb")  # noqa: SIM115

    def close(self) -> None:
        try:
            self._f.close()
        finally:
            super().close()


def _decimal_prefixes(lo: float, hi: float) -> list[str]:
    """One-decimal string prefixes covering [lo, hi] (``41.7`` covers 41.70…41.79)."""
    out, x = [], math.floor(lo * 10)
    while x <= math.floor(hi * 10):
        out.append(f"{x / 10:.1f}")
        x += 1
    return out


def _prefilter() -> re.Pattern[bytes]:
    """A byte regex that matches ``,<lat>,<lon>,`` near any of the ten airports.

    A trace point serialises as ``[offset,lat,lon,alt,…]``, so the pair is adjacent. The
    regex is a cheap superset test on the decompressed file (≈160 MB/s); the exact
    distance test runs only on the files it passes.
    """
    pats = []
    for lat, lon in AIRPORT_LATLON.values():
        la = "|".join(
            re.escape(p) for p in _decimal_prefixes(lat - PREFILTER_DEG, lat + PREFILTER_DEG)
        )
        lo = "|".join(
            re.escape(p) for p in _decimal_prefixes(lon - PREFILTER_DEG, lon + PREFILTER_DEG)
        )
        pats.append(f",(?:{la})\\d*,(?:{lo})\\d*,")
    return re.compile("|".join(pats).encode())


_AP_NAMES = list(AIRPORT_LATLON)
_AP_LAT = np.array([AIRPORT_LATLON[a][0] for a in _AP_NAMES])
_AP_LON = np.array([AIRPORT_LATLON[a][1] for a in _AP_NAMES])
_KM_LAT = 111.32
_DLAT = KEEP_RADIUS_KM / _KM_LAT
_DLON = np.array([KEEP_RADIUS_KM / (_KM_LAT * math.cos(math.radians(la))) for la in _AP_LAT])

POINT_SCHEMA = {
    "airport": pl.Categorical,
    "icao24": pl.String,
    "ts": pl.Float64,  # unix seconds, UTC
    "lat": pl.Float64,
    "lon": pl.Float64,
    "on_ground": pl.Boolean,
    "alt_ft": pl.Float32,
    "gs_kt": pl.Float32,
    "track_deg": pl.Float32,
    "flight": pl.String,  # callsign, only on points that carry the aircraft object
}


def _points_from_trace(obj: dict) -> list[tuple]:
    """Points of one aircraft-day that are within ``KEEP_RADIUS_KM`` of an airport and
    either on the ground or below ``KEEP_ALT_FT``."""
    t0 = obj.get("timestamp")
    icao = obj.get("icao")
    if t0 is None or icao is None:
        return []
    out: list[tuple] = []
    for p in obj.get("trace", ()):
        lat, lon = p[1], p[2]
        if lat is None or lon is None:
            continue
        k = -1
        for j in range(len(_AP_NAMES)):
            if abs(lat - _AP_LAT[j]) <= _DLAT and abs(lon - _AP_LON[j]) <= _DLON[j]:
                k = j
                break
        if k < 0:
            continue
        alt = p[3]
        ground = alt == "ground"
        alt_ft = None if ground or not isinstance(alt, (int, float)) else float(alt)
        if not ground and alt_ft is not None and alt_ft > KEEP_ALT_FT:
            continue
        # exact distance (equirectangular; the boxes above are already tight)
        dx = (lon - _AP_LON[k]) * _KM_LAT * math.cos(math.radians(lat))
        dy = (lat - _AP_LAT[k]) * _KM_LAT
        if dx * dx + dy * dy > KEEP_RADIUS_KM * KEEP_RADIUS_KM:
            continue
        ac = p[8] if len(p) > 8 else None
        flight = ac.get("flight", "").strip() if isinstance(ac, dict) else None
        out.append(
            (
                _AP_NAMES[k],
                icao,
                t0 + p[0],
                lat,
                lon,
                ground,
                alt_ft,
                float(p[4]) if p[4] is not None else None,
                float(p[5]) if p[5] is not None else None,
                flight or None,
            )
        )
    return out


def filter_day(parts: list[Path], day: date, out: Path) -> dict:
    """Stream the concatenated tar once, keep only near-airport low-altitude points."""
    rx = _prefilter()
    rows: list[tuple] = []
    chunks: list[pl.DataFrame] = []
    aircraft: dict[str, tuple[str | None, str | None]] = {}
    n_files = n_cand = n_hit = 0
    n_bytes = 0
    t0 = time.time()
    stream = io.BufferedReader(ChainReader(parts), buffer_size=1 << 24)
    with tarfile.open(fileobj=stream, mode="r|") as tf:
        for member in tf:
            if not member.isfile() or "trace_full_" not in member.name:
                continue
            n_files += 1
            fh = tf.extractfile(member)
            if fh is None:
                continue
            body = fh.read()
            try:
                raw = gzip.decompress(body) if body[:2] == b"\x1f\x8b" else body
            except (OSError, EOFError, zlib.error):
                continue
            n_bytes += len(raw)
            if n_files % 20000 == 0:
                print(
                    f"    {n_files} traces, {n_bytes / 1e9:.1f} GB json, {n_cand} candidates, "
                    f"{sum(c.height for c in chunks) + len(rows)} points, "
                    f"{time.time() - t0:.0f}s",
                    flush=True,
                )
            if not rx.search(raw):
                continue
            n_cand += 1
            try:
                obj = json.loads(raw)
            except ValueError:
                continue
            pts = _points_from_trace(obj)
            if pts:
                n_hit += 1
                rows.extend(pts)
                aircraft[obj["icao"]] = (obj.get("r"), obj.get("t"))
                if len(rows) >= 1_000_000:  # keep python-object memory bounded
                    chunks.append(pl.DataFrame(rows, schema=POINT_SCHEMA, orient="row"))
                    rows = []
    if rows:
        chunks.append(pl.DataFrame(rows, schema=POINT_SCHEMA, orient="row"))
        rows = []
    df = (
        pl.concat(chunks) if chunks else pl.DataFrame([], schema=POINT_SCHEMA, orient="row")
    ).sort("airport", "icao24", "ts")
    (out / "points").mkdir(parents=True, exist_ok=True)
    (out / "aircraft").mkdir(parents=True, exist_ok=True)
    df.write_parquet(out / "points" / f"{day:%Y-%m-%d}.parquet", compression="zstd")
    pl.DataFrame(
        [(k, v[0], v[1]) for k, v in aircraft.items()],
        schema={"icao24": pl.String, "reg": pl.String, "type": pl.String},
        orient="row",
    ).write_parquet(out / "aircraft" / f"{day:%Y-%m-%d}.parquet", compression="zstd")
    stat = {
        "traces": n_files,
        "json_bytes": n_bytes,
        "prefilter_candidates": n_cand,
        "aircraft_kept": n_hit,
        "points": df.height,
        "seconds": round(time.time() - t0, 1),
    }
    print(f"    filtered: {stat}", flush=True)
    return stat


def cmd_fetch(args: argparse.Namespace) -> None:
    out = outdir()
    out.mkdir(parents=True, exist_ok=True)
    TRANSIENT.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    manifest.setdefault("source", "https://github.com/adsblol/globe_history_<year>")
    manifest.setdefault("licence", LICENCE)
    manifest.setdefault("attribution", ATTRIBUTION)
    manifest.setdefault(
        "filter",
        {
            "radius_km": KEEP_RADIUS_KM,
            "alt_max_ft": KEEP_ALT_FT,
            "airports": AIRPORT_LATLON,
            "note": "kept: on-ground points, and airborne points at or below alt_max_ft",
        },
    )
    manifest.setdefault("days", {})
    for day_s in args.days:
        day = date.fromisoformat(day_s)
        if day_s in manifest["days"] and not args.force:
            print(f"{day_s}: already in the manifest, skipping (use --force)")
            continue
        print(f"{day_s}: resolving release …", flush=True)
        tag, assets = release_assets(day)
        total = sum(a.size for a in assets)
        print(f"  {tag}: {len(assets)} assets, {total / 1e9:.2f} GB", flush=True)
        files, records = [], []
        for a in assets:
            dest = TRANSIENT / a.name
            sha = download(a, dest)
            files.append(dest)
            records.append({"name": a.name, "url": a.url, "bytes": a.size, "sha256": sha})
            time.sleep(1.0)  # be polite to GitHub
        stat = filter_day(files, day, out)
        manifest["days"][day_s] = {
            "tag": tag,
            "assets": records,
            "download_bytes": total,
            **stat,
            "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
        if not args.keep_tars:
            for f in files:
                f.unlink(missing_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"manifest -> {manifest_path}")


# --------------------------------------------------------------------------------------
# 3. sessions and matching


def _load_points(out: Path, days: list[str] | None = None) -> pl.DataFrame:
    files = sorted((out / "points").glob("*.parquet"))
    if days:
        files = [f for f in files if f.stem in days]
    if not files:
        raise SystemExit(f"no filtered points under {out / 'points'} — run `fetch` first")
    return pl.concat([pl.read_parquet(f) for f in files]).sort("airport", "icao24", "ts")


def sessions(pts: pl.DataFrame) -> pl.DataFrame:
    """One row per (airport, aircraft, visit): the ADS-B signals of one ground movement.

    A visit is a maximal run of points of one aircraft at one airport separated by less
    than ``SESSION_GAP_S``. A visit can be an arrival, a departure, or an arrival followed
    by a departure; the departure part is everything after the landing.

    ``t_liftoff``       first airborne point after the last ground→air transition
    ``liftoff_censored`` no ground point at all: the aircraft was first seen already
                        airborne, so ``t_liftoff`` is the first point and is an upper bound
    ``t_first_seen_dep`` first point of the departure part (transponders are often switched
                        on at pushback, so this is itself a signal)
    ``t_first_gnd``     first on-ground point of the departure part
    ``t_first_move``    start of the first sustained run of on-ground points at ≥ MOVE_KT
    ``t_last_stop``     last on-ground point below STOP_KT before that
    ``saw_stationary``  the aircraft was observed stopped before it first moved
    """
    pts = pts.with_columns(
        (
            (pl.col("icao24") != pl.col("icao24").shift())
            | (pl.col("airport").cast(pl.String) != pl.col("airport").cast(pl.String).shift())
            | ((pl.col("ts") - pl.col("ts").shift()) > SESSION_GAP_S)
        )
        .fill_null(True)
        .cum_sum()
        .alias("visit")
    )
    rows = []
    for (airport, icao, visit), g in pts.group_by(
        ["airport", "icao24", "visit"], maintain_order=True
    ):
        ts = g["ts"].to_numpy()
        gnd = g["on_ground"].to_numpy()
        gs = g["gs_kt"].fill_null(0.0).to_numpy()
        alt = g["alt_ft"].fill_null(0.0).to_numpy()
        flights = [f for f in g["flight"].to_list() if f]
        g2a = np.flatnonzero(gnd[:-1] & ~gnd[1:])  # ground -> air: a take-off
        a2g = np.flatnonzero(~gnd[:-1] & gnd[1:])  # air -> ground: a landing
        n_air = int((~gnd).sum())
        t_landing = float(ts[a2g[0] + 1]) if a2g.size else None
        censored = False
        if g2a.size:
            lift_i = int(g2a[-1]) + 1
            t_liftoff = float(ts[lift_i])
        elif not gnd.any() and n_air >= 3 and alt[-1] > alt[0]:
            # never seen on the ground, first seen low and climbing: an upper bound on lift-off
            lift_i = 0
            t_liftoff = float(ts[0])
            censored = True
        else:
            lift_i = len(ts)
            t_liftoff = None
        # the departure part starts after the landing that precedes the take-off
        dep0 = 0
        if a2g.size and a2g[0] + 1 < lift_i:
            dep0 = int(a2g[0]) + 1
        gi = np.flatnonzero(gnd[dep0:lift_i]) + dep0  # ground points of the departure part
        t_first_move = t_last_stop = None
        saw_stationary = False
        if gi.size:
            moving = gs[gi] >= MOVE_KT
            run = np.convolve(moving.astype(np.int8), np.ones(3, dtype=np.int8), mode="valid")
            starts = np.flatnonzero(run >= 3)
            k = (
                int(starts[0])
                if starts.size
                else (int(np.flatnonzero(moving)[0]) if moving.any() else None)
            )
            if k is not None:
                t_first_move = float(ts[gi[k]])
                stopped = np.flatnonzero(gs[gi[:k]] < STOP_KT)
                if stopped.size:
                    saw_stationary = True
                    t_last_stop = float(ts[gi[int(stopped[-1])]])
        gaps = np.diff(ts[gi]) if gi.size > 1 else np.array([0.0])
        rows.append(
            {
                "airport": str(airport),
                "icao24": icao,
                "visit": int(visit),
                "flight": max(set(flights), key=flights.count) if flights else None,
                "n_points": int(ts.size),
                "n_gnd": int(gi.size),
                "t_first_seen": float(ts[0]),
                "t_last_seen": float(ts[-1]),
                "t_first_seen_dep": float(ts[dep0]),
                "t_first_gnd": float(ts[gi[0]]) if gi.size else None,
                "t_last_gnd": float(ts[gi[-1]]) if gi.size else None,
                "t_first_move": t_first_move,
                "t_last_stop": t_last_stop,
                "t_liftoff": t_liftoff,
                "liftoff_censored": censored,
                "t_landing": t_landing,
                "saw_stationary": saw_stationary,
                "gnd_span_s": float(ts[gi[-1]] - ts[gi[0]]) if gi.size > 1 else 0.0,
                "gnd_max_gap_s": float(gaps.max()),
            }
        )
    return pl.DataFrame(rows)


def _norm_callsign(col: str) -> pl.Expr:
    return pl.col(col).cast(pl.Utf8).str.strip_chars().str.to_uppercase().replace("", None)


def assign(mv: pl.DataFrame, sess: pl.DataFrame) -> pl.DataFrame:
    """One-to-one assignment of departures to ADS-B visits.

    Candidates are visits at the same airport whose lift-off is within
    ``MATCH_WINDOW_S`` of ``MVT_TIME_UTC_mvt``. Pairs whose callsigns agree are taken
    first, then the rest nearest-first; a movement and a visit are each used once. The
    one-to-one rule matters: at EDDF a departure leaves every 40 s, so *any* window wide
    enough to absorb stamp noise contains several other departures' take-offs.
    """
    cand = sess.filter(pl.col("t_liftoff").is_not_null()).select(
        "airport",
        "icao24",
        "visit",
        "t_liftoff",
        "liftoff_censored",
        _norm_callsign("flight").alias("adsb_flight"),
    )
    pairs = (
        mv.select(S.MVT_ID, S.AIRPORT, "mvt_ts", _norm_callsign(S.CALLSIGN).alias("cs"))
        .join(cand, left_on=S.AIRPORT, right_on="airport", how="inner")
        .with_columns((pl.col("t_liftoff") - pl.col("mvt_ts")).alias("dt"))
        .filter(pl.col("dt").abs() <= MATCH_WINDOW_S)
        .with_columns(
            (
                pl.col("cs").is_not_null()
                & pl.col("adsb_flight").is_not_null()
                & (pl.col("cs") == pl.col("adsb_flight"))
            ).alias("cs_hit")
        )
        .sort([pl.col("cs_hit").not_(), pl.col("dt").abs()])
    )
    used_mvt: set[float] = set()
    used_sess: set[tuple[str, str, int]] = set()
    keep = []
    for mid, ic, vi, dtv, hit, af, cens in zip(
        pairs[S.MVT_ID].to_list(),
        pairs["icao24"].to_list(),
        pairs["visit"].to_list(),
        pairs["dt"].to_list(),
        pairs["cs_hit"].to_list(),
        pairs["adsb_flight"].to_list(),
        pairs["liftoff_censored"].to_list(),
        strict=True,
    ):
        key = (ic, vi)
        if mid in used_mvt or key in used_sess:
            continue
        used_mvt.add(mid)
        used_sess.add(key)
        keep.append((mid, ic, vi, dtv, hit, af, cens))
    return pl.DataFrame(
        keep,
        schema={
            S.MVT_ID: pl.Float64,
            "icao24": pl.String,
            "visit": pl.Int64,
            "liftoff_minus_mvt_s": pl.Float64,
            "callsign_hit": pl.Boolean,
            "adsb_flight": pl.String,
            "liftoff_censored": pl.Boolean,
        },
        orient="row",
    )


def _rate_table(m: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    return (
        m.group_by(by)
        .agg(
            pl.len().alias("dep"),
            pl.col("matched").sum().alias("matched"),
            (pl.col("matched").mean() * 100).round(1).alias("pct"),
            (pl.col("has_gnd").sum()).alias("with_gnd"),
            (pl.col("usable").sum()).alias("usable"),
        )
        .sort(by)
    )


def cmd_match(args: argparse.Namespace) -> None:
    out = outdir()
    pts = _load_points(out)
    days = sorted({f.stem for f in (out / "points").glob("*.parquet")})
    print(f"{pts.height} points over {len(days)} days: {days}")
    sess = sessions(pts)
    sess.write_parquet(out / "sessions.parquet", compression="zstd")
    print(
        f"{sess.height} aircraft visits; {sess['t_liftoff'].is_not_null().sum()} with a lift-off "
        f"({sess['liftoff_censored'].sum()} of them never seen on the ground)"
    )

    mv = _dep(
        [
            S.MVT_ID,
            S.PHASE,
            S.FLIGHT_NO,
            S.CALLSIGN,
            S.AIRPORT,
            S.MVT_TIME,
            S.BLOCK_TIME,
            S.SCHED_TIME,
            S.TARGET,
            S.AOBT_3,
            S.SPLIT_MONTH,
            S.IS_TEST,
            S.STAND,
            S.RUNWAY,
            S.AIRCRAFT_TYPE,
        ]
    ).with_columns(pl.col(S.MVT_TIME).dt.date().cast(pl.String).alias("day"))
    mv = mv.filter(pl.col("day").is_in(days)).with_columns(
        pl.col(S.MVT_TIME).dt.epoch("s").cast(pl.Float64).alias("mvt_ts")
    )
    print(f"{mv.height} departures on those days")

    a = assign(mv, sess)
    m = mv.join(a, on=S.MVT_ID, how="left").with_columns(
        pl.col("icao24").is_not_null().alias("matched")
    )
    m = m.join(
        sess.drop("flight").rename({"airport": "_ap"}), on=["icao24", "visit"], how="left"
    ).drop("_ap")
    m = m.with_columns(
        (pl.col("n_gnd").fill_null(0) > 0).alias("has_gnd"),
        (
            pl.col("saw_stationary").fill_null(False)
            & (pl.col("n_gnd").fill_null(0) >= 5)
            & (pl.col("gnd_max_gap_s").fill_null(1e9) <= 300)
            & pl.col("t_first_move").is_not_null()
        ).alias("usable"),
        pl.when(pl.col(S.AOBT_3).is_null())
        .then(pl.lit("no-NM"))
        .otherwise(pl.lit("with-NM"))
        .alias("lane"),
    )
    m.write_parquet(out / "matched.parquet", compression="zstd")

    print("\n=== match rate by airport (matched / with ground points / usable first-move) ===")
    print(_rate_table(m, [S.AIRPORT]))
    print("\n=== by lane ===")
    print(_rate_table(m, ["lane", S.IS_TEST]))
    print("\n=== LIRF by lane ===")
    print(_rate_table(m.filter(pl.col(S.AIRPORT) == "LIRF"), ["lane", S.IS_TEST]))
    both = m.filter(
        pl.col("matched") & pl.col(S.CALLSIGN).is_not_null() & pl.col("adsb_flight").is_not_null()
    )
    if both.height:
        print(
            f"\ncallsign cross-check: {both['callsign_hit'].mean() * 100:.1f}% of {both.height} "
            "matched rows that carry both callsigns agree"
        )
    mt = m.filter(pl.col("matched"))
    if mt.height:
        q = mt["liftoff_minus_mvt_s"].quantile
        print(
            "lift-off − MVT_TIME (s): "
            f"p05 {q(0.05):.0f}  p25 {q(0.25):.0f}  p50 {q(0.5):.0f}  p75 {q(0.75):.0f}  "
            f"p95 {q(0.95):.0f}"
        )


# --------------------------------------------------------------------------------------
# 4. signal extraction


def cmd_extract(args: argparse.Namespace) -> None:
    """Turn each matched visit into the numbers a taxi-out model would use."""
    out = outdir()
    d = pl.read_parquet(out / "matched.parquet").filter(pl.col("matched"))
    d = d.with_columns(
        # MVT_TIME is the airport's own take-off stamp, so this is the ADS-B taxi-out
        (pl.col("mvt_ts") - pl.col("t_first_move")).alias("adsb_taxi_from_move_s"),
        (pl.col("t_liftoff") - pl.col("t_first_move")).alias("adsb_taxi_liftoff_move_s"),
        (pl.col("mvt_ts") - pl.col("t_first_seen_dep")).alias("mvt_minus_first_seen_s"),
        (pl.col("mvt_ts") - pl.col("t_first_gnd")).alias("mvt_minus_first_gnd_s"),
        (pl.col("mvt_ts") - pl.col("t_last_stop")).alias("mvt_minus_last_stop_s"),
        (pl.col("t_first_move") - pl.col(S.BLOCK_TIME).dt.epoch("s")).alias(
            "first_move_minus_block_s"
        ),
        (pl.col("t_first_seen_dep") - pl.col(S.BLOCK_TIME).dt.epoch("s")).alias(
            "first_seen_minus_block_s"
        ),
        (pl.col("mvt_ts") - pl.col(S.AOBT_3).dt.epoch("s")).alias("mvt_minus_aobt_s"),
        (pl.col("t_first_move") - pl.col(S.AOBT_3).dt.epoch("s")).alias("first_move_minus_aobt_s"),
        (pl.col(S.BLOCK_TIME).dt.epoch("s") - pl.col(S.AOBT_3).dt.epoch("s")).alias(
            "block_minus_aobt_s"
        ),
    )
    d.write_parquet(out / "signals.parquet", compression="zstd")
    print(f"{d.height} matched departures -> {out / 'signals.parquet'}")
    print("\n=== ground coverage of matched departures, by airport ===")
    with pl.Config(tbl_width_chars=200, tbl_rows=40):
        print(
            d.group_by(S.AIRPORT)
            .agg(
                pl.len().alias("matched"),
                pl.col("has_gnd").mean().mul(100).round(1).alias("any_gnd_%"),
                pl.col("saw_stationary").mean().mul(100).round(1).alias("saw_stop_%"),
                pl.col("usable").mean().mul(100).round(1).alias("usable_%"),
                pl.col("n_gnd").median().alias("med_n_gnd"),
                pl.col("gnd_span_s").median().round(0).alias("med_gnd_span_s"),
                pl.col("gnd_max_gap_s").median().round(0).alias("med_gap_s"),
                pl.col("mvt_minus_first_seen_s").median().round(0).alias("med_MVT-firstseen"),
                pl.col("first_seen_minus_block_s").median().round(0).alias("med_firstseen-BLOCK"),
                pl.col("first_move_minus_block_s").median().round(0).alias("med_firstmove-BLOCK"),
            )
            .sort(S.AIRPORT)
        )
    print("\n=== same, LIRF by lane ===")
    with pl.Config(tbl_width_chars=200):
        print(
            d.filter(pl.col(S.AIRPORT) == "LIRF")
            .group_by("lane", S.IS_TEST)
            .agg(
                pl.len().alias("matched"),
                pl.col("has_gnd").mean().mul(100).round(1).alias("any_gnd_%"),
                pl.col("usable").mean().mul(100).round(1).alias("usable_%"),
                pl.col("n_gnd").median().alias("med_n_gnd"),
                pl.col("first_move_minus_block_s").median().round(0).alias("med_firstmove-BLOCK"),
            )
            .sort("lane", S.IS_TEST)
        )


# --------------------------------------------------------------------------------------
# 5. evaluation


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    e = p - y
    ok = np.isfinite(e)
    if ok.sum() < 2:
        return {"n": int(ok.sum()), "rmse": None, "mae": None, "medae": None, "corr": None}
    e, yy, pp = e[ok], y[ok], p[ok]
    corr = float(np.corrcoef(yy, pp)[0, 1]) if yy.std() > 0 and pp.std() > 0 else float("nan")
    return {
        "n": int(ok.sum()),
        "rmse": round(float(np.sqrt(np.mean(e**2))), 1),
        "mae": round(float(np.mean(np.abs(e))), 1),
        "medae": round(float(np.median(np.abs(e))), 1),
        "corr": round(corr, 3),
    }


def cmd_evaluate(args: argparse.Namespace) -> None:
    out = outdir()
    d = pl.read_parquet(out / "signals.parquet").filter(~pl.col(S.IS_TEST))
    oof = pl.read_parquet(paths.RUNS / args.run / "oof.parquet").select(S.MVT_ID, "pred")
    d = d.join(oof, on=S.MVT_ID, how="left")
    d = d.with_columns(
        ((pl.col(S.BLOCK_TIME) - pl.col(S.SCHED_TIME)).dt.total_seconds().abs() <= 10).alias(
            "sched_default"
        ),
        (pl.col(S.TARGET) > 3600).alias("gt1h"),
    )
    y = d[S.TARGET].cast(pl.Float64).to_numpy()
    usable = d["usable"].to_numpy()
    groups: dict[str, np.ndarray] = {
        "all": np.ones(d.height, dtype=bool),
        "with-NM": (d["lane"] == "with-NM").to_numpy(),
        "no-NM": (d["lane"] == "no-NM").to_numpy(),
        "LIRF": (d[S.AIRPORT] == "LIRF").to_numpy(),
        "LIRF no-NM": ((d[S.AIRPORT] == "LIRF") & (d["lane"] == "no-NM")).to_numpy(),
        "LIRF no-NM usable": (
            (d[S.AIRPORT] == "LIRF") & (d["lane"] == "no-NM") & d["usable"]
        ).to_numpy(),
        "sched-default": d["sched_default"].fill_null(False).to_numpy(),
        "> 1 h": d["gt1h"].to_numpy(),
        "usable ADS-B": usable,
        "no-NM usable": ((d["lane"] == "no-NM") & d["usable"]).to_numpy(),
    }
    pred = d["pred"].to_numpy()
    adsb = d["adsb_taxi_from_move_s"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    preds = {
        "ADS-B MVT − first move": adsb,
        "ADS-B liftoff − first move": d["adsb_taxi_liftoff_move_s"]
        .cast(pl.Float64)
        .fill_null(np.nan)
        .to_numpy(),
        "ADS-B MVT − first seen": d["mvt_minus_first_seen_s"]
        .cast(pl.Float64)
        .fill_null(np.nan)
        .to_numpy(),
        f"model {args.run.split('_')[0]}": pred,
    }
    # fold-free combination rules (no fitting, nothing learned from the label)
    preds["combo: ADS-B if usable else model"] = np.where(usable & np.isfinite(adsb), adsb, pred)
    preds["combo: mean(ADS-B, model) if usable"] = np.where(
        usable & np.isfinite(adsb), 0.5 * (adsb + pred), pred
    )

    rows = []
    for gname, gmask in groups.items():
        for pname, p in preds.items():
            mm = _metrics(y[gmask], np.asarray(p, dtype=float)[gmask])
            rows.append({"group": gname, "estimator": pname, **mm})
    t = pl.DataFrame(rows)
    with pl.Config(tbl_rows=300, tbl_width_chars=160, fmt_str_lengths=40):
        print(t)

    # --- does the first ADS-B movement pin BLOCK_TIME? the cleanest statement ------------
    err = pl.col("first_move_minus_block_s")
    print("\n=== t_first_move − BLOCK_TIME (s), matched departures with a first movement ===")
    with pl.Config(tbl_width_chars=200, tbl_rows=40):
        print(
            d.filter(err.is_not_null())
            .group_by(S.AIRPORT)
            .agg(
                pl.len().alias("n"),
                err.quantile(0.25).round(0).alias("p25"),
                err.median().round(0).alias("p50"),
                err.quantile(0.75).round(0).alias("p75"),
                (err.abs() <= 60).mean().mul(100).round(1).alias("within_60s_%"),
                (err.abs() <= 180).mean().mul(100).round(1).alias("within_180s_%"),
                pl.col("usable").sum().alias("usable"),
            )
            .sort(S.AIRPORT)
        )
        print("\n=== same, on the `usable` subset only ===")
        print(
            d.filter(err.is_not_null() & pl.col("usable"))
            .group_by(S.AIRPORT)
            .agg(
                pl.len().alias("n"),
                err.quantile(0.25).round(0).alias("p25"),
                err.median().round(0).alias("p50"),
                err.quantile(0.75).round(0).alias("p75"),
                (err.abs() <= 60).mean().mul(100).round(1).alias("within_60s_%"),
                (err.abs() <= 180).mean().mul(100).round(1).alias("within_180s_%"),
            )
            .sort(S.AIRPORT)
        )

    # --- which of the two off-block stamps does ADS-B agree with? -----------------------
    print("\n=== medians (s) of first_move − BLOCK, first_move − AOBT_3, BLOCK − AOBT_3 ===")
    with pl.Config(tbl_width_chars=200, tbl_rows=40):
        print(
            d.filter(pl.col("first_move_minus_block_s").is_not_null())
            .group_by(S.AIRPORT)
            .agg(
                pl.len().alias("n"),
                pl.col("first_move_minus_block_s").median().round(0).alias("move−BLOCK"),
                pl.col("first_move_minus_aobt_s").median().round(0).alias("move−AOBT"),
                pl.col("block_minus_aobt_s").median().round(0).alias("BLOCK−AOBT"),
                (pl.col("first_move_minus_aobt_s").abs() <= 60)
                .mean()
                .mul(100)
                .round(1)
                .alias("move≈AOBT_60s_%"),
                (pl.col("first_move_minus_block_s").abs() <= 60)
                .mean()
                .mul(100)
                .round(1)
                .alias("move≈BLOCK_60s_%"),
            )
            .sort(S.AIRPORT)
        )

    # --- per-airport RMSE, usable rows only ---------------------------------------------
    rows2 = []
    for ap in sorted(set(d[S.AIRPORT].to_list())):
        k = ((d[S.AIRPORT] == ap) & d["usable"]).to_numpy()
        if k.sum() < 5:
            continue
        rows2.append(
            {
                "airport": ap,
                "n_usable": int(k.sum()),
                "rmse_adsb": _metrics(y[k], adsb[k])["rmse"],
                "rmse_model": _metrics(y[k], pred[k])["rmse"],
                "rmse_mean": _metrics(y[k], 0.5 * (adsb[k] + pred[k]))["rmse"],
                "medae_adsb": _metrics(y[k], adsb[k])["medae"],
                "medae_model": _metrics(y[k], pred[k])["medae"],
            }
        )
    print("\n=== RMSE on `usable` rows, per airport ===")
    with pl.Config(tbl_width_chars=200, tbl_rows=40):
        print(pl.DataFrame(rows2))
    t.write_parquet(out / "evaluation.parquet", compression="zstd")
    print(f"\n-> {out / 'evaluation.parquet'}")


# --------------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("days").set_defaults(fn=cmd_days)
    f = sub.add_parser("fetch")
    f.add_argument("--days", nargs="+", required=True, help="YYYY-MM-DD, processed one at a time")
    f.add_argument("--keep-tars", action="store_true", help="do not delete the tar parts")
    f.add_argument("--force", action="store_true", help="re-fetch a day already in the manifest")
    f.set_defaults(fn=cmd_fetch)
    sub.add_parser("match").set_defaults(fn=cmd_match)
    sub.add_parser("extract").set_defaults(fn=cmd_extract)
    e = sub.add_parser("evaluate")
    e.add_argument("--run", default="exp0068_xgb_r4_d12_mcw3000__s1-2-3__month_loo")
    e.set_defaults(fn=cmd_evaluate)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
