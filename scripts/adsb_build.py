"""Production build of the ADS-B ground-track features (ADSB.lol, ODbL 1.0).

One resumable, idempotent pass per UTC day:

    release  ->  heatmap/ (739 MB, 20 s)   or, when that day ships none, traces/ (3.1 GB, ~300 s)
             ->  points within 12 km of a challenge airport, on the ground or <= 4000 ft
             ->  visits (first movement, stops, path, lift-off)
             ->  global runway-aware assignment to the day's departures
             ->  data/external/adsblol_ground/<YYYY-MM-DD>.parquet keyed by MVT_ID_mvt

Only **derived** features are stored — never positions, never a track — so the output is
tiny (a few hundred kB per day) and can be rsynced to the cluster like any other archive.

The extraction, the visit logic and the matcher are imported from the two pilot scripts
(`adsb_pilot.py`, `adsb_pilot_phase2.py`) rather than copied, so there is exactly one
implementation of each and the pilot's measurements keep applying:
internal note 2026-09-19_adsb-pilot, internal note 2026-09-19_adsb-pilot-phase2.

    uv run python scripts/adsb_build.py plan                     # day order, what is done
    uv run python scripts/adsb_build.py run --stage 1            # Jan+Jul 2026 then Jan+Jul 2025
    uv run python scripts/adsb_build.py run --stage 2            # the remaining 2025 months
    uv run python scripts/adsb_build.py run --months 2026-01 …   # any explicit set
    uv run python scripts/adsb_build.py status                   # progress + coverage so far
    uv run python scripts/adsb_build.py run --source traces --days 2025-01-01 2025-01-10
                                        # rebuild a day range from the full traces

``--source traces`` skips the heatmap and takes the full traces (2–5 s cadence) for every
day. It exists for 2025-01-01..2025-04-20, where the heatmap drops ground tracks at EGLL,
EDDF and EHAM. With it, a day counts as done
only when its manifest entry already says ``source: traces``, so re-running is a resume,
not a second download. Several processes (a Slurm array over ``--days`` ranges) may run at
once: every manifest update re-reads the file under an exclusive lock.

Resumability: a day is skipped when its parquet exists and its manifest entry records the
same builder version. Kill the process at any point and re-run the same command; nothing is
recomputed. Progress is appended to ``outputs/adsb_build/progress.jsonl``.

Compute: downloads run in a thread pool (``--workers``, default 6) under a transient-disk
budget; **filtering and feature building are single-threaded and serial**, one day at a
time, so only one CPU-heavy process is ever active. Payloads are deleted as soon as a day
is written.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import queue
import struct
import sys
import tarfile
import threading
import time
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from adsb_pilot import ATTRIBUTION, LICENCE, UA, Asset, download, filter_day, preferred_urls
from adsb_pilot_phase2 import (
    POINT_SCHEMA,
    _assign,
    _norm_desig,
    _runway_thresholds,
    decode_slice,
    points_from_slice,
    visits,
)
from prc import paths
from prc.data import schema as S

#: "1" = derived features only. "2" additionally keeps the filtered points of the day
#: (``POINTS_DIR``), so a new feature idea costs a re-read of ~2 GB of parquet instead of a
#: 400 GB re-download. Bumping this invalidates every day in the manifest: that is the
#: point, the archive has to be rebuilt once to gain the points.
BUILDER_VERSION = "2"
OUT_DIR = paths.EXTERNAL / "adsblol_ground"
#: Filtered points, all aircraft (arrivals included), within 12 km of a challenge airport
#: and on the ground or <= 4000 ft. ~2-10 MB/day zstd.
POINTS_DIR = paths.EXTERNAL / "adsblol_points"
PROGRESS = paths.OUT / "adsb_build" / "progress.jsonl"
TRANSIENT = Path(os.environ.get("PRC_ADSB_TRANSIENT", "~/.cache/prc-adsb-build")).expanduser()
DISK_BUDGET_BYTES = 9_000_000_000  # < the 10 GB the task allows, across all in-flight days
HEATMAP_RESERVE = 1_100_000_000
TRACE_RESERVE = 4_400_000_000

#: The eight own-track features that survived the 2025-vs-2026 drift check
#: (internal note 2026-09-19_adsb-pilot-phase2 §5: |median shift| and |mean shift| both
#: <= 0.25 sd). Deliberately absent: `n_stops` (median shift 0.87), `mvt_minus_first_seen_s`
#: (0.35), `n_gnd` (0.45), `cov_airport_day` (1.36) and every airport-state feature (the
#: state-only control was +4.8 s, i.e. harmful).
FEATURES = (
    "adsb_mvt_minus_first_move_s",
    "adsb_mvt_minus_first_gnd_s",
    "adsb_gnd_path_m",
    "adsb_stop_total_s",
    "adsb_final_hold_s",
    "adsb_mean_gs_kt",
    "adsb_first_seen_dist_thr_m",
    "adsb_gnd_max_gap_s",
)
FLAGS = ("adsb_matched", "adsb_has_gnd")
OUT_SCHEMA: dict[str, pl.DataType] = {
    S.MVT_ID: pl.Float64,
    **{c: pl.Int8 for c in FLAGS},
    **{c: pl.Float32 for c in FEATURES},
}

STAGE1 = ("2026-01", "2026-07", "2025-01", "2025-07")
STAGE2 = (
    "2025-02",
    "2025-03",
    "2025-04",
    "2025-05",
    "2025-06",
    "2025-08",
    "2025-09",
    "2025-10",
    "2025-11",
    "2025-12",
)

KM_LAT = 111.32


# --------------------------------------------------------------------------------------
# day list


def month_days(month: str) -> list[date]:
    y, m = int(month[:4]), int(month[5:])
    d = date(y, m, 1)
    out = []
    while d.month == m:
        out.append(d)
        d += timedelta(days=1)
    return out


def done_days(source: str = "auto") -> set[str]:
    man = load_manifest()
    return {
        d
        for d, v in man.get("days", {}).items()
        if v.get("builder") == BUILDER_VERSION
        and (source == "auto" or v.get("source") == source)
        and (OUT_DIR / f"{d}.parquet").is_file()
        and (POINTS_DIR / f"{d}.parquet").is_file()
    }


def load_manifest() -> dict:
    p = OUT_DIR / "manifest.json"
    return json.loads(p.read_text()) if p.is_file() else {}


def save_manifest(man: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT_DIR / f"manifest.json.tmp{os.getpid()}"
    tmp.write_text(json.dumps(man, indent=2, sort_keys=True))
    tmp.replace(OUT_DIR / "manifest.json")


def record_day(rec: dict) -> None:
    """Add one day to the manifest. Re-reads it under an exclusive lock, so parallel
    builders (a Slurm array over day ranges) never overwrite each other's entries."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "manifest.lock").open("w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        man = load_manifest()
        man.setdefault("source", "https://github.com/adsblol/globe_history_<year>")
        man.setdefault("licence", LICENCE)
        man.setdefault("attribution", ATTRIBUTION)
        man.setdefault("builder", BUILDER_VERSION)
        man.setdefault("days", {})
        man["days"][rec["day"]] = rec
        save_manifest(man)


# --------------------------------------------------------------------------------------
# payload acquisition (network; runs in the prefetch pool)


@dataclass
class Payload:
    day: date
    kind: str = ""  # "heatmap" | "traces" | "none"
    files: list[Path] = field(default_factory=list)
    assets: list[dict] = field(default_factory=list)
    reserved: int = 0
    error: str = ""


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _replica_urls(day: date) -> list[str]:
    """Preferred replica first, then prod and staging — the heatmap is missing from some."""
    urls: list[str] = []
    pref = preferred_urls(day)
    if pref is not None:
        urls.append(pref[1][0])
    for replica in ("prod", "staging"):
        tag = f"v{day:%Y.%m.%d}-planes-readsb-{replica}-0"
        u = (
            f"https://github.com/adsblol/globe_history_{day.year}/releases/download/"
            f"{tag}/{tag}.tar.aa"
        )
        if u not in urls:
            urls.append(u)
    return urls


def grab_heatmap(day: date, dest: Path) -> tuple[int, str] | None:
    """Copy the `heatmap/` members out of the head of a tar into one framed file.

    Each member is stored as ``<u32 length><raw gzip bytes>``; the connection is dropped at
    the first ``traces/`` member, so only the head of ``.tar.aa`` crosses the network. No
    decompression happens here — this runs in the download pool.
    """
    for url in _replica_urls(day):
        n = 0
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with (
                urllib.request.urlopen(req, timeout=180) as resp,
                dest.open("wb") as out,
                tarfile.open(fileobj=resp, mode="r|") as tf,  # type: ignore[arg-type]
            ):
                for member in tf:
                    if "traces/" in member.name:
                        break
                    if not member.isfile() or "heatmap/" not in member.name:
                        continue
                    fh = tf.extractfile(member)
                    if fh is None:
                        continue
                    body = fh.read()
                    out.write(struct.pack("<I", len(body)))
                    out.write(body)
                    n += 1
        except Exception:  # noqa: BLE001 — a bad replica is not fatal, try the next
            n = 0
        if n:
            return n, url.rsplit("/", 1)[-1]
        dest.unlink(missing_ok=True)
    return None


def iter_heatmap(path: Path) -> Iterator[bytes]:
    with path.open("rb") as fh:
        while True:
            head = fh.read(4)
            if len(head) < 4:
                return
            yield fh.read(struct.unpack("<I", head)[0])


def grab_traces(day: date, tmpdir: Path) -> list[dict] | None:
    """Full tar parts to disk (resumable, sha256 recorded) — the fallback route."""
    pref = preferred_urls(day)
    if pref is None:
        return None
    _, urls = pref
    recs = []
    for u in urls:
        name = u.rsplit("/", 1)[-1]
        dest = tmpdir / name
        try:
            size = _asset_size(u)
            sha = download(Asset(name, u, size), dest)
        except SystemExit:
            return None
        recs.append({"name": name, "url": u, "bytes": size, "sha256": sha})
    return recs


def _asset_size(url: str) -> int:
    import re

    req = urllib.request.Request(url, headers={"User-Agent": UA, "Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        cr = r.headers.get("Content-Range", "")
    m = re.search(r"/(\d+)$", cr)
    if not m:
        raise SystemExit(f"no Content-Range for {url}")
    return int(m.group(1))


# --------------------------------------------------------------------------------------
# per-day build (CPU; serial, main thread only)


def _movements() -> pl.DataFrame:
    """Departures with the columns the matcher and the features need. Read once."""
    return (
        pl.read_parquet(
            paths.INTERIM / "movements.parquet",
            columns=[S.MVT_ID, S.PHASE, S.AIRPORT, S.MVT_TIME, S.RUNWAY],
        )
        .filter(pl.col(S.PHASE) == "DEP")
        .with_columns(
            pl.col(S.MVT_TIME).dt.date().cast(pl.String).alias("day"),
            pl.col(S.MVT_TIME).dt.epoch("s").cast(pl.Float64).alias("mvt_ts"),
            _norm_desig(S.RUNWAY).alias("_desig"),
        )
        .drop(S.PHASE, S.RUNWAY)
    )


def points_from_heatmap(path: Path) -> pl.DataFrame:
    frames = []
    for body in iter_heatmap(path):
        import gzip

        raw = gzip.decompress(body) if body[:2] == b"\x1f\x8b" else body
        r = decode_slice(raw)
        if r is None:
            continue
        p = points_from_slice(*r)
        if p is not None:
            frames.append(p)
    df = pl.concat(frames) if frames else pl.DataFrame([], schema=POINT_SCHEMA)  # type: ignore[arg-type]
    return df.sort("airport", "icao24", "ts")


def points_from_traces(day: date, files: list[Path], scratch: Path) -> pl.DataFrame:
    """Reuse the pilot's trace filter, then read back the day it wrote."""
    filter_day(files, day, scratch)
    f = scratch / "points" / f"{day:%Y-%m-%d}.parquet"
    df = pl.read_parquet(f).with_columns(pl.col("airport").cast(pl.String))
    f.unlink(missing_ok=True)
    (scratch / "aircraft" / f"{day:%Y-%m-%d}.parquet").unlink(missing_ok=True)
    return df.select(list(POINT_SCHEMA)).sort("airport", "icao24", "ts")


def day_features(day: date, pts: pl.DataFrame, mv: pl.DataFrame, thr: pl.DataFrame) -> pl.DataFrame:
    """One row per departure of that day, keyed by ``MVT_ID_mvt``."""
    m = mv.filter(pl.col("day") == f"{day:%Y-%m-%d}")
    if m.height == 0:
        return pl.DataFrame([], schema=OUT_SCHEMA)  # type: ignore[arg-type]
    if pts.height == 0:
        vis = pl.DataFrame(
            [], schema={"airport": pl.String, "icao24": pl.String, "visit": pl.Int64}
        )
        a = pl.DataFrame(
            [],
            schema={
                S.MVT_ID: pl.Float64,
                "icao24": pl.String,
                "visit": pl.Int64,
                "liftoff_minus_mvt_s": pl.Float64,
            },
        )
        d = m.join(a, on=S.MVT_ID, how="left")
        for c in (
            "n_gnd",
            "t_first_move",
            "t_first_gnd",
            "gnd_path_m",
            "stop_total_s",
            "final_hold_s",
            "mean_gs_kt",
            "gnd_max_gap_s",
            "first_seen_lat",
            "first_seen_lon",
        ):
            d = d.with_columns(pl.lit(None, dtype=pl.Float64).alias(c))
    else:
        vis = visits(pts)
        a = _assign(m, vis, thr)
        d = m.join(a, on=S.MVT_ID, how="left").join(
            vis.drop("airport"), on=["icao24", "visit"], how="left"
        )
    d = d.join(thr, left_on=[S.AIRPORT, "_desig"], right_on=["airport", "_desig"], how="left")
    return d.select(
        pl.col(S.MVT_ID),
        pl.col("icao24").is_not_null().cast(pl.Int8).alias("adsb_matched"),
        (pl.col("n_gnd").fill_null(0) > 0).cast(pl.Int8).alias("adsb_has_gnd"),
        (pl.col("mvt_ts") - pl.col("t_first_move"))
        .cast(pl.Float32)
        .alias("adsb_mvt_minus_first_move_s"),
        (pl.col("mvt_ts") - pl.col("t_first_gnd"))
        .cast(pl.Float32)
        .alias("adsb_mvt_minus_first_gnd_s"),
        pl.col("gnd_path_m").cast(pl.Float32).alias("adsb_gnd_path_m"),
        pl.col("stop_total_s").cast(pl.Float32).alias("adsb_stop_total_s"),
        pl.col("final_hold_s").cast(pl.Float32).alias("adsb_final_hold_s"),
        pl.col("mean_gs_kt").cast(pl.Float32).alias("adsb_mean_gs_kt"),
        (
            (
                (pl.col("first_seen_lon") - pl.col("thr_lon"))
                * KM_LAT
                * (pl.col("first_seen_lat") * math.pi / 180).cos()
                * 1000
            ).pow(2)
            + ((pl.col("first_seen_lat") - pl.col("thr_lat")) * KM_LAT * 1000).pow(2)
        )
        .sqrt()
        .cast(pl.Float32)
        .alias("adsb_first_seen_dist_thr_m"),
        pl.col("gnd_max_gap_s").cast(pl.Float32).alias("adsb_gnd_max_gap_s"),
    )


def build_day(pay: Payload, mv: pl.DataFrame, thr: pl.DataFrame, scratch: Path) -> dict:
    day = pay.day
    t0 = time.time()
    if pay.kind == "heatmap":
        pts = points_from_heatmap(pay.files[0])
    elif pay.kind == "traces":
        pts = points_from_traces(day, pay.files, scratch)
    else:
        pts = pl.DataFrame([], schema=POINT_SCHEMA)  # type: ignore[arg-type]
    feats = day_features(day, pts, mv, thr)
    # Keep the filtered points: every later feature generation (stand context, arrival
    # tracks, airframe rotation) reads these instead of re-downloading the release.
    POINTS_DIR.mkdir(parents=True, exist_ok=True)
    ppath = POINTS_DIR / f"{day:%Y-%m-%d}.parquet"
    pts.sort("airport", "icao24", "ts").write_parquet(ppath, compression="zstd")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{day:%Y-%m-%d}.parquet"
    feats.write_parquet(out, compression="zstd")
    return {
        "day": f"{day:%Y-%m-%d}",
        "builder": BUILDER_VERSION,
        "source": pay.kind,
        "assets": pay.assets,
        "points": pts.height,
        "rows": feats.height,
        "matched": int(feats["adsb_matched"].sum()) if feats.height else 0,
        "has_gnd": int(feats["adsb_has_gnd"].sum()) if feats.height else 0,
        "out_bytes": out.stat().st_size,
        "points_bytes": ppath.stat().st_size,
        "out_sha256": _sha256(out),
        "seconds": round(time.time() - t0, 1),
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


# --------------------------------------------------------------------------------------
# prefetch pool + serial build loop


class DiskBudget:
    """Bytes reserved by in-flight payloads, so the pool cannot fill the disk."""

    def __init__(self, cap: int) -> None:
        self._cap = cap
        self._used = 0
        self._cv = threading.Condition()

    def acquire(self, n: int) -> None:
        with self._cv:
            while self._used + n > self._cap and self._used > 0:
                self._cv.wait()
            self._used += n

    def release(self, n: int) -> None:
        with self._cv:
            self._used -= n
            self._cv.notify_all()


def prefetch(day: date, budget: DiskBudget, tmpdir: Path, source: str = "auto") -> Payload:
    pay = Payload(day=day)
    tmpdir.mkdir(parents=True, exist_ok=True)
    if source == "traces":
        return _prefetch_traces(pay, budget, tmpdir)
    budget.acquire(HEATMAP_RESERVE)
    pay.reserved = HEATMAP_RESERVE
    hm = tmpdir / f"{day:%Y-%m-%d}.hmcat"
    got = grab_heatmap(day, hm)
    if got is not None:
        n, asset = got
        pay.kind = "heatmap"
        pay.files = [hm]
        pay.assets = [
            {"name": asset, "slices": n, "bytes": hm.stat().st_size, "sha256": _sha256(hm)}
        ]
        return pay
    budget.release(pay.reserved)
    pay.reserved = 0
    return _prefetch_traces(pay, budget, tmpdir)


def _prefetch_traces(pay: Payload, budget: DiskBudget, tmpdir: Path) -> Payload:
    budget.acquire(TRACE_RESERVE)
    pay.reserved = TRACE_RESERVE
    recs = grab_traces(pay.day, tmpdir)
    if recs:
        pay.kind = "traces"
        pay.files = [tmpdir / r["name"] for r in recs]
        pay.assets = recs
    else:
        pay.kind = "none"
        pay.error = "no release with usable data"
    return pay


def cleanup(pay: Payload, budget: DiskBudget) -> None:
    for f in pay.files:
        f.unlink(missing_ok=True)
    if pay.reserved:
        budget.release(pay.reserved)
        pay.reserved = 0


def cmd_run(args: argparse.Namespace) -> None:
    months = list(args.months) if args.months else list(STAGE1 if args.stage == 1 else STAGE2)
    days = [d for m in months for d in month_days(m)]
    if args.days:
        lo, hi = (date.fromisoformat(x) for x in args.days)
        days = []
        d = lo
        while d <= hi:
            days.append(d)
            d += timedelta(days=1)
        months = [f"{lo:%Y-%m-%d}..{hi:%Y-%m-%d}"]
    today = datetime.now(UTC).date()
    days = [d for d in days if d < today]
    done = set() if args.force else done_days(args.source)
    todo = [d for d in days if f"{d:%Y-%m-%d}" not in done]
    n_have = len(days) - len(todo)
    if args.max_days:
        todo = todo[: args.max_days]
    print(
        f"months {months}: {len(days)} days, {n_have} already built, {len(todo)} to do",
        flush=True,
    )
    if not todo:
        return
    TRANSIENT.mkdir(parents=True, exist_ok=True)
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    mv = _movements()
    thr = _runway_thresholds()
    if DISK_BUDGET_BYTES < TRACE_RESERVE:
        raise SystemExit("disk budget is below one trace payload")
    budget = DiskBudget(DISK_BUDGET_BYTES)

    q: queue.Queue[Payload] = queue.Queue(maxsize=max(1, args.workers))
    stop = threading.Event()

    def producer() -> None:
        """Download in parallel and queue each payload **as it completes**.

        An earlier version waited for days in strict order and deadlocked: several workers
        held heatmap reservations for days that could not be queued while another, whose day
        needed the 4.4 GB trace fallback, waited for disk that only frees when the main loop
        consumes a payload. Nothing here depends on day order — the manifest is keyed by day
        — so completion order is the right order.
        """
        import concurrent.futures as cf

        try:
            with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(prefetch, d, budget, TRANSIENT, args.source): d for d in todo}
                for fut in cf.as_completed(futs):
                    d = futs[fut]
                    if stop.is_set():
                        break
                    try:
                        q.put(fut.result())
                    except Exception as e:  # noqa: BLE001
                        q.put(Payload(day=d, kind="none", error=f"{type(e).__name__}: {e}"))
        finally:
            q.put(Payload(day=date(1970, 1, 1), kind="__end__"))

    threading.Thread(target=producer, daemon=True).start()
    n_ok = 0
    t_start = time.time()
    try:
        while True:
            pay = q.get()
            if pay.kind == "__end__":
                break
            if pay.kind == "none":
                rec = {
                    "day": f"{pay.day:%Y-%m-%d}",
                    "builder": BUILDER_VERSION,
                    "source": "none",
                    "error": pay.error,
                    "rows": 0,
                    "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
                }
            else:
                rec = build_day(pay, mv, thr, TRANSIENT)
            cleanup(pay, budget)
            if pay.kind == "none" and args.source == "traces":
                # keep the day's existing (heatmap) entry and files; a re-run retries it
                print(f"{rec['day']}: no traces ({pay.error}); left as it was", flush=True)
            else:
                record_day(rec)
            with PROGRESS.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            n_ok += 1
            el = time.time() - t_start
            print(
                f"[{n_ok}/{len(todo)}] {rec['day']} {rec['source']:7s} "
                f"rows={rec.get('rows', 0)} matched={rec.get('matched', 0)} "
                f"gnd={rec.get('has_gnd', 0)} {rec.get('seconds', 0)}s "
                f"| elapsed {el / 60:.0f}m eta {el / n_ok * (len(todo) - n_ok) / 60:.0f}m",
                flush=True,
            )
    finally:
        stop.set()
    write_readme()
    print(f"done: {n_ok} days in {(time.time() - t_start) / 60:.0f} min", flush=True)


# --------------------------------------------------------------------------------------


def write_readme() -> None:
    POINTS_DIR.mkdir(parents=True, exist_ok=True)
    (POINTS_DIR / "README.md").write_text(
        """# ADS-B filtered points — ADSB.lol

**Gitignored.** One parquet per UTC day, `<YYYY-MM-DD>.parquet`: every position report
within 12 km of a challenge airport that is on the ground or at or below 4000 ft, for
**all** aircraft (arrivals included), sorted by (airport, icao24, ts). Columns: `airport`,
`icao24`, `ts` (epoch s), `lat`, `lon`, `on_ground`, `alt_ft`, `gs_kt` (+ `track_deg` and
`flight` on the days that came from the traces rather than the heatmap). ~4 MB/day.

Kept since builder version 2 so that a new feature generation costs a parquet read instead
of re-downloading ~550 GB of releases. The derived
per-day feature files live in `../adsblol_ground/` (v1) and `../adsblol_ground_v2/` (v2,
written by `scripts/adsb_v2.py archive`).

## Source and licence

ADSB.lol `globe_history_<year>` daily releases, **ODbL 1.0**. Attribution and share-alike
apply exactly as for `../adsblol_ground/README.md`. Registered in
`docs/data-sources.md`.
"""
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "README.md").write_text(
        f"""# ADS-B ground features (derived) — ADSB.lol

**Gitignored.** One parquet per UTC day, `<YYYY-MM-DD>.parquet`, keyed by `MVT_ID_mvt`,
holding only **derived** features — no positions, no tracks, no callsigns. Built by
`scripts/adsb_build.py` (builder version {BUILDER_VERSION}); `manifest.json` records, per
day, which release asset was used, its sha256, and the sha256 of the output parquet.

## Source and licence

* Upstream: https://github.com/adsblol/globe_history_2025 and `_2026`, one release per UTC
  day per replica. The `heatmap/` files are used where a day ships them (739 MB, ~20 s);
  the per-aircraft `traces/` are the fallback (~3.1 GB, ~300 s) on the ~24 % of days that
  do not.
* **Licence: {LICENCE}.** Attribution is required wherever a derived table is published:

  > {ATTRIBUTION}

  ODbL share-alike applies to a published *derived database*; features consumed by a model
  are a produced work and need the notice above.
* Organizer clarification (2026-09-17): open data is allowed if declared, and ADSB.lol is
  named as acceptable. Register the row in `docs/data-sources.md` before a
  feature block ships.

## Columns

| column | meaning |
|---|---|
| `MVT_ID_mvt` | the departure |
| `adsb_matched` | a trace was assigned to this departure (runway-aware global assignment) |
| `adsb_has_gnd` | the assigned visit has at least one on-ground point |
| `adsb_mvt_minus_first_move_s` | take-off stamp minus the first sustained ground movement |
| `adsb_mvt_minus_first_gnd_s` | take-off stamp minus the first on-ground point |
| `adsb_gnd_path_m` | ground track length from first movement to lift-off |
| `adsb_stop_total_s` | time below 1 kt after the first movement |
| `adsb_final_hold_s` | the last stop before lift-off (the holding-point queue) |
| `adsb_mean_gs_kt` | mean ground speed while moving |
| `adsb_first_seen_dist_thr_m` | distance from the first point of the departure to its runway threshold |
| `adsb_gnd_max_gap_s` | largest gap between consecutive on-ground points (quality) |

Everything is anchored on `MVT_TIME_UTC_mvt`, which is populated on the ranking rows, so
every column is computable at test time. Nothing reads `BLOCK_TIME_UTC_mvt` or the target.

The eight numeric features are exactly those whose 2025-vs-2026 distribution shift is
within 0.25 sd. `n_stops`,
`mvt_minus_first_seen_s`, `n_gnd`, `cov_airport_day` and all airport-state features are
deliberately excluded.

## Re-running

```bash
uv run python scripts/adsb_build.py run --stage 1   # Jan+Jul 2026, Jan+Jul 2025
uv run python scripts/adsb_build.py run --stage 2   # the remaining 2025 months
uv run python scripts/adsb_build.py status
```

Idempotent per day and resumable: a day already in `manifest.json` at this builder version
is skipped.
"""
    )


def cmd_plan(args: argparse.Namespace) -> None:
    done = done_days()
    for stage, months in (("1", STAGE1), ("2", STAGE2)):
        for m in months:
            days = [d for d in month_days(m) if d < datetime.now(UTC).date()]
            have = sum(1 for d in days if f"{d:%Y-%m-%d}" in done)
            print(f"stage {stage}  {m}: {have}/{len(days)} days built")
    print(f"total built: {len(done)}")


def cmd_status(args: argparse.Namespace) -> None:
    man = load_manifest()
    days = man.get("days", {})
    if not days:
        print("nothing built yet")
        return
    ok = [v for v in days.values() if v.get("rows")]
    src = {}
    for v in days.values():
        src[v.get("source", "?")] = src.get(v.get("source", "?"), 0) + 1
    tot = sum(v.get("out_bytes", 0) for v in days.values())
    print(f"{len(days)} days in the manifest, sources {src}, output {tot / 1e6:.1f} MB")
    print(
        f"rows {sum(v.get('rows', 0) for v in ok):,}  matched {sum(v.get('matched', 0) for v in ok):,}"
        f"  with ground {sum(v.get('has_gnd', 0) for v in ok):,}"
    )
    files = sorted(OUT_DIR.glob("*.parquet"))
    if not files:
        return
    d = pl.concat([pl.read_parquet(f).with_columns(pl.lit(f.stem).alias("day")) for f in files])
    mv = pl.read_parquet(
        paths.INTERIM / "movements.parquet", columns=[S.MVT_ID, S.AIRPORT, S.SPLIT_MONTH]
    )
    d = d.join(mv, on=S.MVT_ID, how="left")
    with pl.Config(tbl_rows=60, tbl_width_chars=160):
        print(
            d.group_by(S.SPLIT_MONTH, S.AIRPORT)
            .agg(
                pl.len().alias("dep"),
                pl.col("adsb_matched").mean().mul(100).round(1).alias("matched_%"),
                pl.col("adsb_has_gnd").mean().mul(100).round(1).alias("gnd_%"),
            )
            .sort(S.SPLIT_MONTH, S.AIRPORT)
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan").set_defaults(fn=cmd_plan)
    r = sub.add_parser("run")
    r.add_argument("--stage", type=int, default=1, choices=(1, 2))
    r.add_argument("--months", nargs="*", default=None)
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--max-days", type=int, default=0)
    r.add_argument("--force", action="store_true")
    r.add_argument(
        "--days",
        nargs=2,
        metavar=("FIRST", "LAST"),
        default=None,
        help="inclusive ISO day range; overrides --stage/--months",
    )
    r.add_argument(
        "--source",
        choices=("auto", "traces"),
        default="auto",
        help="auto = heatmap, traces as fallback (default); traces = full traces only",
    )
    r.set_defaults(fn=cmd_run)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
