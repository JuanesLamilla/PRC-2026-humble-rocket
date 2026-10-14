"""ADSB.lol pilot, phase 2 — ADS-B as model *features* on ordinary rows.

Phase 1 (`scripts/adsb_pilot.py`) scored the raw
measurement `MVT_TIME − t_first_move` as if it were a prediction of the label, and a naive
50/50 mean with the model. That is the wrong test for a per-airport offset that a learner
can absorb. Phase 2 asks the right one: build test-observable ADS-B features, fit a
residual model on `label − exp0068 OOF pred`, cross-validate by day, and see what is left.

Everything here is anchored on `MVT_TIME_UTC_mvt`, which is populated on the ranking rows,
so every feature is computable at test time.

Source change: phase 2 reads the **heatmap** files instead of the per-aircraft traces.
They sit at the front of the daily tar, are ~750 MB/day instead of 3.3 GB, decode with one
`np.frombuffer`, and (phase 1 §7) reproduce the trace-derived first-movement time within
15 s on 97.4 % of visits. `fetch` streams the tar and drops the connection at the first
`traces/` member, so only the head of `.tar.aa` is ever transferred.

    uv run python scripts/adsb_pilot_phase2.py plan                  # the day sample
    uv run python scripts/adsb_pilot_phase2.py fetch [--days …]      # heatmap -> points_hm/
    uv run python scripts/adsb_pilot_phase2.py compare               # heatmap vs traces
    uv run python scripts/adsb_pilot_phase2.py features              # -> features.parquet
    uv run python scripts/adsb_pilot_phase2.py residual              # the decision tables
    uv run python scripts/adsb_pilot_phase2.py drift                 # 2025 vs 2026

Compute rules: desktop only, CPU only, one day at a time, nothing kept but parquet.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import random
import sys
import tarfile
import time
import urllib.request
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from adsb_pilot import (
    AIRPORT_LATLON,
    ATTRIBUTION,
    KEEP_RADIUS_KM,
    LICENCE,
    UA,
    outdir,
    preferred_urls,
)
from prc import paths
from prc.data import schema as S

MAGIC = 0x0E7F7C9D
HEAT_DT = np.dtype([("hex", "<u4"), ("lat", "<i4"), ("lon", "<i4"), ("alt", "<i2"), ("gs", "<i2")])
ALT_MAX_UNITS = 160  # alt is stored as feet/25; 160 = 4000 ft
GROUND_SENTINEL = -123
KM_LAT = 111.32

_AP = list(AIRPORT_LATLON)
_LAT = np.array([AIRPORT_LATLON[a][0] for a in _AP])
_LON = np.array([AIRPORT_LATLON[a][1] for a in _AP])

POINT_SCHEMA = {
    "airport": pl.String,
    "icao24": pl.String,
    "ts": pl.Float64,
    "lat": pl.Float64,
    "lon": pl.Float64,
    "on_ground": pl.Boolean,
    "alt_ft": pl.Float32,
    "gs_kt": pl.Float32,
}

SESSION_GAP_S = 1800.0
MOVE_KT = 3.0
STOP_KT = 1.0  # phase 2 uses < 1 kt for "stopped" (the brief's definition)
STATIONARY_KT = 2.0
MATCH_WINDOW_S = 180.0
RUNWAY_COST_S_PER_KM = 60.0  # 1 km off the filed runway costs as much as a minute of time
RUNWAY_CAP_KM = 6.0
BIG_COST = 1e6


# --------------------------------------------------------------------------------------
# 1. day sample


def plan_days(seed: int = 20260919) -> dict[str, list[str]]:
    """20 random 2025 days stratified over the months that have a preferred release, and
    three days in each ranking month. The three phase-1 days are kept for continuity but
    flagged, because they were chosen by LIRF error mass and are not a random sample."""
    rng = random.Random(seed)
    avail: dict[str, list[str]] = {}
    for year in (2025, 2026):
        cache = Path(f"~/.cache/prc-adsb-pilot/PREFERRED_RELEASES_{year}.txt").expanduser()
        if not cache.is_file():
            preferred_urls(date(year, 1, 1))
        for line in cache.read_text().splitlines():
            if "-planes-readsb-" not in line:
                continue
            tag = line.split("/download/")[1].split("/")[0]
            d = tag[1:11].replace(".", "-")
            if "mlatonly" in tag:  # MLAT only, no ADS-B — unusable for surface work
                continue
            avail.setdefault(d[:7], []).append(d)
    phase1 = ["2025-01-05", "2025-07-13", "2025-07-14", "2026-01-20", "2026-07-15"]
    random_2025: list[str] = []
    months = [f"2025-{m:02d}" for m in range(1, 13)]
    per = [2] * 8 + [1] * 4  # 20 days over 12 months
    rng.shuffle(per)
    for mo, k in zip(months, per, strict=True):
        pool = sorted(set(avail.get(mo, [])) - set(phase1))
        random_2025 += rng.sample(pool, min(k, len(pool)))
    rank_2026: list[str] = []
    for mo in ("2026-01", "2026-07"):
        pool = sorted(set(avail.get(mo, [])) - set(phase1))
        rank_2026 += rng.sample(pool, 2)
    return {
        "random_2025": sorted(random_2025),
        "rank_2026": sorted(rank_2026 + [d for d in phase1 if d.startswith("2026")]),
        "phase1_2025_error_mass": [d for d in phase1 if d.startswith("2025")],
    }


def cmd_plan(args: argparse.Namespace) -> None:
    p = plan_days()
    for k, v in p.items():
        print(f"{k} ({len(v)}): {' '.join(v)}")
    print(f"total days: {sum(len(v) for v in p.values())}")


# --------------------------------------------------------------------------------------
# 2. heatmap fetch


def decode_slice(raw: bytes) -> tuple[np.ndarray, np.ndarray] | None:
    """One heatmap slice -> (records, unix seconds). Records before the first timestamp
    marker are the file's own index and are dropped."""
    a = np.frombuffer(raw[: len(raw) // 16 * 16], dtype=HEAT_DT)
    mk = np.flatnonzero(a["hex"] == MAGIC)
    if mk.size == 0:
        return None
    t_ms = (a["lat"][mk].astype(np.int64) << 32) + a["lon"][mk].astype(np.uint32).astype(np.int64)
    which = np.searchsorted(mk, np.arange(a.size), side="right") - 1
    keep = (which >= 0) & (a["hex"] != MAGIC) & (a["hex"] != 0)
    return a[keep], t_ms[which[keep]] / 1000.0


def points_from_slice(a: np.ndarray, ts: np.ndarray) -> pl.DataFrame | None:
    lat = a["lat"] / 1e6
    lon = a["lon"] / 1e6
    ground = a["alt"] == GROUND_SENTINEL
    frames = []
    for j, icao in enumerate(_AP):
        dlon = KEEP_RADIUS_KM / (KM_LAT * math.cos(math.radians(_LAT[j])))
        sel = (np.abs(lat - _LAT[j]) <= KEEP_RADIUS_KM / KM_LAT) & (np.abs(lon - _LON[j]) <= dlon)
        sel &= ground | (a["alt"] <= ALT_MAX_UNITS)
        if not sel.any():
            continue
        dx = (lon[sel] - _LON[j]) * KM_LAT * np.cos(np.radians(lat[sel]))
        dy = (lat[sel] - _LAT[j]) * KM_LAT
        near = dx * dx + dy * dy <= KEEP_RADIUS_KM**2
        if not near.any():
            continue
        idx = np.flatnonzero(sel)[near]
        frames.append(
            pl.DataFrame(
                {
                    "airport": np.full(idx.size, icao),
                    "icao24": np.array([f"{h & 0xFFFFFF:06x}" for h in a["hex"][idx]]),
                    "ts": ts[idx],
                    "lat": lat[idx],
                    "lon": lon[idx],
                    "on_ground": ground[idx],
                    "alt_ft": np.where(ground[idx], np.nan, a["alt"][idx] * 25.0).astype(
                        np.float32
                    ),
                    "gs_kt": (a["gs"][idx] / 10.0).astype(np.float32),
                },
                schema=POINT_SCHEMA,
            )
        )
    return pl.concat(frames) if frames else None


def _stream_heatmap(url: str) -> tuple[list[pl.DataFrame], int, int]:
    """Decode `heatmap/` out of the head of one tar part, then drop the connection."""
    frames: list[pl.DataFrame] = []
    n_slices = n_bytes = 0
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=180) as resp:
        stream = io.BufferedReader(resp, buffer_size=1 << 23)  # type: ignore[arg-type]
        with tarfile.open(fileobj=stream, mode="r|") as tf:
            for member in tf:
                if "traces/" in member.name:
                    break  # everything we want is behind us
                if not member.isfile() or "heatmap/" not in member.name:
                    continue
                fh = tf.extractfile(member)
                if fh is None:
                    continue
                body = fh.read()
                n_bytes += len(body)
                raw = gzip.decompress(body) if body[:2] == b"\x1f\x8b" else body
                r = decode_slice(raw)
                if r is None:
                    continue
                n_slices += 1
                pts = points_from_slice(*r)
                if pts is not None:
                    frames.append(pts)
    return frames, n_slices, n_bytes


def fetch_heatmap_day(day: date, out: Path) -> dict:
    """The day's `heatmap/`, from whichever replica actually ships one.

    Not every release has a heatmap: on 2025-07-15 the prod tar goes ``acas/`` ->
    ``LICENSE-cc0.txt`` -> ``traces/`` with no ``heatmap/`` at all, so the preferred
    replica alone is not enough and the others have to be tried.
    """
    pref = preferred_urls(day)
    tried: list[str] = []
    urls: list[str] = []
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
    t0 = time.time()
    frames: list[pl.DataFrame] = []
    n_slices = n_bytes = 0
    used = ""
    for u in urls:
        tried.append(u.rsplit("/", 1)[-1])
        try:
            frames, n_slices, n_bytes = _stream_heatmap(u)
        except Exception as e:  # noqa: BLE001 — a missing replica is not fatal
            print(f"    {u.rsplit('/', 1)[-1]}: {type(e).__name__}", flush=True)
            continue
        if n_slices:
            used = u.rsplit("/", 1)[-1]
            break
    df = (
        pl.concat(frames) if frames else pl.DataFrame([], schema=POINT_SCHEMA)  # type: ignore[arg-type]
    ).sort("airport", "icao24", "ts")
    (out / "points_hm").mkdir(parents=True, exist_ok=True)
    df.write_parquet(out / "points_hm" / f"{day:%Y-%m-%d}.parquet", compression="zstd")
    stat = {
        "asset": used,
        "tried": tried,
        "slices": n_slices,
        "heatmap_bytes": n_bytes,
        "points": df.height,
        "seconds": round(time.time() - t0, 1),
    }
    print(f"  {day}: {stat}", flush=True)
    return stat


def cmd_fetch(args: argparse.Namespace) -> None:
    out = outdir()
    mpath = out / "manifest_heatmap.json"
    man = json.loads(mpath.read_text()) if mpath.is_file() else {}
    man.setdefault("source", "https://github.com/adsblol/globe_history_<year> (heatmap/ only)")
    man.setdefault("licence", LICENCE)
    man.setdefault("attribution", ATTRIBUTION)
    man.setdefault(
        "note",
        "only the head of <tag>.tar.aa is transferred: the connection is dropped at the "
        "first traces/ member. 10 s cadence, no callsign, no track.",
    )
    man.setdefault("days", {})
    p = plan_days()
    days = args.days or sorted(set(p["random_2025"] + p["rank_2026"] + p["phase1_2025_error_mass"]))
    total = 0
    for i, d in enumerate(days, 1):
        if d in man["days"] and man["days"][d].get("slices", 0) > 0 and not args.force:
            print(f"{d}: already fetched, skipping")
            total += man["days"][d]["heatmap_bytes"]
            continue
        print(f"[{i}/{len(days)}] {d}", flush=True)
        try:
            st = fetch_heatmap_day(date.fromisoformat(d), out)
        except Exception as e:  # noqa: BLE001 — one bad day must not abort the sample
            print(f"  {d}: SKIPPED ({type(e).__name__}: {e})", flush=True)
            continue
        st["fetched_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        man["days"][d] = st
        total += st["heatmap_bytes"]
        mpath.write_text(json.dumps(man, indent=2))
        time.sleep(1.0)  # polite
    mpath.write_text(json.dumps(man, indent=2))
    print(f"{len(days)} days, {total / 1e9:.1f} GB of heatmap transferred -> {mpath}")


# --------------------------------------------------------------------------------------
# 3. visits and features


def _runway_thresholds() -> pl.DataFrame:
    """(airport, designator) -> threshold lat/lon, from the archived OurAirports pull."""
    dirs = sorted(paths.EXTERNAL.glob("ourairports_*"))
    f = dirs[-1] / "runways.csv" if dirs else None
    if f is None or not f.is_file():
        return pl.DataFrame(
            [],
            schema={
                "airport": pl.String,
                "_desig": pl.String,
                "thr_lat": pl.Float64,
                "thr_lon": pl.Float64,
            },
        )
    rw = pl.read_csv(f, ignore_errors=True).filter(
        pl.col("airport_ident").is_in(list(AIRPORT_LATLON))
    )
    ends = []
    for side in ("le", "he"):
        ends.append(
            rw.select(
                pl.col("airport_ident").alias("airport"),
                pl.col(f"{side}_ident")
                .cast(pl.Utf8)
                .str.strip_chars()
                .str.to_uppercase()
                .alias("_desig"),
                pl.col(f"{side}_latitude_deg").cast(pl.Float64).alias("thr_lat"),
                pl.col(f"{side}_longitude_deg").cast(pl.Float64).alias("thr_lon"),
            )
        )
    return (
        pl.concat(ends)
        .drop_nulls(["_desig", "thr_lat", "thr_lon"])
        .unique(subset=["airport", "_desig"], keep="first")
    )


def _norm_desig(col: str) -> pl.Expr:
    c = pl.col(col).cast(pl.Utf8).str.strip_chars().str.to_uppercase()
    num = c.str.extract(r"^(\d{1,2})", 1)
    side = c.str.extract(r"^\d{1,2}([LCR])$", 1).fill_null("")
    return pl.when(num.is_null()).then(c).otherwise(num.str.zfill(2) + side)


def _climbing(g: pl.DataFrame) -> bool:
    a = g["alt_ft"].to_numpy().astype(float)
    a = a[np.isfinite(a)]
    return a.size >= 3 and a[-1] > a[0]


def visits(pts: pl.DataFrame) -> pl.DataFrame:
    """One row per (airport, aircraft, visit) with the phase-2 ground descriptors."""
    pts = pts.with_columns(
        (
            (pl.col("icao24") != pl.col("icao24").shift())
            | (pl.col("airport") != pl.col("airport").shift())
            | ((pl.col("ts") - pl.col("ts").shift()) > SESSION_GAP_S)
        )
        .fill_null(True)
        .cum_sum()
        .alias("visit")
    )
    rows = []
    blank: dict = dict.fromkeys(
        [
            "t_liftoff",
            "liftoff_censored",
            "n_gnd",
            "t_first_seen_dep",
            "t_first_gnd",
            "t_first_move",
            "saw_stationary",
            "gnd_max_gap_s",
            "gnd_path_m",
            "n_stops",
            "stop_total_s",
            "final_hold_s",
            "mean_gs_kt",
            "first_seen_lat",
            "first_seen_lon",
        ]
    )
    for (airport, icao, visit), g in pts.group_by(
        ["airport", "icao24", "visit"], maintain_order=True
    ):
        ts = g["ts"].to_numpy()
        gnd = g["on_ground"].to_numpy()
        gs = np.nan_to_num(g["gs_kt"].to_numpy().astype(float))
        lat = g["lat"].to_numpy()
        lon = g["lon"].to_numpy()
        g2a = np.flatnonzero(gnd[:-1] & ~gnd[1:])
        a2g = np.flatnonzero(~gnd[:-1] & gnd[1:])
        censored = False
        if g2a.size:
            lift_i = int(g2a[-1]) + 1
        elif not gnd.any() and (~gnd).sum() >= 3 and _climbing(g):
            # never seen on the ground but first seen low and climbing: a censored take-off.
            # The climb test is what keeps arrivals out of the departure candidate pool.
            lift_i, censored = 0, True
        else:
            rows.append({"airport": str(airport), "icao24": icao, "visit": int(visit), **blank})
            continue
        t_liftoff = float(ts[lift_i])
        dep0 = int(a2g[0]) + 1 if (a2g.size and a2g[0] + 1 < lift_i) else 0
        gi = np.flatnonzero(gnd[dep0:lift_i]) + dep0
        rec: dict = {
            "airport": str(airport),
            "icao24": icao,
            "visit": int(visit),
            "t_liftoff": t_liftoff,
            "liftoff_censored": censored,
            "n_gnd": int(gi.size),
            "t_first_seen_dep": float(ts[dep0]),
            "t_first_gnd": float(ts[gi[0]]) if gi.size else None,
            "t_first_move": None,
            "saw_stationary": False,
            "gnd_max_gap_s": float(np.diff(ts[gi]).max()) if gi.size > 1 else None,
            "gnd_path_m": None,
            "n_stops": None,
            "stop_total_s": None,
            "final_hold_s": None,
            "mean_gs_kt": None,
            "first_seen_lat": float(lat[dep0]),
            "first_seen_lon": float(lon[dep0]),
            "liftoff_lat": float(lat[lift_i]),
            "liftoff_lon": float(lon[lift_i]),
        }
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
                rec["t_first_move"] = float(ts[gi[k]])
                rec["saw_stationary"] = bool((gs[gi[:k]] < STATIONARY_KT).any())
                seg = gi[k:]
                if seg.size > 1:
                    dx = np.diff(lon[seg]) * KM_LAT * np.cos(np.radians(lat[seg][:-1])) * 1000
                    dy = np.diff(lat[seg]) * KM_LAT * 1000
                    rec["gnd_path_m"] = float(np.sqrt(dx * dx + dy * dy).sum())
                    dt = np.diff(ts[seg])
                    stopped = gs[seg] < STOP_KT
                    rec["stop_total_s"] = float(dt[stopped[:-1]].sum())
                    edges = np.diff(np.concatenate(([0], stopped.astype(np.int8), [0])))
                    rec["n_stops"] = int((edges == 1).sum())
                    # the final hold: the last stopped run before lift-off
                    on = np.flatnonzero(edges == 1)
                    off = np.flatnonzero(edges == -1)
                    if on.size:
                        a_, b_ = int(on[-1]), min(int(off[-1]), seg.size - 1)
                        rec["final_hold_s"] = float(ts[seg[b_]] - ts[seg[a_]])
                    rec["mean_gs_kt"] = (
                        float(gs[seg][gs[seg] >= MOVE_KT].mean()) if moving.any() else None
                    )
        rows.append(rec)
    return pl.DataFrame(rows, infer_schema_length=None)


def _assign(mv: pl.DataFrame, vis: pl.DataFrame, thr: pl.DataFrame | None = None) -> pl.DataFrame:
    """Globally optimal assignment of departures to ADS-B visits, per airport-day.

    The traces carry a callsign and phase 1 used it as the primary key; the heatmap does
    not, so the match has to come from geometry and time alone. Two things matter and
    neither is optional:

    * **global, not greedy.** Nearest-first greedy picks the right aircraft only 78.8 % of
      the time (measured against phase 1's callsign-confirmed matches on 2025-07-13),
      because at a busy airport several take-offs sit inside the ±10 s quantisation of the
      heatmap; a minimum-cost bipartite assignment lets the competing movements claim
      their own aircraft.
    * **runway, not only time.** Where an airport departs off several runways at once
      (LFPG) time cannot separate the candidates at all. The cost adds the distance from
      the ADS-B lift-off point to the threshold of the movement's own ``RUNWAY_mvt``.

    cost = |lift-off − MVT_TIME| seconds + ``RUNWAY_COST_S_PER_KM`` × distance (km),
    capped, and pairs outside ``MATCH_WINDOW_S`` are forbidden.
    """
    from scipy.optimize import linear_sum_assignment

    cand = vis.filter(pl.col("t_liftoff").is_not_null())
    thr_map: dict[tuple[str, str], tuple[float, float]] = {}
    if thr is not None:
        thr_map = {
            (r["airport"], r["_desig"]): (r["thr_lat"], r["thr_lon"]) for r in thr.to_dicts()
        }
    keep = []
    for ap, mg in mv.group_by(S.AIRPORT, maintain_order=True):
        ap = ap[0] if isinstance(ap, tuple) else ap
        vg = cand.filter(pl.col("airport") == ap)
        if vg.height == 0 or mg.height == 0:
            continue
        t = mg["mvt_ts"].to_numpy()
        lt = vg["t_liftoff"].to_numpy()
        cost = np.abs(lt[None, :] - t[:, None])
        forbid = cost > MATCH_WINDOW_S
        vlat = vg["liftoff_lat"].to_numpy().astype(float)
        vlon = vg["liftoff_lon"].to_numpy().astype(float)
        desigs = mg["_desig"].to_list()
        tl = np.array([thr_map.get((ap, d), (np.nan, np.nan))[0] for d in desigs])
        tn = np.array([thr_map.get((ap, d), (np.nan, np.nan))[1] for d in desigs])
        if np.isfinite(tl).any() and np.isfinite(vlat).any():
            dx = (vlon[None, :] - tn[:, None]) * KM_LAT * np.cos(np.radians(vlat[None, :]))
            dy = (vlat[None, :] - tl[:, None]) * KM_LAT
            dist = np.sqrt(dx * dx + dy * dy)
            pen = RUNWAY_COST_S_PER_KM * np.minimum(np.nan_to_num(dist, nan=0.0), RUNWAY_CAP_KM)
            cost = cost + np.where(np.isfinite(dist), pen, 0.0)
        cost = np.where(forbid, BIG_COST, cost)
        r, c = linear_sum_assignment(cost)
        ids = mg[S.MVT_ID].to_list()
        ic = vg["icao24"].to_list()
        vv = vg["visit"].to_list()
        for i, j in zip(r, c, strict=True):
            if cost[i, j] < BIG_COST:
                keep.append((ids[i], ic[j], vv[j], float(lt[j] - t[i])))
    return pl.DataFrame(
        keep,
        schema={
            S.MVT_ID: pl.Float64,
            "icao24": pl.String,
            "visit": pl.Int64,
            "liftoff_minus_mvt_s": pl.Float64,
        },
        orient="row",
    )


def airport_state(pts: pl.DataFrame, mv: pl.DataFrame, thr: pl.DataFrame) -> pl.DataFrame:
    """Congestion seen by ADS-B, per departure: how many other aircraft were moving on the
    ground in the last 10/20 min, and how many sat near the departure runway threshold at
    MVT − 5 min. These exist for every row at a covered airport, even when the departure's
    own track is missing."""
    gnd = pts.filter(pl.col("on_ground"))
    rows = []
    thr_map = {(r["airport"], r["_desig"]): (r["thr_lat"], r["thr_lon"]) for r in thr.to_dicts()}
    for ap, g in gnd.group_by("airport", maintain_order=True):
        ap = ap[0] if isinstance(ap, tuple) else ap
        gg = g.sort("ts")
        ts = gg["ts"].to_numpy()
        gs = np.nan_to_num(gg["gs_kt"].to_numpy().astype(float))
        lat = gg["lat"].to_numpy()
        lon = gg["lon"].to_numpy()
        ic = gg["icao24"].to_numpy()
        m = mv.filter(pl.col(S.AIRPORT) == ap)
        for mid, t, des in zip(
            m[S.MVT_ID].to_list(), m["mvt_ts"].to_list(), m["_desig"].to_list(), strict=True
        ):
            lo10, lo20 = np.searchsorted(ts, [t - 600, t - 1200])
            hi = np.searchsorted(ts, t)
            mv10 = ic[lo10:hi][gs[lo10:hi] >= MOVE_KT]
            mv20 = ic[lo20:hi][gs[lo20:hi] >= MOVE_KT]
            q = None
            key = (ap, des)
            if key in thr_map:
                tl, tn = thr_map[key]
                a_, b_ = np.searchsorted(ts, [t - 360, t - 240])
                if b_ > a_:
                    dx = (lon[a_:b_] - tn) * KM_LAT * np.cos(np.radians(lat[a_:b_])) * 1000
                    dy = (lat[a_:b_] - tl) * KM_LAT * 1000
                    near = (dx * dx + dy * dy) <= 1500.0**2
                    q = int(np.unique(ic[a_:b_][near & (gs[a_:b_] < 5)]).size)
            rows.append(
                {
                    S.MVT_ID: mid,
                    "n_moving_10m": int(np.unique(mv10).size),
                    "n_moving_20m": int(np.unique(mv20).size),
                    "n_queued_thr": q,
                }
            )
    return (
        pl.DataFrame(rows)
        if rows
        else pl.DataFrame(
            [],
            schema={
                S.MVT_ID: pl.Float64,
                "n_moving_10m": pl.Int64,
                "n_moving_20m": pl.Int64,
                "n_queued_thr": pl.Int64,
            },
        )
    )


def cmd_features(args: argparse.Namespace) -> None:
    out = outdir()
    files = [
        f
        for f in sorted((out / "points_hm").glob("*.parquet"))
        if pl.scan_parquet(f).select(pl.len()).collect().item() > 0
    ]
    if not files:
        raise SystemExit("no heatmap points — run `fetch` first")
    print(f"{len(files)} days with a heatmap")
    days = [f.stem for f in files]
    mv = (
        pl.read_parquet(
            paths.INTERIM / "movements.parquet",
            columns=[
                S.MVT_ID,
                S.PHASE,
                S.AIRPORT,
                S.MVT_TIME,
                S.BLOCK_TIME,
                S.TARGET,
                S.AOBT_3,
                S.SPLIT_MONTH,
                S.IS_TEST,
                S.RUNWAY,
                S.STAND,
                S.AIRCRAFT_TYPE,
            ],
        )
        .filter(pl.col(S.PHASE) == "DEP")
        .with_columns(pl.col(S.MVT_TIME).dt.date().cast(pl.String).alias("day"))
        .filter(pl.col("day").is_in(days))
        .with_columns(
            pl.col(S.MVT_TIME).dt.epoch("s").cast(pl.Float64).alias("mvt_ts"),
            _norm_desig(S.RUNWAY).alias("_desig"),
        )
    )
    thr = _runway_thresholds()
    parts = []
    for f in files:
        pts = pl.read_parquet(f).sort("airport", "icao24", "ts")
        day = f.stem
        m = mv.filter(pl.col("day") == day)
        vis = visits(pts)
        a = _assign(m, vis, thr)
        d = m.join(a, on=S.MVT_ID, how="left").join(
            vis.drop("airport"), on=["icao24", "visit"], how="left"
        )
        d = d.join(airport_state(pts, m, thr), on=S.MVT_ID, how="left")
        parts.append(d)
        print(f"  {day}: {m.height} departures, {a.height} matched", flush=True)
    d = pl.concat(parts)
    d = d.join(thr, left_on=[S.AIRPORT, "_desig"], right_on=["airport", "_desig"], how="left")
    d = d.with_columns(
        pl.col("icao24").is_not_null().alias("matched"),
        (pl.col("n_gnd").fill_null(0) > 0).alias("has_gnd"),
        (pl.col("mvt_ts") - pl.col("t_first_move")).alias("mvt_minus_first_move_s"),
        (pl.col("mvt_ts") - pl.col("t_first_gnd")).alias("mvt_minus_first_gnd_s"),
        (pl.col("mvt_ts") - pl.col("t_first_seen_dep")).alias("mvt_minus_first_seen_s"),
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
        .alias("first_seen_dist_thr_m"),
    ).with_columns(
        (
            pl.col("saw_stationary").fill_null(False)
            & (pl.col("n_gnd").fill_null(0) >= 3)  # 10 s cadence: ~2.2x fewer points than a trace
            & (pl.col("gnd_max_gap_s").fill_null(1e9) <= 300)
            & pl.col("mvt_minus_first_move_s").is_not_null()
        ).alias("usable")
    )
    # per (airport, day) coverage share — a legal feature, it needs no labels
    d = d.join(
        d.group_by(S.AIRPORT, "day").agg(pl.col("has_gnd").mean().alias("cov_airport_day")),
        on=[S.AIRPORT, "day"],
        how="left",
    )
    d.write_parquet(out / "features_phase2.parquet", compression="zstd")
    print(f"{d.height} departures -> {out / 'features_phase2.parquet'}")
    with pl.Config(tbl_rows=30, tbl_width_chars=180):
        print(
            d.group_by(S.AIRPORT)
            .agg(
                pl.len().alias("dep"),
                pl.col("matched").mean().mul(100).round(1).alias("matched_%"),
                pl.col("has_gnd").mean().mul(100).round(1).alias("gnd_%"),
                pl.col("usable").mean().mul(100).round(1).alias("usable_%"),
                pl.col("n_queued_thr").is_not_null().mean().mul(100).round(1).alias("queue_%"),
            )
            .sort(S.AIRPORT)
        )


FEATS_OWN = [
    "mvt_minus_first_move_s",
    "mvt_minus_first_gnd_s",
    "mvt_minus_first_seen_s",
    "gnd_path_m",
    "n_stops",
    "stop_total_s",
    "final_hold_s",
    "mean_gs_kt",
    "first_seen_dist_thr_m",
    "n_gnd",
    "gnd_max_gap_s",
    "usable",
    "liftoff_censored",
]
FEATS_STATE = ["n_moving_10m", "n_moving_20m", "n_queued_thr", "cov_airport_day"]


def _fit_residual(
    d: pl.DataFrame, feats: list[str], seed: int = 1, folds: int = 5
) -> tuple[np.ndarray, pl.DataFrame]:
    """Grouped-by-day K-fold LightGBM on the model residual.

    Days are the groups, so no day is ever split across train and test: a day's ADS-B
    coverage is a shared, strongly autocorrelated condition and a row-wise split would
    leak it. Returns the out-of-fold correction and the mean gain importance.
    """
    import lightgbm as lgb

    X = d.select(
        [pl.col(f).cast(pl.Float64) for f in feats]
        + [
            pl.col(S.AIRPORT).cast(pl.Categorical).to_physical().cast(pl.Float64).alias("_ap"),
            pl.col("pred").alias("_pred"),
        ]
    ).to_numpy()
    names = [*feats, "_ap", "_pred"]
    y = (d[S.TARGET].cast(pl.Float64) - d["pred"]).to_numpy()
    days = d["day"].to_numpy()
    u = np.unique(days)
    rng = np.random.default_rng(seed)
    assign = {dd: i % folds for i, dd in enumerate(rng.permutation(u))}
    fold = np.array([assign[dd] for dd in days])
    corr = np.zeros(len(y))
    imps = np.zeros(len(names))
    for k in range(folds):
        te, tr = fold == k, fold != k
        if tr.sum() < 500 or te.sum() == 0:
            continue
        m = lgb.LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=200,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=seed,
            verbose=-1,
            n_jobs=6,
        )
        m.fit(X[tr], y[tr], categorical_feature=[len(feats)])
        corr[te] = m.predict(X[te])
        imps += m.booster_.feature_importance("gain")
    imp = (
        pl.DataFrame({"feature": names, "gain": imps / max(folds, 1)})
        .with_columns((pl.col("gain") / pl.col("gain").sum() * 100).round(1).alias("gain_%"))
        .sort("gain", descending=True)
    )
    return corr, imp


def _boot(y: np.ndarray, a: np.ndarray, b: np.ndarray, days: np.ndarray, n: int = 400) -> str:
    """Day-block bootstrap CI on RMSE(b) − RMSE(a)."""
    rng = np.random.default_rng(7)
    u = np.unique(days)
    idx = {d: np.flatnonzero(days == d) for d in u}
    out = []
    for _ in range(n):
        pick = np.concatenate([idx[d] for d in rng.choice(u, u.size, replace=True)])
        out.append(
            np.sqrt(np.mean((y[pick] - b[pick]) ** 2)) - np.sqrt(np.mean((y[pick] - a[pick]) ** 2))
        )
    lo, hi = np.percentile(out, [2.5, 97.5])
    return f"[{lo:+.2f}, {hi:+.2f}]"


def cmd_residual(args: argparse.Namespace) -> None:
    out = outdir()
    d = pl.read_parquet(out / "features_phase2.parquet").filter(~pl.col(S.IS_TEST))
    oof = pl.read_parquet(paths.RUNS / args.run / "oof.parquet").select(S.MVT_ID, "pred")
    d = d.join(oof, on=S.MVT_ID, how="inner")
    if args.random_days_only:
        keep = set(plan_days()["random_2025"])
        d = d.filter(pl.col("day").is_in(list(keep)))
    print(f"{d.height} labelled departures on {d['day'].n_unique()} days")
    y = d[S.TARGET].cast(pl.Float64).to_numpy()
    base = d["pred"].to_numpy()
    days = d["day"].to_numpy()
    ap = d[S.AIRPORT].to_numpy()
    has_gnd = d["has_gnd"].to_numpy()
    matched = d["matched"].to_numpy()
    aobt = d[S.AOBT_3].is_not_null().to_numpy()
    big = y > 3600
    variants = {
        "own + state": FEATS_OWN + FEATS_STATE,
        "state only": FEATS_STATE,
        "own only": FEATS_OWN,
    }
    rows = []
    preds = {}
    top_imp = None
    for name, feats in variants.items():
        corr, imp = _fit_residual(d, feats, seed=args.seed)
        preds[name] = base + corr
        if name == "own + state":
            top_imp = imp
    # permutation control (ADR 2026-09-17): the same fit with the ADS-B columns shuffled
    # inside each (airport, day). Anything the learner still finds is bookkeeping, not signal.
    rng = np.random.default_rng(args.seed)
    perm = d.with_columns(pl.int_range(pl.len()).alias("_i"))
    order = (
        perm.select("_i", S.AIRPORT, "day")
        .with_columns(pl.lit(rng.random(d.height)).alias("_r"))
        .sort(S.AIRPORT, "day", "_r")["_i"]
        .to_numpy()
    )
    src = perm.sort(S.AIRPORT, "day")["_i"].to_numpy()
    shuffled = d.clone()
    cols = FEATS_OWN + FEATS_STATE
    take = np.empty(d.height, dtype=np.int64)
    take[src] = order
    shuffled = shuffled.with_columns([pl.col(c).gather(pl.Series(take)).alias(c) for c in cols])
    corr, _ = _fit_residual(shuffled, cols, seed=args.seed)
    preds["permuted control"] = base + corr
    # the correction is only trustworthy where the aircraft was actually seen on the ground,
    # and the > 1 h lane is where a residual learner does damage; gate on both.
    own_state = preds["own + state"]
    gate = d["has_gnd"].to_numpy() & (base <= 3600)
    preds["own + state, gated"] = np.where(gate, own_state, base)

    def rm(m, p):
        return round(float(np.sqrt(np.mean((y[m] - p[m]) ** 2))), 2)

    scopes = {
        "(a) rows with ground points": has_gnd,
        "(b) all matched rows": matched,
        "(c) all departures on those days": np.ones(len(y), bool),
        "ordinary (has AOBT, <= 1 h)": aobt & ~big,
        "no-AOBT": ~aobt,
        "> 1 h": big,
    }
    for mo in sorted(set(d[S.SPLIT_MONTH].to_list())):
        scopes[f"month {mo} (ordinary)"] = (d[S.SPLIT_MONTH] == mo).to_numpy() & aobt & ~big
    for sname, mask in scopes.items():
        if mask.sum() < 20:
            continue
        row = {"scope": sname, "n": int(mask.sum()), "exp0068": rm(mask, base)}
        for name, p in preds.items():
            row[name] = rm(mask, p)
            row[name + " Δ CI"] = _boot(y[mask], base[mask], p[mask], days[mask])
        rows.append(row)
    res = pl.DataFrame(rows)
    res.write_parquet(out / "residual_phase2.parquet", compression="zstd")
    with pl.Config(tbl_rows=30, tbl_width_chars=250, fmt_str_lengths=40):
        for half in (
            ["scope", "n", "exp0068", "own + state", "own + state Δ CI"],
            [
                "scope",
                "own only",
                "own only Δ CI",
                "state only",
                "permuted control",
                "permuted control Δ CI",
            ],
        ):
            print(res.select(half))
        best = preds["own + state"]
        per_ap = []
        for a in sorted(set(ap)):
            m = ap == a
            if m.sum() < 50:
                continue
            per_ap.append(
                {
                    "airport": a,
                    "n": int(m.sum()),
                    "gnd_%": round(float(has_gnd[m].mean() * 100), 1),
                    "exp0068": rm(m, base),
                    "own+state": rm(m, best),
                    "state only": rm(m, preds["state only"]),
                    "Δ CI (own+state)": _boot(y[m], base[m], best[m], days[m]),
                }
            )
        print(pl.DataFrame(per_ap))
        print("\n=== gain importance, own + state ===")
        print(top_imp.head(12))


def cmd_drift(args: argparse.Namespace) -> None:
    out = outdir()
    d = pl.read_parquet(out / "features_phase2.parquet").with_columns(
        pl.when(pl.col(S.SPLIT_MONTH).str.starts_with("2025"))
        .then(pl.lit("2025"))
        .otherwise(pl.lit("2026"))
        .alias("yr"),
        pl.col(S.SPLIT_MONTH).str.slice(5, 2).alias("mo"),
    )
    d = d.filter(pl.col("mo").is_in(["01", "07"]))
    feats = [f for f in FEATS_OWN + FEATS_STATE if f not in ("usable", "liftoff_censored")]
    rows = []
    for f in feats:
        g = d.group_by("yr").agg(
            pl.col(f).cast(pl.Float64).median().alias("med"),
            pl.col(f).cast(pl.Float64).std().alias("sd"),
            pl.col(f).is_not_null().mean().mul(100).alias("present"),
        )
        m = {r["yr"]: r for r in g.to_dicts()}
        if "2025" not in m or "2026" not in m:
            continue
        sd = np.nanmean([m["2025"]["sd"] or np.nan, m["2026"]["sd"] or np.nan])
        delta = (m["2026"]["med"] or np.nan) - (m["2025"]["med"] or np.nan)
        rows.append(
            {
                "feature": f,
                "med_2025": round(m["2025"]["med"] or float("nan"), 1),
                "med_2026": round(m["2026"]["med"] or float("nan"), 1),
                "std_shift": round(float(delta / sd) if sd and sd > 0 else float("nan"), 2),
                "present_2025_%": round(m["2025"]["present"], 1),
                "present_2026_%": round(m["2026"]["present"], 1),
            }
        )
    with pl.Config(tbl_rows=30, tbl_width_chars=180):
        print(pl.DataFrame(rows))
        print(
            d.group_by(S.AIRPORT, "yr")
            .agg(
                pl.col("has_gnd").mean().mul(100).round(1).alias("gnd_%"),
                pl.col("usable").mean().mul(100).round(1).alias("usable_%"),
                pl.len().alias("n"),
            )
            .pivot(on="yr", index=S.AIRPORT, values=["gnd_%", "usable_%"])
            .sort(S.AIRPORT)
        )


def cmd_compare(args: argparse.Namespace) -> None:
    """Heatmap vs traces on the days phase 1 also pulled: do the features agree?"""
    out = outdir()
    common = sorted(
        {f.stem for f in (out / "points_hm").glob("*.parquet")}
        & {f.stem for f in (out / "points").glob("*.parquet")}
    )
    if not common:
        raise SystemExit("no day has both sources")
    print("days with both sources:", common)
    for day in common:
        hm = visits(
            pl.read_parquet(out / "points_hm" / f"{day}.parquet").sort("airport", "icao24", "ts")
        )
        tr_pts = pl.read_parquet(out / "points" / f"{day}.parquet").drop("track_deg", "flight")
        tr = visits(
            tr_pts.with_columns(pl.col("airport").cast(pl.String)).sort("airport", "icao24", "ts")
        )

        def uniq(x):
            return (
                x.filter(pl.col("t_first_move").is_not_null())
                .group_by("airport", "icao24")
                .agg(pl.len().alias("k"), pl.all().first())
                .filter(pl.col("k") == 1)
            )

        j = uniq(hm).join(uniq(tr), on=["airport", "icao24"], suffix="_tr")
        j = j.filter((pl.col("t_first_move") - pl.col("t_first_move_tr")).abs() < 3600)
        print(f"\n{day}: {j.height} aircraft-visits present in both")
        cols = [
            "t_first_move",
            "t_liftoff",
            "gnd_path_m",
            "n_stops",
            "stop_total_s",
            "final_hold_s",
            "mean_gs_kt",
            "n_gnd",
        ]
        rows = []
        for c in cols:
            a = j[c].cast(pl.Float64).to_numpy()
            b = j[c + "_tr"].cast(pl.Float64).to_numpy()
            k = np.isfinite(a) & np.isfinite(b)
            if k.sum() < 10:
                continue
            rows.append(
                {
                    "feature": c,
                    "n": int(k.sum()),
                    "median_heatmap−trace": round(float(np.median(a[k] - b[k])), 2),
                    "corr": round(float(np.corrcoef(a[k], b[k])[0, 1]), 4),
                    "median_trace": round(float(np.median(b[k])), 1),
                }
            )
        with pl.Config(tbl_rows=20, tbl_width_chars=160):
            print(pl.DataFrame(rows))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan").set_defaults(fn=cmd_plan)
    f = sub.add_parser("fetch")
    f.add_argument("--days", nargs="*", default=None)
    f.add_argument("--force", action="store_true")
    f.set_defaults(fn=cmd_fetch)
    sub.add_parser("compare").set_defaults(fn=cmd_compare)
    sub.add_parser("features").set_defaults(fn=cmd_features)
    r = sub.add_parser("residual")
    r.add_argument("--run", default="exp0068_xgb_r4_d12_mcw3000__s1-2-3__month_loo")
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--random-days-only", action="store_true")
    r.set_defaults(fn=cmd_residual)
    sub.add_parser("drift").set_defaults(fn=cmd_drift)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
