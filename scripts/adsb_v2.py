"""ADS-B ground features, v2 — stand context, pushback signature, airframe rotation.

Phase 2 (`scripts/adsb_pilot_phase2.py`) built
the departure's *own* track: time from first observed movement to take-off, path length,
stops, final hold. That shipped as `adsb_ground` and is worth -14 s in validation / -21 s on
the board. This script asks what else the same points contain, on three axes:

* **A. Context of the first sighting.** Was the aircraft seen stationary before it first
  moved (then the first movement really is the pushback, not a mid-taxi pickup)? How far was
  its first ground point from the stand the movement table assigns it? What fraction of the
  stand -> threshold distance was already behind it? The stand coordinate table is learned
  from the points themselves: the median position of the pre-movement stationary points of
  every departure off that stand, and of the on-stand stop of every arrival to it.
* **B. Airframe rotation through `icao24`.** The movement table has no registration, but
  every ADS-B track does. Matching the ARRIVAL rows to tracks as well (same global
  assignment, keyed on touchdown + threshold) links a departure to the inbound leg of the
  same airframe. That gives the inbound's in-block stamp and stand (visible on test rows),
  the ground time, and — the point of the exercise — a *physical* measurement of the
  airport's stamp offset: ADS-B on-stand time of the inbound minus the airport's own
  in-block stamp, aggregated per airport / day / 3 h / stand area / operator, as a proxy
  for the hidden `BLOCK - true pushback` offset on the departure.
* **C. The no-flight-plan lane**, where `AOBT_3_flt` is null and the ADS-B pushback estimate
  is the only off-block observation there is.

Nothing here reads the departure's own `BLOCK_TIME_UTC_mvt` or `TAXITIME_SEC_mvt`.

    uv run python scripts/adsb_v2.py sessions     # points_hm/ -> visits_v2/<day>.parquet
    uv run python scripts/adsb_v2.py features     # -> features_v2.parquet
    uv run python scripts/adsb_v2.py residual     # the increment of v2 over v1
    uv run python scripts/adsb_v2.py drift        # 2025 vs 2026 parity per feature

Compute rules: desktop only, CPU only, one day at a time, nothing kept but parquet.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from adsb_pilot_phase2 import (
    KM_LAT,
    MOVE_KT,
    SESSION_GAP_S,
    STATIONARY_KT,
    STOP_KT,
    _assign,
    _climbing,
    _norm_desig,
    _runway_thresholds,
)
from prc import paths
from prc.data import schema as S

ARR_STOP_MIN_S = 180.0  # a stationary run this long after touchdown = parked on stand
PB_MAX_S = 480.0  # a stop this soon after the first movement can end a pushback
PB_MIN_STOP_S = 10.0  # one heatmap sample
MIN_STAND_OBS = 5  # a stand needs this many observations to get a coordinate


def outdir() -> Path:
    """The pilot pull whose filtered points are on disk (``adsb_pilot.outdir`` now resolves
    to the production archive ``adsblol_ground/``, which keeps no points)."""
    return paths.EXTERNAL / "adsblol_2026-09-19"


# --------------------------------------------------------------------------------------
# 1. visits: departure descriptors (as phase 2) + stand context + arrival descriptors


def _median_ll(lat: np.ndarray, lon: np.ndarray) -> tuple[float | None, float | None]:
    if lat.size == 0:
        return None, None
    return float(np.median(lat)), float(np.median(lon))


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) index pairs of the True runs of ``mask``."""
    if mask.size == 0:
        return []
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    on = np.flatnonzero(edges == 1)
    off = np.flatnonzero(edges == -1) - 1
    return list(zip(on.tolist(), off.tolist(), strict=True))


def visits_v2(pts: pl.DataFrame) -> pl.DataFrame:
    """One row per (airport, aircraft, visit) with the departure and arrival descriptors.

    The departure block reproduces `adsb_pilot_phase2.visits` (same constants, same
    definitions) so that v1 and v2 are measured on exactly the same tracks; the extra
    columns are the v2 candidates.
    """
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
    rows: list[dict] = []
    for (airport, icao, visit), g in pts.group_by(
        ["airport", "icao24", "visit"], maintain_order=True
    ):
        ts = g["ts"].to_numpy()
        gnd = g["on_ground"].to_numpy()
        gs = np.nan_to_num(g["gs_kt"].to_numpy().astype(float))
        lat = g["lat"].to_numpy()
        lon = g["lon"].to_numpy()
        rec: dict = {"airport": str(airport), "icao24": icao, "visit": int(visit)}
        g2a = np.flatnonzero(gnd[:-1] & ~gnd[1:])
        a2g = np.flatnonzero(~gnd[:-1] & gnd[1:])

        # ---- arrival side ------------------------------------------------------------
        if a2g.size:
            land_i = int(a2g[0]) + 1
            rec["t_landing"] = float(ts[land_i])
            rec["landing_lat"] = float(lat[land_i])
            rec["landing_lon"] = float(lon[land_i])
            gj = np.flatnonzero(gnd)
            gj = gj[gj >= land_i]
            onblock_i = None
            if gj.size:
                still = gs[gj] < STATIONARY_KT
                for a_, b_ in _runs(still):
                    if ts[gj[b_]] - ts[gj[a_]] >= ARR_STOP_MIN_S:
                        onblock_i = (a_, b_)
                        break
                if onblock_i is not None:
                    a_, b_ = onblock_i
                    rec["t_arr_onblock"] = float(ts[gj[a_]])
                    rec["arr_onblock_censored"] = False
                    la, lo = _median_ll(lat[gj[a_ : b_ + 1]], lon[gj[a_ : b_ + 1]])
                else:  # never seen to settle: the last ground point is the best we have
                    rec["t_arr_onblock"] = float(ts[gj[-1]])
                    rec["arr_onblock_censored"] = True
                    la, lo = float(lat[gj[-1]]), float(lon[gj[-1]])
                rec["arr_stand_lat"], rec["arr_stand_lon"] = la, lo
                rec["arr_taxi_in_s"] = rec["t_arr_onblock"] - rec["t_landing"]
                rec["arr_n_gnd"] = int(gj.size)
                # The *last movement* after touchdown is the robust on-stand measurement:
                # a stationary run of >= ARR_STOP_MIN_S is observed on only ~4 % of visits
                # (the transponder goes quiet on stand), but the moment the wheels stop is
                # visible whenever the track survives one sample longer than the last move.
                mv_j = np.flatnonzero(gs[gj] >= MOVE_KT)
                if mv_j.size:
                    last = int(gj[mv_j[-1]])
                    rec["t_arr_last_move"] = float(ts[last])
                    rec["arr_last_move_lat"] = float(lat[last])
                    rec["arr_last_move_lon"] = float(lon[last])
                    # how much track we still have after it stopped: 0 = the track simply
                    # ended while it was still moving, so the stop was never observed
                    rec["arr_tail_s"] = float(ts[gj[-1]] - ts[last])
                    rec["arr_moved_after_landing_s"] = float(ts[last] - ts[land_i])

        # ---- departure side (phase-2 definitions) ------------------------------------
        censored = False
        if g2a.size:
            lift_i = int(g2a[-1]) + 1
        elif not gnd.any() and (~gnd).sum() >= 3 and _climbing(g):
            lift_i, censored = 0, True
        else:
            rows.append(rec)
            continue
        rec["t_liftoff"] = float(ts[lift_i])
        rec["liftoff_censored"] = censored
        rec["liftoff_lat"] = float(lat[lift_i])
        rec["liftoff_lon"] = float(lon[lift_i])
        dep0 = int(a2g[0]) + 1 if (a2g.size and a2g[0] + 1 < lift_i) else 0
        gi = np.flatnonzero(gnd[dep0:lift_i]) + dep0
        rec["n_gnd"] = int(gi.size)
        rec["t_first_seen_dep"] = float(ts[dep0])
        rec["t_first_gnd"] = float(ts[gi[0]]) if gi.size else None
        rec["first_seen_lat"] = float(lat[dep0])
        rec["first_seen_lon"] = float(lon[dep0])
        rec["gnd_max_gap_s"] = float(np.diff(ts[gi]).max()) if gi.size > 1 else None
        rec["first_gnd_lat"] = float(lat[gi[0]]) if gi.size else None
        rec["first_gnd_lon"] = float(lon[gi[0]]) if gi.size else None
        rec["saw_stationary"] = False
        if not gi.size:
            rows.append(rec)
            continue
        moving = gs[gi] >= MOVE_KT
        run = np.convolve(moving.astype(np.int8), np.ones(3, dtype=np.int8), mode="valid")
        starts = np.flatnonzero(run >= 3)
        k = (
            int(starts[0])
            if starts.size
            else (int(np.flatnonzero(moving)[0]) if moving.any() else None)
        )
        if k is None:
            rows.append(rec)
            continue
        rec["t_first_move"] = float(ts[gi[k]])
        rec["first_move_lat"] = float(lat[gi[k]])
        rec["first_move_lon"] = float(lon[gi[k]])
        pre = gi[:k]
        still_pre = pre[gs[pre] < STATIONARY_KT] if pre.size else pre
        rec["saw_stationary"] = bool(still_pre.size)
        rec["n_pre_stationary"] = int(still_pre.size)
        if still_pre.size:
            # only the last unbroken stationary block counts as "waiting on the stand"
            runs = _runs(gs[pre] < STATIONARY_KT)
            a_, b_ = runs[-1]
            rec["pre_move_stationary_s"] = float(ts[gi[k]] - ts[pre[a_]])
            rec["t_last_stationary"] = float(ts[pre[b_]])
            la, lo = _median_ll(lat[pre[a_ : b_ + 1]], lon[pre[a_ : b_ + 1]])
            rec["stand_lat"], rec["stand_lon"] = la, lo
        seg = gi[k:]
        if seg.size > 1:
            dx = np.diff(lon[seg]) * KM_LAT * np.cos(np.radians(lat[seg][:-1])) * 1000
            dy = np.diff(lat[seg]) * KM_LAT * 1000
            step = np.sqrt(dx * dx + dy * dy)
            rec["gnd_path_m"] = float(step.sum())
            dt = np.diff(ts[seg])
            stopped = gs[seg] < STOP_KT
            rec["stop_total_s"] = float(dt[stopped[:-1]].sum())
            sruns = _runs(stopped)
            rec["n_stops"] = len(sruns)
            if sruns:
                a_, b_ = sruns[-1]
                rec["final_hold_s"] = float(ts[seg[b_]] - ts[seg[a_]])
            rec["mean_gs_kt"] = float(gs[seg][moving[k:]].mean()) if moving[k:].any() else None
            # --- pushback signature: first stop soon after the first movement -----------
            for a_, b_ in sruns:
                dur = ts[seg[b_]] - ts[seg[a_]]
                since = ts[seg[a_]] - ts[seg[0]]
                if since <= PB_MAX_S and dur >= PB_MIN_STOP_S and a_ > 0:
                    rec["pushback_dur_s"] = float(since)
                    rec["t_taxi_start"] = float(ts[seg[b_]])
                    rec["pushback_path_m"] = float(step[:a_].sum())
                    ddx = (
                        (lon[seg[a_]] - lon[seg[0]]) * KM_LAT * math.cos(math.radians(lat[seg[0]]))
                    )
                    ddy = (lat[seg[a_]] - lat[seg[0]]) * KM_LAT
                    rec["pushback_disp_m"] = float(math.hypot(ddx, ddy) * 1000)
                    mg = gs[seg[:a_]]
                    rec["pushback_mean_gs_kt"] = float(mg.mean()) if mg.size else None
                    break
        rows.append(rec)
    return pl.DataFrame(rows, infer_schema_length=None)


def cmd_sessions(args: argparse.Namespace) -> None:
    out = outdir()
    dst = out / "visits_v2"
    dst.mkdir(parents=True, exist_ok=True)
    files = [
        f
        for f in sorted((out / "points_hm").glob("*.parquet"))
        if pl.scan_parquet(f).select(pl.len()).collect().item() > 0
    ]
    print(f"{len(files)} days with heatmap points")
    for f in files:
        o = dst / f.name
        if o.exists() and not args.force:
            continue
        pts = pl.read_parquet(f).sort("airport", "icao24", "ts")
        v = visits_v2(pts)
        v.write_parquet(o, compression="zstd")
        print(f"  {f.stem}: {pts.height} points -> {v.height} visits", flush=True)


# --------------------------------------------------------------------------------------
# 2. features


def _dist_m(la1: str, lo1: str, la2: str, lo2: str) -> pl.Expr:
    return (
        (
            ((pl.col(lo1) - pl.col(lo2)) * KM_LAT * (pl.col(la1) * math.pi / 180).cos() * 1000) ** 2
            + ((pl.col(la1) - pl.col(la2)) * KM_LAT * 1000) ** 2
        )
        .sqrt()
        .cast(pl.Float64)
    )


#: The A block's columns, as expressions over a matched departure table. Shared by the
#: pilot evaluation (`features`) and the production archive (`archive`) so that what ships
#: is bit-for-bit what was measured.
A_COLUMNS = [
    _dist_m("first_gnd_lat", "first_gnd_lon", "stand_tab_lat", "stand_tab_lon").alias(
        "a_first_gnd_dist_stand_m"
    ),
    _dist_m("first_move_lat", "first_move_lon", "stand_tab_lat", "stand_tab_lon").alias(
        "a_first_move_dist_stand_m"
    ),
    _dist_m("stand_tab_lat", "stand_tab_lon", "thr_lat", "thr_lon").alias("a_stand_thr_dist_m"),
    pl.col("pre_move_stationary_s").alias("a_pre_move_stationary_s"),
    pl.col("n_pre_stationary").alias("a_n_pre_stationary"),
    pl.col("saw_stationary").cast(pl.Int8).alias("a_saw_stationary"),
    pl.col("pushback_dur_s").alias("a_pushback_dur_s"),
    pl.col("pushback_path_m").alias("a_pushback_path_m"),
    pl.col("pushback_disp_m").alias("a_pushback_disp_m"),
    pl.col("pushback_mean_gs_kt").alias("a_pushback_mean_gs_kt"),
    (pl.col("mvt_ts") - pl.col("t_taxi_start")).alias("a_mvt_minus_taxi_start_s"),
    (pl.col("mvt_ts") - pl.col("t_last_stationary")).alias("a_mvt_minus_last_still_s"),
]
A_DERIVED = [
    (pl.col("first_seen_dist_thr_m") / pl.col("a_stand_thr_dist_m")).alias("a_frac_remaining")
]
A_DERIVED2 = [
    (
        (pl.col("mvt_ts") - pl.col("t_first_seen_dep")) / pl.col("a_frac_remaining").clip(0.05, 1.0)
    ).alias("a_extrap_full_taxi_s")
]

MV_COLS = [
    S.MVT_ID,
    S.PHASE,
    S.AIRPORT,
    S.MVT_TIME,
    S.BLOCK_TIME,
    S.SCHED_TIME,
    S.TARGET,
    S.AOBT_3,
    S.SPLIT_MONTH,
    S.IS_TEST,
    S.RUNWAY,
    S.STAND,
    S.AIRCRAFT_TYPE,
    S.OPERATOR,
]


def _movements(days: list[str]) -> pl.DataFrame:
    return (
        pl.read_parquet(paths.INTERIM / "movements.parquet", columns=MV_COLS)
        .with_columns(pl.col(S.MVT_TIME).dt.date().cast(pl.String).alias("day"))
        .filter(pl.col("day").is_in(days))
        .with_columns(
            pl.col(S.MVT_TIME).dt.epoch("s").cast(pl.Float64).alias("mvt_ts"),
            pl.col(S.BLOCK_TIME).dt.epoch("s").cast(pl.Float64).alias("block_ts"),
            pl.col(S.SCHED_TIME).dt.epoch("s").cast(pl.Float64).alias("sched_ts"),
            _norm_desig(S.RUNWAY).alias("_desig"),
        )
    )


def cmd_features(args: argparse.Namespace) -> None:
    out = outdir()
    files = sorted((out / "visits_v2").glob("*.parquet"))
    if not files:
        raise SystemExit("no visits — run `sessions` first")
    days = [f.stem for f in files]
    mv = _movements(days)
    thr = _runway_thresholds()
    dep_parts, arr_parts = [], []
    for f in files:
        day = f.stem
        vis = pl.read_parquet(f)
        m = mv.filter(pl.col("day") == day)
        md, ma = m.filter(pl.col(S.PHASE) == "DEP"), m.filter(pl.col(S.PHASE) == "ARR")
        # departures: assign on lift-off (phase-2 matcher, unchanged)
        a = _assign(md, vis.filter(pl.col("t_liftoff").is_not_null()), thr)
        dep_parts.append(
            md.join(a, on=S.MVT_ID, how="left").join(
                vis.drop("airport"), on=["icao24", "visit"], how="left"
            )
        )
        # arrivals: the same assignment, keyed on touchdown instead of lift-off
        av = vis.filter(pl.col("t_landing").is_not_null()).select(
            "airport",
            "icao24",
            "visit",
            pl.col("t_landing").alias("t_liftoff"),
            pl.col("landing_lat").alias("liftoff_lat"),
            pl.col("landing_lon").alias("liftoff_lon"),
        )
        b = _assign(ma, av, thr).rename({"liftoff_minus_mvt_s": "landing_minus_mvt_s"})
        arr_parts.append(
            ma.join(b, on=S.MVT_ID, how="left").join(
                vis.select(
                    "icao24",
                    "visit",
                    "t_landing",
                    "t_arr_onblock",
                    "arr_onblock_censored",
                    "t_arr_last_move",
                    "arr_last_move_lat",
                    "arr_last_move_lon",
                    "arr_tail_s",
                    "arr_moved_after_landing_s",
                    "arr_stand_lat",
                    "arr_stand_lon",
                    "arr_taxi_in_s",
                    "arr_n_gnd",
                ),
                on=["icao24", "visit"],
                how="left",
            )
        )
        print(f"  {day}: dep {md.height}/{a.height}  arr {ma.height}/{b.height}", flush=True)
    dep = pl.concat(dep_parts, how="diagonal_relaxed")
    arr = pl.concat(arr_parts, how="diagonal_relaxed")

    # ---- stand coordinate table, learned from the tracks themselves -------------------
    stand_src = pl.concat(
        [
            dep.select(
                S.AIRPORT, S.STAND, pl.col("stand_lat").alias("la"), pl.col("stand_lon").alias("lo")
            ),
            arr.filter(pl.col("arr_tail_s").fill_null(0) >= 20).select(
                S.AIRPORT,
                S.STAND,
                pl.col("arr_last_move_lat").alias("la"),
                pl.col("arr_last_move_lon").alias("lo"),
            ),
        ]
    ).drop_nulls(["la", "lo", S.STAND])
    stands = (
        stand_src.group_by(S.AIRPORT, S.STAND)
        .agg(
            pl.col("la").median().alias("stand_tab_lat"),
            pl.col("lo").median().alias("stand_tab_lon"),
            pl.len().alias("stand_tab_n"),
            pl.col("la").std().alias("_sd_la"),
        )
        .filter(pl.col("stand_tab_n") >= MIN_STAND_OBS)
    )
    stands.write_parquet(out / "stand_table_v2.parquet", compression="zstd")
    print(f"stand table: {stands.height} (airport, stand) coordinates from {stand_src.height} obs")

    # ---- arrival-side stamp offset: physical on-stand time vs the airport's stamp ------
    arr = arr.with_columns(
        (pl.col("t_arr_last_move") - pl.col("block_ts")).alias("arr_offset_s"),
        (pl.col("block_ts") == pl.col("sched_ts")).alias("arr_is_default"),
    ).with_columns(
        pl.when(
            pl.col("arr_offset_s").abs() <= 1800,
            pl.col("arr_tail_s").fill_null(0) >= 20,  # the wheels were seen to stop
            ~pl.col("arr_is_default").fill_null(False),
        )
        .then(pl.col("arr_offset_s"))
        .otherwise(None)
        .alias("_off"),
        (pl.col("mvt_ts") // 10800).alias("_bucket"),
        pl.col(S.STAND).cast(pl.Utf8).str.slice(0, 1).alias("_area"),
    )
    aggs = {
        "off_ap_day": [S.AIRPORT, "day"],
        "off_ap_bucket": [S.AIRPORT, "_bucket"],
        "off_ap_area": [S.AIRPORT, "_area"],
        "off_ap_stand": [S.AIRPORT, S.STAND],
        "off_ap_op": [S.AIRPORT, S.OPERATOR],
    }
    off_tabs = {
        name: arr.group_by(keys).agg(
            pl.col("_off").median().alias(f"b_{name}_med"),
            pl.col("_off").count().alias(f"b_{name}_n"),
        )
        for name, keys in aggs.items()
    }
    # feed-outage signal: how many of the airport's arrivals that day carry a default stamp
    feed = arr.group_by(S.AIRPORT, "day").agg(
        pl.col("arr_is_default").mean().alias("b_arr_default_share_day"),
        pl.col("arr_offset_s").is_not_null().mean().alias("b_arr_cov_day"),
    )

    # ---- rotation link: the inbound leg of the same airframe ---------------------------
    inb = (
        arr.filter(pl.col("icao24").is_not_null())
        .select(
            S.AIRPORT,
            "icao24",
            pl.col("t_landing").alias("rot_t_landing"),
            pl.col("t_arr_last_move").alias("rot_t_onblock"),
            pl.col("arr_tail_s").alias("b_rot_arr_tail_s"),
            pl.col("arr_taxi_in_s").alias("b_rot_arr_taxi_adsb_s"),
            pl.col("block_ts").alias("rot_block_ts"),
            pl.col(S.STAND).alias("rot_stand"),
            pl.col(S.TARGET).alias("rot_taxi_in_s"),
            pl.col("arr_offset_s").alias("b_rot_arr_offset_s"),
            pl.col("arr_is_default").alias("b_rot_arr_is_default"),
            pl.col("arr_onblock_censored").alias("b_rot_onblock_censored"),
        )
        .drop_nulls("rot_t_onblock")
        .sort("rot_t_onblock")
    )
    dep_k = dep.filter(pl.col("icao24").is_not_null()).select(
        S.MVT_ID, S.AIRPORT, "icao24", "mvt_ts"
    )
    rot = dep_k.sort("mvt_ts").join_asof(
        inb,
        left_on="mvt_ts",
        right_on="rot_t_onblock",
        by=[S.AIRPORT, "icao24"],
        strategy="backward",
    )
    dep = dep.join(rot.drop(S.AIRPORT, "icao24", "mvt_ts"), on=S.MVT_ID, how="left")

    # ---- assemble ---------------------------------------------------------------------
    dep = (
        dep.join(thr, left_on=[S.AIRPORT, "_desig"], right_on=["airport", "_desig"], how="left")
        .join(stands, on=[S.AIRPORT, S.STAND], how="left")
        .join(feed, on=[S.AIRPORT, "day"], how="left")
        .with_columns(
            (pl.col("mvt_ts") // 10800).alias("_bucket"),
            pl.col(S.STAND).cast(pl.Utf8).str.slice(0, 1).alias("_area"),
        )
    )
    for name, keys in aggs.items():
        dep = dep.join(off_tabs[name], on=keys, how="left")
    dep = dep.with_columns(
        # --- v1 columns, recomputed identically so both generations share one table -----
        pl.col("icao24").is_not_null().alias("matched"),
        (pl.col("n_gnd").fill_null(0) > 0).alias("has_gnd"),
        (pl.col("mvt_ts") - pl.col("t_first_move")).alias("mvt_minus_first_move_s"),
        (pl.col("mvt_ts") - pl.col("t_first_gnd")).alias("mvt_minus_first_gnd_s"),
        (pl.col("mvt_ts") - pl.col("t_first_seen_dep")).alias("mvt_minus_first_seen_s"),
        _dist_m("first_seen_lat", "first_seen_lon", "thr_lat", "thr_lon").alias(
            "first_seen_dist_thr_m"
        ),
    ).with_columns(
        (
            pl.col("saw_stationary").fill_null(False)
            & (pl.col("n_gnd").fill_null(0) >= 3)
            & (pl.col("gnd_max_gap_s").fill_null(1e9) <= 300)
            & pl.col("mvt_minus_first_move_s").is_not_null()
        ).alias("usable"),
        *A_COLUMNS,
    )
    dep = dep.with_columns(*A_DERIVED).with_columns(
        *A_DERIVED2,
        # --- B: rotation ------------------------------------------------------------------
        pl.col("rot_t_onblock").is_not_null().cast(pl.Int8).alias("b_rot_found"),
        (pl.col("mvt_ts") - pl.col("rot_block_ts")).alias("b_rot_ground_time_s"),
        (pl.col("mvt_ts") - pl.col("rot_t_onblock")).alias("b_rot_ground_time_adsb_s"),
        (pl.col("rot_stand") == pl.col(S.STAND)).cast(pl.Int8).alias("b_rot_same_stand"),
        pl.col("rot_taxi_in_s").alias("b_rot_taxi_in_s"),
        pl.col("b_rot_arr_is_default").cast(pl.Int8),
        pl.col("b_rot_onblock_censored").cast(pl.Int8),
    )
    dep = dep.with_columns(
        # the estimated taxi time if the airport's stamp offset were the inbound-measured one
        (pl.col("mvt_minus_first_move_s") + pl.col("b_off_ap_day_med")).alias("b_est_taxi_day_s"),
        (pl.col("mvt_minus_first_move_s") + pl.col("b_off_ap_area_med")).alias("b_est_taxi_area_s"),
    )
    dep.write_parquet(out / "features_v2.parquet", compression="zstd")
    print(f"{dep.height} departures -> {out / 'features_v2.parquet'}")
    with pl.Config(tbl_rows=30, tbl_width_chars=200):
        print(
            dep.group_by(S.AIRPORT)
            .agg(
                pl.len().alias("dep"),
                pl.col("has_gnd").mean().mul(100).round(1).alias("gnd_%"),
                pl.col("stand_tab_lat").is_not_null().mean().mul(100).round(1).alias("stand_%"),
                pl.col("a_first_move_dist_stand_m").median().round(0).alias("d_move_stand_m"),
                pl.col("b_rot_found").mean().mul(100).round(1).alias("rot_%"),
                pl.col("b_off_ap_day_med").median().round(0).alias("off_day_s"),
                pl.col("b_off_ap_day_n").median().alias("off_n"),
            )
            .sort(S.AIRPORT)
        )
        print(
            arr.group_by(S.AIRPORT)
            .agg(
                pl.len().alias("arr"),
                pl.col("icao24").is_not_null().mean().mul(100).round(1).alias("matched_%"),
                pl.col("_off").is_not_null().mean().mul(100).round(1).alias("offset_%"),
                pl.col("_off").median().round(0).alias("offset_med_s"),
                pl.col("_off").quantile(0.25).round(0).alias("offset_p25"),
                pl.col("_off").quantile(0.75).round(0).alias("offset_p75"),
            )
            .sort(S.AIRPORT)
        )


# --------------------------------------------------------------------------------------
# 2b. production archive: data/external/adsblol_points -> data/external/adsblol_ground_v2

POINTS_DIR = paths.EXTERNAL / "adsblol_points"
ARCH_DIR = paths.EXTERNAL / "adsblol_ground_v2"
CACHE_DIR = paths.EXTERNAL / "adsblol_v2_cache"
#: The columns the feature block ships. `a_extrap_full_taxi_s` is deliberately absent: it
#: moves 0.20 sd between 2025 and 2026 and is coverage-scale dependent by construction
#: The B (rotation / stamp-offset) columns are
#: absent because they did not pay: v1+B was flat-to-worse on every scope.
ARCHIVE_COLS: tuple[str, ...] = (
    "adsb2_first_gnd_dist_stand_m",
    "adsb2_first_move_dist_stand_m",
    "adsb2_stand_thr_dist_m",
    "adsb2_frac_remaining",
    "adsb2_pre_move_stationary_s",
    "adsb2_n_pre_stationary",
    "adsb2_saw_stationary",
    "adsb2_pushback_dur_s",
    "adsb2_pushback_path_m",
    "adsb2_pushback_disp_m",
    "adsb2_pushback_mean_gs_kt",
    "adsb2_mvt_minus_taxi_start_s",
    "adsb2_mvt_minus_last_still_s",
    "adsb2_stand_known",
)


def _match_dep_day(vis: pl.DataFrame, md: pl.DataFrame, thr: pl.DataFrame) -> pl.DataFrame:
    a = _assign(md, vis.filter(pl.col("t_liftoff").is_not_null()), thr)
    return md.join(a, on=S.MVT_ID, how="left").join(
        vis.drop("airport"), on=["icao24", "visit"], how="left"
    )


def _arr_stand_obs(vis: pl.DataFrame, ma: pl.DataFrame, thr: pl.DataFrame) -> pl.DataFrame:
    """(airport, stand, lat, lon) observations from the arrivals that parked that day."""
    av = vis.filter(pl.col("t_landing").is_not_null()).select(
        "airport",
        "icao24",
        "visit",
        pl.col("t_landing").alias("t_liftoff"),
        pl.col("landing_lat").alias("liftoff_lat"),
        pl.col("landing_lon").alias("liftoff_lon"),
    )
    if av.height == 0 or ma.height == 0:
        return pl.DataFrame(
            [],
            schema={S.AIRPORT: pl.String, S.STAND: pl.String, "la": pl.Float64, "lo": pl.Float64},
        )
    b = _assign(ma, av, thr)
    return (
        ma.join(b, on=S.MVT_ID, how="left")
        .join(
            vis.select("icao24", "visit", "arr_last_move_lat", "arr_last_move_lon", "arr_tail_s"),
            on=["icao24", "visit"],
            how="left",
        )
        .filter(pl.col("arr_tail_s").fill_null(0) >= 20)
        .select(
            S.AIRPORT,
            S.STAND,
            pl.col("arr_last_move_lat").alias("la"),
            pl.col("arr_last_move_lon").alias("lo"),
        )
        .drop_nulls()
    )


def cmd_archive(args: argparse.Namespace) -> None:
    """Turn the kept points into the per-day v2 archive the feature block joins.

    Three passes, because the stand coordinate table is global while everything else is
    per-day: (1) match each day's departures and arrivals to tracks and cache the result,
    (2) build the stand table from every day's observations, (3) emit one parquet per day
    with the shipped columns keyed by ``MVT_ID_mvt``. Passes 2 and 3 are cheap; pass 1 is
    the only expensive one and it is resumable — a cached day is skipped.
    """
    files = sorted(POINTS_DIR.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no points under {POINTS_DIR} — run `scripts/adsb_build.py run` first")
    if args.max_days:
        files = files[: args.max_days]
    days = [f.stem for f in files]
    thr = _runway_thresholds()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ARCH_DIR.mkdir(parents=True, exist_ok=True)

    # ---- pass 1: match, per day, cached -------------------------------------------------
    for f in files:
        day = f.stem
        dep_c, arr_c = CACHE_DIR / f"{day}.dep.parquet", CACHE_DIR / f"{day}.arr.parquet"
        if dep_c.is_file() and arr_c.is_file() and not args.force:
            continue
        mv = _movements([day])
        pts = pl.read_parquet(f).sort("airport", "icao24", "ts")
        vis = visits_v2(pts) if pts.height else pl.DataFrame([])
        md = mv.filter(pl.col(S.PHASE) == "DEP")
        ma = mv.filter(pl.col(S.PHASE) == "ARR")
        if vis.height == 0:
            md.select(S.MVT_ID).write_parquet(dep_c, compression="zstd")
            pl.DataFrame(
                [],
                schema={
                    S.AIRPORT: pl.String,
                    S.STAND: pl.String,
                    "la": pl.Float64,
                    "lo": pl.Float64,
                },
            ).write_parquet(arr_c, compression="zstd")
            continue
        _match_dep_day(vis, md, thr).write_parquet(dep_c, compression="zstd")
        _arr_stand_obs(vis, ma, thr).write_parquet(arr_c, compression="zstd")
        print(f"  matched {day}", flush=True)

    # ---- pass 2: the global stand table -------------------------------------------------
    if args.keep_stand_table:
        # Rebuilding some days (e.g. from full traces) must not move the stand coordinates
        # of every other day: reuse the existing table, so untouched days stay byte-equal.
        stands = pl.read_parquet(ARCH_DIR / "stand_table.parquet")
        print(f"stand table kept: {stands.height} (airport, stand) coordinates")
    else:
        stands = _stand_table(days)
        stands.write_parquet(ARCH_DIR / "stand_table.parquet", compression="zstd")
        print(f"stand table: {stands.height} (airport, stand) coordinates")
    _archive_pass3(days, thr, stands)


def _stand_table(days: list[str]) -> pl.DataFrame:
    obs = []
    for day in days:
        d = pl.read_parquet(CACHE_DIR / f"{day}.dep.parquet")
        if "stand_lat" in d.columns:
            obs.append(
                d.select(
                    S.AIRPORT,
                    S.STAND,
                    pl.col("stand_lat").alias("la"),
                    pl.col("stand_lon").alias("lo"),
                ).drop_nulls()
            )
        obs.append(pl.read_parquet(CACHE_DIR / f"{day}.arr.parquet"))
    stands = (
        pl.concat(obs, how="vertical_relaxed")
        .group_by(S.AIRPORT, S.STAND)
        .agg(
            pl.col("la").median().alias("stand_tab_lat"),
            pl.col("lo").median().alias("stand_tab_lon"),
            pl.len().alias("stand_tab_n"),
        )
        .filter(pl.col("stand_tab_n") >= MIN_STAND_OBS)
    )
    return stands


def _archive_pass3(days: list[str], thr: pl.DataFrame, stands: pl.DataFrame) -> None:
    # ---- pass 3: the per-day feature files ----------------------------------------------
    for day in days:
        dep = pl.read_parquet(CACHE_DIR / f"{day}.dep.parquet")
        out = ARCH_DIR / f"{day}.parquet"
        if "t_liftoff" not in dep.columns:  # a day with no usable release
            dep.select(S.MVT_ID).with_columns(
                [pl.lit(None, dtype=pl.Float32).alias(c) for c in ARCHIVE_COLS]
            ).write_parquet(out, compression="zstd")
            continue
        dep = (
            dep.join(thr, left_on=[S.AIRPORT, "_desig"], right_on=["airport", "_desig"], how="left")
            .join(stands, on=[S.AIRPORT, S.STAND], how="left")
            .with_columns(
                _dist_m("first_seen_lat", "first_seen_lon", "thr_lat", "thr_lon").alias(
                    "first_seen_dist_thr_m"
                )
            )
            .with_columns(*A_COLUMNS)
            .with_columns(*A_DERIVED)
        )
        dep = dep.select(
            S.MVT_ID,
            *[
                pl.col(f"a_{c.removeprefix('adsb2_')}").cast(pl.Float32).alias(c)
                for c in ARCHIVE_COLS
                if c != "adsb2_stand_known"
            ],
            pl.col("stand_tab_lat").is_not_null().cast(pl.Int8).alias("adsb2_stand_known"),
        )
        dep.write_parquet(out, compression="zstd")
    print(f"{len(days)} days -> {ARCH_DIR}")


# --------------------------------------------------------------------------------------
# 3. evaluation

FEATS_V1 = [
    "mvt_minus_first_move_s",
    "mvt_minus_first_gnd_s",
    "gnd_path_m",
    "stop_total_s",
    "final_hold_s",
    "mean_gs_kt",
    "first_seen_dist_thr_m",
    "gnd_max_gap_s",
    "matched",
    "has_gnd",
]
FEATS_A = [
    "a_saw_stationary",
    "a_pre_move_stationary_s",
    "a_n_pre_stationary",
    "a_first_gnd_dist_stand_m",
    "a_first_move_dist_stand_m",
    "a_stand_thr_dist_m",
    "a_frac_remaining",
    "a_extrap_full_taxi_s",
    "a_pushback_dur_s",
    "a_pushback_path_m",
    "a_pushback_disp_m",
    "a_pushback_mean_gs_kt",
    "a_mvt_minus_taxi_start_s",
    "a_mvt_minus_last_still_s",
]
#: A minus the two columns the parity check flags: ``a_extrap_full_taxi_s`` moves 0.20 sd
#: and is coverage-scale dependent by construction (better coverage -> the aircraft is seen
#: earlier -> a different extrapolation factor), which is exactly the failure mode the
#: protocol warns about. This is the set the feature block ships.
FEATS_A_SAFE = [f for f in FEATS_A if f != "a_extrap_full_taxi_s"]

FEATS_B = [
    "b_rot_found",
    "b_rot_ground_time_s",
    "b_rot_ground_time_adsb_s",
    "b_rot_same_stand",
    "b_rot_taxi_in_s",
    "b_rot_arr_offset_s",
    "b_rot_arr_tail_s",
    "b_rot_arr_taxi_adsb_s",
    "b_rot_arr_is_default",
    "b_rot_onblock_censored",
    "b_off_ap_day_med",
    "b_off_ap_day_n",
    "b_off_ap_bucket_med",
    "b_off_ap_bucket_n",
    "b_off_ap_area_med",
    "b_off_ap_area_n",
    "b_off_ap_stand_med",
    "b_off_ap_stand_n",
    "b_off_ap_op_med",
    "b_off_ap_op_n",
    "b_est_taxi_day_s",
    "b_est_taxi_area_s",
    "b_arr_default_share_day",
    "b_arr_cov_day",
]


def _fit(d: pl.DataFrame, feats: list[str], seed: int = 1, folds: int = 5):
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


def _boot(y, a, b, days, n: int = 400) -> str:
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
    d = pl.read_parquet(out / "features_v2.parquet").filter(~pl.col(S.IS_TEST))
    oof = pl.read_parquet(paths.RUNS / args.run / "oof.parquet").select(S.MVT_ID, "pred")
    d = d.join(oof, on=S.MVT_ID, how="inner")
    d = d.with_columns(pl.col("matched").cast(pl.Int8), pl.col("has_gnd").cast(pl.Int8))
    print(f"{d.height} labelled departures on {d['day'].n_unique()} days, base = {args.run}")
    y = d[S.TARGET].cast(pl.Float64).to_numpy()
    base = d["pred"].to_numpy()
    days = d["day"].to_numpy()
    ap = d[S.AIRPORT].to_numpy()
    has_gnd = d["has_gnd"].to_numpy().astype(bool)
    rot = d["b_rot_found"].fill_null(0).to_numpy().astype(bool)
    aobt = d[S.AOBT_3].is_not_null().to_numpy()
    big = y > 3600
    variants = {
        "v1": FEATS_V1,
        "v1+A": FEATS_V1 + FEATS_A,
        "v1+Asafe": FEATS_V1 + FEATS_A_SAFE,
        "v1+B": FEATS_V1 + FEATS_B,
        "v1+A+B": FEATS_V1 + FEATS_A + FEATS_B,
    }
    preds, imps = {}, {}
    for name, feats in variants.items():
        corr, imp = _fit(d, feats, seed=args.seed)
        gate = (has_gnd | rot) & (base <= 3600)
        preds[name] = np.where(gate, base + corr, base)
        imps[name] = imp
        print(f"  fitted {name} ({len(feats)} features)", flush=True)
    # permutation control: shuffle the *new* columns inside each (airport, day)
    rng = np.random.default_rng(args.seed)
    perm = d.with_columns(pl.int_range(pl.len()).alias("_i"))
    order = (
        perm.select("_i", S.AIRPORT, "day")
        .with_columns(pl.lit(rng.random(d.height)).alias("_r"))
        .sort(S.AIRPORT, "day", "_r")["_i"]
        .to_numpy()
    )
    src = perm.sort(S.AIRPORT, "day")["_i"].to_numpy()
    take = np.empty(d.height, dtype=np.int64)
    take[src] = order
    newc = FEATS_A + FEATS_B
    sh = d.with_columns([pl.col(c).gather(pl.Series(take)).alias(c) for c in newc])
    corr, _ = _fit(sh, FEATS_V1 + newc, seed=args.seed)
    gate = (has_gnd | rot) & (base <= 3600)
    preds["permuted A+B"] = np.where(gate, base + corr, base)

    def rm(m, p):
        return round(float(np.sqrt(np.mean((y[m] - p[m]) ** 2))), 2)

    scopes = {
        "(a) pooled": np.ones(len(y), bool),
        "(b) has ground track": has_gnd,
        "(c) rotation linked": rot,
        "ordinary (AOBT, <= 1 h)": aobt & ~big,
        "no-AOBT lane": ~aobt,
        "> 1 h": big,
    }
    for mo in sorted(set(d[S.SPLIT_MONTH].to_list())):
        m = (d[S.SPLIT_MONTH] == mo).to_numpy() & aobt & ~big
        if m.sum() >= 2000:
            scopes[f"month {mo} (ordinary)"] = m
    rows = []
    for sname, mask in scopes.items():
        if mask.sum() < 20:
            continue
        row = {"scope": sname, "n": int(mask.sum()), "base": rm(mask, base)}
        for name, p in preds.items():
            row[name] = rm(mask, p)
        for name in ("v1", "v1+A", "v1+Asafe", "v1+B", "v1+A+B"):
            row[name + " CI"] = _boot(y[mask], base[mask], preds[name][mask], days[mask])
        row["incr CI"] = _boot(y[mask], preds["v1"][mask], preds["v1+Asafe"][mask], days[mask])
        rows.append(row)
    res = pl.DataFrame(rows)
    res.write_parquet(out / "residual_v2.parquet", compression="zstd")
    with pl.Config(tbl_rows=40, tbl_width_chars=260, fmt_str_lengths=40):
        print(
            res.select(
                "scope", "n", "base", "v1", "v1+A", "v1+Asafe", "v1+B", "v1+A+B", "permuted A+B"
            )
        )
        print(res.select("scope", "v1 CI", "v1+A CI", "v1+Asafe CI", "v1+B CI", "v1+A+B CI"))
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
                    "rot_%": round(float(rot[m].mean() * 100), 1),
                    "base": rm(m, base),
                    "v1": rm(m, preds["v1"]),
                    "v1+A": rm(m, preds["v1+A"]),
                    "v1+Asafe": rm(m, preds["v1+Asafe"]),
                    "v1+A+B": rm(m, preds["v1+A+B"]),
                    "Asafe CI vs base": _boot(y[m], base[m], preds["v1+Asafe"][m], days[m]),
                }
            )
        print(pl.DataFrame(per_ap))
        print("\n=== gain importance, v1+A+B ===")
        print(imps["v1+A+B"].head(20))
    imps["v1+A+B"].write_parquet(out / "importance_v2.parquet", compression="zstd")


def cmd_drift(args: argparse.Namespace) -> None:
    out = outdir()
    d = pl.read_parquet(out / "features_v2.parquet").with_columns(
        pl.when(pl.col(S.SPLIT_MONTH).str.starts_with("2025"))
        .then(pl.lit("2025"))
        .otherwise(pl.lit("2026"))
        .alias("yr")
    )
    d = d.filter(pl.col(S.SPLIT_MONTH).str.contains("-01|-07"))
    rows = []
    for f in FEATS_A + FEATS_B:
        a = d.filter(pl.col("yr") == "2025")[f].cast(pl.Float64)
        b = d.filter(pl.col("yr") == "2026")[f].cast(pl.Float64)
        if a.drop_nulls().len() < 50 or b.drop_nulls().len() < 50:
            rows.append({"feature": f, "verdict": "too few"})
            continue
        sd = float(np.nanmean([a.std() or np.nan, b.std() or np.nan]))
        shift = (float(b.median()) - float(a.median())) / sd if sd and sd > 0 else None
        rows.append(
            {
                "feature": f,
                "med_2025": round(float(a.median()), 2),
                "med_2026": round(float(b.median()), 2),
                "std_shift": round(shift, 3) if shift is not None else None,
                "present_2025": round(float(a.is_not_null().mean() * 100), 1),
                "present_2026": round(float(b.is_not_null().mean() * 100), 1),
                "verdict": "drop"
                if shift is None or abs(shift) > 0.25
                else ("safe" if abs(shift) <= 0.20 else "borderline"),
            }
        )
    t = pl.DataFrame(rows, infer_schema_length=None).sort("std_shift", nulls_last=True)
    t.write_parquet(out / "drift_v2.parquet", compression="zstd")
    with pl.Config(tbl_rows=60, tbl_width_chars=200):
        print(t)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sessions")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_sessions)
    s = sub.add_parser("features")
    s.set_defaults(fn=cmd_features)
    s = sub.add_parser("residual")
    s.add_argument("--run", default="exp0147_final_xgb_sqrt_adsb__s1-2-3__month_loo")
    s.add_argument("--seed", type=int, default=1)
    s.set_defaults(fn=cmd_residual)
    s = sub.add_parser("archive")
    s.add_argument("--max-days", type=int, default=0)
    s.add_argument("--force", action="store_true")
    s.add_argument(
        "--keep-stand-table",
        action="store_true",
        help="reuse the existing stand_table.parquet instead of rebuilding it",
    )
    s.set_defaults(fn=cmd_archive)
    s = sub.add_parser("drift")
    s.set_defaults(fn=cmd_drift)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
