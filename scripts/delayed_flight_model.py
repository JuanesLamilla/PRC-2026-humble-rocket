"""The delayed-flight model (DF, plan item P1 of internal note 2026-09-23_synthesis), as a
post-processor over a finished member run.

Departures with ``C = MVT − SCHED > 1 h`` are ~195 k 2025 rows and ~43 k ranking rows, and hold
~39 % of the 12-month ex-rollover squared error. On them the label ``y = MVT − BLOCK = C − d``
(``d = BLOCK − SCHED``, the reported off-block delay) falls in one of four stamp classes:

    default  |d| <= 10 s                y = C exactly
    near     10 s < |d| <= 15 min       y = C − d, y/C ≈ 0.90-0.95
    mid      15 min < |d| <= 1 h        (and the rare d < −1 h)
    late     d > 1 h                    the aircraft left late: an ordinary taxi

Today three separate, under-trained pieces answer "was this delayed flight stamped at the
schedule?": the AUC-stopped corruption head (it only knows ``default``), the lane head
(fitted on the 22 k no-NM rows) and the member's normal expert. DF is one model of the whole
population, NM and no-NM rows together with a ``no_nm`` flag, trained on the raw-scale squared
error of its final prediction ``pred = Σ_k p_k · v_k`` with

    v_default = C,  v_near = C − δ_near,  v_mid = C − δ_mid,  v_late = a normal taxi,

and an auxiliary class log-loss so that the probabilities do not collapse. Three formulations:

    net   a small PyTorch MLP, softmax over the four classes and three bounded value heads;
          loss = raw MSE (labels capped at 24 h) + λ · class log-loss
    gbdt  a LightGBM multiclass classifier with gap weights clip(|C − ŷ_normal|, 5 min, 2 h)²,
          plus LightGBM regressors for δ_near, δ_mid and the late taxi, mixed on the raw scale
    reg   a plain LightGBM L2 regression of y on the same inputs (the no-structure stacker)

Inputs are test-time information only: C, the schedule / MVT clock, airport, airline (callsign
prefix), stand, runway, destination, aircraft type, the NM times when present (with their
missingness), contemporaneous state from *other* rows (NM off-block delays around this row's
schedule and takeoff, arrivals, the airport-day), the stand rotation, weather / winter-ops /
feed-health / turnaround feature blocks, nested fold-safe target encodings of the stamp classes
by airline, stand and destination, and the member's own ``pred``, ``y_hat_normal`` and
``p_sched_default``. Nothing is derived from the row's own ``BLOCK`` or label; ``d`` and the
class are training targets only.

Folds: nested leave-one-month-out over the source run's months. The OOF prediction of month m
comes from models fitted on the other eleven months' delayed rows, whose target encodings in
turn exclude month m *and* their own month. The ranking side is fitted on all twelve months.
Rollover rows (|y − 86,400| < 3,600) are excluded from fitting and from the encodings; they are
still predicted (they are scored in ``all``).

    # 0. the member-independent frame of delayed departures (cached; ~2 min, < 3 GB)
    uv run python scripts/delayed_flight_model.py build

    # 1. fit against a run (bare run name) -> outputs/delayed/<run>__<method>__t<thr>.parquet
    uv run python scripts/delayed_flight_model.py fit --run exp0234_r7c_tabm_slog_nomask_lane \
        --method net

    # 2. write a run-shaped directory outputs/runs/<out>/ with the DF rows swapped in
    uv run python scripts/delayed_flight_model.py apply --run exp0234_r7c_tabm_slog_nomask_lane \
        --method net --out exp0234_r7c_tabm_slog_nomask_lane_df

    # 3. score it (the standard tool, unchanged) and the DF-specific table
    uv run python scripts/paired_delta.py exp0234_r7c_tabm_slog_nomask_lane \
        exp0234_r7c_tabm_slog_nomask_lane_df --ex-rollovers --bootstrap 2000
    uv run python scripts/delayed_flight_model.py report \
        --base exp0234_r7c_tabm_slog_nomask_lane --cand exp0234_r7c_tabm_slog_nomask_lane_df

`apply` replaces ``pred`` on the DF rows and shifts every ``pred_s<seed>`` by the same amount
(``pred_s + (df − pred)``), so the seed spread of the source run is kept and the seed mean is
exactly the DF prediction.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
import warnings
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import eda_no_nm_block_model as A
import paired_delta as PD
from prc import paths

warnings.filterwarnings("ignore", message=".*'eval_set' is deprecated.*")
warnings.filterwarnings("ignore", message=".*Sortedness of columns cannot be checked.*")

DF_DIR = paths.OUT / "delayed"
FRAME_VERSION = 1
#: the cached frame holds every departure with C above this (the smallest threshold we try)
FRAME_THR_S = 1800.0
DEFAULT_THR_S = 3600.0
SCHED_TOL_S = 10.0
NEAR_S = 900.0
LATE_S = 3600.0
LABEL_CAP_S = 86_400.0
ROLLOVER_S, ROLLOVER_WIN_S = 86_400.0, 3_600.0
CLASSES = ("default", "near", "mid", "late")
GATE_MONTHS = ("2025-01", "2025-07")
METHODS = ("net", "gbdt", "reg")
TE_PRIOR = 20.0

CAT_FEATURES = [
    "f_airport",
    "f_airline",
    "f_stand",
    "f_runway",
    "f_ades",
    "f_actype",
    "f_segment",
    "f_ftype",
]
ROW_FEATURES = [
    "f_c_sched",
    "f_no_nm",
    "f_hour",
    "f_dow",
    "f_sched_hour",
    "f_sched_moh",
    # NM times against the schedule / the takeoff (null without an NM record)
    "f_obs_delay",
    "f_proxy_taxi",
    "f_eobt1_minus_sched",
    "f_lobt_minus_sched",
    "f_iobt_minus_sched",
    "f_aobt3_minus_eobt1",
    "f_obs_ratio",
]
STATE_FEATURES = [
    *A.STATE_FEATURES,
    "f_win_cmed_3600",
    "f_win_delayed_share_3600",
    "f_win_ratio_med_3600",
    "f_day_ratio_med",
    "f_day_delayed_share",
    "f_day_delayed_nonm_share",
]
#: feature blocks joined on MVT_ID_mvt (newest parquet of the block that has the columns)
BLOCK_COLUMNS: dict[str, list[str]] = {
    "weather_openmeteo": [
        "om_temp",
        "om_prcp",
        "om_snowfall",
        "om_snow_depth",
        "om_wspd",
        "om_gust",
        "om_deicing",
        "om_frost",
        "om_fog",
        "om_prcp_3h",
        "om_frost_hours_24h",
    ],
    "winter_ops": ["wo_frozen_now", "wo_frozen_3h", "wo_deicing_likely", "wo_hours_since_frozen"],
    "metar_visibility": ["mtr_vis_m", "mtr_lvp", "mtr_snow", "mtr_freezing"],
    "feed_health": [
        "fh_arr_sched_share_1h",
        "fh_arr_sched_share_6h",
        "fh_nofp_excess_1h",
        "fh_nofp_excess_6h",
        "fh_spike_share_1h",
        "fh_spike_share_6h",
        "fh_nofp_run",
    ],
    "turnaround_link": [
        "turn_arr_inblock_gap_s",
        "turn_arr_taxi_in_s",
        "turn_arr_has_fpl",
        "turn_arr_inblock_eq_sched",
        "turn_arr_day_sched_share",
    ],
    "stand_bounds_v2": [
        "stand_prev_inblock_gap_s",
        "stand_next_inblock_gap_s",
        "taxi_upper_bound_s",
        "taxi_lower_bound_s",
        "stand_prev_dep_gap_s",
    ],
    "throughput": ["n_dep_tko_30m", "n_arr_ldg_30m", "n_dep_tko_60m", "n_rwy_tko_60m"],
    "runway_gaps_v2": ["gap_prev_tko", "gap_next_tko"],
}
BLOCK_FEATURES = [c for cols in BLOCK_COLUMNS.values() for c in cols]
DERIVED_FEATURES = ["f_c_minus_ub", "f_c_minus_prev_arr"]
MEMBER_FEATURES = ["m_pred", "m_yhn", "m_psd", "m_pred_over_c", "m_gap"]
#: nested target encodings: key -> the class rates encoded (mid is 1 − the others)
TE_KEYS: dict[str, list[str]] = {
    "airline": ["airport", "airline"],
    "airline_nm": ["airport", "airline", "no_nm"],
    "stand": ["airport", "STAND_mvt"],
    "ades": ["airport", "ADES_mvt"],
}
TE_CLASSES = ("default", "near", "late")
TE_FEATURES = [f"te_{k}_{c}" for k in TE_KEYS for c in TE_CLASSES] + [
    f"te_{k}_logn" for k in TE_KEYS
]


def base_features() -> list[str]:
    return ROW_FEATURES + STATE_FEATURES + A.ROTATION_FEATURES + BLOCK_FEATURES + DERIVED_FEATURES


def all_features() -> list[str]:
    return CAT_FEATURES + base_features() + TE_FEATURES + MEMBER_FEATURES


def _maxrss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def _log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f}s | rss {_maxrss_gb():.2f} GB] {msg}", flush=True)


# --------------------------------------------------------------------------- labels


def stamp_class(d: np.ndarray) -> np.ndarray:
    """0 default, 1 near, 2 mid (15 min – 1 h either way, and BLOCK > 1 h before SCHED),
    3 late (BLOCK > 1 h after SCHED); -1 where ``d`` is missing (ranking rows)."""
    d = np.asarray(d, dtype=np.float64)
    ad = np.abs(d)
    out = np.full(len(d), 2, dtype=np.int64)
    out[d > LATE_S] = 3
    out[ad <= NEAR_S] = 1
    out[ad <= SCHED_TOL_S] = 0
    out[~np.isfinite(d)] = -1
    return out


def is_rollover(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    return np.isfinite(y) & (np.abs(y - ROLLOVER_S) < ROLLOVER_WIN_S)


# --------------------------------------------------------------------------- frame


def _newest_block(name: str, cols: list[str]) -> Path:
    cands = [
        p
        for p in paths.FEATURES.glob(f"{name}__*.parquet")
        if set(cols) <= set(pl.read_parquet_schema(p))
    ]
    if not cands:
        raise SystemExit(f"no {name} parquet with {cols} in {paths.FEATURES}")
    return max(cands, key=lambda p: p.stat().st_mtime)


def add_extra_state(dep: pl.DataFrame) -> pl.DataFrame:
    """Delay-regime state from other rows' test-observable stamps (all departures).

    ``obs_ratio = (AOBT_3 − SCHED) / C`` on NM departures delayed > 30 min: the share of the
    delay spent at the gate. Near 1 the flights around this one left late; near 0 they pushed
    on time and were held (de-icing, snow, ATC), which is when a no-NM row's label is ≈ C.
    """
    ratio = (
        pl.when(pl.col("c_sched") > FRAME_THR_S)
        .then(pl.col("obs_delay") / pl.col("c_sched"))
        .otherwise(None)
    )
    dep = dep.with_columns(
        ratio.alias("_ratio"),
        (pl.col("c_sched") > DEFAULT_THR_S).cast(pl.Float64).alias("_delayed"),
    )
    dep = A.roll(
        dep,
        "SCHED_TIME_UTC_mvt",
        "7200s",
        "-3600s",
        "airport",
        [
            pl.col("c_sched").median().alias("f_win_cmed_3600"),
            pl.col("_delayed").mean().alias("f_win_delayed_share_3600"),
            pl.col("_ratio").median().alias("f_win_ratio_med_3600"),
        ],
    )
    day = dep.group_by("airport", "date").agg(
        pl.col("_ratio").median().alias("f_day_ratio_med"),
        pl.col("_delayed").mean().alias("f_day_delayed_share"),
        pl.col("no_nm").filter(pl.col("_delayed") > 0).mean().alias("f_day_delayed_nonm_share"),
    )
    return dep.join(day, on=["airport", "date"], how="left").drop("_ratio", "_delayed")


def build_frame(cache: bool = True) -> pl.DataFrame:
    """Every departure (train + test) with ``C > FRAME_THR_S`` and the member-independent
    features. The state windows need *all* departures, so they are computed on a narrow frame
    of the whole table first; everything wide is joined after the cut."""
    src = paths.INTERIM / "movements.parquet"
    stamp = f"{int(src.stat().st_mtime)}_{src.stat().st_size}"
    cached = DF_DIR / "_frames" / f"delayed_frame__v{FRAME_VERSION}__{stamp}.parquet"
    if cache and cached.is_file():
        return pl.read_parquet(cached)
    t0 = time.time()
    lf = pl.scan_parquet(src)

    def secs(a: str, b: str) -> pl.Expr:
        return (pl.col(a) - pl.col(b)).dt.total_seconds().cast(pl.Float64)

    dep = (
        lf.filter(pl.col("PHASE_mvt") == "DEP")
        .select(
            "MVT_ID_mvt",
            "airport",
            "month",
            "is_test",
            "MVT_TIME_UTC_mvt",
            "SCHED_TIME_UTC_mvt",
            "AOBT_3_flt",
        )
        .with_columns(
            secs("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt").alias("c_sched"),
            secs("AOBT_3_flt", "SCHED_TIME_UTC_mvt").alias("obs_delay"),
            pl.col("AOBT_3_flt").is_null().alias("no_nm"),
        )
        .collect()
    )
    arr = (
        lf.filter(pl.col("PHASE_mvt") == "ARR")
        .select(
            "PHASE_mvt",
            "airport",
            "STAND_mvt",
            "MVT_TIME_UTC_mvt",
            "SCHED_TIME_UTC_mvt",
            "BLOCK_TIME_UTC_mvt",
        )
        .collect()
    )
    _log(f"narrow departures {dep.height:,}, arrivals {arr.height:,}", t0)
    dep = A.add_state_features(dep, arr)
    dep = add_extra_state(dep)
    _log("state features", t0)
    dep = dep.filter(pl.col("c_sched") > FRAME_THR_S)
    ids = dep["MVT_ID_mvt"]
    wide = (
        lf.filter(pl.col("MVT_ID_mvt").is_in(ids.implode()))
        .select(
            "MVT_ID_mvt",
            "FLIGHT_mvt",
            "ADES_mvt",
            "AIRCRAFT_TYPE_mvt",
            "RUNWAY_mvt",
            "STAND_mvt",
            "TAXITIME_SEC_mvt",
            "BLOCK_TIME_UTC_mvt",
            "LOBT_flt",
            "IOBT_flt",
            "EOBT_1_flt",
            "MARKET_SEGMENT_flt",
            "FLIGHT_TYPE_flt",
        )
        .collect()
    )
    dep = dep.join(wide, on="MVT_ID_mvt", how="left")
    del wide
    arr = arr.filter(pl.col("STAND_mvt").is_not_null()).select(
        "PHASE_mvt", "airport", "STAND_mvt", "SCHED_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt"
    )
    dep = A.add_rotation_features(dep, arr)
    del arr
    _log(f"cut to C > {FRAME_THR_S:.0f} s: {dep.height:,} rows; rotation features", t0)
    for name, cols in BLOCK_COLUMNS.items():
        p = _newest_block(name, cols)
        blk = (
            pl.scan_parquet(p)
            .select("MVT_ID_mvt", *cols)
            .filter(pl.col("MVT_ID_mvt").is_in(ids.implode()))
        )
        dep = dep.join(
            blk.collect().with_columns([pl.col(c).cast(pl.Float64) for c in cols]),
            on="MVT_ID_mvt",
            how="left",
        )
        print(f"  block {name}: {p.name}")
    c = pl.col("c_sched")
    dep = dep.with_columns(
        pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("y"),
        secs("BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt").alias("d"),
        pl.col("FLIGHT_mvt").str.extract(r"^([A-Z]+)", 1).alias("airline"),
        c.alias("f_c_sched"),
        pl.col("no_nm").cast(pl.Float64).alias("f_no_nm"),
        pl.col("MVT_TIME_UTC_mvt").dt.hour().cast(pl.Float64).alias("f_hour"),
        pl.col("MVT_TIME_UTC_mvt").dt.weekday().cast(pl.Float64).alias("f_dow"),
        pl.col("SCHED_TIME_UTC_mvt").dt.hour().cast(pl.Float64).alias("f_sched_hour"),
        pl.col("SCHED_TIME_UTC_mvt").dt.minute().cast(pl.Float64).alias("f_sched_moh"),
        pl.col("obs_delay").alias("f_obs_delay"),
        secs("MVT_TIME_UTC_mvt", "AOBT_3_flt").alias("f_proxy_taxi"),
        secs("EOBT_1_flt", "SCHED_TIME_UTC_mvt").alias("f_eobt1_minus_sched"),
        secs("LOBT_flt", "SCHED_TIME_UTC_mvt").alias("f_lobt_minus_sched"),
        secs("IOBT_flt", "SCHED_TIME_UTC_mvt").alias("f_iobt_minus_sched"),
        secs("AOBT_3_flt", "EOBT_1_flt").alias("f_aobt3_minus_eobt1"),
        (pl.col("obs_delay") / c).alias("f_obs_ratio"),
        (c - pl.col("taxi_upper_bound_s")).alias("f_c_minus_ub"),
        (c - pl.col("f_prev_arr_gap")).alias("f_c_minus_prev_arr"),
    ).with_columns(
        pl.col("airport").alias("f_airport"),
        pl.col("airline").alias("f_airline"),
        (pl.col("airport") + "|" + pl.col("STAND_mvt")).alias("f_stand"),
        (pl.col("airport") + "|" + pl.col("RUNWAY_mvt")).alias("f_runway"),
        pl.col("ADES_mvt").alias("f_ades"),
        pl.col("AIRCRAFT_TYPE_mvt").alias("f_actype"),
        pl.col("MARKET_SEGMENT_flt").alias("f_segment"),
        pl.col("FLIGHT_TYPE_flt").alias("f_ftype"),
    )
    dep = dep.with_columns(
        [pl.col(f).cast(pl.Categorical).to_physical().cast(pl.Int32) for f in CAT_FEATURES]
    )
    keep = [
        "MVT_ID_mvt",
        "airport",
        "month",
        "is_test",
        "date",
        "no_nm",
        "c_sched",
        "y",
        "d",
        "airline",
        "STAND_mvt",
        "ADES_mvt",
        *CAT_FEATURES,
        *base_features(),
    ]
    dep = dep.select(keep).with_columns(
        pl.Series("cls", stamp_class(dep["d"].to_numpy()), dtype=pl.Int8)
    )
    dep = dep.sort("MVT_ID_mvt")
    cached.parent.mkdir(parents=True, exist_ok=True)
    # write-then-rename: array tasks that start together all build the frame, and a plain
    # write let one task read another's half-written file (seen in a job array, 2026-09-23)
    tmp = cached.with_name(f"{cached.stem}.{os.getpid()}.tmp")
    dep.write_parquet(tmp)
    tmp.replace(cached)
    _log(f"frame {dep.height:,} x {dep.width} -> {cached}", t0)
    return dep


def leak_audit(frame: pl.DataFrame, thr: float) -> pl.DataFrame:
    """Null rate of every feature on the training vs the ranking rows with C > thr, split by
    ``no_nm``. A feature present in 2025 and absent in 2026 would be a label-side artefact."""
    f = frame.filter(pl.col("c_sched") > thr)
    rows = []
    for col in [*CAT_FEATURES, *base_features()]:
        r = {"feature": col}
        for is_test in (False, True):
            for nonm in (False, True):
                s = f.filter((pl.col("is_test") == is_test) & (pl.col("no_nm") == nonm))[col]
                null = s.is_null() | (s.is_nan() if s.dtype.is_float() else s.is_null())
                r[f"{'test' if is_test else 'train'}_{'nonm' if nonm else 'nm'}"] = round(
                    float(null.mean()) if len(s) else float("nan"), 4
                )
        rows.append(r)
    return pl.DataFrame(rows)


# --------------------------------------------------------------------------- member


def run_months(run: str) -> list[str]:
    return sorted(
        pl.read_parquet(paths.RUNS / run / "oof.parquet", columns=["month"])["month"]
        .unique()
        .to_list()
    )


def attach_member(frame: pl.DataFrame, run: str) -> pl.DataFrame:
    """Join the member's ``pred`` / ``y_hat_normal`` / ``p_sched_default``: OOF for the
    training rows, the test prediction for the ranking rows. Training rows the run did not
    predict are dropped."""
    cols = ["MVT_ID_mvt", "pred", "y_hat_normal", "p_sched_default"]
    src = paths.RUNS / run
    oof = pl.read_parquet(src / "oof.parquet", columns=cols)
    tst = pl.read_parquet(src / "test_pred.parquet", columns=cols)
    mem = pl.concat([oof, tst], how="vertical").rename(
        {"pred": "m_pred", "y_hat_normal": "m_yhn", "p_sched_default": "m_psd"}
    )
    out = frame.join(mem, on="MVT_ID_mvt", how="inner")
    c = pl.col("c_sched")
    return out.with_columns(
        [pl.col(x).cast(pl.Float64) for x in ("m_pred", "m_yhn", "m_psd")]
    ).with_columns(
        (pl.col("m_pred") / c).alias("m_pred_over_c"),
        (c - pl.col("m_yhn")).alias("m_gap"),
    )


# --------------------------------------------------------------------------- encodings


def nested_te(
    src: pl.DataFrame, rows: pl.DataFrame, held: str | None, prior: float = TE_PRIOR
) -> pl.DataFrame:
    """Smoothed class rates by key for ``rows``, never from the row's own month nor ``held``.

    ``src`` are the fitting rows (months, keys, ``cls``). For a row of month k the statistic
    uses ``src`` minus months {k, held}: the training rows of outer fold ``held`` therefore see
    neither their own month nor the held-out month, the held-out rows see the other eleven
    months, and a ranking row (a month not in ``src``) with ``held=None`` sees all of them.
    Returns ``rows["MVT_ID_mvt"]`` plus :data:`TE_FEATURES`, in ``rows`` order.
    """
    ind = [pl.col("cls").cast(pl.Int64).eq(k).cast(pl.Float64).alias(f"_c{k}") for k in (0, 1, 3)]
    s = src.with_columns(ind)
    base = s if held is None else s.filter(pl.col("month") != held)
    pri = {c: float(base[f"_c{k}"].mean()) for c, k in zip(TE_CLASSES, (0, 1, 3), strict=True)}
    out = rows.select("MVT_ID_mvt", "month").with_row_index("_i")
    for name, keys in TE_KEYS.items():
        agg = [pl.len().alias("_n")] + [pl.col(f"_c{k}").sum().alias(f"_s{k}") for k in (0, 1, 3)]
        per = s.group_by([*keys, "month"]).agg(agg)
        tot = per.group_by(keys).agg([pl.col(c).sum() for c in ("_n", "_s0", "_s1", "_s3")])
        own = per.rename({c: f"{c}_own" for c in ("_n", "_s0", "_s1", "_s3")})
        hel = (
            per.filter(pl.col("month") == held)
            .drop("month")
            .rename({c: f"{c}_held" for c in ("_n", "_s0", "_s1", "_s3")})
            if held is not None
            else None
        )
        r = rows.select("MVT_ID_mvt", "month", *keys).with_row_index("_i")
        r = r.join(tot, on=keys, how="left").join(own, on=[*keys, "month"], how="left")
        if hel is not None:
            r = r.join(hel, on=keys, how="left")
        r = r.sort("_i")

        def left(col: str, r: pl.DataFrame = r) -> pl.Expr:
            e = pl.col(col).fill_null(0) - pl.col(f"{col}_own").fill_null(0)
            if held is not None:
                e = e - pl.when(pl.col("month") == held).then(0).otherwise(
                    pl.col(f"{col}_held").fill_null(0)
                )
            return e.cast(pl.Float64)

        n = left("_n")
        feats = [
            ((left(f"_s{k}") + prior * pri[c]) / (n + prior)).alias(f"te_{name}_{c}")
            for c, k in zip(TE_CLASSES, (0, 1, 3), strict=True)
        ]
        enc = r.select("_i", *feats, (n.clip(0, None) + 1).log().alias(f"te_{name}_logn"))
        out = out.join(enc, on="_i", how="left")
    return out.sort("_i").select("MVT_ID_mvt", *TE_FEATURES)


# --------------------------------------------------------------------------- models


def _holdout(df: pl.DataFrame, frac: float, seed: int) -> np.ndarray:
    """Boolean mask of an inner early-stopping holdout: ``frac`` of the (airport, date) blocks
    of the rows given (never a held-out month's rows)."""
    key = (df["airport"] + "|" + df["date"].cast(pl.Utf8)).to_numpy()
    uniq, inv = np.unique(key, return_inverse=True)
    rng = np.random.default_rng(seed)
    pick = rng.random(len(uniq)) < frac
    return pick[inv]


#: model-matrix dtype; ``fit --float32`` switches it (halves memory; the 30-min threshold's
#: 731 k training rows do not fit a 3 GB memory cap in float64)
_NUM_DTYPE: list[type[pl.DataType]] = [pl.Float64]


def _num_matrix(df: pl.DataFrame, feats: list[str]) -> np.ndarray:
    dt = _NUM_DTYPE[0]
    return df.select([pl.col(f).cast(dt).fill_nan(None) for f in feats]).to_numpy()


def _targets(tr: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    c = tr["c_sched"].to_numpy()
    y = np.minimum(tr["y"].to_numpy(), LABEL_CAP_S)
    d = tr["d"].to_numpy()
    cls = tr["cls"].to_numpy().astype(np.int64)
    return c, y, d, cls


def _ref(df: pl.DataFrame) -> np.ndarray:
    """The late branch's reference: the member's normal expert, clipped to a plausible taxi."""
    return np.clip(np.nan_to_num(df["m_yhn"].to_numpy(), nan=900.0), 120.0, 3600.0)


class Fitted(dict):
    """``{"pred", "p" (n, 4), "v" (n, 4)}`` for one set of rows (``v`` may be absent)."""


def fit_reg(tr: pl.DataFrame, evs: list[pl.DataFrame], seed: int = 0, **_: object) -> list[Fitted]:
    """(c): plain LightGBM L2 on the capped raw label — the no-structure stacker."""
    import lightgbm as lgb

    feats = all_features()
    cat_idx = [i for i, f in enumerate(feats) if f in CAT_FEATURES]
    _, y, _, _ = _targets(tr)
    hold = _holdout(tr, 0.1, seed)
    x = _num_matrix(tr, feats)
    m = lgb.LGBMRegressor(**_lgb_params(seed), objective="regression").fit(
        x[~hold],
        y[~hold],
        eval_set=[(x[hold], y[hold])],
        categorical_feature=cat_idx,
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return [Fitted(pred=np.maximum(m.predict(_num_matrix(e, feats)), 0.0)) for e in evs]


def _lgb_params(seed: int) -> dict:
    return {
        "n_estimators": 3000,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_child_samples": 50,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "cat_smooth": 20.0,
        "cat_l2": 10.0,
        "min_data_per_group": 50,
        "max_cat_threshold": 64,
        "verbose": -1,
        "num_threads": 8,
        "random_state": seed,
    }


def gap_weights(c: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """clip(|C − ŷ_normal|, 5 min, 2 h)², mean 1: the cost of a wrong class at this row."""
    w = np.clip(np.abs(c - ref), 300.0, 7200.0) ** 2
    return w / w.mean()


def fit_gbdt(tr: pl.DataFrame, evs: list[pl.DataFrame], seed: int = 0, **_: object) -> list[Fitted]:
    """(b): gap-weighted LightGBM multiclass + three LightGBM value regressors, mixed raw."""
    import lightgbm as lgb

    feats = all_features()
    cat_idx = [i for i, f in enumerate(feats) if f in CAT_FEATURES]
    c, y, d, cls = _targets(tr)
    hold = _holdout(tr, 0.1, seed)
    x = _num_matrix(tr, feats)
    w = gap_weights(c, _ref(tr))
    clf = lgb.LGBMClassifier(**_lgb_params(seed), objective="multiclass", num_class=4).fit(
        x[~hold],
        cls[~hold],
        sample_weight=w[~hold],
        eval_set=[(x[hold], cls[hold])],
        eval_sample_weight=[w[hold]],
        categorical_feature=cat_idx,
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    regs = {}
    for k, target in ((1, np.clip(d, -NEAR_S, NEAR_S)), (2, np.clip(d, -LATE_S, LATE_S)), (3, y)):
        sel = cls == k
        regs[k] = lgb.LGBMRegressor(**_lgb_params(seed), objective="regression").fit(
            x[sel & ~hold],
            target[sel & ~hold],
            eval_set=[(x[sel & hold], target[sel & hold])],
            categorical_feature=cat_idx,
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
    out = []
    for e in evs:
        xe = _num_matrix(e, feats)
        ce = e["c_sched"].to_numpy()
        p = clf.predict_proba(xe)
        v = np.stack(
            [
                ce,
                ce - np.clip(regs[1].predict(xe), -NEAR_S, NEAR_S),
                ce - np.clip(regs[2].predict(xe), -LATE_S, LATE_S),
                np.maximum(regs[3].predict(xe), 0.0),
            ],
            axis=1,
        )
        out.append(Fitted(pred=np.maximum((p * v).sum(axis=1), 0.0), p=p, v=v))
    return out


# ---- (a) the network


class _Prep:
    """Numeric: asinh(x / scale), standardised, NaN -> 0 plus a missing flag where the training
    rows have NaNs. Categoricals: codes seen >= ``min_count`` times in training, else 0."""

    def __init__(self, tr: pl.DataFrame, num: list[str], cat: list[str], min_count: int = 20):
        self.num, self.cat = num, cat
        x = _num_matrix(tr, num)
        self.scale = np.nanmedian(np.abs(x), axis=0)
        self.scale = np.where(np.isfinite(self.scale) & (self.scale > 0), self.scale, 1.0)
        z = np.arcsinh(x / self.scale)
        self.mu = np.nan_to_num(np.nanmean(z, axis=0))
        self.sd = np.nan_to_num(np.nanstd(z, axis=0), nan=1.0)
        self.sd = np.where(self.sd > 1e-6, self.sd, 1.0)
        self.flag = np.flatnonzero(np.isnan(x).any(axis=0))
        self.vocab = []
        for f in cat:
            v = tr[f].fill_null(-1).to_numpy()
            u, n = np.unique(v, return_counts=True)
            keep = u[n >= min_count]
            self.vocab.append({int(k): i + 1 for i, k in enumerate(keep)})

    @property
    def cards(self) -> list[int]:
        return [len(v) + 1 for v in self.vocab]

    def num_x(self, df: pl.DataFrame) -> np.ndarray:
        x = _num_matrix(df, self.num)
        z = (np.arcsinh(x / self.scale) - self.mu) / self.sd
        miss = np.isnan(x[:, self.flag]).astype(np.float64)
        return np.concatenate([np.nan_to_num(z, nan=0.0), miss], axis=1).astype(np.float32)

    def cat_x(self, df: pl.DataFrame) -> np.ndarray:
        cols = []
        for f, voc in zip(self.cat, self.vocab, strict=True):
            v = df[f].fill_null(-1).to_numpy()
            cols.append(np.array([voc.get(int(k), 0) for k in v], dtype=np.int64))
        return np.stack(cols, axis=1) if cols else np.zeros((df.height, 0), dtype=np.int64)


def _atanh_init(x: np.ndarray, scale: float) -> float:
    return float(np.arctanh(np.clip(np.nanmedian(x) / scale, -0.95, 0.95))) if len(x) else 0.0


def fit_net(
    tr: pl.DataFrame,
    evs: list[pl.DataFrame],
    seed: int = 0,
    lam: float = 0.2,
    n_seeds: int = 3,
    epochs: int = 60,
    patience: int = 6,
    width: int = 256,
    batch: int = 1024,
    lr: float = 1e-3,
    device: str = "cpu",
    **_: object,
) -> list[Fitted]:
    """(a): softmax over the four classes and three bounded value heads, trained on
    ``mean((Σ p_k v_k − y)²)`` in ks² plus ``lam`` · class log-loss; ``n_seeds`` nets averaged.

    ``v_near = C − 900·tanh(a₁)``, ``v_mid = C − 3600·tanh(a₂)``,
    ``v_late = ŷ_ref·exp(clamp(a₃, −3, 3))`` with ``ŷ_ref`` the member's normal expert.
    Early stopping on the raw MSE of an inner holdout of airport-days of the training rows.
    """
    import torch
    from torch import nn

    torch.set_num_threads(8)
    feats = [f for f in all_features() if f not in CAT_FEATURES]
    prep = _Prep(tr, feats, CAT_FEATURES)
    _, y, d, cls = _targets(tr)
    ref = _ref(tr)
    init = np.array(
        [
            _atanh_init(d[cls == 1], NEAR_S),
            _atanh_init(d[cls == 2], LATE_S),
            float(np.log(np.nanmedian(y[cls == 3] / ref[cls == 3]))) if (cls == 3).any() else 0.0,
        ]
    )
    prior = np.log(np.bincount(cls, minlength=4) / len(cls) + 1e-4)

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embs = nn.ModuleList([nn.Embedding(k, min(16, k)) for k in prep.cards])
            d_in = len(feats) + len(prep.flag) + sum(min(16, k) for k in prep.cards)
            self.body = nn.Sequential(
                nn.Linear(d_in, width),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(width, width),
                nn.SiLU(),
                nn.Dropout(0.1),
            )
            self.head = nn.Linear(width, 7)
            with torch.no_grad():
                self.head.weight.mul_(0.1)
                self.head.bias.copy_(
                    torch.tensor(np.concatenate([prior, init]), dtype=torch.float32)
                )

        def forward(self, xn, xc, ck, refk):
            h = torch.cat([xn] + [e(xc[:, i]) for i, e in enumerate(self.embs)], dim=1)
            o = self.head(self.body(h))
            logits, a = o[:, :4], o[:, 4:]
            v = torch.stack(
                [
                    ck,
                    ck - 0.9 * torch.tanh(a[:, 0]),
                    ck - 3.6 * torch.tanh(a[:, 1]),
                    refk * torch.exp(torch.clamp(a[:, 2], -3.0, 3.0)),
                ],
                dim=1,
            )
            p = torch.softmax(logits, dim=1)
            return (p * v).sum(dim=1), logits, p, v

    dev = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")

    def tens(df: pl.DataFrame) -> tuple:
        return (
            torch.tensor(prep.num_x(df), device=dev),
            torch.tensor(prep.cat_x(df), device=dev),
            torch.tensor(df["c_sched"].to_numpy() / 1000.0, dtype=torch.float32, device=dev),
            torch.tensor(_ref(df) / 1000.0, dtype=torch.float32, device=dev),
        )

    xtr = tens(tr)
    yk = torch.tensor(y / 1000.0, dtype=torch.float32, device=dev)
    ct = torch.tensor(cls, device=dev)
    ev_t = [tens(e) for e in evs]
    acc = [
        {"pred": np.zeros(e.height), "p": np.zeros((e.height, 4)), "v": np.zeros((e.height, 4))}
        for e in evs
    ]

    def predict(model, t) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, t[0].shape[0], 65536):
                sl = [z[i : i + 65536] for z in t]
                pr, _, p, v = model(*sl)
                outs.append((pr.cpu().numpy(), p.cpu().numpy(), v.cpu().numpy()))
        return tuple(
            np.concatenate([o[j] for o in outs]) * (1 if j == 1 else 1000.0) for j in range(3)
        )  # type: ignore[return-value]

    for s in range(n_seeds):
        torch.manual_seed(seed * 1000 + s)
        hold = _holdout(tr, 0.1, seed * 1000 + s)
        itr, iho = np.flatnonzero(~hold), np.flatnonzero(hold)
        model = Net().to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        best, best_state, bad, best_ep = np.inf, None, 0, -1
        rng = np.random.default_rng(seed * 1000 + s)
        ho_t = [z[iho] for z in xtr]
        for ep in range(epochs):
            model.train()
            perm = rng.permutation(itr)
            for i in range(0, len(perm), batch):
                b = torch.tensor(perm[i : i + batch], device=dev)
                pr, logits, _, _ = model(*(z[b] for z in xtr))
                loss = torch.mean((pr - yk[b]) ** 2) + lam * nn.functional.cross_entropy(
                    logits, ct[b]
                )
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            ph, _, _ = predict(model, ho_t)
            mse = float(np.mean((ph - y[iho]) ** 2))
            if mse < best - 1e-6:
                best, bad, best_ep = mse, 0, ep
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        print(f"    net seed {s}: best epoch {best_ep + 1}, holdout rmse {np.sqrt(best):.1f}")
        for a_, t in zip(acc, ev_t, strict=True):
            pr, p, v = predict(model, t)
            a_["pred"] += pr / n_seeds
            a_["p"] += p / n_seeds
            a_["v"] += v / n_seeds
    return [Fitted(pred=np.maximum(a_["pred"], 0.0), p=a_["p"], v=a_["v"]) for a_ in acc]


FITTERS: dict[str, Callable[..., list[Fitted]]] = {"net": fit_net, "gbdt": fit_gbdt, "reg": fit_reg}


# --------------------------------------------------------------------------- nested fit


def population(frame: pl.DataFrame, thr: float) -> pl.DataFrame:
    return frame.filter(pl.col("c_sched") > thr)


def fit_mask(df: pl.DataFrame) -> np.ndarray:
    """Rows a model may be fitted on: labelled, not a day rollover."""
    y = df["y"].to_numpy()
    return (~df["is_test"].to_numpy()) & np.isfinite(y) & ~is_rollover(y)


def nested_predict(
    pop: pl.DataFrame,
    months: list[str],
    fitter: Callable[..., list[Fitted]],
    log: Callable[[str], None] = print,
    **kw: object,
) -> tuple[pl.DataFrame, list[dict]]:
    """Leave-one-month-out DF predictions for the training rows of ``months`` and an
    all-months fit for the ranking rows. Returns one frame (``MVT_ID_mvt, is_test, df_pred,
    df_p_<class>, df_v_<class>``) and per-fold fit info."""
    fm = fit_mask(pop)
    src = pop.filter(pl.Series(fm)).select("MVT_ID_mvt", "month", "cls", *_te_key_cols())
    pieces, info = [], []
    for held in [*months, None]:
        tr_mask = fm & (pop["month"] != held).to_numpy() if held else fm
        tr = pop.filter(pl.Series(tr_mask))
        ev = pop.filter(pl.col("month") == held) if held else pop.filter(pl.col("is_test"))
        if ev.height == 0:
            continue
        tr = tr.join(nested_te(src, tr, held), on="MVT_ID_mvt", how="left")
        ev = ev.join(nested_te(src, ev, held), on="MVT_ID_mvt", how="left")
        assert held is None or not (tr["month"] == held).any(), "held-out month in training"
        (res,) = fitter(tr, [ev], **kw)
        piece = ev.select("MVT_ID_mvt", "is_test").with_columns(pl.Series("df_pred", res["pred"]))
        if "p" in res:
            piece = piece.with_columns(
                [pl.Series(f"df_p_{k}", res["p"][:, i]) for i, k in enumerate(CLASSES)]
                + [pl.Series(f"df_v_{k}", res["v"][:, i]) for i, k in enumerate(CLASSES)]
            )
        pieces.append(piece)
        info.append({"held": held or "test", "n_train": tr.height, "n_eval": ev.height})
        log(f"fold {held or 'test'}: fitted on {tr.height:,}, predicted {ev.height:,}")
    return pl.concat(pieces, how="diagonal_relaxed"), info


def _te_key_cols() -> list[str]:
    return sorted({k for keys in TE_KEYS.values() for k in keys} - {"month"})


def df_path(run: str, method: str, thr: float, tag: str = "") -> Path:
    return DF_DIR / f"{run}__{method}__t{int(thr)}{('__' + tag) if tag else ''}.parquet"


def _sse_table(df: pl.DataFrame, pred_col: str) -> pl.DataFrame:
    return (
        df.with_columns(((pl.col("y") - pl.col(pred_col)) ** 2).alias("_se"))
        .group_by("cls_name", "no_nm")
        .agg(pl.len().alias("rows"), (pl.col("_se").sum() / 1e9).alias(f"sse_{pred_col}"))
        .sort("cls_name", "no_nm")
    )


def fit(
    run: str,
    method: str,
    thr: float,
    out: Path,
    months_subset: list[str] | None = None,
    float32: bool = False,
    **kw: object,
) -> Path:
    t0 = time.time()
    _NUM_DTYPE[0] = pl.Float32 if float32 else pl.Float64
    frame = build_frame()
    _log(f"frame {frame.height:,} rows", t0)
    months = run_months(run)
    pop = attach_member(population(frame, thr), run)
    del frame
    if float32:
        feats = [f for f in base_features() + MEMBER_FEATURES if pop.schema[f] == pl.Float64]
        pop = pop.with_columns([pl.col(f).cast(pl.Float32) for f in feats])
    missing = set(months) - set(pop.filter(~pl.col("is_test"))["month"].unique().to_list())
    if missing:
        raise SystemExit(f"{run}: months {sorted(missing)} have no delayed rows")
    pop = pop.filter(~pl.col("is_test") & pl.col("month").is_in(months) | pl.col("is_test"))
    if months_subset:  # smoke only: fewer held-out months (all months still in training)
        months = [m for m in months if m in months_subset]
    _log(
        f"population C > {thr:.0f} s: {pop.height:,} rows ({int(pop['is_test'].sum()):,} test)", t0
    )
    preds, info = nested_predict(pop, months, FITTERS[method], log=lambda s: _log(s, t0), **kw)
    res = pop.select("MVT_ID_mvt", "airport", "month", "no_nm", "c_sched", "is_test").join(
        preds.drop("is_test"), on="MVT_ID_mvt", how="inner"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    res.write_parquet(out)

    # OOF diagnostics on the fitting population (ex-rollover, labelled)
    tr = (
        pop.filter(pl.Series(fit_mask(pop)))
        .select("MVT_ID_mvt", "month", "no_nm", "y", "cls", "m_pred")
        .join(res.select("MVT_ID_mvt", "df_pred"), on="MVT_ID_mvt", how="inner")
        .with_columns(
            pl.col("cls").replace_strict(dict(enumerate(CLASSES)), default="?").alias("cls_name")
        )
    )
    tab = _sse_table(tr, "m_pred").join(
        _sse_table(tr, "df_pred").drop("rows"), on=["cls_name", "no_nm"]
    )
    y, mp, dp = (tr[c].to_numpy() for c in ("y", "m_pred", "df_pred"))
    jj = tr["month"].is_in(GATE_MONTHS).to_numpy()
    stats = {
        "run": run,
        "method": method,
        "thr_s": thr,
        "kw": {k: v for k, v in kw.items()},
        "float32": float32,
        "months": months,
        "folds": info,
        "oof_rows_ex_ro": len(y),
        "sse_member_1e9": float(((y - mp) ** 2).sum() / 1e9),
        "sse_df_1e9": float(((y - dp) ** 2).sum() / 1e9),
        "jj_sse_member_1e9": float(((y - mp)[jj] ** 2).sum() / 1e9),
        "jj_sse_df_1e9": float(((y - dp)[jj] ** 2).sum() / 1e9),
        "by_class": tab.to_dicts(),
        "seconds": round(time.time() - t0, 1),
        "max_rss_gb": round(_maxrss_gb(), 2),
    }
    if "df_p_default" in res.columns:
        pc = tr.join(res.select("MVT_ID_mvt", *[f"df_p_{k}" for k in CLASSES]), on="MVT_ID_mvt")
        stats["mean_p_by_true_class"] = (
            pc.group_by("cls_name")
            .agg([pl.col(f"df_p_{k}").mean().round(4) for k in CLASSES])
            .sort("cls_name")
            .to_dicts()
        )
        p_true = pc.select(
            pl.concat_list([pl.col(f"df_p_{k}") for k in CLASSES]).list.get(pl.col("cls"))
        ).to_series()
        stats["class_logloss"] = float(-np.log(np.clip(p_true.to_numpy(), 1e-9, 1)).mean())
    (out.parent / f"{out.stem}.json").write_text(json.dumps(stats, indent=2, default=str))
    with pl.Config(tbl_rows=20, tbl_cols=10):
        print(tab)
    print(
        f"OOF ex-ro SSE (1e9 s²) on C > {thr:.0f}: member {stats['sse_member_1e9']:.2f} -> DF "
        f"{stats['sse_df_1e9']:.2f}; Jan+Jul {stats['jj_sse_member_1e9']:.2f} -> "
        f"{stats['jj_sse_df_1e9']:.2f}"
    )
    _log(f"wrote {out}", t0)
    return out


# --------------------------------------------------------------------------- apply


def best_alpha(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """argmin_w Σ (y − (w·a + (1 − w)·b))², clipped to [0, 1] (closed form)."""
    d = a - b
    den = float(np.sum(d * d))
    return float(np.clip(np.sum((y - b) * d) / den, 0.0, 1.0)) if den > 0 else 0.0


def nested_alphas(
    y: np.ndarray, df_pred: np.ndarray, member: np.ndarray, month: np.ndarray
) -> tuple[dict[str, float], float]:
    """Fold-safe weight of DF against the member on the DF rows: month m gets the α that
    minimises the ex-rollover SSE of the *other* months' OOF rows; the ranking side gets the α
    of all months. Returns ``({month: α}, α_test)``."""
    ok = np.isfinite(y) & ~is_rollover(y) & np.isfinite(df_pred) & np.isfinite(member)
    per = {}
    for m in sorted(set(month[ok].tolist())):
        tr = ok & (month != m)
        per[m] = best_alpha(y[tr], df_pred[tr], member[tr])
    return per, best_alpha(y[ok], df_pred[ok], member[ok])


def rollover_term(df: pl.DataFrame) -> np.ndarray:
    """The member's rollover branch inside its ``pred``: ``p_rollover · (C_ro − ŷ)`` with
    ``C_ro = 86,400 + ŷ`` (`prc.mixture.corruption.corrupt_values`), i.e. ``86,400 · p``."""
    if "p_rollover" not in df.columns:
        return np.zeros(df.height)
    return ROLLOVER_S * np.nan_to_num(df["p_rollover"].to_numpy().astype(np.float64), nan=0.0)


def apply(
    run: str,
    df_file: Path,
    out_name: str,
    only: str = "all",
    blend: str = "nested",
    keep_rollover: bool = True,
) -> Path:
    """Write ``outputs/runs/<out_name>/`` = the source run with ``pred`` replaced on the DF
    rows and every ``pred_s<seed>`` shifted by the same amount (clipped at 0).

    ``blend="nested"`` (the default) replaces with ``α·DF + (1 − α)·pred`` and α from
    :func:`nested_alphas` (leave-one-month-out on the OOF side); ``blend="none"`` with DF.

    DF is fitted ex-rollover, so it estimates the taxi given *no* day rollover. With
    ``keep_rollover`` (the default) the member's own rollover branch ``86,400 · p_rollover``
    is taken out of ``pred`` before the blend and added back after it, so the rollover hedge of
    the source run is kept exactly: ``new = α·DF + (1 − α)·(pred − R) + R``."""
    src, dst = paths.RUNS / run, paths.RUNS / out_name
    if not (src / "oof.parquet").is_file():
        raise SystemExit(f"{src}: no oof.parquet")
    if dst.resolve() == src.resolve():
        raise SystemExit("--out must differ from --run")
    dfp = pl.read_parquet(df_file, columns=["MVT_ID_mvt", "is_test", "no_nm", "df_pred"])
    if only == "no_nm":
        dfp = dfp.filter(pl.col("no_nm"))
    elif only == "nm":
        dfp = dfp.filter(~pl.col("no_nm"))
    if blend not in ("nested", "none"):
        raise SystemExit(f"--blend must be nested or none, not {blend!r}")
    alpha_m, alpha_t = None, 1.0
    if blend == "nested":
        o = pl.read_parquet(src / "oof.parquet").join(
            dfp.filter(~pl.col("is_test")).select("MVT_ID_mvt", "df_pred"), on="MVT_ID_mvt"
        )
        ro = rollover_term(o) if keep_rollover else 0.0
        alpha_m, alpha_t = nested_alphas(
            o["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy(),
            o["df_pred"].to_numpy(),
            o["pred"].to_numpy().astype(np.float64) - ro,
            o["month"].to_numpy(),
        )
        del o
        print(
            "apply: nested alpha per held-out month "
            + ", ".join(f"{m} {a:.3f}" for m, a in alpha_m.items())
            + f"; test {alpha_t:.3f}"
        )
    dst.mkdir(parents=True, exist_ok=True)
    counts = {}
    for which, is_test in (("oof", False), ("test_pred", True)):
        path = src / f"{which}.parquet"
        if not path.is_file():
            print(f"apply: {path} missing, skipped")
            continue
        df = pl.read_parquet(path)
        sub = dfp.filter(pl.col("is_test") == is_test).select("MVT_ID_mvt", "df_pred")
        merged = df.join(sub, on="MVT_ID_mvt", how="left", maintain_order="left")
        use = merged["df_pred"].is_not_null().to_numpy()
        base = merged["pred"].to_numpy().astype(np.float64)
        if alpha_m is None:
            alpha = np.ones(merged.height)
        elif is_test:
            alpha = np.full(merged.height, alpha_t)
        else:
            alpha = merged["month"].replace_strict(alpha_m, default=None).to_numpy()
            alpha = np.asarray(alpha, dtype=np.float64)
            if np.isnan(alpha[use]).any():
                raise SystemExit("apply: a DF row's month has no nested alpha")
        dfv = np.nan_to_num(merged["df_pred"].to_numpy().astype(np.float64), nan=0.0)
        ro = rollover_term(merged) if keep_rollover else np.zeros(merged.height)
        new_pred = np.where(use, alpha * dfv + (1 - alpha) * (base - ro) + ro, base)
        shift = np.where(use, new_pred - merged["pred"].to_numpy(), 0.0)
        cols = {"pred": np.maximum(new_pred, 0.0)}
        for s in (c for c in df.columns if c.startswith("pred_s")):
            cols[s] = np.maximum(merged[s].to_numpy().astype(np.float64) + shift, 0.0)
        out = merged.with_columns(
            [pl.Series(k, v, dtype=df.schema[k]) for k, v in cols.items()]
        ).select(df.columns)
        out.write_parquet(dst / f"{which}.parquet")
        counts[which] = {
            "rows": df.height,
            "replaced": int(use.sum()),
            "rmsd_replaced": float(np.sqrt(np.mean(shift[use] ** 2))) if use.any() else 0.0,
            "mean_shift_replaced": float(shift[use].mean()) if use.any() else 0.0,
        }
        print(
            f"apply: {which} {df.height:,} rows, {int(use.sum()):,} replaced "
            f"(rmsd vs source {counts[which]['rmsd_replaced']:.1f} s, mean shift "
            f"{counts[which]['mean_shift_replaced']:+.1f} s) -> {dst / f'{which}.parquet'}"
        )
    (dst / "meta.json").write_text(
        json.dumps(
            {
                "source_run": run,
                "df_predictions": str(df_file),
                "only": only,
                "blend": blend,
                "keep_rollover": keep_rollover,
                "alpha_oof": alpha_m,
                "alpha_test": alpha_t,
                "counts": counts,
                "written": datetime.now(UTC).isoformat(timespec="seconds"),
                "note": "post-processed delayed-flight (DF) predictions; not a registry run",
            },
            indent=2,
        )
        + "\n"
    )
    return dst


# --------------------------------------------------------------------------- report


def _boot(w, blk, nb, se_r, se_c, m) -> tuple[float, float, float]:
    b = PD.bootstrap_delta(w, blk, nb, se_r, se_c, m)
    return b["boot_lo"], b["boot_hi"], b["boot_p_improve"]


def report(base: str, cand: str, thr: float, n_boot: int, write: Path | None) -> str:
    """Scorecard slices for a DF run against its source: all / ex-ro, Jan, Jul, Jan+Jul pooled
    (the number that tracks the board), the no-NM lane, per airport, SSE on C > thr by stamp
    class, and the test-side rmsd on the replaced rows. Intervals: `paired_delta`'s day-block
    bootstrap on the seed-mean predictions."""
    cols = ["MVT_ID_mvt", "month", "airport", "TAXITIME_SEC_mvt", "pred"]
    r = pl.read_parquet(paths.RUNS / base / "oof.parquet", columns=cols)
    c = pl.read_parquet(paths.RUNS / cand / "oof.parquet", columns=["MVT_ID_mvt", "pred"])
    mv = (
        pl.scan_parquet(paths.INTERIM / "movements.parquet")
        .select(
            "MVT_ID_mvt",
            "MVT_TIME_UTC_mvt",
            "AOBT_3_flt",
            (pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt"))
            .dt.total_seconds()
            .cast(pl.Float64)
            .alias("c_sched"),
            (pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt"))
            .dt.total_seconds()
            .cast(pl.Float64)
            .alias("d"),
        )
        .join(r.select("MVT_ID_mvt").lazy(), on="MVT_ID_mvt", how="semi")
        .collect()
    )
    d = r.join(c.rename({"pred": "pred_c"}), on="MVT_ID_mvt", how="inner").join(
        mv, on="MVT_ID_mvt", how="left"
    )
    y = d["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy()
    pr, pc = d["pred"].to_numpy(), d["pred_c"].to_numpy()
    se_r, se_c = (y - pr) ** 2, (y - pc) ** 2
    month, ap = d["month"].to_numpy(), d["airport"].to_numpy()
    nonm = d["AOBT_3_flt"].is_null().to_numpy()
    exro = ~is_rollover(y)
    csch = d["c_sched"].to_numpy()
    delayed = csch > thr
    jj = np.isin(month, GATE_MONTHS)
    blk, groups = PD.block_index(d)
    nb = int(blk.max()) + 1
    w = PD.block_weights(groups, nb, n_boot, 0) if n_boot else None

    def row(name: str, m: np.ndarray) -> dict:
        rr, cc = float(np.sqrt(se_r[m].mean())), float(np.sqrt(se_c[m].mean()))
        out = {
            "slice": name,
            "rows": int(m.sum()),
            "base_rmse": round(rr, 2),
            "df_rmse": round(cc, 2),
            "delta": round(cc - rr, 2),
        }
        if w is not None:
            lo, hi, pi = _boot(w, blk, nb, se_r, se_c, m)
            out |= {"lo": round(lo, 2), "hi": round(hi, 2), "p_improve": round(pi, 3)}
        return out

    slices = {
        "all": np.ones_like(exro),
        "all|ex_ro": exro,
        "2025-01|ex_ro": (month == "2025-01") & exro,
        "2025-07|ex_ro": (month == "2025-07") & exro,
        "Jan+Jul": jj,
        "Jan+Jul|ex_ro": jj & exro,
        "no_aobt|ex_ro": nonm & exro,
        "Jan+Jul no_aobt|ex_ro": jj & nonm & exro,
        f"C>{int(thr)}|ex_ro": delayed & exro,
        f"C<={int(thr)}|ex_ro": ~delayed & exro,
    }
    main = pl.DataFrame([row(k, m) for k, m in slices.items()])
    per_ap = pl.DataFrame(
        [row(f"ap:{a}|ex_ro", (ap == a) & exro) for a in sorted(set(ap.tolist()))]
    )
    per_month = pl.DataFrame(
        [row(f"{m}|ex_ro", (month == m) & exro) for m in sorted(set(month.tolist()))]
    )
    # SSE on C > thr by stamp class (ex-ro), in 1e9 s², and its share of all 12-month ex-ro SSE
    cls = stamp_class(d["d"].to_numpy())
    tot = float(se_r[exro].sum())
    cl_rows = []
    for k, name in enumerate(CLASSES):
        for nn_ in (False, True):
            m = delayed & exro & (cls == k) & (nonm == nn_)
            cl_rows.append(
                {
                    "class": name,
                    "no_nm": nn_,
                    "rows": int(m.sum()),
                    "sse_base_1e9": round(float(se_r[m].sum()) / 1e9, 3),
                    "sse_df_1e9": round(float(se_c[m].sum()) / 1e9, 3),
                    "delta_1e9": round(float(se_c[m].sum() - se_r[m].sum()) / 1e9, 3),
                    "base_share_of_all_exro": round(float(se_r[m].sum()) / tot, 4),
                }
            )
    by_cls = pl.DataFrame(cl_rows)
    # test side: how far the new predictions moved on the replaced rows
    tb = pl.read_parquet(paths.RUNS / base / "test_pred.parquet", columns=["MVT_ID_mvt", "pred"])
    tc = pl.read_parquet(paths.RUNS / cand / "test_pred.parquet", columns=["MVT_ID_mvt", "pred"])
    tmv = (
        pl.scan_parquet(paths.INTERIM / "movements.parquet")
        .select("MVT_ID_mvt", "month", "AOBT_3_flt")
        .join(tb.select("MVT_ID_mvt").lazy(), on="MVT_ID_mvt", how="semi")
        .collect()
    )
    t = tb.join(tc.rename({"pred": "pred_c"}), on="MVT_ID_mvt").join(tmv, on="MVT_ID_mvt")
    t = t.with_columns((pl.col("pred_c") - pl.col("pred")).alias("shift"))
    moved = t.filter(pl.col("shift") != 0)
    test_rows = []
    for name, sub in (
        ("all replaced", moved),
        ("NM", moved.filter(pl.col("AOBT_3_flt").is_not_null())),
        ("no NM", moved.filter(pl.col("AOBT_3_flt").is_null())),
        ("2026-01", moved.filter(pl.col("month") == "2026-01")),
        ("2026-07", moved.filter(pl.col("month") == "2026-07")),
    ):
        sh = sub["shift"].to_numpy()
        test_rows.append(
            {
                "test slice": name,
                "rows": len(sh),
                "rmsd": round(float(np.sqrt(np.mean(sh**2))) if len(sh) else 0.0, 1),
                "mean_shift": round(float(sh.mean()) if len(sh) else 0.0, 1),
                "rmsd_over_all_test": round(float(np.sqrt((sh**2).sum() / t.height)), 2),
            }
        )
    test_tab = pl.DataFrame(test_rows)
    parts = [
        f"# DF report — {cand} vs {base}",
        "",
        (
            "Generated by `scripts/delayed_flight_model.py report`. Negative delta = DF better. "
            f"Seed-mean predictions; {n_boot} day-block bootstrap replicates (airport × date "
            "within airport-month, `paired_delta.py`'s resampler, seed 0)."
        ),
        "",
        "## Headline slices",
        "",
        A.md(main),
        "",
        "## Per airport (ex-ro)",
        "",
        A.md(per_ap),
        "",
        "## Per month (ex-ro)",
        "",
        A.md(per_month),
        "",
        f"## SSE on C > {int(thr)} s by stamp class (ex-ro, 1e9 s²)",
        "",
        A.md(by_cls),
        "",
        "## Test side: movement of the replaced predictions",
        "",
        A.md(test_tab),
        "",
    ]
    text = "\n".join(parts)
    print(text)
    if write:
        write.parent.mkdir(parents=True, exist_ok=True)
        write.write_text(text)
        print("wrote", write)
    return text


# --------------------------------------------------------------------------- CLI


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build (or rebuild) the cached delayed-departure frame")
    b.add_argument("--rebuild", action="store_true")
    b.add_argument("--audit", action="store_true", help="print the train/test null-rate audit")
    f = sub.add_parser("fit", help="nested leave-one-month-out DF fit against a run")
    f.add_argument("--run", required=True, help="bare run name under outputs/runs/")
    f.add_argument("--method", choices=METHODS, default="net")
    f.add_argument("--thr", type=float, default=DEFAULT_THR_S, help="C threshold, s")
    f.add_argument("--tag", default="", help="suffix for the output file (variants)")
    f.add_argument("--lam", type=float, default=0.2, help="net: class log-loss weight")
    f.add_argument("--n-seeds", type=int, default=3, help="net: seeds averaged per fold")
    f.add_argument("--epochs", type=int, default=60)
    f.add_argument("--width", type=int, default=256)
    f.add_argument("--batch", type=int, default=1024)
    f.add_argument("--lr", type=float, default=1e-3)
    f.add_argument("--patience", type=int, default=6)
    f.add_argument("--device", default="cpu")
    f.add_argument("--months", nargs="*", default=None, help="smoke: held-out months subset")
    f.add_argument(
        "--float32", action="store_true", help="float32 features and matrices (half the memory)"
    )
    a = sub.add_parser("apply", help="write outputs/runs/<out>/ with the DF rows swapped in")
    a.add_argument("--run", required=True, help="bare source run name")
    a.add_argument("--out", required=True, help="bare output run name")
    a.add_argument("--df", type=Path, default=None, help="DF parquet (default: from fit args)")
    a.add_argument("--fit-run", default=None, help="run the DF was fitted on (default --run)")
    a.add_argument("--method", choices=METHODS, default="net")
    a.add_argument("--thr", type=float, default=DEFAULT_THR_S)
    a.add_argument("--tag", default="")
    a.add_argument("--only", choices=("all", "nm", "no_nm"), default="all")
    a.add_argument(
        "--blend",
        choices=("nested", "none"),
        default="nested",
        help="nested: α·DF + (1 − α)·member, α leave-one-month-out on the OOF; none: DF alone",
    )
    a.add_argument(
        "--drop-rollover",
        action="store_true",
        help="do not carry the member's rollover branch (86,400·p_rollover) onto the DF rows",
    )
    r = sub.add_parser("report", help="DF scorecard slices of a DF run against its source")
    r.add_argument("--base", required=True)
    r.add_argument("--cand", required=True)
    r.add_argument("--thr", type=float, default=DEFAULT_THR_S)
    r.add_argument("--bootstrap", type=int, default=2000)
    r.add_argument("--write", type=Path, default=None)
    args = ap.parse_args()
    for name in ("run", "out", "base", "cand", "fit_run"):
        v = getattr(args, name, None)
        if v and ("/" in v or v.endswith(".parquet")):
            raise SystemExit(f"--{name.replace('_', '-')} takes a bare run name, not a path: {v}")
    if args.cmd == "build":
        t0 = time.time()
        frame = build_frame(cache=not args.rebuild)
        _log(f"frame {frame.height:,} x {frame.width}", t0)
        if args.audit:
            with pl.Config(tbl_rows=200, tbl_cols=10):
                print(leak_audit(frame, DEFAULT_THR_S))
    elif args.cmd == "fit":
        kw = {}
        if args.method == "net":
            kw = {
                "lam": args.lam,
                "n_seeds": args.n_seeds,
                "epochs": args.epochs,
                "width": args.width,
                "batch": args.batch,
                "lr": args.lr,
                "patience": args.patience,
                "device": args.device,
            }
        fit(
            args.run,
            args.method,
            args.thr,
            df_path(args.run, args.method, args.thr, args.tag),
            months_subset=args.months,
            float32=args.float32,
            **kw,
        )
    elif args.cmd == "apply":
        dfp = args.df or df_path(args.fit_run or args.run, args.method, args.thr, args.tag)
        apply(
            args.run,
            dfp,
            args.out,
            only=args.only,
            blend=args.blend,
            keep_rollover=not args.drop_rollover,
        )
    else:
        report(args.base, args.cand, args.thr, args.bootstrap, args.write)


if __name__ == "__main__":
    main()
