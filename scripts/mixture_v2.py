"""Mixture v2 (plan R1 of internal note 2026-09-24_recording-process): the delayed-flight model
(DF, `scripts/delayed_flight_model.py`) extended to **every departure**, with the stamp classes
the 2026-09-23 audit found uncovered. A post-processor over a finished member run, like DF.

DF covers departures with ``C = MVT − SCHED`` above 30 min / 1 h. v2 is one mixture model over
all 2.08 M 2025 departures (and the 345 k ranking rows), with six stamp classes. They are
training targets only, derived from ``BLOCK`` and the label, never features; the partition is
exclusive and applied in this priority order (rollovers are excluded from fitting, as in DF):

    0 default  |BLOCK − SCHED| <= 10 s                         value C (exact)
    4 far      BLOCK more than 1 h before SCHED, NM or not      value max(C, 0) + 3600·e^a
    1 near     C > 30 min and 10 s < |BLOCK − SCHED| <= 15 min  value C − 900·tanh(a)
    2 early    NM record, y > 1 h (the runner's `stamp_early`)  value (MVT − AOBT_3) + Δ_E(x)
    3 bfp      NM record, AOBT_3 − BLOCK > 40 min, NM taxi      value (MVT − AOBT_3) + Δ_B(x)
               MVT − AOBT_3 in [0, 1 h] ("block ≪ flight plan")
    5 normal   the rest                                         value ŷ_ref·e^a (a normal taxi)

``Δ_E ∈ [−600, 6000] s`` and ``Δ_B ∈ [2400, 3600] s`` are regressed per row (the runner fixes
the early-stamp offset at a constant). Every value head is bounded. Classes that cannot occur
are masked out of the softmax from test-observable facts only: ``near`` needs C > 30 min;
``early`` needs an NM record; ``bfp`` needs an NM record with ``MVT − AOBT_3`` in [0, 1 h].

Inputs: DF's feature set (row, state, rotation, feature blocks, nested class-rate encodings,
the member's ``pred`` / ``y_hat_normal`` / ``p_sched_default``), plus the member's
``p_stamp_early`` and four burst counts computed identically on train and test from the
movements table: the no-NM share among delayed (C > 1 h) departures and the count of delayed
departures, per airport-hour (hour of MVT) and per airport-day (the day share is DF's own).
Nothing reads the row's own BLOCK or label.

Loss: ``mean((Σ p_k v_k − y)²)`` in ks² (labels capped at 24 h) + λ · class log-loss
(+ an optional ``--value-lam`` · squared error of the true class's value head, off by default).
Folds: DF's nested leave-one-month-out, unchanged; the ranking side is fitted on all 12 months.
``apply`` blends with the member at DF's fold-safe α (optionally one α per band of C).

    # 0. the member-independent frame of every departure (cached; ~10 min, ~15 GB)
    uv run python scripts/mixture_v2.py build
    # 1. fit -> outputs/mixture_v2/<run>__v2[__<tag>].{parquet,json}
    uv run python scripts/mixture_v2.py fit --run exp0227_r7c_realmlp_nomask_lane --device cuda
    # 2. write outputs/runs/<out>/ (DF's apply formula; --alpha-bands 1800 for two α groups)
    uv run python scripts/mixture_v2.py apply --run exp0227_r7c_realmlp_nomask_lane \
        --out exp0227_r7c_realmlp_nomask_lane_mx2
    # 3. score: paired_delta.py as usual, plus the per-class / test-share report
    uv run python scripts/mixture_v2.py report --base exp0227_r7c_realmlp_nomask_lane_df8t1800 \
        --cand exp0227_r7c_realmlp_nomask_lane_mx2 --fit-run exp0227_r7c_realmlp_nomask_lane

Smoke (desktop): ``build --airports LIRF EGLL --months 2025-01 2025-07 2026-01 2026-07`` and the
same two flags on ``fit``.

Extra label-free feature blocks (``--extra-blocks stamp_anomaly ...`` on ``build`` and ``fit``,
default none): every numeric column of each registered block joins the nets' numeric inputs
(the shared body of the class head and the value heads), scaled and imputed like the other
numerics (``DF._Prep``: asinh(x / median|x|), standardised, NaN -> 0 plus a missing flag when
training rows have nulls; ±inf -> null at build). The frame file name carries the blocks and
their source / external hashes, so it never collides with the default frame; ``fit`` with extra
blocks needs a ``--tag``. On the full table a block comes from (or is written to) the registry
cache ``data/features/``; on a smoke subset it is computed on the subset and not cached.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import delayed_flight_model as DF
import eda_no_nm_block_model as A
import paired_delta as PD
from prc import paths

MIX_DIR = paths.OUT / "mixture_v2"
FRAME_VERSION = 1
SCHED_TOL_S = DF.SCHED_TOL_S  # 10 s
NEAR_S = DF.NEAR_S  # 15 min
NEAR_MIN_C_S = 1800.0  # the near class exists on delayed rows only (DF's t1800 population)
FAR_EARLY_S = 3600.0
STAMP_EARLY_MIN_S = 3600.0  # prc.config.CorruptionConfig.stamp_early_min_s
BFP_GAP_S = 2400.0  # AOBT_3 − BLOCK above this ...
BFP_TAXI_MAX_S = 3600.0  # ... with the NM taxi MVT − AOBT_3 in [0, this]
DELAYED_S = 3600.0  # "delayed" for the burst counts (DF's `_delayed`)
RATIO_MIN_C_S = 600.0  # ratios over C are null below this (C ~ 0 on on-time rows)
LABEL_CAP_S = DF.LABEL_CAP_S
CLASSES = ("default", "near", "early", "bfp", "far", "normal")
K = len(CLASSES)
#: value-head bounds, seconds: Δ_E = E_MID + E_HALF·tanh(a), Δ_B = B_MID + B_HALF·tanh(a)
E_MID, E_HALF = 2700.0, 3300.0
B_MID, B_HALF = 3000.0, 600.0
FAR_SCALE_S = 3600.0
GATE_MONTHS = DF.GATE_MONTHS

BURST_FEATURES = ["f_hr_delayed_nonm_share", "f_hr_n_delayed", "f_day_n_delayed"]
MEMBER_FEATURES = [*DF.MEMBER_FEATURES, "m_pse"]
#: nested target encodings: DF's keys, the five non-normal class rates (normal = 1 − the rest)
TE_CODES: dict[str, int] = {"default": 0, "near": 1, "early": 2, "bfp": 3, "far": 4}
TE_PREFIX = "te2"
TE_FEATURES = [f"{TE_PREFIX}_{k}_{c}" for k in DF.TE_KEYS for c in TE_CODES] + [
    f"{TE_PREFIX}_{k}_logn" for k in DF.TE_KEYS
]
#: label-side columns kept in the frame for targets and diagnostics; never model inputs
LABEL_COLUMNS = ("y", "d", "cls")


def base_features() -> list[str]:
    return DF.base_features() + BURST_FEATURES


def frame_columns() -> list[str]:
    """The default frame's columns (``build_frame`` without extra blocks)."""
    return [
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
        *DF.CAT_FEATURES,
        *base_features(),
        "cls",
    ]


def frame_extra_columns(frame: pl.DataFrame) -> list[str]:
    """The extra-block columns of a frame, in frame order (empty for the default frame)."""
    known = set(frame_columns())
    return [c for c in frame.columns if c not in known]


def norm_blocks(names: list[str] | tuple[str, ...] | None) -> list[str]:
    """Sorted, de-duplicated block names (the build order and the cache key)."""
    return sorted(set(names or []))


def all_features() -> list[str]:
    return DF.CAT_FEATURES + base_features() + TE_FEATURES + MEMBER_FEATURES


def _log(msg: str, t0: float) -> None:
    DF._log(msg, t0)


# --------------------------------------------------------------------------- classes


def class_v2(d: np.ndarray, y: np.ndarray, c: np.ndarray, obs_delay: np.ndarray) -> np.ndarray:
    """v2 stamp class codes (see the module docstring), ``-1`` where unlabelled.

    ``d = BLOCK − SCHED``, ``y = MVT − BLOCK``, ``c = MVT − SCHED``, ``obs_delay = AOBT_3 − SCHED``
    (NaN without an NM record). ``AOBT_3 − BLOCK = obs_delay − d`` and the NM taxi
    ``MVT − AOBT_3 = c − obs_delay``. Later assignments win, so the order below is the reverse
    of the priority order."""
    d, y, c, o = (np.asarray(v, dtype=np.float64) for v in (d, y, c, obs_delay))
    nm = np.isfinite(o)
    proxy, gap = c - o, o - d
    with np.errstate(invalid="ignore"):
        out = np.full(len(d), 5, dtype=np.int64)
        out[nm & (gap > BFP_GAP_S) & (proxy >= 0) & (proxy <= BFP_TAXI_MAX_S)] = 3
        out[nm & (y > STAMP_EARLY_MIN_S)] = 2
        out[(c > NEAR_MIN_C_S) & (np.abs(d) <= NEAR_S)] = 1
        out[d < -FAR_EARLY_S] = 4
        out[np.abs(d) <= SCHED_TOL_S] = 0
    out[~(np.isfinite(d) & np.isfinite(y))] = -1
    return out


def class_mask(c: np.ndarray, obs_delay: np.ndarray) -> np.ndarray:
    """``(n, K)`` bool: the classes a row can be in, from test-observable columns only."""
    c, o = np.asarray(c, dtype=np.float64), np.asarray(obs_delay, dtype=np.float64)
    nm = np.isfinite(o)
    proxy = c - o
    m = np.ones((len(c), K), dtype=bool)
    with np.errstate(invalid="ignore"):
        m[:, 1] = c > NEAR_MIN_C_S
        m[:, 2] = nm
        m[:, 3] = nm & (proxy >= 0) & (proxy <= BFP_TAXI_MAX_S)
    return m


# --------------------------------------------------------------------------- frame


def add_burst_features(dep: pl.DataFrame) -> pl.DataFrame:
    """Per airport-hour (hour of MVT) and airport-day: delayed departures and the no-NM share
    among them. Uses C and the NM match only, both present on train and test rows."""
    dep = dep.with_columns(
        (pl.col("c_sched") > DELAYED_S).alias("_dl"),
        pl.col("MVT_TIME_UTC_mvt").dt.hour().alias("_hr"),
    )
    hr = dep.group_by("airport", "date", "_hr").agg(
        pl.col("no_nm").filter(pl.col("_dl")).mean().cast(pl.Float64).alias(BURST_FEATURES[0]),
        pl.col("_dl").sum().cast(pl.Float64).alias(BURST_FEATURES[1]),
    )
    day = dep.group_by("airport", "date").agg(
        pl.col("_dl").sum().cast(pl.Float64).alias(BURST_FEATURES[2])
    )
    return (
        dep.join(hr, on=["airport", "date", "_hr"], how="left")
        .join(day, on=["airport", "date"], how="left")
        .drop("_dl", "_hr")
    )


def extra_blocks_key(names: list[str] | tuple[str, ...] | None) -> str:
    """``""`` without extra blocks (the default frame name, unchanged); otherwise
    ``__xb_<name>.<source hash>[.<external hash>]-...`` so that a changed block, or a block built
    against another archive, never serves a stale frame."""
    names = norm_blocks(names)
    if not names:
        return ""
    from prc.features import REGISTRY

    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise SystemExit(f"unknown feature blocks {unknown}; known: {sorted(REGISTRY)}")
    parts = []
    for n in names:
        b = REGISTRY[n]
        ext = b.external_hash()
        parts.append(f"{n}.{b.source_hash()}{('.' + ext) if ext else ''}")
    return "__xb_" + "-".join(parts)


def frame_path(
    src: Path,
    airports: list[str] | None,
    months: list[str] | None,
    extra_blocks: list[str] | tuple[str, ...] | None = None,
) -> Path:
    stamp = f"{int(src.stat().st_mtime)}_{src.stat().st_size}"
    sub = ""
    if airports or months:
        sub = (
            "__sub_" + "-".join(sorted(airports or ["all"])) + "_" + "-".join(sorted(months or []))
        )
    xb = extra_blocks_key(extra_blocks)
    return MIX_DIR / "_frames" / f"mixture_frame__v{FRAME_VERSION}__{stamp}{sub}{xb}.parquet"


def add_extra_blocks(
    dep: pl.DataFrame,
    names: list[str],
    src: Path,
    airports: list[str] | None = None,
    months: list[str] | None = None,
) -> tuple[pl.DataFrame, list[str]]:
    """Join the numeric columns of the registered feature blocks ``names`` onto ``dep`` (cast
    to Float64, ±inf -> null). The blocks see the movement table the frame was built from: the
    full table (registry cache in ``data/features/`` used or written) or the smoke subset
    (computed, not cached)."""
    from prc.features import build as build_blocks

    subset = bool(airports or months)
    lf = pl.scan_parquet(src)
    if airports:
        lf = lf.filter(pl.col("airport").is_in(airports))
    if months:
        lf = lf.filter(pl.col("month").is_in(months))
    mv = lf.collect()
    blk = build_blocks(mv, names, use_cache=not subset)
    del mv
    blk = blk.join(dep.select("MVT_ID_mvt"), on="MVT_ID_mvt", how="semi")
    cols = [c for c in blk.columns if c != "MVT_ID_mvt"]
    bad = [c for c in cols if not blk.schema[c].is_numeric()]
    if bad:
        raise SystemExit(f"extra blocks {names}: non-numeric columns {bad}")
    clash = sorted(set(cols) & set(dep.columns))
    if clash:
        raise SystemExit(f"extra blocks {names}: columns already in the frame {clash}")
    blk = blk.with_columns(
        [
            pl.when(pl.col(c).cast(pl.Float64).is_infinite())
            .then(None)
            .otherwise(pl.col(c).cast(pl.Float64))
            .alias(c)
            for c in cols
        ]
    )
    return dep.join(blk, on="MVT_ID_mvt", how="left", maintain_order="left"), cols


def build_frame(
    cache: bool = True,
    airports: list[str] | None = None,
    months: list[str] | None = None,
    extra_blocks: list[str] | tuple[str, ...] | None = None,
) -> pl.DataFrame:
    """Every departure (train + test) with the member-independent features: DF's
    :func:`delayed_flight_model.build_frame` without the C cut, plus the burst counts and the
    v2 class labels. ``airports`` / ``months`` restrict the movement table first (smoke only:
    the state windows then see only that subset). ``extra_blocks``: registered label-free
    feature blocks whose numeric columns are appended (:func:`add_extra_blocks`)."""
    src = paths.INTERIM / "movements.parquet"
    extra_blocks = norm_blocks(extra_blocks)
    cached = frame_path(src, airports, months, extra_blocks)
    if cache and cached.is_file():
        return pl.read_parquet(cached)
    t0 = time.time()
    lf = pl.scan_parquet(src)
    if airports:
        lf = lf.filter(pl.col("airport").is_in(airports))
    if months:
        lf = lf.filter(pl.col("month").is_in(months))

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
    _log(f"departures {dep.height:,}, arrivals {arr.height:,}", t0)
    dep = A.add_state_features(dep, arr)
    dep = DF.add_extra_state(dep)
    dep = add_burst_features(dep)
    _log("state + burst features", t0)
    ids = dep.select("MVT_ID_mvt").lazy()
    wide = (
        lf.select(
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
        .join(ids, on="MVT_ID_mvt", how="semi")
        .collect()
    )
    dep = dep.join(wide, on="MVT_ID_mvt", how="left")
    del wide
    arr = arr.filter(pl.col("STAND_mvt").is_not_null()).select(
        "PHASE_mvt", "airport", "STAND_mvt", "SCHED_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt"
    )
    dep = A.add_rotation_features(dep, arr)
    del arr
    _log(f"rotation features, {dep.height:,} rows", t0)
    for name, cols in DF.BLOCK_COLUMNS.items():
        p = DF._newest_block(name, cols)
        blk = (
            pl.scan_parquet(p)
            .select("MVT_ID_mvt", *cols)
            .join(ids, on="MVT_ID_mvt", how="semi")
            .collect()
        )
        dep = dep.join(
            blk.with_columns([pl.col(c).cast(pl.Float64) for c in cols]),
            on="MVT_ID_mvt",
            how="left",
        )
        print(f"  block {name}: {p.name}")
    c = pl.col("c_sched")
    big_c = c > RATIO_MIN_C_S
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
        # DF divides by C > 30 min only; over all rows C can be 0, so the ratio needs a floor
        pl.when(big_c).then(pl.col("obs_delay") / c).otherwise(None).alias("f_obs_ratio"),
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
        [pl.col(f).cast(pl.Categorical).to_physical().cast(pl.Int32) for f in DF.CAT_FEATURES]
    )
    keep = [c for c in frame_columns() if c != "cls"]
    cls = class_v2(
        dep["d"].to_numpy(), dep["y"].to_numpy(), dep["c_sched"].to_numpy(),
        dep["f_obs_delay"].to_numpy(),
    )  # fmt: skip
    dep = dep.select(keep).with_columns(pl.Series("cls", cls, dtype=pl.Int8)).sort("MVT_ID_mvt")
    num = [f for f in base_features() if dep.schema[f].is_float()]
    dep = dep.with_columns(
        [pl.when(pl.col(f).is_infinite()).then(None).otherwise(pl.col(f)).alias(f) for f in num]
    )
    if extra_blocks:
        dep, xcols = add_extra_blocks(dep, extra_blocks, src, airports, months)
        _log(f"extra blocks {extra_blocks}: {len(xcols)} columns {xcols}", t0)
    cached.parent.mkdir(parents=True, exist_ok=True)
    tmp = cached.with_name(f"{cached.stem}.{os.getpid()}.tmp")  # write-then-rename (DF, 35853)
    dep.write_parquet(tmp)
    tmp.replace(cached)
    _log(f"frame {dep.height:,} x {dep.width} -> {cached}", t0)
    return dep


def class_counts(frame: pl.DataFrame) -> pl.DataFrame:
    """Rows per v2 class × NM on the labelled, ex-rollover 2025 rows (and per airport)."""
    f = frame.filter(pl.Series(DF.fit_mask(frame)))
    return (
        f.with_columns(
            pl.col("cls").replace_strict(dict(enumerate(CLASSES)), default="?").alias("class")
        )
        .group_by("class", "no_nm")
        .agg(pl.len().alias("rows"))
        .sort("class", "no_nm")
    )


def leak_audit(frame: pl.DataFrame) -> pl.DataFrame:
    """Null rate of every frame feature (extra-block columns included) on training vs ranking
    rows, by ``no_nm``. A feature present in 2025 and absent in 2026 would be a label-side
    artefact."""
    cols = [*DF.CAT_FEATURES, *base_features(), *frame_extra_columns(frame)]
    null = [
        (pl.col(c).is_null() | pl.col(c).is_nan()) if frame.schema[c].is_float()
        else pl.col(c).is_null()
        for c in cols
    ]  # fmt: skip
    g = (
        frame.group_by("is_test", "no_nm")
        .agg([e.mean().round(4).alias(c) for e, c in zip(null, cols, strict=True)])
        .with_columns(
            (
                pl.when("is_test").then(pl.lit("test")).otherwise(pl.lit("train"))
                + "_"
                + pl.when("no_nm").then(pl.lit("nonm")).otherwise(pl.lit("nm"))
            ).alias("grp")
        )
        .drop("is_test", "no_nm")
    )
    out = g.unpivot(index="grp", variable_name="feature").pivot("grp", index="feature")
    order = [k for k in ("train_nm", "train_nonm", "test_nm", "test_nonm") if k in out.columns]
    return out.select("feature", *order)


# --------------------------------------------------------------------------- member


def attach_member(frame: pl.DataFrame, run: str) -> pl.DataFrame:
    """DF's member columns plus ``m_pse`` (the member's ``p_stamp_early``; 0 when absent)."""
    out = DF.attach_member(frame, run)
    src = paths.RUNS / run
    parts = []
    for f in ("oof.parquet", "test_pred.parquet"):
        schema = pl.read_parquet_schema(src / f)
        if "p_stamp_early" in schema:
            parts.append(pl.read_parquet(src / f, columns=["MVT_ID_mvt", "p_stamp_early"]))
    if parts:
        pse = pl.concat(parts).rename({"p_stamp_early": "m_pse"})
        out = out.join(pse, on="MVT_ID_mvt", how="left")
    else:
        print(f"attach_member: {run} has no p_stamp_early; m_pse = 0")
        out = out.with_columns(pl.lit(None, pl.Float64).alias("m_pse"))
    c = pl.col("c_sched")
    return out.with_columns(
        pl.col("m_pse").cast(pl.Float64).fill_null(0.0),
        # DF's pred / C over all rows: null where C is ~0 (on-time rows)
        pl.when(c > RATIO_MIN_C_S)
        .then(pl.col("m_pred") / c)
        .otherwise(None)
        .alias("m_pred_over_c"),
    )


# --------------------------------------------------------------------------- encodings


def nested_te(
    src: pl.DataFrame,
    rows: pl.DataFrame,
    held: str | None,
    prior: float = DF.TE_PRIOR,
    codes: dict[str, int] | None = None,
    prefix: str = TE_PREFIX,
) -> pl.DataFrame:
    """DF's :func:`delayed_flight_model.nested_te` for an arbitrary set of class codes: the
    smoothed rate of each class by key, never from the row's own month nor ``held``. With
    ``codes={"default": 0, "near": 1, "late": 3}`` and ``prefix="te"`` it reproduces DF's
    encodings exactly (tested)."""
    codes = TE_CODES if codes is None else codes
    names = list(codes)
    ind = [
        pl.col("cls").cast(pl.Int64).eq(k).cast(pl.Float64).alias(f"_c{k}") for k in codes.values()
    ]
    s = src.with_columns(ind)
    base = s if held is None else s.filter(pl.col("month") != held)
    pri = {c: float(base[f"_c{k}"].mean()) for c, k in codes.items()}
    stat_cols = ["_n", *[f"_s{k}" for k in codes.values()]]
    out = rows.select("MVT_ID_mvt", "month").with_row_index("_i")
    for name, keys in DF.TE_KEYS.items():
        agg = [pl.len().alias("_n")] + [
            pl.col(f"_c{k}").sum().alias(f"_s{k}") for k in codes.values()
        ]
        per = s.group_by([*keys, "month"]).agg(agg)
        tot = per.group_by(keys).agg([pl.col(c).sum() for c in stat_cols])
        own = per.rename({c: f"{c}_own" for c in stat_cols})
        hel = (
            per.filter(pl.col("month") == held)
            .drop("month")
            .rename({c: f"{c}_held" for c in stat_cols})
            if held is not None
            else None
        )
        r = rows.select("MVT_ID_mvt", "month", *keys).with_row_index("_i")
        r = r.join(tot, on=keys, how="left").join(own, on=[*keys, "month"], how="left")
        if hel is not None:
            r = r.join(hel, on=keys, how="left")
        r = r.sort("_i")

        def left(col: str) -> pl.Expr:
            e = pl.col(col).fill_null(0) - pl.col(f"{col}_own").fill_null(0)
            if held is not None:
                e = e - pl.when(pl.col("month") == held).then(0).otherwise(
                    pl.col(f"{col}_held").fill_null(0)
                )
            return e.cast(pl.Float64)

        n = left("_n")
        feats = [
            ((left(f"_s{k}") + prior * pri[c]) / (n + prior)).alias(f"{prefix}_{name}_{c}")
            for c, k in codes.items()
        ]
        enc = r.select("_i", *feats, (n.clip(0, None) + 1).log().alias(f"{prefix}_{name}_logn"))
        out = out.join(enc, on="_i", how="left")
    cols = [f"{prefix}_{k}_{c}" for k in DF.TE_KEYS for c in names] + [
        f"{prefix}_{k}_logn" for k in DF.TE_KEYS
    ]
    return out.sort("_i").select("MVT_ID_mvt", *cols)


# --------------------------------------------------------------------------- the network


def _obs(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(C, AOBT_3 − SCHED, MVT − AOBT_3 with the member's normal taxi where missing)``."""
    c = df["c_sched"].to_numpy().astype(np.float64)
    o = df["f_obs_delay"].cast(pl.Float64).to_numpy()
    proxy = np.where(np.isfinite(o), c - o, DF._ref(df))
    return c, o, proxy


def value_inits(y: np.ndarray, d: np.ndarray, c: np.ndarray, o: np.ndarray, cls: np.ndarray,
                ref: np.ndarray) -> np.ndarray:  # fmt: skip
    """Bias inits of the five value heads, at each class's median on the training rows."""
    gap = o - d

    def med(x: np.ndarray) -> float:
        x = x[np.isfinite(x)]
        return float(np.median(x)) if len(x) else float("nan")

    def atanh_at(x: np.ndarray, mid: float, half: float) -> float:
        m = med(x)
        return float(np.arctanh(np.clip((m - mid) / half, -0.95, 0.95))) if np.isfinite(m) else 0.0

    def log_at(x: np.ndarray) -> float:
        m = med(x)
        return float(np.clip(np.log(m), -3.0, 3.0)) if np.isfinite(m) and m > 0 else 0.0

    return np.array(
        [
            atanh_at(d[cls == 1], 0.0, NEAR_S),
            atanh_at(gap[cls == 2], E_MID, E_HALF),
            atanh_at(gap[cls == 3], B_MID, B_HALF),
            log_at((y - np.maximum(c, 0.0))[cls == 4] / FAR_SCALE_S),
            log_at(y[cls == 5] / ref[cls == 5]),
        ]
    )


def fit_net(
    tr: pl.DataFrame,
    evs: list[pl.DataFrame],
    seed: int = 0,
    lam: float = 0.2,
    value_lam: float = 0.0,
    n_seeds: int = 3,
    epochs: int = 60,
    patience: int = 6,
    width: int = 256,
    batch: int = 1024,
    lr: float = 1e-3,
    device: str = "cpu",
    extra_features: list[str] | tuple[str, ...] = (),
    **_: object,
) -> list[DF.Fitted]:
    """DF's network with six classes: masked softmax, five bounded value heads, trained on
    ``mean((Σ p_k v_k − y)²)`` in ks² + ``lam`` · class log-loss (+ ``value_lam`` · the true
    class head's squared error); ``n_seeds`` nets averaged, early stopping on the raw MSE of
    an inner holdout of airport-days of the training rows (DF's). ``extra_features``: numeric
    extra-block columns appended to the numeric inputs (same scaling and imputation)."""
    import torch
    from torch import nn

    torch.set_num_threads(8)
    feats = [f for f in all_features() if f not in DF.CAT_FEATURES] + list(extra_features)
    prep = DF._Prep(tr, feats, DF.CAT_FEATURES)
    _, y, d, cls = DF._targets(tr)
    c, o, _ = _obs(tr)
    ref = DF._ref(tr)
    init = value_inits(y, d, c, o, cls, ref)
    prior = np.log(np.bincount(cls, minlength=K) / len(cls) + 1e-4)
    em, eh, bm, bh, fs = (v / 1000.0 for v in (E_MID, E_HALF, B_MID, B_HALF, FAR_SCALE_S))
    near = NEAR_S / 1000.0

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
            self.head = nn.Linear(width, K + 5)
            with torch.no_grad():
                self.head.weight.mul_(0.1)
                self.head.bias.copy_(
                    torch.tensor(np.concatenate([prior, init]), dtype=torch.float32)
                )

        def forward(self, xn, xc, ck, refk, pk, mk):
            h = torch.cat([xn] + [e(xc[:, i]) for i, e in enumerate(self.embs)], dim=1)
            out = self.head(self.body(h))
            logits = out[:, :K].masked_fill(~mk, -1e4)
            a = out[:, K:]
            v = torch.stack(
                [
                    ck,
                    ck - near * torch.tanh(a[:, 0]),
                    pk + em + eh * torch.tanh(a[:, 1]),
                    pk + bm + bh * torch.tanh(a[:, 2]),
                    torch.relu(ck) + fs * torch.exp(torch.clamp(a[:, 3], -3.0, 3.0)),
                    refk * torch.exp(torch.clamp(a[:, 4], -3.0, 3.0)),
                ],
                dim=1,
            )
            p = torch.softmax(logits, dim=1)
            return (p * v).sum(dim=1), logits, p, v

    dev = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")

    def tens(df: pl.DataFrame) -> tuple:
        ce, oe, pe = _obs(df)
        return (
            torch.tensor(prep.num_x(df), device=dev),
            torch.tensor(prep.cat_x(df), device=dev),
            torch.tensor(ce / 1000.0, dtype=torch.float32, device=dev),
            torch.tensor(DF._ref(df) / 1000.0, dtype=torch.float32, device=dev),
            torch.tensor(pe / 1000.0, dtype=torch.float32, device=dev),
            torch.tensor(class_mask(ce, oe), device=dev),
        )

    xtr = tens(tr)
    if not xtr[5].cpu().numpy()[np.arange(len(cls)), cls].all():
        raise AssertionError("a training row's class is masked out: class_mask/class_v2 disagree")
    yk = torch.tensor(y / 1000.0, dtype=torch.float32, device=dev)
    ct = torch.tensor(cls, device=dev)
    ev_t = [tens(e) for e in evs]
    acc = [
        {"pred": np.zeros(e.height), "p": np.zeros((e.height, K)), "v": np.zeros((e.height, K))}
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
            np.concatenate([o_[j] for o_ in outs]) * (1 if j == 1 else 1000.0) for j in range(3)
        )  # type: ignore[return-value]

    for s in range(n_seeds):
        torch.manual_seed(seed * 1000 + s)
        hold = DF._holdout(tr, 0.1, seed * 1000 + s)
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
                pr, logits, _, v = model(*(z[b] for z in xtr))
                loss = torch.mean((pr - yk[b]) ** 2) + lam * nn.functional.cross_entropy(
                    logits, ct[b]
                )
                if value_lam > 0:
                    vt = v.gather(1, ct[b][:, None])[:, 0]
                    loss = loss + value_lam * torch.mean((vt - yk[b]) ** 2)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            ph, _, _ = predict(model, ho_t)
            mse = float(np.mean((ph - y[iho]) ** 2))
            if mse < best - 1e-6:
                best, bad, best_ep = mse, 0, ep
                best_state = {k: v_.detach().clone() for k, v_ in model.state_dict().items()}
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
    return [DF.Fitted(pred=np.maximum(a_["pred"], 0.0), p=a_["p"], v=a_["v"]) for a_ in acc]


# --------------------------------------------------------------------------- nested fit


def _te_key_cols() -> list[str]:
    return sorted({k for keys in DF.TE_KEYS.values() for k in keys} - {"month"})


def nested_predict(
    pop: pl.DataFrame,
    months: list[str],
    fitter: Callable[..., list[DF.Fitted]],
    log: Callable[[str], None] = print,
    test_foldavg: bool = False,
    **kw: object,
) -> tuple[pl.DataFrame, list[dict]]:
    """DF's leave-one-month-out loop with the v2 encodings: month m's rows are predicted by a
    model fitted on the other months (their encodings exclude m and their own month); the
    ranking rows by a model fitted on every month. Columns: ``MVT_ID_mvt, is_test, df_pred,
    df_p_<class>, df_v_<class>``.

    ``test_foldavg``: every month-held fit also predicts the ranking rows (with encodings that
    exclude its held month, as for its own rows), and the ranking rows get the mean of those
    fits and the full fit instead of the full fit alone. The OOF rows are unchanged. One
    full fit's refit noise lands on every ranking row at once (replicate fits 2026-09-25: test
    rmsd 26–43 s between seeds vs 16–25 s on OOF); the mean over 13 fits damps it at no extra
    fitting cost. The full fit's own prediction is kept as ``df_pred_full``."""
    fm = DF.fit_mask(pop)
    src = pop.filter(pl.Series(fm)).select("MVT_ID_mvt", "month", "cls", *_te_key_cols())
    pieces, info = [], []
    te_acc: list[tuple[np.ndarray, DF.Fitted]] = []  # (row ids, fit) per held month
    te_rows = pop.filter(pl.col("is_test")) if test_foldavg else None
    for held in [*months, None]:
        tr_mask = fm & (pop["month"] != held).to_numpy() if held else fm
        tr = pop.filter(pl.Series(tr_mask))
        ev = pop.filter(pl.col("month") == held) if held else pop.filter(pl.col("is_test"))
        if ev.height == 0:
            continue
        tr = tr.join(nested_te(src, tr, held), on="MVT_ID_mvt", how="left")
        ev = ev.join(nested_te(src, ev, held), on="MVT_ID_mvt", how="left")
        assert held is None or not (tr["month"] == held).any(), "held-out month in training"
        if test_foldavg and held is not None and te_rows is not None and te_rows.height:
            te = te_rows.join(nested_te(src, te_rows, held), on="MVT_ID_mvt", how="left")
            res, res_te = fitter(tr, [ev, te], **kw)
            te_acc.append((te["MVT_ID_mvt"].to_numpy(), res_te))
            del te
        else:
            (res,) = fitter(tr, [ev], **kw)
        if held is None and te_acc:
            full = res["pred"]
            # align every fold's ranking-row predictions to this piece's row order by id
            pos = pl.DataFrame({"MVT_ID_mvt": ev["MVT_ID_mvt"]}).with_row_index("_j")
            fits = []
            for ids, f in te_acc:
                j = (
                    pl.DataFrame({"MVT_ID_mvt": ids})
                    .join(pos, on="MVT_ID_mvt", how="left", maintain_order="left")["_j"]
                    .to_numpy()
                )
                assert len(j) == ev.height and not np.isnan(j.astype(float)).any()
                inv = np.empty_like(j)
                inv[j] = np.arange(len(j))
                g = DF.Fitted(pred=f["pred"][inv])
                for k in ("p", "v"):
                    if k in f:
                        g[k] = f[k][inv]
                fits.append(g)
            fits.append(res)
            res = DF.Fitted(pred=np.mean([f["pred"] for f in fits], axis=0))
            for k in ("p", "v"):
                if all(k in f for f in fits):
                    res[k] = np.mean([f[k] for f in fits], axis=0)
            log(
                f"test: mean of {len(fits)} fits; rmsd to the full fit "
                f"{float(np.sqrt(np.mean((res['pred'] - full) ** 2))):.1f} s"
            )
        piece = ev.select("MVT_ID_mvt", "is_test").with_columns(pl.Series("df_pred", res["pred"]))
        if held is None and te_acc:
            piece = piece.with_columns(pl.Series("df_pred_full", full))
        if "p" in res:
            piece = piece.with_columns(
                [pl.Series(f"df_p_{k}", res["p"][:, i]) for i, k in enumerate(CLASSES)]
                + [pl.Series(f"df_v_{k}", res["v"][:, i]) for i, k in enumerate(CLASSES)]
            )
        pieces.append(piece)
        info.append({"held": held or "test", "n_train": tr.height, "n_eval": ev.height})
        log(f"fold {held or 'test'}: fitted on {tr.height:,}, predicted {ev.height:,}")
        del tr, ev
    return pl.concat(pieces, how="diagonal_relaxed"), info


def fit_path(run: str, tag: str = "") -> Path:
    return MIX_DIR / f"{run}__v2{('__' + tag) if tag else ''}.parquet"


def _sse_by_class(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    return (
        df.with_columns([((pl.col("y") - pl.col(c)) ** 2).alias(f"_se_{c}") for c in cols])
        .group_by("class", "no_nm")
        .agg(
            pl.len().alias("rows"),
            *[(pl.col(f"_se_{c}").sum() / 1e9).round(3).alias(f"sse_{c}_1e9") for c in cols],
        )
        .sort("class", "no_nm")
    )


def fit(
    run: str,
    out: Path,
    months_subset: list[str] | None = None,
    float32: bool = False,
    airports: list[str] | None = None,
    frame_months: list[str] | None = None,
    test_foldavg: bool = False,
    extra_blocks: list[str] | tuple[str, ...] | None = None,
    **kw: object,
) -> Path:
    t0 = time.time()
    DF._NUM_DTYPE[0] = pl.Float32 if float32 else pl.Float64
    extra_blocks = norm_blocks(extra_blocks)
    frame = build_frame(airports=airports, months=frame_months, extra_blocks=extra_blocks)
    extras = frame_extra_columns(frame)
    if bool(extras) != bool(extra_blocks):
        raise SystemExit(f"frame extra columns {extras} do not match --extra-blocks {extra_blocks}")
    if extras:
        kw["extra_features"] = extras
    _log(f"frame {frame.height:,} rows" + (f", extra columns {extras}" if extras else ""), t0)
    months = DF.run_months(run)
    if airports or frame_months:  # smoke frame: only the months it holds
        months = [m for m in months if m in set(frame["month"].unique().to_list())]
    pop = attach_member(frame, run)
    del frame
    if float32:
        feats = [
            f for f in base_features() + MEMBER_FEATURES + extras if pop.schema[f] == pl.Float64
        ]
        pop = pop.with_columns([pl.col(f).cast(pl.Float32) for f in feats])
    missing = set(months) - set(pop.filter(~pl.col("is_test"))["month"].unique().to_list())
    if missing:
        raise SystemExit(f"{run}: months {sorted(missing)} have no rows")
    pop = pop.filter(~pl.col("is_test") & pl.col("month").is_in(months) | pl.col("is_test"))
    if months_subset:  # smoke only: fewer held-out months (all months still in training)
        months = [m for m in months if m in months_subset]
    _log(f"population: {pop.height:,} rows ({int(pop['is_test'].sum()):,} test)", t0)
    preds, info = nested_predict(
        pop, months, fit_net, log=lambda s: _log(s, t0), test_foldavg=test_foldavg, **kw
    )
    res = pop.select("MVT_ID_mvt", "airport", "month", "no_nm", "c_sched", "is_test").join(
        preds.drop("is_test"), on="MVT_ID_mvt", how="inner"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f"{out.stem}.{os.getpid()}.tmp")
    pv = [c for c in res.columns if c.startswith(("df_p_", "df_v_"))]
    res.with_columns([pl.col(c).cast(pl.Float32) for c in pv]).write_parquet(tmp)  # ~half size
    tmp.replace(out)

    name = pl.col("cls").replace_strict(dict(enumerate(CLASSES)), default="?").alias("class")
    tr = (
        pop.filter(pl.Series(DF.fit_mask(pop)))
        .select("MVT_ID_mvt", "month", "no_nm", "y", "cls", "m_pred")
        .join(res.select("MVT_ID_mvt", "df_pred"), on="MVT_ID_mvt", how="inner")
        .with_columns(name)
    )
    tab = _sse_by_class(tr, ["m_pred", "df_pred"])
    tab_jj = _sse_by_class(tr.filter(pl.col("month").is_in(GATE_MONTHS)), ["m_pred", "df_pred"])
    y, mp, dp = (tr[c].to_numpy() for c in ("y", "m_pred", "df_pred"))
    jj = tr["month"].is_in(GATE_MONTHS).to_numpy()
    pcols = [f"df_p_{k}" for k in CLASSES]
    pc = tr.join(res.select("MVT_ID_mvt", *pcols), on="MVT_ID_mvt")
    p_true = pc.select(pl.concat_list(pcols).list.get(pl.col("cls"))).to_series().to_numpy()
    test_shares = (
        res.filter(pl.col("is_test"))
        .group_by("month")
        .agg(pl.len().alias("rows"), *[pl.col(c).mean().round(5) for c in pcols])
        .sort("month")
    )
    oof_shares = (
        pc.group_by("month")
        .agg(
            pl.len().alias("rows"),
            *[pl.col(c).mean().round(5) for c in pcols],
            *[
                (pl.col("cls") == i).mean().round(5).alias(f"true_{k}")
                for i, k in enumerate(CLASSES)
            ],
        )
        .sort("month")
    )
    stats = {
        "run": run,
        "kw": dict(kw),
        "float32": float32,
        "airports": airports,
        "frame_months": frame_months,
        **({"extra_blocks": extra_blocks} if extra_blocks else {}),
        "months": months,
        "folds": info,
        "class_counts_2025_ex_ro": class_counts(pop).to_dicts(),
        "oof_rows_ex_ro": len(y),
        "sse_member_1e9": float(((y - mp) ** 2).sum() / 1e9),
        "sse_v2_1e9": float(((y - dp) ** 2).sum() / 1e9),
        "jj_sse_member_1e9": float(((y - mp)[jj] ** 2).sum() / 1e9),
        "jj_sse_v2_1e9": float(((y - dp)[jj] ** 2).sum() / 1e9),
        "by_class": tab.to_dicts(),
        "by_class_jj": tab_jj.to_dicts(),
        "mean_p_by_true_class": pc.group_by("class")
        .agg([pl.col(c).mean().round(4) for c in pcols])
        .sort("class")
        .to_dicts(),
        "class_logloss": float(-np.log(np.clip(p_true, 1e-9, 1)).mean()),
        "oof_predicted_vs_true_shares": oof_shares.to_dicts(),
        "test_predicted_shares": test_shares.to_dicts(),
        "seconds": round(time.time() - t0, 1),
        "max_rss_gb": round(DF._maxrss_gb(), 2),
    }
    (out.parent / f"{out.stem}.json").write_text(json.dumps(stats, indent=2, default=str))
    with pl.Config(tbl_rows=30, tbl_cols=12):
        print(tab)
        print(test_shares)
    print(
        f"OOF ex-ro SSE (1e9 s²), all rows: member {stats['sse_member_1e9']:.2f} -> v2 "
        f"{stats['sse_v2_1e9']:.2f}; Jan+Jul {stats['jj_sse_member_1e9']:.2f} -> "
        f"{stats['jj_sse_v2_1e9']:.2f}; class logloss {stats['class_logloss']:.4f}"
    )
    _log(f"wrote {out}", t0)
    return out


# --------------------------------------------------------------------------- apply


def apply(
    run: str,
    fit_file: Path,
    out_name: str,
    bands: tuple[float, ...] = (),
    blend: str = "nested",
    keep_rollover: bool = True,
) -> Path:
    """Write ``outputs/runs/<out_name>/``. Without ``bands`` this *is* DF's
    :func:`delayed_flight_model.apply` (called directly). With ``bands`` (C thresholds in s),
    the fold-safe α is fitted separately for each band of C, with the same formula
    ``new = α·v2 + (1 − α)·(pred − R) + R`` and the same seed shift."""
    if not bands:
        return DF.apply(run, fit_file, out_name, blend=blend, keep_rollover=keep_rollover)
    if blend != "nested":
        raise SystemExit("--alpha-bands needs --blend nested")
    src, dst = paths.RUNS / run, paths.RUNS / out_name
    if dst.resolve() == src.resolve():
        raise SystemExit("--out must differ from --run")
    edges = np.array([-np.inf, *sorted(bands), np.inf])
    fp = pl.read_parquet(fit_file, columns=["MVT_ID_mvt", "is_test", "c_sched", "df_pred"])
    fp = fp.with_columns(
        pl.Series("band", np.searchsorted(edges, fp["c_sched"].to_numpy(), side="left") - 1)
    )
    o = pl.read_parquet(src / "oof.parquet").join(
        fp.filter(~pl.col("is_test")).select("MVT_ID_mvt", "df_pred", "band"), on="MVT_ID_mvt"
    )
    ro = DF.rollover_term(o) if keep_rollover else 0.0
    yv = o["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy()
    mem = o["pred"].to_numpy().astype(np.float64) - ro
    alphas: dict[int, tuple[dict[str, float], float]] = {}
    for b in range(len(edges) - 1):
        m = o["band"].to_numpy() == b
        if m.any():
            alphas[b] = DF.nested_alphas(
                yv[m], o["df_pred"].to_numpy()[m], mem[m], o["month"].to_numpy()[m]
            )
            print(
                f"apply: band {b} (C in ({edges[b]:.0f}, {edges[b + 1]:.0f}]) alpha "
                + ", ".join(f"{k} {a:.3f}" for k, a in alphas[b][0].items())
                + f"; test {alphas[b][1]:.3f}"
            )
    del o
    dst.mkdir(parents=True, exist_ok=True)
    counts = {}
    for which, is_test in (("oof", False), ("test_pred", True)):
        path = src / f"{which}.parquet"
        df = pl.read_parquet(path)
        sub = fp.filter(pl.col("is_test") == is_test).select("MVT_ID_mvt", "df_pred", "band")
        merged = df.join(sub, on="MVT_ID_mvt", how="left", maintain_order="left")
        use = merged["df_pred"].is_not_null().to_numpy()
        band = merged["band"].fill_null(-1).to_numpy()
        alpha = np.zeros(merged.height)
        for b, (per, a_t) in alphas.items():
            m = band == b
            if is_test:
                alpha[m] = a_t
            else:
                am = merged["month"].replace_strict(per, default=None).to_numpy()
                am = np.asarray(am, dtype=np.float64)
                if np.isnan(am[m & use]).any():
                    raise SystemExit("apply: a v2 row's month has no nested alpha")
                alpha[m] = am[m]
        base = merged["pred"].to_numpy().astype(np.float64)
        dfv = np.nan_to_num(merged["df_pred"].to_numpy().astype(np.float64), nan=0.0)
        ro_ = DF.rollover_term(merged) if keep_rollover else np.zeros(merged.height)
        new_pred = np.where(use, alpha * dfv + (1 - alpha) * (base - ro_) + ro_, base)
        shift = np.where(use, new_pred - merged["pred"].to_numpy(), 0.0)
        cols = {"pred": np.maximum(new_pred, 0.0)}
        for s in (c for c in df.columns if c.startswith("pred_s")):
            cols[s] = np.maximum(merged[s].to_numpy().astype(np.float64) + shift, 0.0)
        merged.with_columns([pl.Series(k, v, dtype=df.schema[k]) for k, v in cols.items()]).select(
            df.columns
        ).write_parquet(dst / f"{which}.parquet")
        counts[which] = {
            "rows": df.height,
            "replaced": int(use.sum()),
            "rmsd_replaced": float(np.sqrt(np.mean(shift[use] ** 2))) if use.any() else 0.0,
            "mean_shift_replaced": float(shift[use].mean()) if use.any() else 0.0,
        }
        print(f"apply: {which} {df.height:,} rows, {int(use.sum()):,} replaced -> {dst}")
    (dst / "meta.json").write_text(
        json.dumps(
            {
                "source_run": run,
                "df_predictions": str(fit_file),
                "blend": "nested",
                "alpha_bands": list(bands),
                "keep_rollover": keep_rollover,
                "alpha_oof": {str(b): a[0] for b, a in alphas.items()},
                "alpha_test": {str(b): a[1] for b, a in alphas.items()},
                "counts": counts,
                "written": datetime.now(UTC).isoformat(timespec="seconds"),
                "note": "post-processed mixture-v2 predictions; not a registry run",
            },
            indent=2,
        )
        + "\n"
    )
    return dst


# --------------------------------------------------------------------------- report


def report(
    base: str,
    cand: str,
    fit_file: Path | None,
    n_boot: int,
    write: Path | None,
    restrict: bool = False,
) -> str:
    """Headline slices, per airport, SSE per v2 class (all months and Jan+Jul, ex-ro), the
    test-side movement and v2's predicted class shares on the ranking months. ``restrict``
    limits the OOF comparison to the rows in ``fit_file`` (smoke fits on a subset)."""
    cols = ["MVT_ID_mvt", "month", "airport", "TAXITIME_SEC_mvt", "pred"]
    r = pl.read_parquet(paths.RUNS / base / "oof.parquet", columns=cols)
    c = pl.read_parquet(paths.RUNS / cand / "oof.parquet", columns=["MVT_ID_mvt", "pred"])
    if restrict and fit_file is not None:
        ids = pl.read_parquet(fit_file, columns=["MVT_ID_mvt", "is_test"]).filter(
            ~pl.col("is_test")
        )
        r = r.join(ids.select("MVT_ID_mvt"), on="MVT_ID_mvt", how="semi")

    def secs(a: str, b: str) -> pl.Expr:
        return (pl.col(a) - pl.col(b)).dt.total_seconds().cast(pl.Float64)

    mv = (
        pl.scan_parquet(paths.INTERIM / "movements.parquet")
        .select(
            "MVT_ID_mvt",
            "MVT_TIME_UTC_mvt",
            "AOBT_3_flt",
            secs("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt").alias("c_sched"),
            secs("BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt").alias("d"),
            secs("AOBT_3_flt", "SCHED_TIME_UTC_mvt").alias("obs_delay"),
        )
        .join(r.select("MVT_ID_mvt").lazy(), on="MVT_ID_mvt", how="semi")
        .collect()
    )
    d = r.join(c.rename({"pred": "pred_c"}), on="MVT_ID_mvt", how="inner").join(
        mv, on="MVT_ID_mvt", how="left"
    )
    y = d["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy()
    se_r = (y - d["pred"].to_numpy()) ** 2
    se_c = (y - d["pred_c"].to_numpy()) ** 2
    month, ap = d["month"].to_numpy(), d["airport"].to_numpy()
    nonm = d["AOBT_3_flt"].is_null().to_numpy()
    exro = ~DF.is_rollover(y)
    csch = d["c_sched"].to_numpy()
    jj = np.isin(month, GATE_MONTHS)
    cls = class_v2(d["d"].to_numpy(), y, csch, d["obs_delay"].to_numpy())
    blk, groups = PD.block_index(d)
    nb = int(blk.max()) + 1
    w = PD.block_weights(groups, nb, n_boot, 0) if n_boot else None

    def row(name: str, m: np.ndarray) -> dict:
        if not m.any():
            return {"slice": name, "rows": 0}
        rr, cc = float(np.sqrt(se_r[m].mean())), float(np.sqrt(se_c[m].mean()))
        out = {"slice": name, "rows": int(m.sum()), "base_rmse": round(rr, 2),
               "cand_rmse": round(cc, 2), "delta": round(cc - rr, 2)}  # fmt: skip
        if w is not None:
            b = PD.bootstrap_delta(w, blk, nb, se_r, se_c, m)
            out |= {"lo": b["boot_lo"], "hi": b["boot_hi"], "p_improve": b["boot_p_improve"]}
        return out

    slices = {
        "all": np.ones_like(exro),
        "all|ex_ro": exro,
        "2025-01|ex_ro": (month == "2025-01") & exro,
        "2025-07|ex_ro": (month == "2025-07") & exro,
        "jj|ex_ro": jj & exro,
        "no_aobt|ex_ro": nonm & exro,
        "jj no_aobt|ex_ro": jj & nonm & exro,
        "C>1800|ex_ro": (csch > 1800) & exro,
        "C<=1800|ex_ro": (csch <= 1800) & exro,
    }
    main_tab = pl.DataFrame([row(k, m) for k, m in slices.items()])
    per_ap = pl.DataFrame(
        [row(f"ap:{a}|ex_ro", (ap == a) & exro) for a in sorted(set(ap.tolist()))]
    )
    cl_rows = []
    tot = float(se_r[exro].sum())
    for k, name in enumerate(CLASSES):
        for nn_ in (False, True):
            for scope, sm in (("12m", np.ones_like(jj)), ("jj", jj)):
                m = exro & (cls == k) & (nonm == nn_) & sm
                cl_rows.append(
                    {
                        "class": name,
                        "no_nm": nn_,
                        "scope": scope,
                        "rows": int(m.sum()),
                        "sse_base_1e9": round(float(se_r[m].sum()) / 1e9, 3),
                        "sse_cand_1e9": round(float(se_c[m].sum()) / 1e9, 3),
                        "delta_1e9": round(float(se_c[m].sum() - se_r[m].sum()) / 1e9, 3),
                        "base_share_of_12m_exro": round(float(se_r[m].sum()) / tot, 4),
                    }
                )
    by_cls = pl.DataFrame(cl_rows)
    tb = pl.read_parquet(paths.RUNS / base / "test_pred.parquet", columns=["MVT_ID_mvt", "pred"])
    tc = pl.read_parquet(paths.RUNS / cand / "test_pred.parquet", columns=["MVT_ID_mvt", "pred"])
    if restrict and fit_file is not None:
        tids = pl.read_parquet(fit_file, columns=["MVT_ID_mvt", "is_test"]).filter("is_test")
        tb = tb.join(tids.select("MVT_ID_mvt"), on="MVT_ID_mvt", how="semi")
    tmv = (
        pl.scan_parquet(paths.INTERIM / "movements.parquet")
        .select("MVT_ID_mvt", "month", "AOBT_3_flt")
        .join(tb.select("MVT_ID_mvt").lazy(), on="MVT_ID_mvt", how="semi")
        .collect()
    )
    t = tb.join(tc.rename({"pred": "pred_c"}), on="MVT_ID_mvt").join(tmv, on="MVT_ID_mvt")
    t = t.with_columns((pl.col("pred_c") - pl.col("pred")).alias("shift"))
    test_rows = []
    for name, sub in (
        ("all test rows", t),
        ("NM", t.filter(pl.col("AOBT_3_flt").is_not_null())),
        ("no NM", t.filter(pl.col("AOBT_3_flt").is_null())),
        ("2026-01", t.filter(pl.col("month") == "2026-01")),
        ("2026-07", t.filter(pl.col("month") == "2026-07")),
    ):
        sh = sub["shift"].to_numpy()
        test_rows.append(
            {
                "test slice": name,
                "rows": len(sh),
                "rmsd": round(float(np.sqrt(np.mean(sh**2))) if len(sh) else 0.0, 1),
                "mean_shift": round(float(sh.mean()) if len(sh) else 0.0, 1),
            }
        )
    parts = [
        f"# Mixture-v2 report — {cand} vs {base}",
        "",
        "Generated by `scripts/mixture_v2.py report`. Negative delta = candidate better. "
        f"Seed-mean predictions; {n_boot} day-block bootstrap replicates (`paired_delta.py`'s "
        "resampler, seed 0)." + (" OOF restricted to the fit file's rows." if restrict else ""),
        "",
        "## Headline slices",
        "",
        A.md(main_tab),
        "",
        "## Per airport (ex-ro)",
        "",
        A.md(per_ap),
        "",
        "## SSE by v2 class (ex-ro, 1e9 s²)",
        "",
        A.md(by_cls),
        "",
        "## Test side: movement of the candidate against the base",
        "",
        A.md(pl.DataFrame(test_rows)),
        "",
    ]
    if fit_file is not None and fit_file.is_file():
        pcols = [f"df_p_{k}" for k in CLASSES]
        fp = pl.read_parquet(fit_file, columns=["month", "is_test", "no_nm", *pcols])
        shares = (
            fp.filter(pl.col("is_test"))
            .group_by("month", "no_nm")
            .agg(pl.len().alias("rows"), *[pl.col(x).mean().round(5) for x in pcols])
            .sort("month", "no_nm")
        )
        parts += ["## v2 predicted class shares on the ranking rows", "", A.md(shares), ""]
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
    b = sub.add_parser("build", help="build (or rebuild) the cached all-departures frame")
    b.add_argument("--rebuild", action="store_true")
    b.add_argument("--audit", action="store_true", help="print the train/test null-rate audit")
    b.add_argument("--airports", nargs="*", default=None, help="smoke: airport subset")
    b.add_argument("--months", nargs="*", default=None, help="smoke: month subset")
    b.add_argument(
        "--extra-blocks",
        nargs="*",
        default=None,
        help="registered label-free feature blocks appended to the frame (default none)",
    )
    f = sub.add_parser("fit", help="nested leave-one-month-out v2 fit against a run")
    f.add_argument("--run", required=True, help="bare run name under outputs/runs/")
    f.add_argument("--tag", default="", help="suffix for the output file (variants)")
    f.add_argument("--lam", type=float, default=0.2, help="class log-loss weight")
    f.add_argument("--value-lam", type=float, default=0.0, help="true-class value-head MSE weight")
    f.add_argument("--n-seeds", type=int, default=3, help="nets averaged per fold")
    f.add_argument(
        "--seed", type=int, default=0, help="base seed of the nets (replicate fits; 0 = default)"
    )
    f.add_argument("--epochs", type=int, default=60)
    f.add_argument("--width", type=int, default=256)
    f.add_argument("--batch", type=int, default=1024)
    f.add_argument("--lr", type=float, default=1e-3)
    f.add_argument("--patience", type=int, default=6)
    f.add_argument("--device", default="cpu")
    f.add_argument("--held-months", nargs="*", default=None, help="smoke: held-out month subset")
    f.add_argument("--airports", nargs="*", default=None, help="smoke: frame airport subset")
    f.add_argument("--months", nargs="*", default=None, help="smoke: frame month subset")
    f.add_argument("--float32", action="store_true", help="float32 features (half the memory)")
    f.add_argument(
        "--test-foldavg",
        action="store_true",
        help="ranking rows = mean of the 12 month-held fits and the full fit (OOF unchanged)",
    )
    f.add_argument(
        "--extra-blocks",
        nargs="*",
        default=None,
        help="feature blocks whose numeric columns join the nets' inputs (needs --tag)",
    )
    a = sub.add_parser("apply", help="write outputs/runs/<out>/ with the v2 rows blended in")
    a.add_argument("--run", required=True, help="bare source run name")
    a.add_argument("--out", required=True, help="bare output run name")
    a.add_argument("--fit", type=Path, default=None, help="v2 parquet (default: from --fit-run)")
    a.add_argument("--fit-run", default=None, help="run the v2 fit was made on (default --run)")
    a.add_argument("--tag", default="")
    a.add_argument("--blend", choices=("nested", "none"), default="nested")
    a.add_argument(
        "--alpha-bands",
        type=float,
        nargs="*",
        default=[],
        help="C thresholds (s): one nested α per band of C (default: DF's single α)",
    )
    a.add_argument("--drop-rollover", action="store_true")
    r = sub.add_parser("report", help="per-class / per-airport / test-share report")
    r.add_argument("--base", required=True)
    r.add_argument("--cand", required=True)
    r.add_argument("--fit", type=Path, default=None, help="v2 parquet (default: from --fit-run)")
    r.add_argument("--fit-run", default=None)
    r.add_argument("--tag", default="")
    r.add_argument("--restrict", action="store_true", help="OOF rows of the fit file only")
    r.add_argument("--bootstrap", type=int, default=2000)
    r.add_argument("--write", type=Path, default=None)
    args = ap.parse_args()
    for name in ("run", "out", "base", "cand", "fit_run"):
        v = getattr(args, name, None)
        if v and ("/" in v or v.endswith(".parquet")):
            raise SystemExit(f"--{name.replace('_', '-')} takes a bare run name, not a path: {v}")
    if args.cmd == "build":
        t0 = time.time()
        frame = build_frame(
            cache=not args.rebuild,
            airports=args.airports,
            months=args.months,
            extra_blocks=args.extra_blocks,
        )
        _log(f"frame {frame.height:,} x {frame.width}", t0)
        with pl.Config(tbl_rows=200, tbl_cols=10):
            print(class_counts(frame))
            if args.audit:
                print(leak_audit(frame))
    elif args.cmd == "fit":
        kw = {
            "lam": args.lam,
            "value_lam": args.value_lam,
            "n_seeds": args.n_seeds,
            "epochs": args.epochs,
            "width": args.width,
            "batch": args.batch,
            "lr": args.lr,
            "patience": args.patience,
            "device": args.device,
        }
        if args.seed:
            kw["seed"] = args.seed
        if args.extra_blocks and not args.tag:
            raise SystemExit(
                "--extra-blocks needs a --tag (the output must not replace the default)"
            )
        fit(
            args.run,
            fit_path(args.run, args.tag),
            months_subset=args.held_months,
            float32=args.float32,
            airports=args.airports,
            frame_months=args.months,
            test_foldavg=args.test_foldavg,
            extra_blocks=args.extra_blocks,
            **kw,
        )
    elif args.cmd == "apply":
        fp = args.fit or fit_path(args.fit_run or args.run, args.tag)
        apply(
            args.run,
            fp,
            args.out,
            bands=tuple(args.alpha_bands),
            blend=args.blend,
            keep_rollover=not args.drop_rollover,
        )
    else:
        fp = args.fit or (fit_path(args.fit_run, args.tag) if args.fit_run else None)
        report(args.base, args.cand, fp, args.bootstrap, args.write, restrict=args.restrict)


if __name__ == "__main__":
    main()
