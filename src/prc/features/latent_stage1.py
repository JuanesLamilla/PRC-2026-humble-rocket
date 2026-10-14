"""Fold-safe stage-1 latent estimates of taxi-out (``stage1: unimpeded_additional``).

Taxi-out = unimpeded taxi (stand, runway, aircraft, geometry) + additional time (queue,
weather, time of day). The main model already gets the pieces as raw features and as group
statistics (``target_enc``: ``p10`` is a crude unimpeded proxy). This module gives it two
*model-based* estimates of the two terms, fitted inside the fold like a target encoding:

``s1_unimpeded``
    A small LightGBM (CPU, :data:`N_ROUNDS` trees, depth :data:`MAX_DEPTH`) on the fit set's
    rows that are **clean** (:func:`prc.features.target_enc.clean_mask` and not the
    schedule-default artefact — the ``te_clean_labels`` definition) **and low-congestion**
    (``sq_taxiing_at_push`` and ``sq_tko_during_taxi`` both at or below their airport's
    25th percentile over the clean fit rows), target ``y`` in seconds, **static features
    only** (:data:`UNIMPEDED_COLS`). Its prediction is the expected unimpeded taxi.
``s1_additional``
    A second small LightGBM on **all** clean fit rows, target ``y − s1_unimpeded`` (the
    unimpeded model's prediction on those rows, in-sample within this stage-1 fit), from the
    queue / weather / time-of-day features (:data:`ADDITIONAL_PREFIXES`,
    :data:`ADDITIONAL_COLS`).
``s1_total``
    Their sum (a tree cannot add two columns).

Fold safety is the whole point, and mirrors ``te_train_mode: inner_month``
(:func:`prc.features.target_enc.encode_inner_month`):

* served rows (the fold's validation rows, the test rows) get stage 1 fitted on **all** of the
  fit frame's rows — :func:`fit_apply`;
* the fit frame's own training rows of month ``m`` get stage 1 fitted on the fit frame's rows
  of the **other** months — :func:`inner_month` — so no training row's stage-1 value has seen
  its own label, or any label of its month.

Every fit is deterministic (fixed seed, ``deterministic``, row-wise histograms, fixed thread
count), and the row cap (:data:`MAX_FIT_ROWS`) draws with a fixed RNG over the candidate
rows, so a stage-1 table is a pure function of its inputs.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

from prc.data import schema as S
from prc.features.target_enc import _sched_default, clean_mask

log = logging.getLogger(__name__)

MODES = ("off", "unimpeded_additional")
OUT_COLS = ("s1_unimpeded", "s1_additional", "s1_total")

#: Static columns the unimpeded model may read: the ``cats`` block's identifiers (or the raw
#: movement-table keys when ``cats`` is not in the run), and the ADS-B stand-to-threshold
#: distance. Nothing that varies with the traffic of the moment.
UNIMPEDED_COLS = (
    "airport",
    "runway",
    "stand",
    "actype",
    "wake",
    "operator",
    "airport_runway",
    "airport_stand",
    "adsb2_stand_thr_dist_m",
)
#: Raw movement-table fallbacks for the identifiers, used only when the ``cats`` name is absent.
RAW_FALLBACK = {
    "airport": S.AIRPORT,
    "runway": S.RUNWAY,
    "stand": S.STAND,
    "actype": S.AIRCRAFT_TYPE,
    "wake": S.WAKE_CAT,
    "operator": S.OPERATOR,
}
#: The additional-time model: ``surface_queue``, ``throughput``, ``metar_visibility``,
#: ``winter_ops`` by prefix; ``runway_gaps_v2`` and the calendar by name; plus
#: :data:`IDENTIFIERS`. ``month_num``
#: and ``doy`` are left out on purpose: under inner leave-one-month-out every served month is
#: unseen, so a month-valued split is extrapolation.
ADDITIONAL_PREFIXES = ("sq_", "n_dep_tko_", "n_arr_ldg_", "n_rwy_tko_", "mtr_", "wo_")
ADDITIONAL_COLS = (
    "gap_prev_tko",
    "gap_next_tko",
    "hour",
    "minute_of_day",
    "dow",
    "is_weekend",
)
#: Identifiers the additional model also reads (a queue of five means different things at two
#: airports); the raw airport code stands in when the ``cats`` block is absent.
IDENTIFIERS = ("airport", "runway", "airport_runway")
#: The two congestion columns the low-congestion subset is defined by, and the quantile.
CONGESTION_COLS = ("sq_taxiing_at_push", "sq_tko_during_taxi")
LOW_Q = 0.25

N_ROUNDS = 200
MAX_DEPTH = 6
#: Cap on the rows one stage-1 model is fitted on (a fixed-seed draw). 2 M-row folds would
#: otherwise dominate the cost; a 31-leaf, 200-tree model does not need more.
MAX_FIT_ROWS = 400_000
#: Below this many rows a stage-1 model is the constant mean of its target.
MIN_FIT_ROWS = 20
PARAMS = {
    "objective": "regression",
    "learning_rate": 0.08,
    "num_leaves": 31,
    "max_depth": MAX_DEPTH,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.9,
    "lambda_l2": 1.0,
    "max_bin": 63,
    "max_cat_to_onehot": 4,
    "cat_smooth": 10.0,
    "min_data_per_group": 50,
    "seed": 0,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}


@dataclass(frozen=True)
class Columns:
    """Which columns of the frame each stage-1 model reads (resolved once per frame)."""

    unimpeded: tuple[str, ...]
    additional: tuple[str, ...]
    congestion: tuple[str, ...]


def resolve_columns(frame: pl.DataFrame) -> Columns:
    cols = set(frame.columns)
    unimp: list[str] = []
    for c in UNIMPEDED_COLS:
        if c in cols:
            unimp.append(c)
        elif c in RAW_FALLBACK and RAW_FALLBACK[c] in cols:
            unimp.append(RAW_FALLBACK[c])
    add = [c for c in frame.columns if c.startswith(ADDITIONAL_PREFIXES)]
    add += [c for c in ADDITIONAL_COLS if c in cols and c not in add and c not in IDENTIFIERS]
    if not unimp:
        raise ValueError("stage1: no static column (airport / stand / runway / ...) in the frame")
    if not add:  # the identifiers alone would make it a second unimpeded model
        raise ValueError(
            "stage1: no queue / weather / calendar column in the frame; add surface_queue, "
            "throughput, runway_gaps_v2, metar_visibility, winter_ops or calendar_v2"
        )
    ids = [c for c in IDENTIFIERS if c in cols]
    add += ids or ([S.AIRPORT] if S.AIRPORT in cols else [])
    cong = tuple(c for c in CONGESTION_COLS if c in cols)
    return Columns(tuple(dict.fromkeys(unimp)), tuple(dict.fromkeys(add)), cong)


def needed_columns(frame: pl.DataFrame) -> list[str]:
    """Every column stage 1 reads from ``frame``: narrow a wide fold frame to these before
    filtering it, so a fold's copy costs ~70 columns rather than the whole matrix."""
    cols = resolve_columns(frame)
    keys = [S.MVT_ID, S.AIRPORT, S.SPLIT_MONTH, S.TARGET, S.MVT_TIME, S.SCHED_TIME]
    want = [*keys, *cols.unimpeded, *cols.additional, *cols.congestion]
    return [c for c in dict.fromkeys(want) if c in frame.columns]


def _clean(frame: pl.DataFrame) -> pl.Expr:
    keep = clean_mask()
    if S.MVT_TIME in frame.columns and S.SCHED_TIME in frame.columns:
        keep = keep & ~_sched_default()
    return keep & pl.col(S.TARGET).is_not_null()


def _low_congestion(fit_clean: pl.DataFrame, cong: tuple[str, ...]) -> np.ndarray:
    """Rows at or below their airport's 25th percentile on every congestion column (the
    percentiles are taken over ``fit_clean`` itself — no target is read). Null congestion
    (no valid own interval) is not low congestion. Without ``surface_queue`` every clean row
    qualifies, and the log says so."""
    if not cong:
        log.warning("stage1: no surface_queue columns; unimpeded model fitted on all clean rows")
        return np.ones(fit_clean.height, dtype=bool)
    exprs = [
        (pl.col(c) <= pl.col(c).quantile(LOW_Q, "nearest").over(S.AIRPORT)).fill_null(False)
        for c in cong
    ]
    return fit_clean.select(pl.all_horizontal(exprs)).to_series().to_numpy()


def _pandas(frame: pl.DataFrame, cols: tuple[str, ...]) -> pd.DataFrame:
    pdf = frame.select(cols).to_pandas()
    for c in pdf.columns:
        d = pdf[c].dtype
        if not (
            pd.api.types.is_numeric_dtype(d) or pd.api.types.is_bool_dtype(d)
        ) and not isinstance(d, pd.CategoricalDtype):
            pdf[c] = pdf[c].astype("category")
    return pdf


def _cap(n: int) -> np.ndarray:
    """Positions of at most :data:`MAX_FIT_ROWS` rows out of ``n``, sorted, fixed seed."""
    if n <= MAX_FIT_ROWS:
        return np.arange(n)
    return np.sort(np.random.default_rng(0).choice(n, MAX_FIT_ROWS, replace=False))


def _fit(X: pd.DataFrame, y: np.ndarray, n_threads: int) -> lgb.Booster | float:
    """One small booster, or the mean when there is too little to learn from."""
    if len(y) < MIN_FIT_ROWS:
        return float(np.mean(y)) if len(y) else 0.0
    # 50 rows per leaf at scale; fewer on a small frame (the synthetic fixture, a sub-sample)
    mdl = int(min(PARAMS["min_data_in_leaf"], max(5, len(y) // 20)))
    params = {**PARAMS, "min_data_in_leaf": mdl, "min_data_per_group": mdl}
    ds = lgb.Dataset(X, label=y, categorical_feature="auto", free_raw_data=True)
    return lgb.train({**params, "num_threads": n_threads}, ds, num_boost_round=N_ROUNDS)


def _predict(model: lgb.Booster | float, X: pd.DataFrame, n_threads: int) -> np.ndarray:
    if isinstance(model, float):
        return np.full(len(X), model)
    return np.asarray(model.predict(X, num_threads=n_threads), dtype=np.float64)


def fit_apply(
    fit_on: pl.DataFrame,
    apply_to: pl.DataFrame,
    n_threads: int = 4,
    cols: Columns | None = None,
) -> pl.DataFrame:
    """Fit both stage-1 models on ``fit_on`` and return ``apply_to``'s estimates, keyed by
    MVT_ID, in ``apply_to``'s order. Reads ``fit_on``'s target only."""
    cols = cols or resolve_columns(fit_on)
    fit_clean = fit_on.filter(_clean(fit_on))
    y = fit_clean.get_column(S.TARGET).to_numpy().astype(np.float64)
    low = _low_congestion(fit_clean, cols.congestion)

    u_rows = np.flatnonzero(low)[_cap(int(low.sum()))]
    Xu_all = _pandas(fit_clean, cols.unimpeded)
    m_u = _fit(Xu_all.iloc[u_rows], y[u_rows], n_threads)

    a_rows = _cap(fit_clean.height)
    Xa_fit = _pandas(fit_clean[a_rows], cols.additional)
    u_hat_fit = _predict(m_u, Xu_all.iloc[a_rows], n_threads)
    del Xu_all
    m_a = _fit(Xa_fit, y[a_rows] - u_hat_fit, n_threads)
    del Xa_fit

    u = _predict(m_u, _pandas(apply_to, cols.unimpeded), n_threads)
    a = _predict(m_a, _pandas(apply_to, cols.additional), n_threads)
    return pl.DataFrame(
        {
            S.MVT_ID: apply_to.get_column(S.MVT_ID),
            "s1_unimpeded": u.astype(np.float32),
            "s1_additional": a.astype(np.float32),
            "s1_total": (u + a).astype(np.float32),
        }
    )


def inner_month(
    fit_on: pl.DataFrame, n_threads: int = 4, month_col: str = S.SPLIT_MONTH
) -> pl.DataFrame:
    """Training-row estimates with served-row semantics: the rows of month ``m`` get
    :func:`fit_apply` fitted on ``fit_on``'s rows of the other months. One row per row of
    ``fit_on``, in its order, keyed by MVT_ID."""
    cols = resolve_columns(fit_on)
    pos = "__s1_inner_pos"
    base = fit_on.with_row_index(pos)
    parts: list[pl.DataFrame] = []
    for m in base.get_column(month_col).unique().sort().to_list():
        t = time.time()
        is_m = pl.col(month_col).eq_missing(m)
        rows_m, fit_m = base.filter(is_m), base.filter(~is_m)
        out = fit_apply(fit_m.drop(pos), rows_m.drop(pos), n_threads, cols)
        parts.append(out.with_columns(rows_m.get_column(pos)))
        log.info(
            "stage1 inner month %s: fit %d rows, apply %d, %.1fs",
            m,
            fit_m.height,
            rows_m.height,
            time.time() - t,
        )
    out = pl.concat(parts, how="vertical").sort(pos)
    if not out.get_column(S.MVT_ID).equals(fit_on.get_column(S.MVT_ID)):
        raise AssertionError("stage1.inner_month: MVT_ID order not preserved")
    return out.drop(pos)
