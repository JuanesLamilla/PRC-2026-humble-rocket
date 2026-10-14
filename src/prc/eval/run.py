"""Run one experiment end-to-end: features -> folds -> train per fold+seed -> OOF ->
test predictions -> artifacts -> registry row.

Artifacts under ``outputs/runs/<name>/``:
    config.yaml            resolved config
    metrics.json           cv rmse (overall, per fold, per airport), best iterations, timings
    oof.parquet            MVT_ID, fold, y, pred          (blendable across teammates)
    test_pred.parquet      MVT_ID, pred                   (mean over folds x seeds, or,
                           under `test_fit: full`, the models fitted on all training rows)
    test_pred_foldavg.parquet                             (`test_fit: full` only, 2026-09-23:
                           the fold-model average that `test_pred` used to be, same schema,
                           same head / mixture / clip; the full fit overwrites it in memory)
    importance.csv         gain importance averaged over folds
    log.txt

The registry ``experiments/registry.csv`` gets one appended row per run.

Three optional layers switch on from the YAML and are no-ops when their keys are absent
(``prc.config``; ``prc.mixture``):

* ``target.mode: residual_proxy`` — regime-A rows (a real NM off-block) learn
  ``y − (MVT − AOBT_3)`` through a per-row offset column and the offset is added back at
  prediction; regime B/C rows keep the raw target (offset 0).
* ``target.mode: residual_proxy_split`` — the same offset, but one model per group of
  regimes (``target.split_groups``, default ``[[A], [B, C]]``) instead of one model with the
  regime as a feature.
* ``mixture.corruption`` — a fold-safe classifier per corruption type; the final prediction is
  ``Σ_k p_k·C_k + (1 − Σ_k p_k)·ŷ_normal``. The mixture is affine in ``ŷ``, so it is applied
  once to the fold/seed-averaged prediction (and to each per-seed column) rather than inside
  the fit loop. Options (all default to the 2026-09-18 ``exp0043`` behaviour):
  ``extra_blocks`` / ``extra_signals`` / ``identity_signals`` widen the *classifier's* matrix
  only, ``calibrate`` fits per-airport isotonic curves on each head's internal holdout,
  ``normal_weight`` replaces the hard all/clean switch with ``1 − Σ_k p_k`` sample weights,
  and ``sched_head: z_regression`` replaces the schedule-default *classifier* with a weighted
  regression of ``z = (y − ŷ)/(C − ŷ)`` — the RMSE-optimal mixture weight, which needs a
  nested in-fold ``ŷ_normal`` (:func:`_nested_yhat`).

Two more options shape the regressor loop and what comes out of it, both default-off and
both byte-neutral at their default:

* ``cv.group_by_airport`` — the normal-row regressor is split into one model per airport (or
  per named airport plus a shared ``rest`` model). The corruption head stays global and the
  target encodings are untouched; a unit's model is fitted on, and predicts, that unit's rows
  only (:func:`prc.eval.fitpool.unit_masks`).
* ``test_fit: full`` — after CV, one model per (seed, regime group, airport unit) is fitted on
  ALL training rows for a fixed number of rounds and writes ``test_pred``; the OOF and every
  metric stay the fold models' (:func:`_run_full_fit`, :func:`_full_fit_rounds`).

The head section runs **before** the regressor loop, because ``normal_weight`` needs the
fold's own probabilities to weight that fold's training rows. Both loops are deterministically
seeded and share no RNG state, so the order does not move any number.
"""

from __future__ import annotations

import csv
import dataclasses
import gc
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import polars as pl
import yaml

from prc import cache as cache_mod
from prc import paths
from prc.config import CorruptionConfig, ExperimentConfig, ModelConfig
from prc.cv.folds import get_folds, n_folds, train_mask
from prc.data import schema as S
from prc.data.load import load_movements
from prc.eval import events
from prc.eval import stack as stack_mod
from prc.eval.fitpool import (
    NN_KINDS,
    FitTask,
    FoldContext,
    SpawnFitPool,
    fit_full,
    fit_one,
    inner_holdout_mask,
    resolve_nn,
    run_fold_fits,
    set_prep_threads,
    unit_masks,
)
from prc.eval.lanes import lane_metrics, lane_rmse
from prc.eval.metrics import bootstrap_rmse_std, rmse, rmse_breakdown
from prc.eval.season import combine as season_combine
from prc.eval.season import season_weights
from prc.eval.timing import Timings
from prc.eval.transform import inverse, transform
from prc.features import build
from prc.features import latent_stage1 as stage1_mod
from prc.features.target_enc import DEFAULT_GROUPS, encode, encode_train
from prc.mixture.clf_features import (
    identity,
    sched_default_encodings,
    sched_default_stats,
    sched_eq_events,
    sched_eq_prev24h,
    signals,
)
from prc.mixture.corruption import (
    LABEL_PREFIX,
    PROB_PREFIX,
    SCHED_VALUE_COL,
    apply_mixture,
    clip_probs,
    corrupt_values,
    corruption_labels,
    sched_default_value,
    z_targets,
)
from prc.mixture.head import (
    ZTargets,
    auc_score,
    calibration_table,
    predict_heads,
    train_heads,
    weighted_z_loss,
)
from prc.mixture.regimes import PROXY_COL, REGIME_COL, assign_regimes
from prc.models import make_model
from prc.models.matrices import MatrixCache
from prc.runtime import Runtime
from prc.runtime import resolve as resolve_runtime

log = logging.getLogger("prc.run")


#: The target transform lives in :mod:`prc.eval.transform` (2026-09-19) so that this module
#: and :mod:`prc.eval.fitpool` cannot drift apart, and so that the negative-label guard the
#: code review asked for (F3) exists in exactly one place. ``sqrt`` is bit-identical to the
#: implementation this replaced; ``log1p`` was previously unrunnable on real labels.
_transform = transform
_inverse = inverse


def _to_pandas(X: pl.DataFrame) -> pd.DataFrame:
    pdf = X.to_pandas()
    for c in pdf.columns:
        d = pdf[c].dtype
        if not (
            pd.api.types.is_numeric_dtype(d) or pd.api.types.is_bool_dtype(d)
        ) and not isinstance(d, pd.CategoricalDtype):
            pdf[c] = pdf[c].astype("category")
    return pdf


def _git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _regime_arrays(frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(regime letters, `MVT − AOBT_3` in seconds with NaN for no NM match) for a set of rows."""
    r = assign_regimes(frame)
    return r.get_column(REGIME_COL).to_numpy(), r.get_column(PROXY_COL).to_numpy().astype(float)


def _proxy_offset(regime: np.ndarray, proxy: np.ndarray, mode: str) -> np.ndarray:
    """The per-row offset the model learns against: the NM proxy on regime-A rows, 0 elsewhere."""
    if mode == "raw":
        return np.zeros(len(regime))
    return np.where((regime == "A") & np.isfinite(proxy), np.nan_to_num(proxy), 0.0)


def _rows(X: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    """Positional boolean row selection, re-indexed so later column assignment lines up."""
    return cast(pd.DataFrame, X[mask]).reset_index(drop=True)


def _align(ids: pl.DataFrame, table: pl.DataFrame) -> pl.DataFrame:
    """``table`` re-ordered onto the rows of ``ids`` (one column: the movement id)."""
    return ids.join(table, on=S.MVT_ID, how="left", maintain_order="left").drop(S.MVT_ID)


def build_clf_extras(corr_cfg: CorruptionConfig | None, mv: pl.DataFrame) -> pl.DataFrame | None:
    """The head-only table keyed by ``MVT_ID_mvt``: ``clfx_*`` (the blocks named in
    ``extra_blocks``, built from the raw movement table and therefore **unmasked** — that is
    the whole point, ``mask_for_no_nm`` nulls those same columns for the regressor) and
    ``clf_*`` (the engineered signals of ``prc.mixture.clf_features``, and with
    ``identity_signals`` the ``FLIGHT_mvt`` parse / stand re-occupation / domestic-destination
    columns). ``None`` when none of them is configured, which is the default and the
    pre-2026-09-18 behaviour.
    """
    if corr_cfg is None or not (
        corr_cfg.extra_blocks or corr_cfg.extra_signals or corr_cfg.identity_signals
    ):
        return None
    parts: list[pl.DataFrame] = []
    if corr_cfg.extra_blocks:
        blocks = build(mv, list(corr_cfg.extra_blocks))
        parts.append(blocks.rename({c: "clfx_" + c for c in blocks.columns if c != S.MVT_ID}))
    if corr_cfg.extra_signals:
        parts.append(signals(mv))
    if corr_cfg.identity_signals:
        parts.append(identity(mv))
    out = parts[0]
    for q in parts[1:]:
        out = out.join(q, on=S.MVT_ID, how="left")
    return out


def _clf_extra(frame: pl.DataFrame, regime: np.ndarray, extra: pl.DataFrame | None) -> pd.DataFrame:
    """The head's own columns for a set of rows: regime, airport and hour — which the head
    needs even when the experiment's feature list does not carry them — followed by any
    configured extra blocks / engineered signals.

    Kept apart from the regressor's matrix on purpose: the head is fitted on one fold's rows at
    a time, so materialising a second full-height copy of the matrix (what this module did until
    2026-09-18) bought nothing and cost a gigabyte.
    """
    out = pd.DataFrame(
        {
            "clf_regime": pd.Categorical(regime, categories=["A", "B", "C"]),
            "clf_airport": pd.Categorical(frame.get_column(S.AIRPORT).to_list()),
            "clf_hour": frame.get_column(S.MVT_TIME).dt.hour().to_numpy(),
        }
    )
    if extra is not None and extra.width:
        out = pd.concat([out, _to_pandas(extra)], axis=1)
    return out


def _clf_frame(
    X: pd.DataFrame,
    cols: list[str],
    extras: list[pd.DataFrame | None],
    mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """The classifier's matrix for ``mask`` rows: the selected regressor columns plus the head's
    own columns. Only the selected rows are ever materialised."""
    base = X if cols == list(X.columns) else X.loc[:, cols]
    if mask is not None:
        base = cast(pd.DataFrame, base[mask])
    parts = [base.reset_index(drop=True)]
    for e in extras:
        if e is None or not len(e.columns):
            continue
        parts.append(
            (cast(pd.DataFrame, e[mask]) if mask is not None else e).reset_index(drop=True)
        )
    return pd.concat(parts, axis=1) if len(parts) > 1 else parts[0]


def _sched_eq_counts(rows: pl.DataFrame, event_rows: pl.DataFrame, tol_s: float) -> pd.DataFrame:
    """The fold-safe ``BLOCK == SCHED`` count in the previous 24 h, aligned to ``rows``.

    ``event_rows`` are the *only* rows whose off-block stamp is read — a fold's training rows
    when building that fold's head, all training rows when building the test-time head — and
    the window is half-open, so no row ever counts itself.
    """
    events = sched_eq_events(event_rows, tol_s)
    counts = sched_eq_prev24h(rows.select(S.MVT_ID, S.AIRPORT, S.MVT_TIME), events)
    return _align(rows.select(S.MVT_ID), counts).to_pandas()


def _sched_default_te(
    rows: pl.DataFrame, fit_rows: pl.DataFrame, tol_s: float, loo: bool = False
) -> pd.DataFrame:
    """The (airport, airline) and (airport, destination) schedule-default encodings, aligned
    to ``rows``.

    Same fold-scoped contract as :func:`_sched_eq_counts`: ``fit_rows`` are the *only* rows
    whose off-block stamp is read — a fold's training departures when building that fold's
    head, all training departures when building the test-time head — so a validation row's own
    label never enters the encoding it is scored with. The statistics are measured on all of
    those departures, not only the no-flight-plan ones (EDA §3.2).

    ``loo`` (``mixture.corruption.sched_default_te_loo``, default off) additionally removes
    each ``fit_rows`` row's own label from its own encoding, which is the other half of the
    same contract: without it the head is *fitted* on encodings that carry a little of the
    answer and *scored* on encodings that carry none (review F4).
    """
    stats = sched_default_stats(fit_rows, tol_s)
    if loo:
        marked = rows.select(
            S.MVT_ID, S.AIRPORT, S.FLIGHT_NO, S.ADES, S.BLOCK_TIME, S.SCHED_TIME
        ).join(
            fit_rows.select(S.MVT_ID).with_columns(pl.lit(True).alias("_in_fit")),
            on=S.MVT_ID,
            how="left",
            maintain_order="left",
        )
        marked = marked.with_columns(pl.col("_in_fit").fill_null(False))
        enc = sched_default_encodings(marked, stats, loo_tol_s=tol_s)
    else:
        enc = sched_default_encodings(rows.select(S.MVT_ID, S.AIRPORT, S.FLIGHT_NO, S.ADES), stats)
    return _align(rows.select(S.MVT_ID), enc).to_pandas()


def _in_fold_weights(
    corr_cfg: CorruptionConfig,
    model_cfg: ModelConfig,
    X_in: pd.DataFrame,
    labels_in: dict[str, np.ndarray],
    seed: int,
) -> np.ndarray:
    """``1 − Σ_k p_k`` for a fold's training rows, from heads fitted on *those rows only*.

    A nested split of the fold's training set gives every row a probability from a head that
    never saw its label, and — unlike reusing the CV out-of-fold probabilities — never saw a
    label from this fold's validation rows either. That is what keeps the weights fold-safe.
    """
    n = len(X_in)
    split = np.random.default_rng(seed + 9_973).integers(0, corr_cfg.weight_inner_splits, size=n)
    p_sum = np.zeros(n)
    for s in range(corr_cfg.weight_inner_splits):
        pred = split == s
        fit = ~pred
        if not pred.any() or not fit.any():
            continue
        heads = train_heads(
            corr_cfg, model_cfg, _rows(X_in, fit), {t: v[fit] for t, v in labels_in.items()}, seed
        )
        probs = clip_probs(predict_heads(heads, _rows(X_in, pred)), corr_cfg.p_max)
        for v in probs.values():
            p_sum[pred] += v
        del heads, probs
        gc.collect()
    return np.clip(1.0 - p_sum, corr_cfg.weight_floor, 1.0)


def _nested_yhat(
    corr_cfg: CorruptionConfig,
    model_cfg: ModelConfig,
    transform: str,
    X: pd.DataFrame,
    rows: np.ndarray,
    y_res: np.ndarray,
    off: np.ndarray,
    w: np.ndarray | None,
    seed: int,
    z_splits: int | None = None,
) -> np.ndarray:
    """``ŷ_normal`` for a set of training rows (positional indices ``rows``), from regressors
    fitted on the *other* rows of that same set.

    The second half of the nested mechanism :func:`_in_fold_weights` already uses, and the
    reason ``sched_head: z_regression`` is fold-safe. ``z = (y - ŷ)/(C - ŷ)`` reads the row's
    own target, so it belongs inside the fold exactly where the 0/1 label does — but it also
    reads ``ŷ``, and a regressor's prediction *on its own training rows* is far too good
    (``y - ŷ`` collapses, ``z`` collapses with it, and the head would learn ``p ≈ 0``). The
    same nested split as the expert weights fixes both: every row's ``ŷ`` comes from a model
    that never saw that row, and every model is fitted on the fold's training rows only, so
    no validation label reaches the head's target either.

    The nested expert is fitted the way the real one is (``w`` = the fold's ``1 - Σ p`` or
    clean-row weights, ``None`` on the default path) and early-stops on a random
    ``holdout_frac`` slice of its own fit rows — never on the rows it is about to predict.

    Cost: ``splits`` regressor fits per fold on ``1 - 1/splits`` of that fold's training
    rows, plus the same again for the test-time head. At the default 2 splits and the primary
    2-fold scheme that is three full-size fits, single-seeded, against the six the run already
    does — about +50 % of ``exp0047``'s regressor time.

    **How many splits (review F5).** ``z = (y - ŷ)/(C - ŷ)`` is built from these half-size
    models but is then applied to the prediction of the *full-size* model, so at 2 splits the
    head is calibrated against a ŷ that is systematically worse than the one it will meet.
    ``mixture.corruption.z_inner_splits`` raises the count for the z head only: at 10 splits
    each nested fit sees 90 % of the fold and the target approaches the full-size ŷ, at 10x
    the cost. Default ``None`` reuses ``weight_inner_splits`` — the pre-2026-09-19 behaviour.
    """
    n = len(rows)
    splits = corr_cfg.weight_inner_splits if z_splits is None else z_splits
    split = np.random.default_rng(seed + 9_973).integers(0, splits, size=n)
    rng = np.random.default_rng(seed + 7_919)
    out = np.full(n, np.nan)
    for s in range(splits):
        pred = np.flatnonzero(split == s)
        fit = np.flatnonzero(split != s)
        if len(pred) == 0 or len(fit) < 2:
            continue
        hold = rng.random(len(fit)) < corr_cfg.holdout_frac
        if not hold.any() or hold.all():  # degenerate slice: keep one row back regardless
            hold = np.zeros(len(fit), dtype=bool)
            hold[0] = True
        core, es = rows[fit[~hold]], rows[fit[hold]]
        m = make_model(model_cfg, seed)
        m.fit(
            X.take(core).reset_index(drop=True),
            _transform(y_res[core], transform),
            X.take(es).reset_index(drop=True),
            _transform(y_res[es], transform),
            None if w is None else w[core],
        )
        gc.collect()
        idx = rows[pred]
        out[pred] = _inverse(m.predict(X.take(idx).reset_index(drop=True)), transform) + off[idx]
        del m
        gc.collect()
    return out


def _expert_weights(
    corr_cfg: CorruptionConfig, clean_only: bool, clean_rows: np.ndarray
) -> np.ndarray | None:
    """The sample weights the normal expert is fitted under, for the two configurations that
    do not need a fitted head: the hard ``train_normal_on: clean`` switch and
    ``normal_weight: clean``. ``None`` on the default path (every row weight 1)."""
    if clean_only or corr_cfg.normal_weight == "clean":
        return clean_rows.astype(np.float64)
    return None


def _rollover_weights(
    corr_cfg: CorruptionConfig, w_all: np.ndarray, rollover_rows: np.ndarray | None
) -> np.ndarray:
    """``rollover_weight: zero``: the normal expert's weights with every rollover-labelled row
    set to 0 (a copy; every other row keeps its soft weight). ``soft`` returns ``w_all``
    itself, untouched, so the default path is the pre-2026-09-22 one object for object.

    ``rollover_rows`` is the head's own training label (``corruption_labels``), which reads
    the target — it is only ever applied to a fit's *training* rows (``w_all[fit_rows]``).
    """
    if corr_cfg.rollover_weight == "soft":
        return w_all
    assert rollover_rows is not None  # the config validator requires `rollover` in types
    out = np.array(w_all, dtype=np.float64, copy=True)
    out[rollover_rows] = 0.0
    return out


def _season_fit_weights(
    cfg: ExperimentConfig,
    w_fit: np.ndarray | None,
    fit_months: np.ndarray,
    target_months: np.ndarray,
    tag: str,
) -> np.ndarray:
    """``season_weight``: the fit's row weights times the seasonal similarity of each fitted
    row's month to the fit's target month(s) (:mod:`prc.eval.season`). Only called when the
    option is set, so the default path never builds a weight array it did not build before."""
    sw = cfg.season_weight
    assert sw is not None
    targets = sorted({str(m) for m in target_months})
    factor = season_weights(fit_months, targets, sw.tau_months, sw.floor)
    out = season_combine(w_fit, factor)
    log.info(
        "season_weight %s: target %s, factor mean %.3f min %.3f (n=%d)",
        tag,
        targets,
        float(factor.mean()) if len(factor) else float("nan"),
        float(factor.min()) if len(factor) else float("nan"),
        len(factor),
    )
    return out


def _fold_z(
    corr_cfg: CorruptionConfig,
    cfg: ExperimentConfig,
    X: pd.DataFrame,
    rows: np.ndarray,
    y: np.ndarray,
    y_res: np.ndarray,
    off: np.ndarray,
    sched: np.ndarray,
    w: np.ndarray | None,
    seed: int,
    tag: str,
) -> ZTargets | None:
    """The z head's target for one set of training rows, or ``None`` when the schedule-default
    head is the classifier (the default path, where nothing extra is fitted at all)."""
    if corr_cfg.sched_head != "z_regression":
        return None
    t0 = time.time()
    y_hat = _nested_yhat(
        corr_cfg,
        cfg.model,
        cfg.target.transform,
        X,
        rows,
        y_res,
        off,
        w,
        seed,
        corr_cfg.z_inner_splits,
    )
    z, zw = z_targets(y[rows], y_hat, sched[rows], corr_cfg)
    log.info(
        "%s z head target: %d/%d rows carry weight, weighted mean z=%.4f, nested yhat in %.0fs",
        tag,
        int((zw > 0).sum()),
        len(rows),
        float((zw * z).sum() / zw.sum()) if zw.sum() > 0 else float("nan"),
        time.time() - t0,
    )
    return ZTargets(z, zw)


def _z_metrics(
    corr_cfg: CorruptionConfig,
    y: np.ndarray,
    y_hat: np.ndarray,
    sched: np.ndarray,
    p: np.ndarray,
    labels: dict[str, np.ndarray],
    airports: np.ndarray,
    lanes: dict[str, np.ndarray],
) -> dict:
    """The value-weighted gate of internal note 2026-09-18_head-paradox §5.1, on OOF rows.

    ``weighted_z_loss`` = ``Σ (C - ŷ)²·(p - z)² / Σ (C - ŷ)²`` — the mixture's own squared
    error per unit of value, which is what AUC fails to see. Reported overall, per airport and
    for the no-flight-plan lanes; ``z`` is the *raw* ratio here (the head trains on the
    winsorised one), so the numerator is exactly the schedule-default branch's contribution to
    the run's SSE on the rows that carry weight. Computed for every head run, classifier or
    not, so the two are comparable.
    """
    z, w = z_targets(y, y_hat, sched, corr_cfg, winsorise=False)
    scored = lanes["scored"]
    grp = np.where(y > 3600.0, "clean_gt_1h", "clean_le_1h").astype(object)
    if "sched_default" in labels:
        grp = np.where(labels["sched_default"] == 1, "sched_default", grp)
    if "rollover" in labels:  # a rollover first, as in the EDA's partition
        grp = np.where(labels["rollover"] == 1, "rollover", grp)

    def cell(m: np.ndarray) -> dict:
        wm = m & (w > 0)
        return {
            "n": int(m.sum()),
            "n_weighted": int(wm.sum()),
            "z_mean": float(z[wm].mean()) if wm.any() else None,
            "z_mean_weighted": float((w[wm] * z[wm]).sum() / w[wm].sum()) if wm.any() else None,
            "p_mean": float(p[m].mean()) if m.any() else None,
        }

    def loss(m: np.ndarray) -> float | None:
        return weighted_z_loss(p[m], z[m], w[m])

    out: dict = {
        "weighted_z_loss": loss(scored),
        "weighted_z_loss_per_airport": {
            str(a): loss(scored & (airports == a))
            for a in sorted({str(v) for v in airports[scored]})
        },
        "n_z_weighted": int((w[scored] > 0).sum()),
        "z_mean_by_group": {
            g: cell(scored & (grp == g))
            for g in ("sched_default", "rollover", "clean_le_1h", "clean_gt_1h")
        },
    }
    for name, m in lanes.items():
        if name != "scored":
            out[f"weighted_z_loss_{name}"] = loss(m)
    return out


@dataclass
class Assembled:
    """What the setup phase produces and everything after it consumes.

    Deliberately closed over: once these exist, the movement table and the feature frames
    are never needed again — which is what lets ``runtime.cache_features`` skip loading a
    244 MB parquet and joining twenty blocks onto it. ``clf_extra_*`` are the head-only
    tables (:func:`build_clf_extras`) for the same reason: they are the last consumer of
    the raw movement table, so they are cached beside the frames rather than left to force
    a load.
    """

    train: pl.DataFrame
    test: pl.DataFrame
    feat_cols: list[str]
    clf_extra_tr: pl.DataFrame | None = None
    clf_extra_te: pl.DataFrame | None = None
    n_masked_cols: int = 0
    n_no_nm: int = 0
    #: the frames cache key, when `runtime.cache_features` is on and every input could be
    #: hashed. The target-encoding and head caches hang off it, so the rows they were
    #: computed from are part of their keys by construction.
    key: str | None = None


def _apply_drop_columns(
    cfg: ExperimentConfig,
    train: pl.DataFrame,
    test: pl.DataFrame,
    feat_cols: list[str],
) -> tuple[pl.DataFrame, pl.DataFrame, list[str]] | None:
    """Remove ``cfg.drop_columns`` from the assembled frames and the feature list.

    Returns ``None`` when the list is empty, so the default path is not merely equivalent to
    the pre-2026-09-20 one — it is the same objects. Raises when a name is not an assembled
    feature column: a silent no-op there would make a typo look like a result.
    """
    if not cfg.drop_columns:
        return None
    unknown = [c for c in cfg.drop_columns if c not in feat_cols]
    if unknown:
        raise ValueError(f"drop_columns names columns that are not features: {unknown}")
    drop = list(dict.fromkeys(cfg.drop_columns))
    log.info("drop_columns: removing %d columns %s", len(drop), drop)
    return train.drop(drop), test.drop(drop), [c for c in feat_cols if c not in set(drop)]


#: One entry per model the regressor is split into by `cv.group_by_airport`: a label for the
#: run's tags, and the airports it owns (None = every airport, the global model).
AirportUnit = tuple[str, tuple[str, ...] | None]


def _airport_units(
    cfg: ExperimentConfig, airports_tr: np.ndarray, airports_te: np.ndarray
) -> list[AirportUnit]:
    """The airport split of the normal-row regressor (``cv.group_by_airport``).

    ``false`` -> ``[("all", None)]``, which is the pre-2026-09-20 single global model and the
    only case that produces no tag suffix anywhere. ``true`` -> one unit per airport present
    in the data. A list -> one unit per named airport plus a shared ``rest`` unit for every
    other airport (omitted when the named airports are all of them).

    The airport set is taken from train **and** test, so an airport that appears only in the
    test set still lands in a unit rather than silently losing its rows.
    """
    sel = cfg.cv.group_by_airport
    if not sel:
        return [("all", None)]
    known = sorted({str(a) for a in airports_tr.tolist()} | {str(a) for a in airports_te.tolist()})
    if sel is True:
        return [(a, (a,)) for a in known]
    unknown = [a for a in sel if a not in known]
    if unknown:
        raise ValueError(f"cv.group_by_airport names airports that are not in the data: {unknown}")
    units: list[AirportUnit] = [(a, (a,)) for a in sel]
    rest = tuple(a for a in known if a not in set(sel))
    if rest:
        units.append(("rest", rest))
    return units


def _fit_tag(k: int | str, seed: int, gi: int, ulabel: str, n_groups: int, n_units: int) -> str:
    """The key of one fit in ``metrics.json['best_iterations']``.

    Suffixes are added only when there is something to disambiguate, so a plain single-model
    run keeps the ``fold0_seed42`` keys every registered run was written with.
    """
    tag = f"fold{k}_seed{seed}"
    if n_groups > 1:
        tag += f"_g{gi}"
    if n_units > 1:
        tag += f"_{ulabel}"
    return tag


def _full_fit_rounds(
    cfg: ExperimentConfig, iters: list[int], fold_rows: list[int], n_all: int
) -> tuple[int, dict]:
    """How many rounds one ``test_fit: full`` model gets, and the numbers behind it.

    **The rule.** A fold model early-stops after ``best_iteration + 1`` trees on
    ``n_fold_train`` rows. The full model sees more rows and has no validation set to stop
    it, so it is given the fold models' own stopping point scaled by how much more data it
    has::

        rounds = round( median_k(best_iteration_k + 1) * n_all / mean_k(n_fold_train_k) )

    the median over the folds of *this* (seed, regime group, airport unit), and the row
    counts of exactly those fits — so a per-airport unit is scaled by its own airport's
    growth, not the run's. ``+1`` because ``best_iteration`` is a 0-based tree index and the
    fold model predicts with ``iteration_range=(0, best_iteration + 1)``.

    It is deliberately crude: with a fixed learning rate, the optimal number of trees grows
    roughly with the number of rows at a fixed leaf size, and no honest early-stopping signal
    exists for a model fitted on everything. The alternative — a held-out slice — would make
    the full model a smaller model, which is the thing this option exists to avoid.

    When no fold of this unit reported a ``best_iteration`` (a constant baseline, or a unit
    that never had training rows) the model falls back to ``model.num_boost_round``.
    """
    info: dict = {"n_all": int(n_all), "n_fold_mean": None, "median_fold_rounds": None}
    if not iters or not fold_rows or sum(fold_rows) == 0:
        info["rounds"] = int(cfg.model.num_boost_round)
        info["rule"] = "fallback: model.num_boost_round (no fold best_iteration)"
        return info["rounds"], info
    median_rounds = float(np.median([i + 1 for i in iters]))
    n_fold_mean = float(np.mean(fold_rows))
    rounds = max(1, round(median_rounds * n_all / n_fold_mean))
    info |= {
        "n_fold_mean": round(n_fold_mean, 1),
        "median_fold_rounds": median_rounds,
        "scale": round(n_all / n_fold_mean, 4),
        "rounds": rounds,
        "rule": "round(median(best_iteration+1) * n_all / mean(n_fold_train))",
    }
    return rounds, info


def _assemble(cfg: ExperimentConfig, T: Timings) -> Assembled:
    """Movement table -> feature blocks -> mask -> folds -> train/test frames.

    Extracted verbatim from ``run_experiment`` on 2026-09-19 so that (a) each step can be
    timed on its own and (b) the whole thing has a cache key
    (:func:`prc.cache.frames_inputs` hashes this function's source, so editing it
    invalidates every cached frame).
    """
    corr_cfg = cfg.corruption()
    with T.phase("load_movements"):
        mv = load_movements()
    with T.phase("build_features"):
        feats = build(mv, cfg.features)
    te_groups = {n: DEFAULT_GROUPS[n] for n in cfg.target_encoding}
    te_keys = sorted({c for cols in te_groups.values() for c in cols} - {S.AIRPORT, S.SPLIT_MONTH})
    mix_keys: list[str] = []
    if cfg.uses_regimes():
        mix_keys = [S.MVT_TIME, S.SCHED_TIME, S.AOBT_3] + ([S.BLOCK_TIME] if corr_cfg else [])
    if corr_cfg and corr_cfg.identity_signals:
        # the two keys the head's fold-scoped label encodings group by; they are not features
        mix_keys = mix_keys + [S.FLIGHT_NO, S.ADES]
    # MVT_TIME / AOBT_3 are always carried: they are the two columns the lane breakdown
    # (`prc.eval.lanes`) is defined by, and keeping them costs two datetime columns instead
    # of re-reading the movement table after the fit loops.
    lane_keys = [S.MVT_TIME, S.AOBT_3]
    keep = list(
        dict.fromkeys(
            [S.MVT_ID, S.AIRPORT, S.SPLIT_MONTH, S.IS_TEST, S.TARGET, *te_keys, *mix_keys]
            + lane_keys
            + ([S.AOBT_3] if cfg.mask_for_no_nm else [])
        )
    )
    with T.phase("join_features"):
        deps = mv.filter(pl.col(S.PHASE) == "DEP").select(keep)
        # `maintain_order="left"` (2026-09-19, review F9): row POSITION is load-bearing
        # downstream — `_in_fold_weights` and `_stratified_holdout` index by it, and
        # `sample_frac` draws by it — but polars does not document left-join order as a
        # guarantee. It happens to be stable in 1.44.1 (verified over 3 repeats, 2,085,047
        # rows), so asking for it explicitly costs nothing today and stops a polars upgrade
        # from silently moving the `exp0047 = 404.63` anchor with no code change.
        data = deps.join(feats, on=S.MVT_ID, how="left", maintain_order="left")
    n_masked_cols = 0
    n_no_nm = 0
    if cfg.mask_for_no_nm:
        with T.phase("mask"):
            unknown = [b for b in cfg.mask_for_no_nm if b not in cfg.features]
            if unknown:
                raise ValueError(f"mask_for_no_nm names blocks not in features: {unknown}")
            masked_cols = [c for c in build(mv, cfg.mask_for_no_nm).columns if c != S.MVT_ID]
            no_nm = pl.col(S.AOBT_3).is_null()
            data = data.with_columns(
                [pl.when(no_nm).then(None).otherwise(pl.col(c)).alias(c) for c in masked_cols]
            )
            n_masked_cols = len(masked_cols)
            n_no_nm = int(data.select(no_nm.sum()).item())
        log.info(
            "masked %d columns from %s on %d no-NM departures",
            n_masked_cols,
            cfg.mask_for_no_nm,
            n_no_nm,
        )
    with T.phase("folds"):
        folds = get_folds(mv, cfg.cv.scheme)
        data = data.join(folds, on=S.MVT_ID, how="left", maintain_order="left").with_columns(
            pl.col("fold").fill_null(-1)
        )

        train = data.filter(~pl.col(S.IS_TEST)).drop_nulls(subset=[S.TARGET])
        test = data.filter(pl.col(S.IS_TEST))
        if cfg.sample_frac:
            if cfg.sample_frac_scope == "train_only":
                # F6: thin only the rows no fold ever validates, so the sub-sampled run
                # scores the SAME population as the gate and its RMSE can be compared to
                # one. Sampling before the fold split (the "all" branch) thins the
                # validation rows too, which is why the sub-sample "learning curve" read
                # backwards. Re-sorted by movement id so the row order is a deterministic
                # function of the sample, not of the concat.
                scored = pl.col("fold") >= 0
                kept, thinned = train.filter(scored), train.filter(~scored)
                train = pl.concat([kept, thinned.sample(fraction=cfg.sample_frac, seed=0)]).sort(
                    S.MVT_ID
                )
                log.info(
                    "sample_frac=%.4g scope=train_only: kept all %d validated rows, "
                    "thinned %d -> %d never-validated rows",
                    cfg.sample_frac,
                    thinned.height,
                    thinned.height,
                    train.height - kept.height,
                )
            else:
                train = train.sample(fraction=cfg.sample_frac, seed=0)

    feat_cols = [c for c in feats.columns if c != S.MVT_ID]

    # Head-only columns, built while the movement table is still in memory. `extra_blocks` are
    # joined UNMASKED under a `clfx_` prefix: the point is to give the classifier exactly the
    # surface columns `mask_for_no_nm` takes away from the regressor.
    clf_extra_tr: pl.DataFrame | None = None
    clf_extra_te: pl.DataFrame | None = None
    with T.phase("clf_extras"):
        clf_extra = build_clf_extras(corr_cfg, mv)
        if clf_extra is not None:
            clf_extra_tr = _align(train.select(S.MVT_ID), clf_extra)
            clf_extra_te = _align(test.select(S.MVT_ID), clf_extra)
            del clf_extra
    del mv, feats, deps, data, folds
    gc.collect()
    return Assembled(
        train=train,
        test=test,
        feat_cols=feat_cols,
        clf_extra_tr=clf_extra_tr,
        clf_extra_te=clf_extra_te,
        n_masked_cols=n_masked_cols,
        n_no_nm=n_no_nm,
    )


_FRAME_FILES = ("train.parquet", "test.parquet")


def assemble_frames(cfg: ExperimentConfig, T: Timings, rt: Runtime) -> Assembled:
    """:func:`_assemble`, optionally served from ``data/cache/frames/<key>/``.

    On a hit nothing above is executed at all: no movement table is read, no block is
    joined, no mask is applied. On a miss the work is done under an exclusive flock so that
    concurrent array tasks on a shared filesystem compute it once, then written temp-file +
    atomic-rename with a ``READY`` marker last.

    The cache fails open in every direction: a key that cannot see one of its inputs
    (:func:`prc.cache.block_inputs`), a lock that never comes free, or a read that raises
    all end in the same place — recompute, and say so in the log.
    """
    if not rt.cache_features:
        return _assemble(cfg, T)
    with T.phase("cache_key"):
        inputs = cache_mod.frames_inputs(cfg, _assemble)
    if inputs is None:
        log.info("frames cache: key incomplete, recomputing (see prc.cache)")
        return _assemble(cfg, T)
    entry = cache_mod.entry("frames", inputs)

    def _load() -> Assembled | None:
        if not entry.ready():
            return None
        try:
            with T.phase("cache_read"):
                meta = cache_mod.read_json(entry, "meta.json")
                out = Assembled(
                    train=cache_mod.read_frame(entry, "train.parquet"),
                    test=cache_mod.read_frame(entry, "test.parquet"),
                    feat_cols=list(meta["feat_cols"]),
                    clf_extra_tr=(
                        cache_mod.read_frame(entry, "clf_extra_tr.parquet")
                        if meta["clf_extra"]
                        else None
                    ),
                    clf_extra_te=(
                        cache_mod.read_frame(entry, "clf_extra_te.parquet")
                        if meta["clf_extra"]
                        else None
                    ),
                    n_masked_cols=int(meta["n_masked_cols"]),
                    n_no_nm=int(meta["n_no_nm"]),
                )
            log.info("frames cache HIT %s", entry.dir)
            return out
        except (OSError, KeyError, ValueError) as exc:
            log.warning("frames cache unreadable (%s); recomputing", exc)
            return None

    hit = _load()
    if hit is not None:
        hit.key = entry.key
        return hit
    with cache_mod.locked(entry) as got:
        if got:  # another task may have finished it while we waited for the lock
            hit = _load()
            if hit is not None:
                hit.key = entry.key
                return hit
        out = _assemble(cfg, T)
        out.key = entry.key
        if got:
            with T.phase("cache_write"):
                cache_mod.write_frame(entry, "train.parquet", out.train)
                cache_mod.write_frame(entry, "test.parquet", out.test)
                if out.clf_extra_tr is not None and out.clf_extra_te is not None:
                    cache_mod.write_frame(entry, "clf_extra_tr.parquet", out.clf_extra_tr)
                    cache_mod.write_frame(entry, "clf_extra_te.parquet", out.clf_extra_te)
                cache_mod.write_json(
                    entry,
                    "meta.json",
                    {
                        "feat_cols": out.feat_cols,
                        "clf_extra": out.clf_extra_tr is not None,
                        "n_masked_cols": out.n_masked_cols,
                        "n_no_nm": out.n_no_nm,
                    },
                )
                entry.mark_ready()
            log.info("frames cache WROTE %s", entry.dir)
    return out


def _stage1_threads(rt: Runtime) -> int:
    """Stage-1 boosters run on the CPU with a fixed thread count (determinism)."""
    return int(rt.n_threads or 4)


def _stage1(
    T: Timings, rt: Runtime, fit_on: pl.DataFrame, apply_to: pl.DataFrame | None, phase: str
) -> pd.DataFrame:
    """``cfg.stage1`` estimates (:mod:`prc.features.latent_stage1`) as a pandas block with a
    fresh RangeIndex: served-row values for ``apply_to`` fitted on all of ``fit_on``, or, with
    ``apply_to=None``, inner leave-one-month-out values for ``fit_on``'s own rows."""
    t = time.time()
    fit_on = fit_on.select(stage1_mod.needed_columns(fit_on))
    with T.phase(phase):
        if apply_to is None:
            out = stage1_mod.inner_month(fit_on, _stage1_threads(rt))
        else:
            out = stage1_mod.fit_apply(fit_on, apply_to, _stage1_threads(rt))
    log.info("stage1 %s: %d rows in %.1fs", phase, out.height, time.time() - t)
    return out.drop(S.MVT_ID).to_pandas()


def _te_cached(
    cfg: ExperimentConfig,
    T: Timings,
    rt: Runtime,
    frames_key: str | None,
    fold: int | str,
    role: str,
    compute,
) -> pd.DataFrame:
    """One fold-local target-encoding table, from ``data/cache/te/`` when it is there.

    Separate entries per (fold, role) rather than one entry for the whole scheme: a
    ``month_loo`` cell would otherwise have to materialise twelve folds' encodings before it
    could write anything, and a resumed cell could use none of them.
    """
    # `prc.cache.te_inputs` does not (and must not, from here) know about `te_clean_labels`,
    # so a clean-label run would collide with a plain one's entry. It fails open instead.
    if not rt.cache_features or frames_key is None or cfg.te_clean_labels:
        with T.phase("te_fold" if isinstance(fold, int) else "te_test"):
            return compute()
    key = cache_mod.te_inputs(frames_key, cfg, fold, role)
    if role == "tr" and cfg.te_train_mode != "self":
        # T1: only the training rows' table depends on the mode (validation and test rows are
        # encoded identically in every mode, so their entries are shared). Added only off the
        # default, so every existing `self` key is unchanged.
        key["train_mode"] = cfg.te_train_mode
    entry = cache_mod.entry("te", key)
    if entry.ready():
        try:
            with T.phase("cache_read"):
                return cache_mod.read_frame(entry, "te.parquet").to_pandas()
        except (OSError, ValueError) as exc:
            log.warning("te cache unreadable (%s); recomputing", exc)
    with T.phase("te_fold" if isinstance(fold, int) else "te_test"):
        out = compute()
    with cache_mod.locked(entry, timeout_s=600.0) as got:
        if got and not entry.ready():
            with T.phase("cache_write"):
                cache_mod.write_frame(entry, "te.parquet", pl.from_pandas(out))
                entry.mark_ready()
    return out


@dataclass
class HeadOutputs:
    """Everything the corruption-head section produces, in a form that survives a round trip
    through ``data/cache/head/``: arrays and scalars, no fitted objects.

    The fitted heads themselves are deliberately *not* cached. They are only ever used to
    predict the rows this run already predicted (the fold's validation rows and the test
    set), so their outputs are the whole of their value, and a pickled booster is a far more
    fragile cache entry than a float array.
    """

    probs_oof: dict[str, np.ndarray] = field(default_factory=dict)
    probs_test: dict[str, np.ndarray] = field(default_factory=dict)
    weights_by_fold: dict[int, np.ndarray] = field(default_factory=dict)
    #: The normal expert's weights over **all** training rows, for ``test_fit: full`` (the
    #: full model's analogue of ``weights_by_fold``). Computed only when the full fit needs
    #: it — or for free under ``sched_head: z_regression``, which already builds it.
    weights_all: np.ndarray | None = None
    holdout_auc: dict[str, float | None] = field(default_factory=dict)
    holdout_wz: dict[str, float | None] = field(default_factory=dict)
    calibrated: dict[str, list[str]] = field(default_factory=dict)
    n_clf_features: int = 0


def _run_head(
    cfg: ExperimentConfig,
    corr_cfg: CorruptionConfig,
    T: Timings,
    out_dir: Path,
    train: pl.DataFrame,
    test: pl.DataFrame,
    X_all: pd.DataFrame,
    X_test: pd.DataFrame,
    regime_tr: np.ndarray,
    regime_te: np.ndarray,
    clf_extra_tr: pl.DataFrame | None,
    clf_extra_te: pl.DataFrame | None,
    labels: dict[str, np.ndarray],
    clean_rows: np.ndarray,
    clean_only: bool,
    fold_id: np.ndarray,
    months: pl.Series,
    y_all: np.ndarray,
    y_res: np.ndarray,
    off_tr: np.ndarray,
    sched_tr: np.ndarray,
) -> HeadOutputs:
    """The corruption-head section, lifted out of ``run_experiment`` unchanged (2026-09-19).

    Moved for two reasons: every phase inside it now gets its own timer — the 2026-09-18
    measurement could only say "≈60 s of setup", not that ~50 s of it was the nested in-fold
    weights — and it now has a return value that is pure data, which is what makes it
    cacheable across the capacity variants that share a head (:func:`head_outputs`).
    """
    out = HeadOutputs()
    # the head sees the regressor's block features (not the fold-local target encodings),
    # plus its own columns; `clf_columns` may narrow the regressor part.
    clf_cols = list(corr_cfg.clf_columns) or list(X_all.columns)
    missing = [c for c in clf_cols if c not in X_all.columns]
    if missing:
        raise ValueError(f"mixture.corruption.clf_columns names unknown columns: {missing}")
    extra_tr = _clf_extra(train, regime_tr, clf_extra_tr)
    extra_te = _clf_extra(test, regime_te, clf_extra_te)
    probs_oof = {t: np.full(len(train), np.nan) for t in corr_cfg.types}
    out.probs_oof = probs_oof
    # F7.1: the head's seed is its own knob. Default None = cfg.seeds[0], which is what an
    # seed-farm job array turned into one head PER CELL — `merge-seeds` then averaged three
    # different mixtures, so a merged run was a 3-head ensemble while a native multi-seed run
    # is a 1-head model. Pinning `mixture.corruption.head_seed` makes every cell of a farm fit
    # the identical head, which is the only way a merge equals the native run.
    head_seed = corr_cfg.head_seed if corr_cfg.head_seed is not None else cfg.seeds[0]
    for k in range(n_folds(cfg.cv.scheme)):
        va = fold_id == k
        tr = train_mask(months, cfg.cv.scheme, k).to_numpy() & ~va
        if va.sum() == 0:
            continue
        t_head = time.time()
        events.emit(out_dir, "head_start", fold=k, n_rows=int(tr.sum()))
        # the 24 h BLOCK == SCHED count for BOTH the fitted and the scored rows comes from
        # this fold's training rows only
        with T.phase("head_signals"):
            fold_rows = train.filter(pl.Series(tr))
            fold_cnt = (
                _sched_eq_counts(train, fold_rows, corr_cfg.sched_default_tol_s)
                if corr_cfg.extra_signals
                else None
            )
            # ... and the same for the label encodings: this fold's training rows only
            fold_te = (
                _sched_default_te(
                    train,
                    fold_rows,
                    corr_cfg.sched_default_tol_s,
                    corr_cfg.sched_default_te_loo,
                )
                if corr_cfg.identity_signals
                else None
            )
            X_in = _clf_frame(X_all, clf_cols, [extra_tr, fold_cnt, fold_te], tr)
        labels_in = {t: v[tr] for t, v in labels.items()}
        # The nested in-fold pass comes first: `normal_weight` weights this fold's
        # training rows with it, and the z head needs an out-of-sample `ŷ_normal` fitted
        # under exactly those weights. Both are deterministic functions of `head_seed`
        # and of this fold's training rows, so running them before the classifiers moves
        # no number (every RNG here is a local generator).
        w_fold = (
            _expert_weights(corr_cfg, clean_only, clean_rows)
            if corr_cfg.sched_head == "z_regression"
            else None
        )
        if corr_cfg.normal_weight == "one_minus_p":
            w_fold = np.ones(len(train))
            with T.phase("head_nested_weights"):
                w_fold[tr] = _in_fold_weights(corr_cfg, cfg.model, X_in, labels_in, head_seed)
            out.weights_by_fold[k] = w_fold
            log.info(
                "fold=%d normal-expert weights: mean=%.4f min=%.4f (n=%d)",
                k,
                float(w_fold[tr].mean()),
                float(w_fold[tr].min()),
                int(tr.sum()),
            )
        with T.phase("head_nested_yhat"):
            z_in = _fold_z(
                corr_cfg,
                cfg,
                X_all,
                np.flatnonzero(tr),
                y_all,
                y_res,
                off_tr,
                sched_tr,
                w_fold,
                head_seed,
                f"fold={k}",
            )
        with T.phase("head_classifiers"):
            heads = train_heads(corr_cfg, cfg.model, X_in, labels_in, head_seed, z_in)
        with T.phase("head_predict"):
            X_sc = _clf_frame(X_all, clf_cols, [extra_tr, fold_cnt, fold_te], va)
            for t, p in predict_heads(heads, X_sc).items():
                probs_oof[t][va] = p
        del heads, X_sc, z_in
        events.emit(
            out_dir,
            "head_done",
            fold=k,
            auc={t: auc_score(labels[t][va], probs_oof[t][va]) for t in corr_cfg.types},
            n_pos={t: int(labels[t][va].sum()) for t in corr_cfg.types},
            elapsed_s=round(time.time() - t_head, 1),
        )
        del X_in, labels_in, fold_cnt, fold_te, fold_rows, w_fold
        gc.collect()
    # the test-time head: fitted on all training rows, so its 24 h counts use all of them
    t_head = time.time()
    events.emit(out_dir, "head_start", fold="test", n_rows=int(train.height))
    with T.phase("head_signals"):
        all_cnt = (
            _sched_eq_counts(train, train, corr_cfg.sched_default_tol_s)
            if corr_cfg.extra_signals
            else None
        )
        all_te = (
            _sched_default_te(
                train, train, corr_cfg.sched_default_tol_s, corr_cfg.sched_default_te_loo
            )
            if corr_cfg.identity_signals
            else None
        )
        X_clf_all = _clf_frame(X_all, clf_cols, [extra_tr, all_cnt, all_te])
    out.n_clf_features = X_clf_all.shape[1]
    # the audit `exp0051` left open: exactly which columns the head is fitted on
    log.info("classifier matrix: %d columns (%s)", out.n_clf_features, list(X_clf_all.columns))
    w_all = None
    # `test_fit: full` needs the expert's weights over all training rows for the same reason
    # z_regression does: the model fitted on everything must carry the weights its fold
    # counterparts carried. It is the same nested in-fold pass, over all training rows.
    if corr_cfg.sched_head == "z_regression" or cfg.test_fit == "full":
        w_all = _expert_weights(corr_cfg, clean_only, clean_rows)
        if corr_cfg.normal_weight == "one_minus_p":
            with T.phase("head_nested_weights"):
                w_all = _in_fold_weights(corr_cfg, cfg.model, X_clf_all, labels, head_seed)
        out.weights_all = w_all
    with T.phase("head_nested_yhat"):
        z_all = _fold_z(
            corr_cfg,
            cfg,
            X_all,
            np.arange(len(train)),
            y_all,
            y_res,
            off_tr,
            sched_tr,
            w_all,
            head_seed,
            "test",
        )
    with T.phase("head_classifiers"):
        heads_all = train_heads(corr_cfg, cfg.model, X_clf_all, labels, head_seed, z_all)
    del X_clf_all, z_all, w_all
    gc.collect()
    with T.phase("head_predict"):
        test_cnt = (
            _sched_eq_counts(test, train, corr_cfg.sched_default_tol_s)
            if corr_cfg.extra_signals
            else None
        )
        # no `loo` here on purpose: no test row is in `train`, so leave-one-out would be a
        # no-op — the test rows are the clean, fully out-of-sample case by construction
        test_te = (
            _sched_default_te(test, train, corr_cfg.sched_default_tol_s)
            if corr_cfg.identity_signals
            else None
        )
        out.probs_test = predict_heads(
            heads_all, _clf_frame(X_test, clf_cols, [extra_te, test_cnt, test_te])
        )
    out.holdout_auc = {t: heads_all[t].holdout_auc for t in corr_cfg.types}
    out.holdout_wz = {t: heads_all[t].holdout_wz for t in corr_cfg.types}
    out.calibrated = {t: sorted(heads_all[t].calibrators) for t in corr_cfg.types}
    events.emit(
        out_dir,
        "head_done",
        fold="test",
        auc=out.holdout_auc,  # internal holdout
        n_pos={t: int(labels[t].sum()) for t in corr_cfg.types},
        elapsed_s=round(time.time() - t_head, 1),
    )
    del extra_tr, extra_te, all_cnt, test_cnt, all_te, test_te, heads_all
    gc.collect()
    return out


def _run_full_fit(
    cfg: ExperimentConfig,
    T: Timings,
    rt: Runtime,
    frames_key: str | None,
    out_dir: Path,
    train: pl.DataFrame,
    X_all: pd.DataFrame,
    X_test: pd.DataFrame,
    te_groups: dict,
    y_res: np.ndarray,
    off_te: np.ndarray,
    regime_tr: np.ndarray,
    regime_te: np.ndarray,
    airports_tr: np.ndarray,
    airports_te: np.ndarray,
    units: list[AirportUnit],
    groups: list[list[str]],
    clean_rows: np.ndarray,
    clean_only: bool,
    corr_cfg: CorruptionConfig | None,
    head: HeadOutputs,
    best_iters: dict[str, int | None],
    fold_fit_rows: dict[tuple[int, int, str], int],
    rollover_rows: np.ndarray | None = None,
    rollover_zeroed: dict[str, int] | None = None,
    nn_pool: SpawnFitPool | None = None,
    months_tr: np.ndarray | None = None,
    months_te: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """``test_fit: full``: one model per (seed, regime group, airport unit) on **all**
    training rows, and the test prediction they produce.

    The head is untouched — the classifiers, the experts and the encodings that score the
    test rows are already fitted on all of 2025 (:func:`_run_head`'s test section), and the
    mixture is applied to this prediction exactly as it was applied to the fold average.
    What changes is only which normal-expert models write ``test_pred``.

    The training matrix is the block features plus the target encodings fitted on **all**
    training rows — the same statistics the test rows already carry, which is what makes the
    full model and its test matrix consistent (the fold models use their own fold's
    statistics). Row weights and the clean-row filter follow the fold path exactly; the
    ``one_minus_p`` weights come from the head's all-row nested pass
    (:attr:`HeadOutputs.weights_all`).

    Nothing here touches the OOF, the metrics or ``importance.csv``: the fold models remain
    the measured object, and a full model has no out-of-sample prediction to measure.

    ``season_weight`` (2026-09-23): one set of full fits **per test month**, each weighted
    towards its own month and writing only that month's test rows. Without it there is one
    target covering every test row and the loop below is the pre-2026-09-23 one.
    """
    log.info("test_fit=full: fitting %d training rows after CV", train.height)
    X_full = X_all
    if te_groups:
        te_full = _te_cached(
            cfg,
            T,
            rt,
            frames_key,
            "full",
            "tr",
            lambda: (
                encode_train(
                    train,
                    te_groups,
                    cfg.te_alpha,
                    tuple(cfg.te_stats),
                    cfg.te_clean_labels,
                    cfg.te_train_mode,
                )
                .drop(S.MVT_ID)
                .to_pandas()
            ),
        )
        X_full = pd.concat([X_all.reset_index(drop=True), te_full], axis=1)
    if cfg.stage1 != "off":  # inner leave-one-month-out over all training rows
        s1_full = _stage1(T, rt, train, None, "stage1_full")
        X_full = pd.concat([X_full.reset_index(drop=True), s1_full], axis=1)
        del s1_full
    if cfg.stack_member is not None:  # H1: the member fitted on the other 11 months per row
        stk_full = stack_mod.StackColumns(cfg, train, None).full()
        X_full = pd.concat([X_full.reset_index(drop=True), stk_full], axis=1)
        del stk_full
    fit_rows = clean_rows.copy() if clean_only else np.ones(len(train), dtype=bool)
    w_fit: np.ndarray | None = None
    if corr_cfg and corr_cfg.normal_weight != "none":
        if corr_cfg.normal_weight == "clean":
            w_all = clean_rows.astype(np.float64)
        elif head.weights_all is None:
            raise ValueError(
                "test_fit: full with normal_weight: one_minus_p needs the head's all-row "
                "weights; HeadOutputs.weights_all is empty (stale head cache entry?)"
            )
        else:
            w_all = np.asarray(head.weights_all, dtype=np.float64)
        w_all = _rollover_weights(corr_cfg, w_all, rollover_rows)
        w_fit = w_all[fit_rows]
        if corr_cfg.rollover_weight == "zero" and rollover_rows is not None:
            n0 = int((rollover_rows & fit_rows).sum())
            log.info("rollover_weight=zero: %d full-fit training rows at weight 0", n0)
            if rollover_zeroed is not None:
                rollover_zeroed["full"] = n0
    ctx0 = FoldContext(
        cfg=cfg,
        groups=groups,
        X_fit=_rows(X_full, fit_rows) if clean_only else X_full,
        X_va=None,
        X_test=X_test,
        y_res=y_res,
        fit_rows=fit_rows,
        va=np.zeros(len(train), dtype=bool),  # unused: the full fit has no validation set
        regime_tr=regime_tr,
        regime_te=regime_te,
        w_fit=w_fit,
        n_threads=rt.n_threads,
        airport_tr=airports_tr if len(units) > 1 else None,
        airport_te=airports_te if len(units) > 1 else None,
    )
    gtags = ["all" if len(groups) == 1 else "".join(g) for g in groups]
    n_fold = n_folds(cfg.cv.scheme)
    info: dict[str, dict] = {}
    pred = np.zeros(len(X_test))

    # (label, ctx, test rows this target writes). One target and every row by default.
    targets: list[tuple[str | None, FoldContext, np.ndarray]] = [
        (None, ctx0, np.ones(len(X_test), dtype=bool))
    ]
    if cfg.season_weight is not None:
        if months_tr is None or months_te is None:
            raise ValueError("season_weight needs the training and test months")
        targets = []
        for m in sorted({str(v) for v in months_te}):
            w_t = _season_fit_weights(cfg, w_fit, months_tr[fit_rows], np.array([m]), f"full {m}")
            targets.append((m, dataclasses.replace(ctx0, w_fit=w_t), months_te == m))

    def _plan(ctx: FoldContext, seed: int, gi: int, ulabel: str, aps) -> tuple:
        task = FitTask(seed=seed, gi=gi, aps=aps)
        gm, gm_te = unit_masks(ctx, task)
        fold_tags = [
            (_fit_tag(k, seed, gi, ulabel, len(groups), len(units)), k) for k in range(n_fold)
        ]
        iters = [best_iters[t] for t, _ in fold_tags if best_iters.get(t) is not None]
        rows = [
            fold_fit_rows[(k, gi, ulabel)]
            for t, k in fold_tags
            if best_iters.get(t) is not None and (k, gi, ulabel) in fold_fit_rows
        ]
        n_all = int(gm[fit_rows].sum())
        rounds, rinfo = _full_fit_rounds(cfg, cast(list[int], iters), rows, n_all)
        return task, gm_te, iters, rows, n_all, rounds, rinfo

    for tlabel, ctx, t_rows in targets:
        # `runtime.nn_fold_workers`: queue every full fit up front (the round counts are
        # already known) and collect them in the sequential loop's order below
        futs: dict[tuple, object] = {}
        handle = None
        if nn_pool is not None:
            handle = nn_pool.put(ctx)
            for seed in cfg.seeds:
                for gi in range(len(groups)):
                    for ulabel, aps in units:
                        task, *_, rounds, _ = _plan(ctx, seed, gi, ulabel, aps)
                        futs[(seed, gi, ulabel)] = nn_pool.submit(handle, task, rounds)
        p_sum = np.zeros(len(X_test))
        n_sets = 0
        for seed in cfg.seeds:
            p_te = np.zeros(len(X_test))
            for gi in range(len(groups)):
                for ulabel, aps in units:
                    task, gm_te, iters, rows, n_all, rounds, rinfo = _plan(
                        ctx, seed, gi, ulabel, aps
                    )
                    tag = _fit_tag("full", seed, gi, ulabel, len(groups), len(units))
                    if tlabel is not None:
                        tag = f"{tag}/target={tlabel}"
                    events.emit(
                        out_dir,
                        "full_fit_start",
                        seed=seed,
                        group=gtags[gi],
                        airport_unit=ulabel,
                        n_rows=n_all,
                        rounds=rounds,
                        **({"target": tlabel} if tlabel is not None else {}),
                    )
                    if nn_pool is not None:
                        r = futs.pop((seed, gi, ulabel)).result()  # type: ignore[attr-defined]
                    else:
                        r = fit_full(ctx, task, rounds)
                    T.merge({f"fit_{n}": v for n, v in r.timings.items()})
                    if r.fallback is not None:
                        p_te[gm_te] = r.fallback
                        log.warning("%s: no training rows; using the mean", tag)
                    elif r.g_te_pred is not None:
                        p_te[gm_te] = r.g_te_pred
                    info[tag] = rinfo | {"fold_rounds": iters, "fold_rows": rows}
                    log.info("%s: %d rounds on %d rows (%s)", tag, rounds, n_all, rinfo["rule"])
                    events.emit(
                        out_dir,
                        "full_fit_done",
                        seed=seed,
                        group=gtags[gi],
                        airport_unit=ulabel,
                        rounds=rounds,
                        elapsed_s=round(r.elapsed_s, 1),
                    )
            p_sum += p_te + off_te
            n_sets += 1
        if nn_pool is not None and handle is not None:
            nn_pool.release(handle)
        pred[t_rows] = (p_sum / max(n_sets, 1))[t_rows]
        del ctx
    del ctx0, targets, X_full
    gc.collect()
    return pred, info


def head_outputs(rt: Runtime, frames_key: str | None, T: Timings, **kw) -> HeadOutputs:
    """:func:`_run_head`, optionally served from ``data/cache/head/<key>/``.

    This is the entry that makes a capacity sweep cheap. ``exp0065..0071`` are seven
    variants of R4 that differ only in the *normal regressor's* ``max_depth`` /
    ``min_child_weight`` / ``num_boost_round``. :func:`prc.mixture.head.classifier_config`
    reads only ``model.kind`` and ``model.device`` from that config, so all seven fit an
    identical head over identical rows — ~50 s of nested in-fold weights plus three
    classifiers per fold, seven times over, in the first seed-farm campaign. The key
    (:func:`prc.cache.head_inputs`) therefore omits the regressor's capacity **except**
    under ``sched_head: z_regression``, where ``_nested_yhat`` fits the normal regressor
    itself to build the head's target and the capacity is genuinely an input again.
    """
    if rt.parallel_fits and not rt.cache_head:
        log.warning(
            "fit_workers=%d with a corruption head and cache_head off: the head trains in "
            "this process, after which forking a fit pool would deadlock, so the fits will "
            "run sequentially (prc.eval.fitpool). Set PRC_CACHE_HEAD=1 to get both.",
            rt.fit_workers,
        )
    if not rt.cache_head or frames_key is None:
        return _run_head(T=T, **kw)
    cfg = kw["cfg"]
    corr = cfg.corruption()
    assert corr is not None
    # the resolved head seed, not cfg.seeds[0]: with `head_seed` pinned, two cells of a seed
    # farm fit the identical head and must therefore share one cache entry
    seed = corr.head_seed if corr.head_seed is not None else cfg.seeds[0]
    entry = cache_mod.entry("head", cache_mod.head_inputs(frames_key, cfg, seed))
    types = list(kw["corr_cfg"].types)

    def _load() -> HeadOutputs | None:
        if not entry.ready():
            return None
        try:
            with T.phase("cache_read"):
                arr = cache_mod.read_arrays(entry, "head.npz")
                meta = cache_mod.read_json(entry, "meta.json")
                out = HeadOutputs(
                    probs_oof={t: arr[f"oof__{t}"] for t in types},
                    probs_test={t: arr[f"test__{t}"] for t in types},
                    weights_by_fold={
                        int(k.split("__")[1]): arr[k] for k in arr if k.startswith("w__")
                    },
                    weights_all=arr.get("wall"),
                    holdout_auc=meta["holdout_auc"],
                    holdout_wz=meta["holdout_wz"],
                    calibrated=meta["calibrated"],
                    n_clf_features=int(meta["n_clf_features"]),
                )
            log.info("head cache HIT %s", entry.dir)
            events.emit(kw["out_dir"], "head_cached", key=entry.key, n_types=len(types))
            return out
        except (OSError, KeyError, ValueError) as exc:
            log.warning("head cache unreadable (%s); recomputing", exc)
            return None

    hit = _load()
    if hit is not None:
        return hit
    with cache_mod.locked(entry) as got:
        if got:
            hit = _load()
            if hit is not None:
                return hit
        out = _run_head(T=T, **kw)
        if got:
            with T.phase("cache_write"):
                arrays = {f"oof__{t}": out.probs_oof[t] for t in types}
                arrays |= {f"test__{t}": out.probs_test[t] for t in types}
                arrays |= {f"w__{k}": v for k, v in out.weights_by_fold.items()}
                if out.weights_all is not None:
                    arrays["wall"] = out.weights_all
                cache_mod.write_arrays(entry, "head.npz", arrays)
                cache_mod.write_json(
                    entry,
                    "meta.json",
                    {
                        "holdout_auc": out.holdout_auc,
                        "holdout_wz": out.holdout_wz,
                        "calibrated": out.calibrated,
                        "n_clf_features": out.n_clf_features,
                    },
                )
                entry.mark_ready()
            log.info("head cache WROTE %s", entry.dir)
    return out


def run_experiment(cfg: ExperimentConfig, out_dir: Path | None = None) -> dict:
    t0 = time.time()
    out_dir = out_dir or (paths.RUNS / cfg.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(out_dir / "log.txt"), logging.StreamHandler()],
    )
    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg.model_dump(), f, sort_keys=False)

    T = Timings()
    rt = resolve_runtime(cfg)
    if rt.n_threads is not None:
        # polars reads this once, at import; setting it later is a no-op, so say so rather
        # than let a config claim a thread count it never got.
        os.environ.setdefault("POLARS_MAX_THREADS", str(rt.n_threads))
    log.info("runtime switches: %s", json.dumps(rt.as_dict()))
    # torch members only (2026-09-22): spawn fold workers and the threaded quantile transform
    nn_rt = resolve_nn(cfg)
    if cfg.model.kind in NN_KINDS:
        log.info("nn runtime switches: %s", json.dumps(nn_rt.as_dict()))
        set_prep_threads(nn_rt.prep_threads)

    corr_cfg = cfg.corruption()
    groups = cfg.regime_groups()
    clean_only = bool(corr_cfg and corr_cfg.train_normal_on == "clean")
    # progress events for `prc status` / the monitor TUI; never load-bearing for the run
    events.emit(
        out_dir,
        "start",
        exp=cfg.name,
        config_digest=cfg.digest(),
        git=_git_rev(),
        scheme=cfg.cv.scheme,
        n_folds=n_folds(cfg.cv.scheme),
        seeds=list(cfg.seeds),
        n_groups=len(groups),
        model_kind=cfg.model.kind,
        device=cfg.model.device,
        pid=os.getpid(),
        host=events.host(),
    )

    te_groups = {n: DEFAULT_GROUPS[n] for n in cfg.target_encoding}
    asm = assemble_frames(cfg, T, rt)
    train, test, feat_cols = asm.train, asm.test, asm.feat_cols
    frames_key = asm.key  # None when the cache is off or a key input could not be hashed
    # `drop_columns` (2026-09-20): after the blocks are assembled, before any matrix is built,
    # and identically on train and test. Applied here rather than inside `_assemble` on
    # purpose — the frames cache key covers `_assemble`, so two configs that differ only in
    # `drop_columns` still share one cached frame build.
    dropped = _apply_drop_columns(cfg, train, test, feat_cols)
    if dropped:
        train, test, feat_cols = dropped
    log.info("features=%d train=%d test=%d", len(feat_cols), train.height, test.height)
    # `cv.group_by_airport`: how many normal-row regressors each (fold, seed, regime group)
    # is split into. One unit — ("all", None) — is the global model and the default.
    airports_tr = train.get_column(S.AIRPORT).to_numpy()
    airports_te = test.get_column(S.AIRPORT).to_numpy()
    units = _airport_units(cfg, airports_tr, airports_te)
    if len(units) > 1:
        log.info(
            "cv.group_by_airport=%s: %d regressors per (fold, seed, regime group): %s",
            cfg.cv.group_by_airport,
            len(units),
            {lab: list(aps or ()) for lab, aps in units},
        )
    events.emit(
        out_dir,
        "features",
        n_features=len(feat_cols),  # block columns; target encodings are added per fold
        n_train=int(train.height),
        n_test=int(test.height),
        blocks=list(cfg.features),
        masked_blocks=list(cfg.mask_for_no_nm),
        n_airport_units=len(units),
        elapsed_s=round(time.time() - t0, 1),
    )

    # Head-only columns, built with the movement table (`_assemble`, or served from the frames
    # cache alongside train/test). `extra_blocks` are joined UNMASKED under a `clfx_` prefix:
    # the point is to give the classifier exactly the surface columns `mask_for_no_nm` takes
    # away from the regressor.
    clf_extra_tr: pl.DataFrame | None = asm.clf_extra_tr
    clf_extra_te: pl.DataFrame | None = asm.clf_extra_te
    if clf_extra_tr is not None:
        log.info("classifier extra columns: %d (%s)", clf_extra_tr.width, clf_extra_tr.columns)
    del asm
    gc.collect()

    y_all = train.get_column(S.TARGET).to_numpy().astype(np.float64)
    fold_id = train.get_column("fold").to_numpy()
    months = train.get_column(S.SPLIT_MONTH)
    months_np = months.to_numpy()
    with T.phase("to_pandas"):
        X_all = _to_pandas(train.select(feat_cols))
        X_test = _to_pandas(test.select(feat_cols))
    if te_groups:  # test-set encodings use all training rows; per-fold ones are added in the loop
        te_test = _te_cached(
            cfg,
            T,
            rt,
            frames_key,
            "test",
            "test",
            lambda: (
                encode(
                    train, test, te_groups, cfg.te_alpha, tuple(cfg.te_stats), cfg.te_clean_labels
                )
                .drop(S.MVT_ID)
                .to_pandas()
            ),
        )
        X_test = pd.concat([X_test, te_test], axis=1)
        feat_cols = feat_cols + list(te_test.columns)
    if cfg.stage1 != "off":  # served-row stage-1 estimates, fitted on all training rows
        s1_test = _stage1(T, rt, train, test, "stage1_test")
        X_test = pd.concat([X_test, s1_test], axis=1)
        feat_cols = feat_cols + list(s1_test.columns)
        del s1_test
    # H1 (`stack_member`, prc.eval.stack): a member's fold-safe prediction as columns, after
    # stage 1 so the column order matches the fold matrices; None on the default path
    stk = stack_mod.StackColumns(cfg, train, test) if cfg.stack_member is not None else None
    if stk is not None:
        X_test = pd.concat([X_test, stk.test()], axis=1)
        feat_cols = feat_cols + stk.names

    # --- regimes, offsets and corruption labels (all no-ops on the default path) ------------
    if cfg.uses_regimes():
        regime_tr, proxy_tr = _regime_arrays(train)
        regime_te, proxy_te = _regime_arrays(test)
    else:
        regime_tr = np.full(train.height, "A")
        regime_te = np.full(test.height, "A")
        proxy_tr = np.full(train.height, np.nan)
        proxy_te = np.full(test.height, np.nan)
    off_tr = _proxy_offset(regime_tr, proxy_tr, cfg.target.mode)
    off_te = _proxy_offset(regime_te, proxy_te, cfg.target.mode)
    y_res = y_all - off_tr

    labels: dict[str, np.ndarray] = {}
    clean_rows = np.ones(train.height, dtype=bool)
    # `rollover_weight: zero` only: the rows the normal expert's fits give weight 0
    rollover_rows: np.ndarray | None = None
    rollover_zeroed: dict[str, int] = {}  # fold (or "full") -> training rows at weight 0
    sched_tr = np.full(train.height, np.nan)
    sched_te = np.full(test.height, np.nan)
    if corr_cfg:
        # `MVT − SCHED`: the corrupt value of the schedule-default branch. Needed before the
        # head section under `sched_head: z_regression` (it is the `C` of `z`) and by the
        # mixture after the regressor loop; it reads no target, so it is not fold-scoped.
        sched_tr = sched_default_value(train).get_column(SCHED_VALUE_COL).to_numpy()
        sched_te = sched_default_value(test).get_column(SCHED_VALUE_COL).to_numpy()
        lab_df = corruption_labels(train, corr_cfg)
        labels = {
            t: lab_df.get_column(LABEL_PREFIX + t).to_numpy().astype(np.float64)
            for t in corr_cfg.types
        }
        for v in labels.values():
            clean_rows &= v == 0
        if corr_cfg.rollover_weight == "zero":
            rollover_rows = labels["rollover"] > 0
        log.info(
            "corruption base rates: %s (clean rows %.4f)",
            {t: round(float(v.mean()), 5) for t, v in labels.items()},
            clean_rows.mean(),
        )

    # --- corruption head: fold-safe probabilities ------------------------------------------
    # This runs *before* the regressor loop because `normal_weight` weights a fold's training
    # rows with probabilities fitted inside that same fold. Both loops are deterministically
    # seeded and share no RNG state, so the order moves no number.
    probs_oof: dict[str, np.ndarray] = {}
    probs_test: dict[str, np.ndarray] = {}
    head_stats: dict[str, dict] = {}
    head_calibration: dict[str, dict] = {}
    weights_by_fold: dict[int, np.ndarray] = {}
    scored_rows = np.zeros(train.height, dtype=bool)  # filled in once the OOF exists
    lane_rows: dict[str, np.ndarray] = {}
    head: HeadOutputs = HeadOutputs()
    n_clf_features = 0
    if corr_cfg:
        with T.phase("head_total"):
            head = head_outputs(
                cfg=cfg,
                corr_cfg=corr_cfg,
                T=T,
                rt=rt,
                frames_key=frames_key,
                out_dir=out_dir,
                train=train,
                test=test,
                X_all=X_all,
                X_test=X_test,
                regime_tr=regime_tr,
                regime_te=regime_te,
                clf_extra_tr=clf_extra_tr,
                clf_extra_te=clf_extra_te,
                labels=labels,
                clean_rows=clean_rows,
                clean_only=clean_only,
                fold_id=fold_id,
                months=months,
                y_all=y_all,
                y_res=y_res,
                off_tr=off_tr,
                sched_tr=sched_tr,
            )
        probs_oof = head.probs_oof
        probs_test = head.probs_test
        weights_by_fold = head.weights_by_fold
        n_clf_features = head.n_clf_features
        del clf_extra_tr, clf_extra_te
        gc.collect()

    oof = np.full(len(train), np.nan)
    oof_by_seed = {seed: np.full(len(train), np.nan) for seed in cfg.seeds}
    test_pred = np.zeros(len(test))
    n_pred_sets = 0
    # `runtime.nn_fold_workers` (2026-09-22): torch members fit their (fold, seed) models in
    # spawn workers sharing the GPU while this process builds the next fold. None = the
    # sequential path, byte for byte. See prc.eval.fitpool.SpawnFitPool.
    nn_pool: SpawnFitPool | None = None
    if nn_rt.fold_workers > 1 and cfg.model.kind in NN_KINDS:
        if rt.parallel_fits:
            log.info(
                "nn_fold_workers=%d overrides fit_workers=%d", nn_rt.fold_workers, rt.fit_workers
            )
        torch_threads = None
        if rt.n_threads is None and str(cfg.model.device).startswith("cuda"):
            # the CPU side of a CUDA fit is its preprocessing; split the cores between workers
            torch_threads = max(1, len(os.sched_getaffinity(0)) // nn_rt.fold_workers)
        nn_pool = SpawnFitPool(
            nn_rt.fold_workers,
            spill_dir=nn_rt.spill_dir,
            log_file=str(out_dir / "log.txt"),
            prep_threads=nn_rt.prep_threads,
            torch_threads=torch_threads,
        )
    elif nn_rt.fold_workers > 1:
        log.warning(
            "nn_fold_workers=%d ignored: model.kind %s is not a torch member (%s)",
            nn_rt.fold_workers,
            cfg.model.kind,
            ", ".join(NN_KINDS),
        )
    # folds submitted to nn_pool and not yet assembled: (k, va, handle, futures), fold order
    nn_pending: list[tuple[int, np.ndarray, str, list]] = []
    importances = []
    best_iters: dict[str, int | None] = {}
    # (fold, gi, unit label) -> rows that fit was fitted on; the denominator of the
    # `test_fit: full` round-count rule, and unused otherwise.
    fold_fit_rows: dict[tuple[int, int, str], int] = {}

    def _assemble_fold(k: int, va: np.ndarray, results: list) -> None:
        """One fold's fit results -> oof / oof_by_seed / test_pred, in seed order. The
        sequential loop calls it right after its fits; the nn pool path calls it for each
        fold in fold order once that fold's fits are back, so the sums run in one order."""
        nonlocal n_pred_sets, test_pred
        gtags = ["all" if len(groups) == 1 else "".join(g) for g in groups]
        by_task = {(r.seed, r.gi, r.aps): r for r in results}
        for seed in cfg.seeds:
            p_va = np.zeros(int(va.sum()))
            p_te = np.zeros(len(test))
            for gi, grp in enumerate(groups):
                for ulabel, aps in units:
                    # exactly the masks `prc.eval.fitpool.unit_masks` gave the fit: the regime
                    # group, narrowed to the unit's airports. Units partition the airports, so
                    # the writes below never overlap.
                    gm = np.isin(regime_tr, grp)
                    gm_te = np.isin(regime_te, grp)
                    if aps is not None:
                        gm = gm & np.isin(airports_tr, aps)
                        gm_te = gm_te & np.isin(airports_te, aps)
                    g_va = gm[va]
                    tag = _fit_tag(k, seed, gi, ulabel, len(groups), len(units))
                    gtag = gtags[gi]
                    r = by_task[(seed, gi, aps)]
                    T.merge({f"fit_{n}": v for n, v in r.timings.items()})
                    if r.fallback is not None:  # nothing to learn from: the fold's mean
                        p_va[g_va] = r.fallback
                        p_te[gm_te] = r.fallback
                        log.warning(
                            "%s: no training rows for regimes %s (airports %s); using the mean",
                            tag,
                            grp,
                            "all" if aps is None else list(aps),
                        )
                    else:
                        if g_va.any() and r.g_va_pred is not None:
                            p_va[g_va] = r.g_va_pred
                        if gm_te.any() and r.g_te_pred is not None:
                            p_te[gm_te] = r.g_te_pred
                        best_iters[tag] = r.best_iteration
                        if r.importance is not None:
                            importances.append(r.importance)
                    # this group's own validation RMSE, in seconds of taxi time (the proxy
                    # offset added back); None when it has no validation rows in this fold
                    events.emit(
                        out_dir,
                        "fit_done",
                        fold=k,
                        seed=seed,
                        group=gtag,
                        airport_unit=ulabel,
                        rmse=float(rmse(y_all[va][g_va], p_va[g_va] + off_tr[va][g_va]))
                        if g_va.any()
                        else None,
                        best_iteration=best_iters.get(tag),
                        elapsed_s=round(r.elapsed_s, 1),
                        total_elapsed_s=round(time.time() - t0, 1),
                    )
            p_va = p_va + off_tr[va]
            p_te = p_te + off_te
            oof[va] = np.where(np.isnan(oof[va]), 0, oof[va]) + p_va / len(cfg.seeds)
            oof_by_seed[seed][va] = p_va
            test_pred += p_te
            n_pred_sets += 1
            log.info(
                "fold=%d seed=%d rmse=%.3f iters=%s",
                k,
                seed,
                rmse(y_all[va], p_va),
                best_iters.get(f"fold{k}_seed{seed}")
                if len(groups) * len(units) == 1
                else {
                    _fit_tag(k, seed, gi, lab, len(groups), len(units)): best_iters.get(
                        _fit_tag(k, seed, gi, lab, len(groups), len(units))
                    )
                    for gi in range(len(groups))
                    for lab, _ in units
                },
            )

    def _drain_nn(limit: int) -> None:
        """Assemble every leading fold whose fits are back; then, while more than ``limit``
        fits are outstanding, wait for the next one. Fold order is kept throughout."""
        from concurrent.futures import FIRST_COMPLETED, wait

        assert nn_pool is not None
        while nn_pending:
            while nn_pending and all(f.done() for f in nn_pending[0][3]):
                k_, va_, handle_, futs_ = nn_pending.pop(0)
                results_ = [f.result() for f in futs_]
                nn_pool.release(handle_)
                _assemble_fold(k_, va_, results_)
                del results_, futs_
            outstanding = [f for *_, fs in nn_pending for f in fs if not f.done()]
            if not nn_pending or len(outstanding) <= limit:
                return
            with T.phase("fit_total"):
                wait(outstanding, return_when=FIRST_COMPLETED)

    # `validation.early_stop: inner` (2026-09-23): a seeded share of the training airport-days
    # is the fold models' early-stopping set instead of the scored month. None = default path.
    es_all: np.ndarray | None = None
    if cfg.inner_early_stop():
        assert cfg.validation is not None
        day_key = (
            train.get_column(S.AIRPORT).cast(pl.Utf8)
            + "|"
            + train.get_column(S.MVT_TIME).dt.date().cast(pl.Utf8).fill_null("NaT")
        ).to_numpy()
        es_all = inner_holdout_mask(day_key, cfg.validation.inner_frac, cfg.validation.inner_seed)
        del day_key
        log.info(
            "validation.early_stop=inner: %d of %d training rows (%.1f %%) on held-out "
            "airport-days, inner_frac=%.3f inner_seed=%d",
            int(es_all.sum()),
            len(es_all),
            100.0 * float(es_all.mean()),
            cfg.validation.inner_frac,
            cfg.validation.inner_seed,
        )
    for k in range(n_folds(cfg.cv.scheme)):
        va = fold_id == k
        tr = train_mask(months, cfg.cv.scheme, k).to_numpy() & ~va
        if va.sum() == 0:
            log.warning("fold %d has no validation rows; skipping", k)
            continue
        X_tr = cast(pd.DataFrame, X_all[tr])
        X_va = cast(pd.DataFrame, X_all[va])
        if te_groups:  # fit statistics on this fold's training rows only
            t_te = time.time()
            tr_rows, va_rows = train.filter(pl.Series(tr)), train.filter(pl.Series(va))
            te_tr = _te_cached(
                cfg,
                T,
                rt,
                frames_key,
                k,
                "tr",
                lambda _t=tr_rows: (
                    encode_train(
                        _t,
                        te_groups,
                        cfg.te_alpha,
                        tuple(cfg.te_stats),
                        cfg.te_clean_labels,
                        cfg.te_train_mode,
                    )
                    .drop(S.MVT_ID)
                    .to_pandas()
                ),
            )
            te_va = _te_cached(
                cfg,
                T,
                rt,
                frames_key,
                k,
                "va",
                lambda _t=tr_rows, _v=va_rows: (
                    encode(
                        _t, _v, te_groups, cfg.te_alpha, tuple(cfg.te_stats), cfg.te_clean_labels
                    )
                    .drop(S.MVT_ID)
                    .to_pandas()
                ),
            )
            X_tr = pd.concat([X_tr.reset_index(drop=True), te_tr], axis=1)
            X_va = pd.concat([X_va.reset_index(drop=True), te_va], axis=1)
            del te_tr, te_va, tr_rows, va_rows
            events.emit(out_dir, "te_done", fold=k, elapsed_s=round(time.time() - t_te, 1))
        if cfg.stage1 != "off":  # after the encodings, so the column order matches X_test
            narrow = train.select(stage1_mod.needed_columns(train))
            tr_rows, va_rows = narrow.filter(pl.Series(tr)), narrow.filter(pl.Series(va))
            del narrow
            s1_tr = _stage1(T, rt, tr_rows, None, "stage1_fold")
            s1_va = _stage1(T, rt, tr_rows, va_rows, "stage1_fold")
            X_tr = pd.concat([X_tr.reset_index(drop=True), s1_tr], axis=1)
            X_va = pd.concat([X_va.reset_index(drop=True), s1_va], axis=1)
            del s1_tr, s1_va, tr_rows, va_rows
        if stk is not None:  # inner-run values for training rows, served values for validation
            stk_tr, stk_va = stk.fold(k, tr, va)
            X_tr = pd.concat([X_tr.reset_index(drop=True), stk_tr], axis=1)
            X_va = pd.concat([X_va.reset_index(drop=True), stk_va], axis=1)
            del stk_tr, stk_va
        # the clean expert drops the rows carrying a corruption label from its *training* set;
        # the validation set is untouched, so no extra validation label enters the fit. Any
        # target encodings stay fitted on all of the fold's training rows (a reference
        # statistic, not a training target), so X_tr is subset rather than rebuilt.
        fit_rows = (tr & clean_rows) if clean_only else tr
        X_fit = _rows(X_tr, clean_rows[tr]) if clean_only else X_tr
        # `normal_weight` is the soft version of that switch: every row stays in, carrying
        # 1 - sum_k p_k. The probabilities come from heads fitted inside this fold's own
        # training rows (`_in_fold_weights`), so no validation label reaches the weights.
        w_fit: np.ndarray | None = None
        if corr_cfg and corr_cfg.normal_weight != "none":
            w_all = (
                clean_rows.astype(np.float64)
                if corr_cfg.normal_weight == "clean"
                else weights_by_fold[k]
            )
            w_all = _rollover_weights(corr_cfg, w_all, rollover_rows)
            w_fit = w_all[fit_rows]
            if corr_cfg.rollover_weight == "zero" and rollover_rows is not None:
                rollover_zeroed[str(k)] = int((rollover_rows & fit_rows).sum())
                log.info(
                    "rollover_weight=zero: fold %d, %d training rows at weight 0",
                    k,
                    rollover_zeroed[str(k)],
                )
        if cfg.season_weight is not None:
            # lever 1: target this fold's fits at the month(s) it validates
            w_fit = _season_fit_weights(cfg, w_fit, months_np[fit_rows], months_np[va], f"fold={k}")
        # One fold's fits are independent functions of (seed, regime group) — see
        # prc.eval.fitpool, which is the single implementation both the sequential and the
        # one-process-per-GPU paths call. `pool.map` returns in task order, so the assembly
        # below runs in exactly the order the sequential loop used and the averaging is
        # deterministic either way.
        ctx = FoldContext(
            cfg=cfg,
            groups=groups,
            X_fit=X_fit,
            X_va=X_va,
            X_test=X_test,
            y_res=y_res,
            fit_rows=fit_rows,
            va=va,
            regime_tr=regime_tr,
            regime_te=regime_te,
            w_fit=w_fit,
            n_threads=rt.n_threads,
            airport_tr=airports_tr if len(units) > 1 else None,
            airport_te=airports_te if len(units) > 1 else None,
            es_rows=None if es_all is None else es_all[fit_rows],
        )
        # (seed, regime group, airport unit). With `cv.group_by_airport` off there is exactly
        # one unit carrying `aps=None`, and both the task list and every tag below are what
        # they were before the option existed.
        tasks = [
            FitTask(seed=seed, gi=gi, aps=aps)
            for seed in cfg.seeds
            for gi in range(len(groups))
            for _, aps in units
        ]
        n_fit_rows = {
            (gi, lab): int(
                (
                    np.isin(regime_tr, grp)
                    & (True if aps is None else np.isin(airports_tr, aps))
                    # under `inner` the held-out airport-days are not fitted on
                    & (True if es_all is None else ~es_all)
                )[fit_rows].sum()
            )
            for gi, grp in enumerate(groups)
            for lab, aps in units
        }
        fold_fit_rows |= {(k, gi, lab): n for (gi, lab), n in n_fit_rows.items()}
        gtags = ["all" if len(groups) == 1 else "".join(g) for g in groups]
        ulabel_of = {aps: lab for lab, aps in units}
        for task in tasks:
            events.emit(
                out_dir,
                "fit_start",
                fold=k,
                seed=task.seed,
                group=gtags[task.gi],
                airport_unit=ulabel_of[task.aps],
                n_rows=n_fit_rows[(task.gi, ulabel_of[task.aps])],
            )
        if nn_pool is not None:
            # spill the fold, queue its fits, and go build the next fold while they run;
            # at most `nn_fold_workers` fits outstanding beyond the one being queued
            handle = nn_pool.put(ctx)
            nn_pending.append((k, va, handle, [nn_pool.submit(handle, t) for t in tasks]))
            del ctx, X_tr, X_va, X_fit, w_fit
            gc.collect()
            _drain_nn(nn_rt.fold_workers)
            continue
        if rt.parallel_fits and len(tasks) > 1:
            for i, task in enumerate(tasks):
                task.device = rt.devices[i % len(rt.devices)]
            with T.phase("fit_total"):
                results = run_fold_fits(ctx, tasks, rt.fit_workers)
        else:
            # one process: the matrices of this fold are built once and reused by every seed
            # (runtime.reuse_fold_matrix), then dropped when the fold ends.
            mc = MatrixCache() if rt.reuse_fold_matrix else None
            results = []
            with T.phase("fit_total"):
                for task in tasks:
                    results.append(
                        fit_one(
                            ctx,
                            task,
                            mc,
                            # the matrix cache is keyed per (fold, regime group, airport
                            # unit): two units of one fold hold DIFFERENT rows, so the unit
                            # has to be in the key or the second one would be served the
                            # first one's matrices.
                            key=None if mc is None else f"fold{k}/g{task.gi}/{ulabel_of[task.aps]}",
                        )
                    )
            if mc is not None:
                log.info("fold=%d matrix cache %s", k, json.dumps(mc.stats()))
                T.add("fit_matrix_cache_hits", 0.0, n=mc.hits)
                mc.clear()
                gc.collect()
        _assemble_fold(k, va, results)
        del ctx, results
        del X_tr, X_va, X_fit, w_fit
        gc.collect()
    if nn_pool is not None:
        _drain_nn(0)
        T.add("nn_spill", nn_pool.spill_s)
    weights_by_fold.clear()
    test_pred /= max(n_pred_sets, 1)
    if nn_pool is not None and cfg.test_fit != "full":
        nn_pool.close()

    # `test_fit: full` (2026-09-20): replace the fold average with models fitted on every
    # training row. Before the mixture, so the head applies to it exactly as it applied to
    # the fold average; after the OOF, which it never touches.
    full_fit_info: dict = {}
    # the fold-model average the full fit replaces, kept so it can be written next to it
    # (`test_pred_foldavg.parquet`); None on the default path, which writes nothing extra
    test_pred_foldavg: np.ndarray | None = None
    if cfg.test_fit == "full":
        test_pred_foldavg = test_pred.copy()
        with T.phase("full_fit_total"):
            test_pred, full_fit_info = _run_full_fit(
                cfg=cfg,
                T=T,
                rt=rt,
                frames_key=frames_key,
                out_dir=out_dir,
                train=train,
                X_all=X_all,
                X_test=X_test,
                te_groups=te_groups,
                y_res=y_res,
                off_te=off_te,
                regime_tr=regime_tr,
                regime_te=regime_te,
                airports_tr=airports_tr,
                airports_te=airports_te,
                units=units,
                groups=groups,
                clean_rows=clean_rows,
                clean_only=clean_only,
                corr_cfg=corr_cfg,
                head=head,
                best_iters=best_iters,
                fold_fit_rows=fold_fit_rows,
                rollover_rows=rollover_rows,
                rollover_zeroed=rollover_zeroed,
                nn_pool=nn_pool,
                months_tr=months_np,
                months_te=test.get_column(S.SPLIT_MONTH).to_numpy(),
            )
    if nn_pool is not None and cfg.test_fit == "full":
        nn_pool.close()

    if corr_cfg:
        scored_rows = ~np.isnan(oof)
        # the lane the identity signals are aimed at: departures with no flight-plan record,
        # at LIRF (where the substitution is a mass phenomenon) and everywhere
        no_nm_rows = train.get_column(S.AOBT_3).is_null().to_numpy() & scored_rows
        lane_rows = {
            "no_nm_lirf": no_nm_rows & (airports_tr == "LIRF"),
            "no_nm_all": no_nm_rows,
        }
        for t in corr_cfg.types:
            head_stats[t] = {
                "base_rate": float(labels[t].mean()),
                "n_pos": int(labels[t].sum()),
                "auc_oof": auc_score(labels[t][scored_rows], probs_oof[t][scored_rows]),
                "holdout_auc_all": head.holdout_auc.get(t),
                "p_mean_oof": float(np.nanmean(probs_oof[t])),
                "p_mean_test": float(np.mean(probs_test[t])),
                "calibrated": head.calibrated.get(t, []),
            }
            if t == "sched_default":
                head_stats[t] |= {
                    f"auc_{name}": auc_score(labels[t][m], probs_oof[t][m])
                    for name, m in lane_rows.items()
                }
                head_stats[t] |= {f"n_{name}": int(m.sum()) for name, m in lane_rows.items()}
            p_s, y_s = probs_oof[t][scored_rows], labels[t][scored_rows]
            a_s = airports_tr[scored_rows]
            head_calibration[t] = {
                "pooled": calibration_table(p_s, y_s),
                "per_airport": {
                    str(a): calibration_table(p_s[a_s == a], y_s[a_s == a])
                    for a in sorted(set(a_s.tolist()))
                },
            }
        log.info("corruption head: %s", json.dumps(head_stats, default=str))

    y_hat_normal, y_hat_normal_test = oof.copy(), test_pred.copy()
    y_hat_normal_foldavg = None if test_pred_foldavg is None else test_pred_foldavg.copy()
    if corr_cfg:
        if "sched_default" in corr_cfg.types:
            # the value-weighted gate: it needs `ŷ_normal`, so it is measured here rather than
            # in the head block above, on the same OOF rows and before any clipping
            head_stats["sched_default"] |= _z_metrics(
                corr_cfg,
                y_all,
                y_hat_normal,
                sched_tr,
                np.nan_to_num(probs_oof["sched_default"]),
                labels,
                airports_tr,
                {"scored": scored_rows} | lane_rows,
            )
            head_stats["sched_default"]["holdout_weighted_z_loss"] = head.holdout_wz.get(
                "sched_default"
            )
            log.info(
                "sched_default value-weighted loss: %s (no-NM LIRF %s)",
                head_stats["sched_default"]["weighted_z_loss"],
                head_stats["sched_default"].get("weighted_z_loss_no_nm_lirf"),
            )
        safe_oof = {t: np.nan_to_num(v) for t, v in probs_oof.items()}
        oof = apply_mixture(oof, safe_oof, corrupt_values(oof, sched_tr, corr_cfg), corr_cfg.p_max)
        for seed, v in oof_by_seed.items():
            oof_by_seed[seed] = apply_mixture(
                v, safe_oof, corrupt_values(v, sched_tr, corr_cfg), corr_cfg.p_max
            )
        test_pred = apply_mixture(
            test_pred,
            probs_test,
            corrupt_values(test_pred, sched_te, corr_cfg),
            corr_cfg.p_max,
        )
        if test_pred_foldavg is not None:  # the same head on the fold average
            test_pred_foldavg = apply_mixture(
                test_pred_foldavg,
                probs_test,
                corrupt_values(test_pred_foldavg, sched_te, corr_cfg),
                corr_cfg.p_max,
            )

    lo, hi = cfg.target.clip_min, cfg.target.clip_max

    def _clip(a: np.ndarray) -> np.ndarray:
        return np.clip(a, lo, hi) if (lo is not None or hi is not None) else a

    oof, test_pred = _clip(oof), _clip(test_pred)
    y_hat_normal, y_hat_normal_test = _clip(y_hat_normal), _clip(y_hat_normal_test)
    if test_pred_foldavg is not None and y_hat_normal_foldavg is not None:
        test_pred_foldavg = _clip(test_pred_foldavg)
        y_hat_normal_foldavg = _clip(y_hat_normal_foldavg)

    scored = ~np.isnan(oof)
    oof_df = (
        train.select(S.MVT_ID, S.AIRPORT, S.SPLIT_MONTH, "fold", S.TARGET)
        .with_columns(
            pl.Series("pred", oof),
            # per-seed predictions (clipped like the mean) so paired seed deltas between runs
            # can be computed from stored OOFs without retraining
            *[pl.Series(f"pred_s{seed}", _clip(v)) for seed, v in oof_by_seed.items()],
            *([pl.Series(REGIME_COL, regime_tr)] if cfg.uses_regimes() else []),
            *(
                [pl.Series("y_hat_normal", y_hat_normal)]
                + [pl.Series(PROB_PREFIX + t, probs_oof[t]) for t in corr_cfg.types]
                if corr_cfg
                else []
            ),
        )
        .filter(pl.Series(scored))
    )
    per_seed = {
        str(seed): float(rmse(oof_df[S.TARGET].to_numpy(), oof_df[f"pred_s{seed}"].to_numpy()))
        for seed in cfg.seeds
    }
    overall = rmse(oof_df[S.TARGET].to_numpy(), oof_df["pred"].to_numpy())
    per_fold = rmse_breakdown(oof_df, S.TARGET, "pred", ["fold"])
    per_airport = rmse_breakdown(oof_df, S.TARGET, "pred", [S.AIRPORT])
    per_month = rmse_breakdown(oof_df, S.TARGET, "pred", [S.SPLIT_MONTH])
    boot_std = bootstrap_rmse_std(oof_df[S.TARGET].to_numpy(), oof_df["pred"].to_numpy())

    metrics = {
        "name": cfg.name,
        "cv_rmse": overall,
        "cv_rmse_boot_std": boot_std,
        "per_month": {r[S.SPLIT_MONTH]: float(r["rmse"]) for r in per_month.iter_rows(named=True)},
        "per_fold": {int(r["fold"]): float(r["rmse"]) for r in per_fold.iter_rows(named=True)},
        "per_airport": {r[S.AIRPORT]: float(r["rmse"]) for r in per_airport.iter_rows(named=True)},
        "per_seed": per_seed,
        "n_train": int(train.height),
        "n_oof": int(scored.sum()),
        "n_test": int(test.height),
        "n_features": len(feat_cols),
        "features": cfg.features,
        "best_iterations": best_iters,
        "elapsed_s": round(time.time() - t0, 1),
        # which models wrote test_pred, and — under "full" — how many rounds each of them
        # got and the fold numbers the rule derived that from
        "test_fit": cfg.test_fit,
        "full_fit": full_fit_info,
        "git": _git_rev(),
        "config_digest": cfg.digest(),
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    # the lane breakdown scripts/lane_report.py prints, computed here from the rows already in
    # memory so readers (registry, `prc status`, the monitor) never re-join parquet files
    metrics["lanes"] = lane_metrics(
        train.select(S.SPLIT_MONTH, S.TARGET, S.MVT_TIME, S.AOBT_3)
        .with_columns(pl.Series("pred", oof))
        .filter(pl.Series(scored))
    )
    if cfg.stage1 != "off":  # absent on the default path, so its metrics.json is unchanged
        metrics["stage1"] = cfg.stage1
    if cfg.stack_member is not None:  # absent on the default path
        metrics["stack_member"] = cfg.stack_member.model_dump()
    if cfg.season_weight is not None:  # absent on the default path
        metrics["season_weight"] = cfg.season_weight.model_dump()
    if len(units) > 1:  # cv.group_by_airport: which airports each regressor owned
        metrics["airport_units"] = {lab: list(aps or ()) for lab, aps in units}
    if cfg.uses_regimes():
        per_regime = rmse_breakdown(oof_df, S.TARGET, "pred", [REGIME_COL])
        metrics["target_mode"] = cfg.target.mode
        metrics["regime_groups"] = groups
        metrics["per_regime"] = {
            r[REGIME_COL]: {
                "n": int(r["n"]),
                "rmse": float(r["rmse"]),
                "sse_share": float(r["sse_share"]),
            }
            for r in per_regime.iter_rows(named=True)
        }
    if corr_cfg:
        metrics["corruption"] = head_stats
        metrics["corruption_calibration"] = head_calibration
        metrics["corruption_train_normal_on"] = corr_cfg.train_normal_on
        metrics["corruption_normal_weight"] = corr_cfg.normal_weight
        metrics["corruption_calibrate"] = corr_cfg.calibrate
        metrics["corruption_extra_blocks"] = list(corr_cfg.extra_blocks)
        metrics["corruption_extra_signals"] = corr_cfg.extra_signals
        metrics["corruption_identity_signals"] = corr_cfg.identity_signals
        metrics["corruption_sched_head"] = corr_cfg.sched_head
        if corr_cfg.rollover_weight != "soft":  # absent at the default: metrics.json unchanged
            metrics["corruption_rollover_weight"] = corr_cfg.rollover_weight
            metrics["rollover_zeroed_rows"] = dict(rollover_zeroed)
        metrics["n_clf_features"] = n_clf_features
        metrics["cv_rmse_no_head"] = float(
            rmse(oof_df[S.TARGET].to_numpy(), oof_df["y_hat_normal"].to_numpy())
        )
        metrics["head_delta_rmse"] = metrics["cv_rmse"] - metrics["cv_rmse_no_head"]
    # Where the wall clock went, and which performance switches were on when it went there.
    # Always written: the 2026-09-18 "=60 s of setup, =300 s of fits" figure had to be
    # reconstructed by hand from log.txt, and the 50 s inside that setup that turned out to
    # be the nested in-fold weights was invisible until someone went looking.
    metrics["timings"] = T.as_dict()
    metrics["runtime"] = rt.as_dict() | (nn_rt.as_dict() if cfg.model.kind in NN_KINDS else {})

    def _test_frame(pred: np.ndarray, y_normal: np.ndarray) -> pl.DataFrame:
        return test.select(S.MVT_ID).with_columns(
            pl.Series("pred", pred),
            *([pl.Series(REGIME_COL, regime_te)] if cfg.uses_regimes() else []),
            *(
                [pl.Series("y_hat_normal", y_normal)]
                + [pl.Series(PROB_PREFIX + t, probs_test[t]) for t in corr_cfg.types]
                if corr_cfg
                else []
            ),
        )

    with T.phase("write_artifacts"):
        oof_df.write_parquet(out_dir / "oof.parquet")
        _test_frame(test_pred, y_hat_normal_test).write_parquet(out_dir / "test_pred.parquet")
        if test_pred_foldavg is not None and y_hat_normal_foldavg is not None:
            # `test_fit: full` only: the fold-model average, in the same schema, for
            # `scripts/blend_fit.py apply --test-source foldavg` (review 2026-09-23 B, #6)
            _test_frame(test_pred_foldavg, y_hat_normal_foldavg).write_parquet(
                out_dir / "test_pred_foldavg.parquet"
            )
        if importances:
            pd.concat(importances, axis=1).fillna(0).mean(axis=1).sort_values(
                ascending=False
            ).to_csv(out_dir / "importance.csv", header=["gain"])
    metrics["timings"] = T.as_dict()  # including the write itself
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    append_registry(metrics, cfg)
    log.info("%s", T.summary(metrics["elapsed_s"]))
    log.info("DONE %s cv_rmse=%.3f (%.0fs)", cfg.name, overall, metrics["elapsed_s"])
    events.emit(
        out_dir,
        "done",
        cv_rmse=metrics["cv_rmse"],
        per_month=metrics["per_month"],
        cv_rmse_no_head=metrics.get("cv_rmse_no_head"),
        elapsed_s=metrics["elapsed_s"],
        peak_rss_mb=events.peak_rss_mb(),
    )
    return metrics


REGISTRY_COLS = [
    "finished_at",
    "name",
    "cv_rmse",
    "cv_scheme",
    "model",
    "n_features",
    "seeds",
    "elapsed_s",
    "git",
    "config_digest",
    "description",
    # appended 2026-09-18; older rows simply have no value in these columns
    "lane_ordinary_rmse",
    "lane_no_aobt_rmse",
]


def _sync_header(f, existing: list[str]) -> None:
    """Extend an older header in place when columns have been appended to ``REGISTRY_COLS``.

    Only the header line is rewritten — every data row is copied back byte for byte, so the
    history is untouched and older rows are simply short (polars fills the new columns with
    null). A header that is not a prefix of the current columns is left alone: that is a
    conflict for a human, not for a run that has just finished.
    """
    if existing == REGISTRY_COLS[: len(existing)] and len(existing) < len(REGISTRY_COLS):
        rest = f.read()
        f.seek(0)
        f.truncate()
        csv.writer(f).writerow(REGISTRY_COLS)
        f.write(rest)


def append_registry(metrics: dict, cfg: ExperimentConfig) -> None:
    """Append one row; an exclusive lock makes concurrent SLURM array tasks safe (the header
    check must sit inside the lock or two tasks can both write one)."""
    import fcntl

    path = paths.EXPERIMENTS / "registry.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.touch()
    lanes = metrics.get("lanes")
    with open(path, "r+", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        header = f.readline()
        if header.strip():
            _sync_header(f, next(csv.reader([header])))
        f.seek(0, 2)
        new = f.tell() == 0
        w = csv.DictWriter(f, fieldnames=REGISTRY_COLS)
        if new:
            w.writeheader()
        w.writerow(
            {
                "finished_at": metrics["finished_at"],
                "name": cfg.name,
                "cv_rmse": round(metrics["cv_rmse"], 4),
                "cv_scheme": cfg.cv.scheme,
                "model": cfg.model.kind,
                "n_features": metrics["n_features"],
                "seeds": len(cfg.seeds),
                "elapsed_s": metrics["elapsed_s"],
                "git": metrics["git"],
                "config_digest": metrics["config_digest"],
                "description": cfg.description,
                "lane_ordinary_rmse": lane_rmse(lanes, "ordinary"),
                "lane_no_aobt_rmse": lane_rmse(lanes, "no_aobt"),
            }
        )
