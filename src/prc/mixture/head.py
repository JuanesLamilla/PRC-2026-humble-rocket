"""The corruption head: one fold-safe model per corruption type — a binary classifier, or for
``sched_default`` under ``mixture.corruption.sched_head: z_regression`` a weighted regression
of the RMSE-optimal mixture weight ``z`` (:func:`fit_z_head`).

The labels come from the target (`prc.mixture.corruption`), so a head may only ever see the
training rows of its own fold. This module therefore fits and predicts, and never decides
*which* rows it is given — the runner owns that and hands it a fold-train matrix.

Each head reuses the project's GBDT wrappers with a binary objective and AUC early stopping on
an internal, stratified 10 % holdout of the rows it was given (never the fold's validation
rows, which would leak the labels we are trying to predict). A type with too few positives
falls back to its base rate, which is the RMSE-optimal constant when nothing is learnable.

With ``mixture.corruption.calibrate: isotonic`` the same internal holdout also fits a monotone
map from the raw score to the observed rate, per ``(type, airport)``, falling back to a pooled
curve where an airport has fewer than ``calibrate_min_positives`` positives. Under RMSE the
mixture weight is a probability, not a ranking, so a head with AUC 0.87 can still cost the
ordinary lane 12 s purely by putting the wrong *level* of mass on ``MVT - SCHED``. The holdout
is the fold's own, so no CV validation row is involved; it is the same holdout the head
early-stopped on, which makes the fitted curve mildly optimistic (documented limitation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd

from prc.config import CorruptionConfig, ModelConfig
from prc.models import make_model

AIRPORT_COL = "clf_airport"  # the classifier matrix always carries it (prc.eval.run._clf_frame)
POOLED = ""  # key of the pooled calibration curve

_XGB_PARAMS = {"objective": "binary:logistic", "eval_metric": "auc"}
_LGBM_PARAMS = {"objective": "binary", "metric": "auc"}
_XGB_Z_PARAMS = {"objective": "reg:squarederror", "eval_metric": "rmse"}
_LGBM_Z_PARAMS = {"objective": "regression", "metric": "rmse"}


@dataclass(frozen=True)
class ZTargets:
    """The continuous target of a ``z_regression`` head and its value weights, aligned to the
    rows the head is fitted on (:func:`prc.mixture.corruption.z_targets`)."""

    z: np.ndarray
    w: np.ndarray


@dataclass
class TypeHead:
    """A fitted (or degenerate) head for one corruption type: a classifier of the type's 0/1
    label, or — for ``sched_default`` under ``sched_head: z_regression`` — a weighted
    regression of ``z`` whose prediction is clipped into ``clip``."""

    name: str
    base_rate: float
    n_pos: int
    model: Any | None = None
    holdout_auc: float | None = None
    #: airport -> isotonic map, plus :data:`POOLED`; empty when calibration is off
    calibrators: dict[str, Any] = field(default_factory=dict)
    kind: str = "classifier"
    #: [lo, hi] the z head's prediction is clipped to (``mixture.corruption.p_clip``)
    clip: tuple[float, float] = (0.0, 1.0)
    #: the z head's weighted squared error on its internal holdout (what it early-stopped on)
    holdout_wz: float | None = None

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        lo, hi = self.clip if self.kind == "z_regression" else (0.0, 1.0)
        if self.model is None:
            return np.full(len(X), self.base_rate, dtype=np.float64)
        p = np.clip(np.asarray(self.model.predict(X), dtype=np.float64), lo, hi)
        if self.calibrators:  # never fitted for a z head: it predicts a weight, not a rate
            p = apply_calibration(p, _airports(X), self.calibrators)
        return np.clip(p, lo, hi)


def classifier_config(corr_cfg: CorruptionConfig, model_cfg: ModelConfig) -> ModelConfig:
    """The regressor's engine and device, a binary objective, and the head's own capacity."""
    kind = model_cfg.kind if model_cfg.kind in ("xgb", "lgbm") else "xgb"
    base = dict(_XGB_PARAMS if kind == "xgb" else _LGBM_PARAMS)
    if kind == "xgb":
        base |= {"max_depth": corr_cfg.max_depth, "learning_rate": corr_cfg.learning_rate}
    else:
        base |= {
            "num_leaves": 2**corr_cfg.max_depth - 1,
            "learning_rate": corr_cfg.learning_rate,
        }
    return ModelConfig(
        kind=kind,
        params=base | dict(corr_cfg.params),
        num_boost_round=corr_cfg.num_boost_round,
        early_stopping_rounds=corr_cfg.early_stopping_rounds,
        device=model_cfg.device,
    )


def z_head_config(corr_cfg: CorruptionConfig, model_cfg: ModelConfig) -> ModelConfig:
    """The regressor's engine and device, a squared-error objective, and the z head's own
    (small) capacity — the same shape as :func:`classifier_config`, different loss."""
    kind = model_cfg.kind if model_cfg.kind in ("xgb", "lgbm") else "xgb"
    base = dict(_XGB_Z_PARAMS if kind == "xgb" else _LGBM_Z_PARAMS)
    if kind == "xgb":
        base |= {
            "max_depth": corr_cfg.z_max_depth,
            "learning_rate": corr_cfg.z_learning_rate,
            "min_child_weight": corr_cfg.z_min_child_weight,
        }
    else:
        base |= {
            "num_leaves": 2**corr_cfg.z_max_depth - 1,
            "learning_rate": corr_cfg.z_learning_rate,
            "min_sum_hessian_in_leaf": corr_cfg.z_min_child_weight,
        }
    return ModelConfig(
        kind=kind,
        params=base | dict(corr_cfg.params),
        num_boost_round=corr_cfg.z_num_boost_round,
        early_stopping_rounds=corr_cfg.early_stopping_rounds,
        device=model_cfg.device,
    )


def weighted_z_loss(p: np.ndarray, z: np.ndarray, w: np.ndarray) -> float | None:
    """``Σ w·(p - z)² / Σ w`` — the mixture's own squared error on the rows that carry weight,
    per unit of ``(C - yhat)²``. ``None`` when nothing carries weight.

    This is the quantity a ``sched_default`` head is actually judged on, and the replacement
    for the AUC gate: AUC weights every row the
    same, the squared error weights it by ``(C - yhat)²``, and ``exp0057`` won the first and
    lost the second.
    """
    p, z, w = (np.asarray(v, dtype=np.float64) for v in (p, z, w))
    ok = np.isfinite(p) & np.isfinite(z) & np.isfinite(w) & (w > 0)
    total = float(w[ok].sum())
    if not ok.any() or total <= 0:
        return None
    return float((w[ok] * (p[ok] - z[ok]) ** 2).sum() / total)


def _stratified_holdout(y: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Boolean mask of holdout rows with at least one row of each class where possible."""
    rng = np.random.default_rng(seed)
    hold = np.zeros(len(y), dtype=bool)
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls)
        if len(idx) == 0:
            continue
        n = min(len(idx) - 1, max(1, round(frac * len(idx))))
        if n <= 0:
            continue
        hold[rng.choice(idx, size=n, replace=False)] = True
    return hold


def _airports(X: pd.DataFrame) -> np.ndarray:
    if AIRPORT_COL not in X.columns:
        return np.full(len(X), POOLED, dtype=object)
    return np.asarray(X[AIRPORT_COL].astype(str), dtype=object)


def fit_calibrators(
    p: np.ndarray, y: np.ndarray, airports: np.ndarray, min_positives: int
) -> dict[str, Any]:
    """Isotonic score -> rate maps: one per airport with enough positives, plus a pooled one.

    ``p`` / ``y`` / ``airports`` are the head's own internal holdout rows. An airport cell with
    fewer than ``min_positives`` positives is not fitted at all and falls back to the pooled
    curve at :func:`apply_calibration` time.
    """
    from sklearn.isotonic import IsotonicRegression

    def _fit(pi: np.ndarray, yi: np.ndarray) -> Any | None:
        if len(pi) < 2 or yi.min() == yi.max() or pi.min() == pi.max():
            return None
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(pi, yi)
        return iso

    out: dict[str, Any] = {}
    pooled = _fit(p, y)
    if pooled is None:
        return out
    out[POOLED] = pooled
    for airport in sorted({str(a) for a in airports}):
        m = airports == airport
        if int(y[m].sum()) < min_positives:
            continue
        cal = _fit(p[m], y[m])
        if cal is not None:
            out[airport] = cal
    return out


def apply_calibration(
    p: np.ndarray, airports: np.ndarray, calibrators: dict[str, Any]
) -> np.ndarray:
    """Per-airport curve where one was fitted, pooled curve everywhere else."""
    pooled = calibrators.get(POOLED)
    out = np.asarray(pooled.predict(p), dtype=np.float64) if pooled is not None else p.copy()
    for airport, cal in calibrators.items():
        if airport == POOLED:
            continue
        m = airports == airport
        if m.any():
            out[m] = np.asarray(cal.predict(p[m]), dtype=np.float64)
    return out


def calibration_table(
    p: np.ndarray, y: np.ndarray, n_bins: int = 10
) -> list[dict[str, float | int]]:
    """Predicted vs actual rate by decile of ``p`` (fewer bins when ``p`` has ties)."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    if len(p) == 0:
        return []
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        return [{"bin": 0, "n": len(p), "p_mean": float(p.mean()), "actual": float(y.mean())}]
    idx = np.clip(np.searchsorted(edges, p, side="left") - 1, 0, len(edges) - 2)
    rows: list[dict[str, float | int]] = []
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        rows.append(
            {
                "bin": int(b),
                "n": int(m.sum()),
                "p_mean": round(float(p[m].mean()), 6),
                "actual": round(float(y[m].mean()), 6),
            }
        )
    return rows


def fit_z_head(
    corr_cfg: CorruptionConfig,
    model_cfg: ModelConfig,
    X: pd.DataFrame,
    y: np.ndarray,
    zt: ZTargets,
    seed: int,
    name: str = "sched_default",
) -> TypeHead:
    """The schedule-default head as a weighted regression of ``z`` instead of a classifier of
    the label.

    Squared-error objective on ``zt.z``, ``sample_weight = zt.w = (C - yhat)²`` (zero below
    ``z_min_gap_s``), early stopping on the **weighted** squared error of the same internal,
    label-stratified holdout the classifier uses — never on AUC, which is the metric that
    made ``exp0057`` look like an improvement. The prediction is clipped to ``p_clip`` at
    :meth:`TypeHead.predict`, so the mixture stays a sub-convex combination.

    Degenerate case (too few rows carrying weight): the weighted mean of ``z``, which is the
    RMSE-optimal constant for this branch, clipped the same way.
    """
    lo, hi = float(corr_cfg.p_clip[0]), float(corr_cfg.p_clip[1])
    z, w = np.asarray(zt.z, dtype=np.float64), np.asarray(zt.w, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n_pos = int(y.sum())
    total = float(w.sum())
    fallback = float(np.clip((w * z).sum() / total, lo, hi)) if total > 0 else lo
    hold = _stratified_holdout(y, corr_cfg.holdout_frac, seed)
    if int((w > 0).sum()) < corr_cfg.min_positives or not hold.any() or hold.all():
        return TypeHead(name, fallback, n_pos, kind="z_regression", clip=(lo, hi))
    model = make_model(z_head_config(corr_cfg, model_cfg), seed)
    w_val = w[hold] if w[hold].sum() > 0 else None
    model.fit(X[~hold], z[~hold], X[hold], z[hold], w[~hold], w_val)
    p_hold = np.clip(np.asarray(model.predict(X[hold]), dtype=np.float64), lo, hi)
    return TypeHead(
        name,
        fallback,
        n_pos,
        model,
        None,  # AUC is not what this head optimises; the OOF AUC is still reported by the run
        {},
        kind="z_regression",
        clip=(lo, hi),
        holdout_wz=weighted_z_loss(p_hold, z[hold], w[hold]),
    )


def train_heads(
    corr_cfg: CorruptionConfig,
    model_cfg: ModelConfig,
    X: pd.DataFrame,
    labels: dict[str, np.ndarray],
    seed: int,
    z: ZTargets | None = None,
) -> dict[str, TypeHead]:
    """Fit one head per configured type on the rows of ``X`` (a fold's training rows, or all
    training rows for the test-time head).

    Every type is a classifier of its own 0/1 label except ``sched_default`` when
    ``corr_cfg.sched_head == "z_regression"`` *and* ``z`` is given, which is then
    :func:`fit_z_head`. Callers that want probabilities whatever the configuration — the
    nested pass behind ``normal_weight: one_minus_p``, whose output is a weight ``1 - Σ p``
    and therefore has to be one — simply pass no ``z``.
    """
    clf_cfg = classifier_config(corr_cfg, model_cfg)
    heads: dict[str, TypeHead] = {}
    for name in corr_cfg.types:
        y = np.asarray(labels[name], dtype=np.float64)
        n_pos = int(y.sum())
        base_rate = float(y.mean()) if len(y) else 0.0
        if name == "sched_default" and corr_cfg.sched_head == "z_regression" and z is not None:
            heads[name] = fit_z_head(corr_cfg, model_cfg, X, y, z, seed, name)
            continue
        if n_pos < corr_cfg.min_positives or n_pos > len(y) - corr_cfg.min_positives:
            heads[name] = TypeHead(name, base_rate, n_pos)
            continue
        hold = _stratified_holdout(y, corr_cfg.holdout_frac, seed)
        model = make_model(clf_cfg, seed)
        model.fit(X[~hold], y[~hold], X[hold], y[hold])
        auc = None
        p_hold = None
        if hold.sum() and 0 < y[hold].sum() < hold.sum():
            from sklearn.metrics import roc_auc_score

            p_hold = np.clip(np.asarray(model.predict(X[hold]), dtype=np.float64), 0.0, 1.0)
            auc = float(roc_auc_score(y[hold], p_hold))
        cals: dict[str, Any] = {}
        if corr_cfg.calibrate == "isotonic" and p_hold is not None:
            cals = fit_calibrators(
                p_hold,
                y[hold],
                _airports(cast(pd.DataFrame, X[hold])),
                corr_cfg.calibrate_min_positives,
            )
        heads[name] = TypeHead(name, base_rate, n_pos, model, auc, cals)
    return heads


def predict_heads(heads: dict[str, TypeHead], X: pd.DataFrame) -> dict[str, np.ndarray]:
    return {name: head.predict(X) for name, head in heads.items()}


def auc_score(y: np.ndarray, p: np.ndarray) -> float | None:
    """ROC AUC, or None when the labels are single-class (AUC is undefined)."""
    y = np.asarray(y)
    if len(y) == 0 or y.min() == y.max():
        return None
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p))
