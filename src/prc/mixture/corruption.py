"""Corruption types, their labels, their corrupt values and the RMSE-optimal mixture
(plan M3 / approach-ideas A2).

Three label-corruption processes are visible in the training targets
(internal note 2026-09-17_literature-analysis §2.1):

=================  ==========================================  ==================================
type               label (train rows only, derived from y)      corrupt value `C` (computable at test)
=================  ==========================================  ==================================
``sched_default``  ``|BLOCK − SCHED| <= 10 s`` — the airport     ``MVT − SCHED`` — *exactly* the label
                   wrote the schedule into the off-block stamp   the organizers then score
``rollover``       ``|y − 86400| < 3600`` — +1 day in the stamp  ``86400 + normal prediction``
``stamp_early``    ``y > 3600`` and neither of the above and     ``normal prediction + 3055 s``
                   an NM match exists                            (median airport-vs-NM gap)
=================  ==========================================  ==================================

Under RMSE the optimal prediction for a possibly-corrupted row is the conditional mean
``Σ_k p_k·C_k + (1 − Σ_k p_k)·ŷ_normal``, so the whole problem is estimating ``p``. The labels
are functions of the target, therefore every classifier that produces ``p`` must be fitted
inside the fold (see ``prc.mixture.head``); nothing in this module fits anything.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from prc.config import CorruptionConfig
from prc.data import schema as S

CORRUPTION_TYPES: tuple[str, ...] = ("sched_default", "rollover", "stamp_early")

LABEL_PREFIX = "lab_"
PROB_PREFIX = "p_"
SCHED_VALUE_COL = "corrupt_sched_default"


def corruption_labels(df: pl.DataFrame, cfg: CorruptionConfig) -> pl.DataFrame:
    """One row per ``MVT_ID_mvt`` with an Int8 0/1 column per corruption type.

    Reads the target and the off-block stamp, so this is a *training-row* quantity only and
    its consumers must keep it inside the fold.
    """
    blk_minus_sched = (pl.col(S.BLOCK_TIME) - pl.col(S.SCHED_TIME)).dt.total_seconds()
    y = pl.col(S.TARGET)
    sched_default = (blk_minus_sched.abs() <= cfg.sched_default_tol_s).fill_null(False)
    rollover = ((y - cfg.rollover_s).abs() < cfg.rollover_window_s).fill_null(False)
    stamp_early = (
        (y > cfg.stamp_early_min_s) & ~sched_default & ~rollover & pl.col(S.AOBT_3).is_not_null()
    ).fill_null(False)
    exprs = {
        "sched_default": sched_default,
        "rollover": rollover,
        "stamp_early": stamp_early,
    }
    return df.select(S.MVT_ID, *[exprs[t].cast(pl.Int8).alias(LABEL_PREFIX + t) for t in cfg.types])


def sched_default_value(df: pl.DataFrame) -> pl.DataFrame:
    """`MVT − SCHED` in seconds: the value the target takes when the airport substituted the
    schedule into the off-block stamp. Both columns exist on the ranking rows."""
    return df.select(
        S.MVT_ID,
        (pl.col(S.MVT_TIME) - pl.col(S.SCHED_TIME))
        .dt.total_seconds()
        .cast(pl.Float64)
        .alias(SCHED_VALUE_COL),
    )


def corrupt_values(
    y_hat: np.ndarray, sched_value: np.ndarray, cfg: CorruptionConfig
) -> dict[str, np.ndarray]:
    """The per-row corrupt value of each configured type, given the normal prediction."""
    y_hat = np.asarray(y_hat, dtype=np.float64)
    all_values = {
        "sched_default": np.where(np.isfinite(sched_value), sched_value, y_hat),
        "rollover": cfg.rollover_s + y_hat,
        "stamp_early": y_hat + cfg.stamp_early_offset_s,
    }
    return {t: all_values[t] for t in cfg.types}


def z_targets(
    y: np.ndarray,
    y_hat: np.ndarray,
    sched_value: np.ndarray,
    cfg: CorruptionConfig,
    winsorise: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """``(z, w)`` for the schedule-default branch: the RMSE-optimal mixture weight and its
    value weight.

    Holding the other branches fixed, the residual the schedule-default branch contributes is
    ``pred - y = (C - yhat)·(p - z)`` with ``z = (y - yhat)/(C - yhat)``, so the ``p`` that
    minimises the squared error is the conditional mean of ``z`` under weights
    ``w = (C - yhat)²`` — not the 0/1 label a classifier fits. ``z ~ 1`` on a schedule
    default (``y = C`` by construction), ``~ 1.26`` on a day rollover and ``~ -1.4`` on a
    clean short taxi, and ``Σ w·(p - z)² = Σ (pred - y)²`` exactly on the rows that carry
    weight, which is what makes the weighted loss the honest gate.

    Reads the target, so this is a *training-row* quantity: ``y_hat`` must be a prediction
    that never saw the row's own label (``prc.eval.run._nested_yhat``).

    Rows where ``|C - yhat| < cfg.z_min_gap_s`` (and rows with no usable ``C`` or ``yhat``)
    get weight 0 and ``z = 0``: there the ratio is numerically meaningless and the branch can
    move the prediction by at most a minute, so letting it set the head's target is how the
    ordinary lane would be paid for the corrupted one. ``w`` is normalised to mean 1 over the
    rows that carry any, which leaves every ratio (the fitted optimum, the weighted loss)
    unchanged and keeps ``min_child_weight`` / ``lambda`` on their usual scale.

    ``winsorise`` clips ``z`` to ``[cfg.z_clip_lo, cfg.z_clip_hi]``; the head trains on the
    winsorised target, the metrics read the raw one.
    """
    y = np.asarray(y, dtype=np.float64)
    y_hat = np.asarray(y_hat, dtype=np.float64)
    c = np.asarray(sched_value, dtype=np.float64)
    c = np.where(np.isfinite(c), c, y_hat)  # the substitution `corrupt_values` makes
    gap = c - y_hat
    ok = np.isfinite(gap) & np.isfinite(y) & (np.abs(gap) >= cfg.z_min_gap_s)
    z = np.where(ok, (y - y_hat) / np.where(ok, gap, 1.0), 0.0)
    if winsorise:
        z = np.clip(z, cfg.z_clip_lo, cfg.z_clip_hi)
    w = np.where(ok, gap**2, 0.0)
    scale = float(w[ok].mean()) if ok.any() else 0.0
    return z, w / scale if scale > 0 else w


def clip_probs(probs: dict[str, np.ndarray], p_max: float = 1.0) -> dict[str, np.ndarray]:
    """Clip each probability to [0, 1] and scale them down together where they sum above
    ``p_max``, so the mixture weights stay a (sub-)convex combination."""
    if not probs:
        return {}
    keys = list(probs)
    stack = np.clip(np.stack([np.nan_to_num(probs[k], nan=0.0) for k in keys]), 0.0, 1.0)
    total = stack.sum(axis=0)
    scale = np.where(total > p_max, p_max / np.maximum(total, 1e-12), 1.0)
    return {k: stack[i] * scale for i, k in enumerate(keys)}


def apply_mixture(
    y_hat: np.ndarray,
    probs: dict[str, np.ndarray],
    values: dict[str, np.ndarray],
    p_max: float = 1.0,
) -> np.ndarray:
    """``Σ_k p_k·C_k + (1 − Σ_k p_k)·ŷ_normal`` with the probabilities clipped by
    :func:`clip_probs`. Affine in ``y_hat``, so it is equivalent to apply it before or after
    averaging predictions over folds and seeds."""
    y_hat = np.asarray(y_hat, dtype=np.float64)
    if not probs:
        return y_hat.copy()
    p = clip_probs(probs, p_max)
    out = y_hat.copy()
    for k, pk in p.items():
        out = out + pk * (np.asarray(values[k], dtype=np.float64) - y_hat)
    return out
