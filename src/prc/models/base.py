"""Thin, uniform wrappers over GBDT libraries.

Every model: ``fit(X, y, X_val, y_val, w=None) -> self``; ``predict(X) -> np.ndarray``;
``importance() -> pd.Series | None``. Inputs are pandas frames (categoricals as
``category`` dtype) so all three libraries consume them natively.

``X_val=None`` (2026-09-20) is the **full-fit** path (``test_fit: full``): no validation set,
no early stopping, exactly ``cfg.num_boost_round`` rounds, and ``best_iteration`` set to the
last tree so ``predict`` uses all of them. Only ``prc.eval.fitpool.fit_full`` passes it, so no
fold model can reach that branch.

``w`` is an optional per-row training weight (``mixture.corruption.normal_weight``) and
``w_val`` the same for the early-stopping set. ``w_val`` defaults to None and the regressor
never passes it, so early stopping keeps measuring the unweighted RMSE we are actually scored
on. Its one consumer is the z-regression corruption head
(``mixture.corruption.sched_head: z_regression``), whose loss *is* a weighted squared error:
its holdout must carry the same ``(C - yhat)²`` weights or early stopping would optimise a
different objective from the fit.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pandas as pd

from prc.config import ModelConfig


class Model(Protocol):
    """``key`` (2026-09-19) is an opaque per-(fold, group) string. With a
    :class:`~prc.models.matrices.MatrixCache` attached, the wrapper caches its built booster
    matrices under it so every seed of a fold reuses them; with no cache, or ``key=None``,
    nothing changes. It never reaches the library and can never move a number."""

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        X_val: pd.DataFrame | None = None,
        y_val: np.ndarray | None = None,
        w: np.ndarray | None = None,
        w_val: np.ndarray | None = None,
        key: str | None = None,
    ) -> Model: ...
    def predict(self, X: pd.DataFrame, key: str | None = None) -> np.ndarray: ...
    def importance(self) -> pd.Series | None: ...

    best_iteration: int | None


class ConstantModel:
    """Baselines: global mean / median of the training target."""

    best_iteration = None

    def __init__(self, stat: str) -> None:
        self.stat = stat
        self.value = 0.0

    def fit(self, X, y, X_val=None, y_val=None, w=None, w_val=None, key=None):
        self.value = float(np.mean(y) if self.stat == "mean" else np.median(y))
        return self

    def predict(self, X, key=None):
        return np.full(len(X), self.value)

    def importance(self):
        return None


class ProxyModel:
    """Predict a single feature column verbatim (e.g. ``tko_minus_aobt3`` = the Network
    Manager naive taxi time), falling back to the training mean where it is null or
    outside [lo, hi]. The day-one calibration submission and the bar every model must beat."""

    best_iteration = None

    def __init__(self, column: str, lo: float = 0.0, hi: float = 14400.0) -> None:
        self.column, self.lo, self.hi = column, lo, hi
        self.fallback = 0.0

    def fit(self, X, y, X_val=None, y_val=None, w=None, w_val=None, key=None):
        self.fallback = float(np.mean(y))
        return self

    def predict(self, X, key=None):
        v = X[self.column].to_numpy(dtype=float)
        ok = np.isfinite(v) & (v >= self.lo) & (v <= self.hi)
        return np.where(ok, v, self.fallback)

    def importance(self):
        return None


#: Has THIS process ever built a gradient-boosting booster?
#:
#: Measured 2026-09-19: a process that has run one XGBoost fit **deadlocks if it later
#: forks and the child calls into the booster** — the child inherits an OpenMP runtime it
#: cannot re-enter. `nthread=1` in the parent does not help, `nthread=1` in the child does
#: not help; only never having trained in the parent does (or `spawn`/`forkserver`, which
#: would have to pickle gigabytes of design matrix per task). `prc.eval.fitpool` reads this
#: flag before forking a fit pool, and `prc.eval.run` runs the corruption head in an
#: isolated child when parallel fits are on, so that the parent stays clean.
_BOOSTER_BUILT = False


def booster_built_here() -> bool:
    """True once this process has constructed a GBDT wrapper (and therefore will train)."""
    return _BOOSTER_BUILT


def make_model(
    cfg: ModelConfig,
    seed: int,
    matrices: Any | None = None,
    device: str | None = None,
    n_threads: int | None = None,
) -> Any:
    """Build a wrapper. The three trailing arguments are performance-only
    (:class:`prc.config.RuntimeConfig`) and every one of them defaults to the behaviour that
    existed before 2026-09-19: no matrix cache, the config's own device, the library's own
    thread count. The constant baselines ignore all three."""
    if cfg.kind == "proxy":
        return ProxyModel(**cfg.params)
    if cfg.kind in (
        "lgbm", "xgb", "cat", "tabm", "setenc", "realmlp", "modernnca", "seqenc"
    ):  # fmt: skip
        # tabm / setenc count too: a process that has initialised CUDA through torch cannot fork a
        # child that uses CUDA either, so the fit pool must stay sequential after it.
        global _BOOSTER_BUILT
        _BOOSTER_BUILT = True
    if cfg.kind == "lgbm":
        from prc.models.lgbm import LGBMModel

        return LGBMModel(cfg, seed, matrices, device, n_threads)
    if cfg.kind == "xgb":
        from prc.models.xgb import XGBModel

        return XGBModel(cfg, seed, matrices, device, n_threads)
    if cfg.kind == "cat":
        from prc.models.cat import CatModel

        return CatModel(cfg, seed, matrices, device, n_threads)
    if cfg.kind == "tabm":
        from prc.models.tabm import TabMModel

        return TabMModel(cfg, seed, matrices, device, n_threads)
    if cfg.kind == "realmlp":
        from prc.models.realmlp import RealMLPModel

        return RealMLPModel(cfg, seed, matrices, device, n_threads)
    if cfg.kind == "modernnca":
        from prc.models.modernnca import ModernNCAModel

        return ModernNCAModel(cfg, seed, matrices, device, n_threads)
    if cfg.kind == "setenc":
        from prc.models.setenc import SetEncModel

        return SetEncModel(cfg, seed, matrices, device, n_threads)
    if cfg.kind == "seqenc":
        from prc.models.seqenc import SeqEncModel

        return SeqEncModel(cfg, seed, matrices, device, n_threads)
    if cfg.kind in ("mean", "median"):
        return ConstantModel(cfg.kind)
    raise ValueError(cfg.kind)
