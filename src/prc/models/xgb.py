from __future__ import annotations

import time

import numpy as np
import pandas as pd
import xgboost as xgb

from prc.config import ModelConfig
from prc.models.matrices import MatrixCache

DEFAULTS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.05,
    "max_depth": 8,
    "min_child_weight": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "max_cat_to_onehot": 1,
    "max_bin": 256,
}


RAW_METRIC_NAME = "rmse_raw"


def raw_rmse(predt: np.ndarray, dmat: xgb.DMatrix) -> tuple[str, float]:
    """``model.eval_metric_space: raw`` — RMSE in seconds for a sqrt member: prediction and
    label are squared (the inverse of ``prc.eval.transform``'s ``sqrt``) before the error.

    Weighted exactly like XGBoost's built-in ``rmse``: ``sqrt(Σ w·e² / Σ w)`` when the matrix
    carries weights, the plain mean otherwise. The runner hands the validation matrix no
    weights (``fitpool.fit_one`` passes no ``w_val``), so on the fold path this is the
    unweighted RMSE over the held-out month, as the built-in metric is today. Labels squared
    are ``max(y, 0)``: the transform clipped the 369 negative labels, and that is not undone.
    """
    y = np.square(dmat.get_label().astype(np.float64))
    p = np.square(np.asarray(predt, dtype=np.float64))
    w = dmat.get_weight()
    se = (p - y) ** 2
    if w.size:
        return RAW_METRIC_NAME, float(np.sqrt(np.sum(w * se) / np.sum(w)))
    return RAW_METRIC_NAME, float(np.sqrt(np.mean(se)))


#: log-link objectives whose XGBoost deviance metric is undefined below 0 (see ``fit``)
DEVIANCE_NEEDS_CLIPPED_LABELS = frozenset({"count:poisson", "reg:gamma"})


def log_link_deviance(params: dict) -> str:
    """XGBoost's own negative log-likelihood metric for a log-link objective."""
    obj = params["objective"]
    if obj == "reg:tweedie":
        return f"tweedie-nloglik@{params.get('tweedie_variance_power', 1.5)}"
    return {"count:poisson": "poisson-nloglik", "reg:gamma": "gamma-nloglik"}[obj]


class XGBModel:
    """XGBoost booster wrapper.

    Three optional, result-neutral performance arguments (2026-09-19, all defaulting to the
    previous behaviour):

    ``matrices``   a :class:`~prc.models.matrices.MatrixCache` shared across the seeds of one
                   fold. With a ``key``, the train / val / predict ``DMatrix`` objects are
                   built once and reused instead of once per seed.
    ``device``     overrides ``cfg.device``, so a parallel fit worker can be pinned to
                   ``cuda:1`` while the config still says ``cuda``.
    ``n_threads``  ``nthread``. On ``tree_method=hist`` the split search is deterministic in
                   the thread count, so this is a speed knob, not a result knob — measured,
                   not assumed.

    ``timings`` accumulates ``matrix`` / ``train`` / ``predict`` seconds for the run's
    ``metrics.json``; :meth:`prc.eval.timing.Timings.absorb` drains it.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        seed: int,
        matrices: MatrixCache | None = None,
        device: str | None = None,
        n_threads: int | None = None,
    ) -> None:
        self.cfg = cfg
        self.params = {**DEFAULTS, **cfg.params, "seed": seed, "device": device or cfg.device}
        if n_threads is not None:
            self.params["nthread"] = int(n_threads)
        self.booster: xgb.Booster | None = None
        self.best_iteration: int | None = None
        self._cats: dict[str, pd.Index] = {}
        self.matrices = matrices
        self.timings: dict[str, float] = {}

    def _t(self, name: str, seconds: float) -> None:
        self.timings[name] = self.timings.get(name, 0.0) + seconds

    def _align(self, X: pd.DataFrame) -> pd.DataFrame:
        """XGBoost >= 3 refuses categories unseen in training; map them to NaN (the same
        thing LightGBM does implicitly) by pinning every categorical to the training levels."""
        X = X.copy()
        for c, cats in self._cats.items():
            X[c] = pd.Categorical(X[c].astype(object), categories=cats)
        return X

    def _cache(self, key: str | None, build) -> object:
        if self.matrices is None:
            t0 = time.perf_counter()
            obj = build()
            self._t("matrix", time.perf_counter() - t0)
            return obj
        obj, dt = self.matrices.get_or_build(key, build)
        self._t("matrix", dt)
        return obj

    def fit(self, X, y, X_val=None, y_val=None, w=None, w_val=None, key=None):
        self._cats = {
            c: X[c].cat.categories for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)
        }
        if self.cfg.label_clip_min is not None:
            # P3a: the training labels only (Poisson / Tweedie reject y < 0); the
            # early-stopping labels stay as they are, so the stopping metric is the real one
            y = np.maximum(np.asarray(y, dtype=np.float64), self.cfg.label_clip_min)
        dtrain = self._cache(
            None if key is None else f"{key}:train",
            lambda: xgb.DMatrix(X, y, weight=w, enable_categorical=True),
        )
        if X_val is None:
            # `test_fit: full` (2026-09-20): no validation set, no early stopping, a fixed
            # number of rounds. Nothing else in the runner passes X_val=None, so this branch
            # cannot move a fold model.
            t0 = time.perf_counter()
            self.booster = xgb.train(
                self.params, dtrain, num_boost_round=self.cfg.num_boost_round, verbose_eval=200
            )
            self._t("train", time.perf_counter() - t0)
            self.best_iteration = self.cfg.num_boost_round - 1  # every tree is used
            return self
        if (
            self.cfg.label_clip_min is not None
            and self.cfg.eval_metric_space == "transformed"
            and "eval_metric" not in self.cfg.params
            and self.cfg.params.get("objective") in DEVIANCE_NEEDS_CLIPPED_LABELS
        ):
            # P3a: `poisson-nloglik` carries lgamma(y + 1) and `gamma-nloglik` log(y), both
            # infinite at the scored month's negative labels (smoke: `inf` from round 0, so
            # early stopping kept 0 trees). Their stopping labels get the training clip too;
            # Tweedie's deviance is finite there and keeps the real labels.
            y_val = np.maximum(np.asarray(y_val, dtype=np.float64), self.cfg.label_clip_min)
        dval = self._cache(
            None if key is None else f"{key}:val",
            lambda: xgb.DMatrix(self._align(X_val), y_val, weight=w_val, enable_categorical=True),
        )
        t0 = time.perf_counter()
        extra = {}
        params = self.params
        if self.cfg.log_link() and self.cfg.eval_metric_space == "transformed":
            # P3a: for a log-link objective the "transformed" space is the loss's own — its
            # deviance on the validation rows — unless the YAML names a metric. The wrapper's
            # DEFAULTS `rmse` would otherwise be the seconds RMSE, which the scored month's
            # schedule-default rows dominate (smoke: stops ~10x earlier, see the eng. note)
            if "eval_metric" not in self.cfg.params:
                params = {**params, "eval_metric": log_link_deviance(self.cfg.params)}
        elif self.cfg.eval_metric_space == "raw" and self.cfg.log_link():
            # P3a: a log-link objective predicts seconds and XGBoost's built-in `rmse` is
            # computed on the response scale (exp of the margin), so the raw RMSE is the
            # built-in metric; making it the only one makes it the one early stopping watches
            params = {**params, "eval_metric": "rmse"}
        elif self.cfg.eval_metric_space == "raw":
            # the custom metric is the only one evaluated, so it is the one early stopping
            # watches (xgb.train stops on the last metric of the last eval set)
            params = {k: v for k, v in params.items() if k != "eval_metric"}
            params["disable_default_eval_metric"] = 1
            extra["custom_metric"] = raw_rmse
        self.booster = xgb.train(
            params,
            dtrain,
            num_boost_round=self.cfg.num_boost_round,
            evals=[(dval, "val")],
            early_stopping_rounds=self.cfg.early_stopping_rounds,
            verbose_eval=200,
            **extra,
        )
        self._t("train", time.perf_counter() - t0)
        self.best_iteration = self.booster.best_iteration
        return self

    def predict(self, X, key=None):
        assert self.booster is not None
        d = self._cache(
            None if key is None else f"{key}:predict",
            lambda: xgb.DMatrix(self._align(X), enable_categorical=True),
        )
        t0 = time.perf_counter()
        out = self.booster.predict(d, iteration_range=(0, (self.best_iteration or 0) + 1))
        self._t("predict", time.perf_counter() - t0)
        return out

    def importance(self):
        assert self.booster is not None
        return pd.Series(self.booster.get_score(importance_type="total_gain")).sort_values(
            ascending=False
        )
