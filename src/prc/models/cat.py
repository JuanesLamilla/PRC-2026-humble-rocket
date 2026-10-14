from __future__ import annotations

import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from prc.config import ModelConfig
from prc.models.matrices import MatrixCache

DEFAULTS = {
    "loss_function": "RMSE",
    "learning_rate": 0.08,
    "depth": 8,
    "l2_leaf_reg": 3.0,
    "border_count": 254,
    "verbose": 200,
}


class RawRMSE:
    """``model.eval_metric_space: raw`` for CatBoost — RMSE in seconds for a sqrt member,
    prediction and label squared before the error; weighted like CatBoost's own RMSE
    (``sqrt(Σ w·e² / Σ w)``; ``weight`` is ``None`` when the pool has none, which is the fold
    path's validation pool). A user-defined metric runs on the CPU even when training runs on
    the GPU (CatBoost 1.2 accepts it and warns that it defaults ``metric_period`` to 5;
    :class:`CatModel` pins ``metric_period: 1`` so the overfitting detector sees every
    iteration, as it does with the built-in metric).

    ``link="log"`` (P3a, 2026-09-23) is the same metric for a log-link loss (``Poisson``) on
    labels in seconds: the approx is log E[y|x], so the prediction is ``exp(approx)`` and the
    label is used as it is. CatBoost's built-in ``RMSE`` eval metric does *not* do this for a
    Poisson loss — it compares the raw log-scale score with the label (measured: 562.7 against
    a true 172.8 on a toy problem), which is why the custom metric is needed."""

    def __init__(self, link: str = "sqrt") -> None:
        if link not in ("sqrt", "log"):
            raise ValueError(f"RawRMSE link must be 'sqrt' or 'log', got {link!r}")
        self.link = link

    def is_max_optimal(self) -> bool:
        return False

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        a = np.asarray(approxes[0], dtype=np.float64)
        t = np.asarray(target, dtype=np.float64)
        if self.link == "log":
            p, y = np.exp(a), t
        else:
            p, y = np.square(a), np.square(t)
        se = (p - y) ** 2
        if weight is None:
            return float(np.sum(se)), float(se.size)
        w = np.asarray(weight, dtype=np.float64)
        return float(np.sum(w * se)), float(np.sum(w))

    def get_final_error(self, error: float, weight: float) -> float:
        return float(np.sqrt(error / weight)) if weight else 0.0


class CatModel:
    """CatBoost wrapper. Same three result-neutral performance arguments as
    :class:`prc.models.xgb.XGBModel`; here the reused object is the ``Pool`` (which owns the
    ``_prep`` copy and the categorical encoding, i.e. most of the per-fit setup cost).

    ``device`` accepts ``cuda`` / ``cuda:<i>``; CatBoost takes the index separately as
    ``devices``, so ``cuda:1`` becomes ``task_type=GPU, devices="1"``.
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
        dev = device or cfg.device
        self.params = {
            **DEFAULTS,
            **cfg.params,
            "random_seed": seed,
            "iterations": cfg.num_boost_round,
            "od_type": "Iter",
            "od_wait": cfg.early_stopping_rounds,
            "task_type": "GPU" if dev.startswith("cuda") else "CPU",
        }
        if dev.startswith("cuda:"):
            self.params["devices"] = dev.split(":", 1)[1]
        if n_threads is not None:
            self.params["thread_count"] = int(n_threads)
        if cfg.eval_metric_space == "raw":
            self.params["eval_metric"] = RawRMSE("log" if cfg.log_link() else "sqrt")
            self.params.setdefault("metric_period", 1)
        self.model: CatBoostRegressor | None = None
        self.best_iteration: int | None = None
        self.cat_cols: list[str] = []
        self.matrices = matrices
        self.timings: dict[str, float] = {}

    def _t(self, name: str, seconds: float) -> None:
        self.timings[name] = self.timings.get(name, 0.0) + seconds

    def _cache(self, key: str | None, build):
        if self.matrices is None:
            t0 = time.perf_counter()
            obj = build()
            self._t("matrix", time.perf_counter() - t0)
            return obj
        obj, dt = self.matrices.get_or_build(key, build)
        self._t("matrix", dt)
        return obj

    @staticmethod
    def _prep(X: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
        X = X.copy()
        for c in cat_cols:
            # pandas >= 3 keeps missing values through astype(str); CatBoost rejects them, so
            # fill explicitly and hand over plain object strings.
            col = X[c].astype(object)
            X[c] = col.where(col.notna(), "NA").astype(object)
        return X

    def fit(self, X, y, X_val=None, y_val=None, w=None, w_val=None, key=None):
        self.cat_cols = [c for c in X.columns if str(X[c].dtype) == "category"]
        cats = self.cat_cols
        if self.cfg.label_clip_min is not None:
            # P3a: the training labels only, as in the xgb wrapper
            y = np.maximum(np.asarray(y, dtype=np.float64), self.cfg.label_clip_min)
        train = self._cache(
            None if key is None else f"{key}:train",
            lambda: Pool(self._prep(X, cats), y, weight=w, cat_features=cats),
        )
        if X_val is None:
            # `test_fit: full` (2026-09-20): no eval set, so the overfitting detector has
            # nothing to watch and is dropped along with `use_best_model`.
            # `eval_metric_space: raw` goes too: with no eval set its metric would only be
            # evaluated (in Python, every iteration) on the learn set, and it steers nothing.
            drop = ("od_type", "od_wait")
            if self.cfg.eval_metric_space == "raw":
                drop += ("eval_metric", "metric_period")
            params = {k: v for k, v in self.params.items() if k not in drop}
            self.model = CatBoostRegressor(**params)
            t0 = time.perf_counter()
            self.model.fit(train)
            self._t("train", time.perf_counter() - t0)
            self.best_iteration = int(self.model.tree_count_) - 1
            return self
        val = self._cache(
            None if key is None else f"{key}:val",
            lambda: Pool(self._prep(X_val, cats), y_val, weight=w_val, cat_features=cats),
        )
        self.model = CatBoostRegressor(**self.params)
        t0 = time.perf_counter()
        self.model.fit(train, eval_set=val, use_best_model=True)
        self._t("train", time.perf_counter() - t0)
        self.best_iteration = int(self.model.get_best_iteration())
        return self

    def predict(self, X, key=None):
        assert self.model is not None
        cats = self.cat_cols
        pool = self._cache(
            None if key is None else f"{key}:predict",
            lambda: Pool(self._prep(X, cats), cat_features=cats),
        )
        t0 = time.perf_counter()
        out = self.model.predict(pool)
        self._t("predict", time.perf_counter() - t0)
        return out

    def importance(self):
        assert self.model is not None
        return pd.Series(
            self.model.get_feature_importance(), index=self.model.feature_names_
        ).sort_values(ascending=False)
