from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from prc.config import ModelConfig
from prc.models.matrices import MatrixCache

DEFAULTS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 0,
}


class LGBMModel:
    """LightGBM wrapper.

    Takes the same result-neutral performance arguments as :class:`prc.models.xgb.XGBModel`
    for a uniform :func:`prc.models.base.make_model` signature. ``matrices`` is accepted and
    ignored: a ``lgb.Dataset`` is bound to the booster that consumed it and cannot be shared
    across fits the way a ``DMatrix`` can, so there is nothing safe to reuse here. ``device``
    and ``n_threads`` are honoured, and ``timings`` is filled the same way.
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
        self.params = {**DEFAULTS, **cfg.params, "seed": seed}
        dev = device or cfg.device
        if dev.startswith("cuda"):
            self.params["device_type"] = "cuda"
            if ":" in dev:
                self.params["gpu_device_id"] = int(dev.split(":", 1)[1])
        if n_threads is not None:
            self.params["num_threads"] = int(n_threads)
        self.booster: lgb.Booster | None = None
        self.best_iteration: int | None = None
        self.matrices = matrices  # accepted for signature parity; see the class docstring
        self.timings: dict[str, float] = {}

    def _t(self, name: str, seconds: float) -> None:
        self.timings[name] = self.timings.get(name, 0.0) + seconds

    def fit(self, X, y, X_val=None, y_val=None, w=None, w_val=None, key=None):
        t0 = time.perf_counter()
        dtrain = lgb.Dataset(X, y, weight=w, free_raw_data=False)
        if X_val is None:
            # `test_fit: full` (2026-09-20): no validation set, no early stopping.
            self._t("matrix", time.perf_counter() - t0)
            t0 = time.perf_counter()
            self.booster = lgb.train(
                self.params,
                dtrain,
                num_boost_round=self.cfg.num_boost_round,
                callbacks=[lgb.log_evaluation(200)],
            )
            self._t("train", time.perf_counter() - t0)
            self.best_iteration = self.cfg.num_boost_round  # every tree is used
            return self
        dval = lgb.Dataset(X_val, y_val, weight=w_val, reference=dtrain, free_raw_data=False)
        self._t("matrix", time.perf_counter() - t0)
        t0 = time.perf_counter()
        self.booster = lgb.train(
            self.params,
            dtrain,
            num_boost_round=self.cfg.num_boost_round,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(self.cfg.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(200),
            ],
        )
        self._t("train", time.perf_counter() - t0)
        self.best_iteration = self.booster.best_iteration
        return self

    def predict(self, X, key=None):
        assert self.booster is not None
        t0 = time.perf_counter()
        out = self.booster.predict(X, num_iteration=self.best_iteration)
        self._t("predict", time.perf_counter() - t0)
        return out

    def importance(self):
        assert self.booster is not None
        return pd.Series(
            self.booster.feature_importance("gain"), index=self.booster.feature_name()
        ).sort_values(ascending=False)


__all__ = ["LGBMModel", "np"]
