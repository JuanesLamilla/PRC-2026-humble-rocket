"""Reuse of built booster matrices across the seeds of one fold.

The 2026-09-18 baseline measurement of ``exp0047`` (3 seeds x 2 folds on one L40S) spent
its 361 s on six sequential GPU fits — and each of those six rebuilt the *same three*
matrices, because ``XGBModel.fit`` constructs its ``DMatrix`` inside ``fit`` and a new
``XGBModel`` is made per (fold, seed). The pandas design matrix was already built once per
fold; the DMatrix was not.

A :class:`MatrixCache` is a plain dict the runner hands to every model of one fold, keyed
by ``fold/group/role``, and cleared when the fold ends so only one fold's matrices are
resident. It changes nothing about *what* is built: the same ``xgb.DMatrix(X, y, weight=w,
enable_categorical=True)`` call, from the same frame — the fold's training rows and weights
do not depend on the seed (``w_fit`` is computed outside the seed loop in
``prc.eval.run``), so the object the second seed would have built is bit-for-bit the object
the first seed built.

Deliberately **not** ``QuantileDMatrix``: that pre-bins on a different code path from
``DMatrix`` + ``hist`` and is a *result-changing* candidate, which belongs in an experiment,
not in a performance switch.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any


class MatrixCache:
    """One fold's booster matrices, addressed by an opaque string key."""

    __slots__ = ("_d", "build_s", "hits", "misses")

    def __init__(self) -> None:
        self._d: dict[str, Any] = {}
        self.hits = 0
        self.misses = 0
        self.build_s = 0.0

    def get_or_build(self, key: str | None, build: Callable[[], Any]) -> tuple[Any, float]:
        """``(matrix, seconds spent building it)``; ``key=None`` bypasses the cache entirely,
        which is exactly what the default (switch-off) path does."""
        if key is not None and key in self._d:
            self.hits += 1
            return self._d[key], 0.0
        t0 = time.perf_counter()
        obj = build()
        dt = time.perf_counter() - t0
        self.build_s += dt
        self.misses += 1
        if key is not None:
            self._d[key] = obj
        return obj, dt

    def clear(self) -> None:
        self._d.clear()

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses, "build_s": round(self.build_s, 3)}

    def __len__(self) -> int:
        return len(self._d)
