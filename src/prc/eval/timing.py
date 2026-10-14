"""Per-phase wall-clock instrumentation for one run.

Always on. A phase is a ``time.perf_counter()`` pair and two dict updates, so the whole
mechanism costs microseconds against phases that cost seconds — there is no "enable
profiling" flag to forget to set, and therefore no run whose timings are missing when the
question finally gets asked.

The contract:

* :meth:`Timings.phase` is a context manager; nested phases are allowed and each one
  accumulates on its own key (they are *not* subtracted from one another, so a parent
  phase's seconds include its children's — read ``phases`` as a flat accumulation table,
  not a tree).
* :meth:`Timings.absorb` folds a model wrapper's own sub-timings (matrix build vs train vs
  predict — :mod:`prc.models.xgb`, :mod:`prc.models.cat`) in under a prefix, so the runner
  gets the split without every call site knowing what a DMatrix is.
* :meth:`Timings.as_dict` is what lands in ``metrics.json`` under ``timings`` and what
  :meth:`Timings.summary` prints into ``log.txt`` at the end of the run.

Timings never feed a prediction, a metric or a cache key. They are observation only.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager

#: Phases printed first in :meth:`Timings.summary`, in the order a run performs them.
#: Anything not listed follows, sorted by cost.
ORDER: tuple[str, ...] = (
    "load_movements",
    "build_features",
    "join_features",
    "mask",
    "folds",
    "cache_key",
    "cache_read",
    "cache_write",
    "to_pandas",
    "te_test",
    "te_fold",
    "head_nested_weights",
    "head_nested_yhat",
    "head_classifiers",
    "head_predict",
    "head_total",
    "fit_matrix",
    "fit_train",
    "fit_predict",
    "fit_total",
    "test_predict",
    "metrics",
    "write_artifacts",
)


class Timings:
    """Accumulating wall-clock table: phase name -> seconds, and how many times it ran."""

    __slots__ = ("counts", "seconds")

    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def add(self, name: str, seconds: float, n: int = 1) -> None:
        self.seconds[name] = self.seconds.get(name, 0.0) + float(seconds)
        self.counts[name] = self.counts.get(name, 0) + int(n)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - t0)

    def absorb(self, model: object, prefix: str = "fit") -> None:
        """Fold a model wrapper's ``timings`` dict in under ``prefix`` and reset it.

        ``{"matrix": 1.2, "train": 40.1, "predict": 0.4}`` becomes
        ``fit_matrix`` / ``fit_train`` / ``fit_predict``. A model that does not expose
        ``timings`` (the constant baselines) is silently skipped.
        """
        t = getattr(model, "timings", None)
        if not isinstance(t, dict):
            return
        for k, v in t.items():
            self.add(f"{prefix}_{k}", float(v))
        t.clear()

    def merge(self, other: Mapping[str, float], counts: Mapping[str, int] | None = None) -> None:
        """Fold another table in — how a parallel fit worker's timings come home."""
        for k, v in other.items():
            self.seconds[k] = self.seconds.get(k, 0.0) + float(v)
        for k, n in (counts or {}).items():
            self.counts[k] = self.counts.get(k, 0) + int(n)

    def as_dict(self) -> dict:
        return {
            "phases_s": {k: round(v, 3) for k, v in self._ordered()},
            "counts": {k: self.counts[k] for k, _ in self._ordered() if k in self.counts},
        }

    def _ordered(self) -> list[tuple[str, float]]:
        known = [(k, self.seconds[k]) for k in ORDER if k in self.seconds]
        rest = sorted(
            ((k, v) for k, v in self.seconds.items() if k not in ORDER),
            key=lambda kv: -kv[1],
        )
        return known + rest

    def summary(self, total_s: float | None = None) -> str:
        parts = [
            f"{k}={v:.1f}s" + (f"x{self.counts[k]}" if self.counts.get(k, 1) > 1 else "")
            for k, v in self._ordered()
        ]
        head = "timings"
        if total_s is not None:
            accounted = sum(self.seconds.get(k, 0.0) for k in ("head_total", "fit_total"))
            head = f"timings total={total_s:.1f}s (head+fit {accounted:.1f}s)"
        return head + ": " + " ".join(parts)
