"""Helpers shared by the Phase 1 screen blocks (``taxi_physics``, ``domain_interactions``,
``weather_process``, ``stamp_anomaly``, ``hour_state``). No block is registered here.

``deps_digest``: a block that calls other blocks (through :func:`prc.features.registry.build`)
must change its cache key when one of them changes. The registry hashes only the block's own
module, so these blocks pass ``external=`` a digest of their own module, the files they read,
and the source + external hashes of every block they depend on.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

import polars as pl

from prc.data import schema as S
from prc.features.daylight_curfew import AIRPORT_TZ
from prc.features.registry import REGISTRY, external_digest


def deps_digest(module: str, deps: Iterable[str], files: Iterable[Path] = ()) -> str:
    h = hashlib.sha1(external_digest(list(files), module).encode())
    for n in deps:
        b = REGISTRY[n]
        h.update(f"{n}:{b.source_hash()}:{b.external_hash()}".encode())
    return h.hexdigest()[:8]


def secs(a: str, b: str) -> pl.Expr:
    """``a − b`` in seconds (Float64; null when either is null)."""
    return (pl.col(a) - pl.col(b)).dt.total_seconds().cast(pl.Float64)


def local_hour(col: str = S.MVT_TIME) -> pl.Expr:
    """Local clock hour of ``col`` at the row's airport (null for an unknown airport)."""
    e: pl.Expr | None = None
    for ap, tz in AIRPORT_TZ.items():
        v = pl.col(col).dt.convert_time_zone(tz).dt.hour()
        cond = pl.col(S.AIRPORT) == ap
        e = pl.when(cond).then(v) if e is None else e.when(cond).then(v)
    assert e is not None
    return e.otherwise(None).cast(pl.Int8)


def safe_div(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    """``num / den`` with a null result where ``den`` is null or zero."""
    return pl.when(den.is_null() | (den == 0)).then(None).otherwise(num / den)
