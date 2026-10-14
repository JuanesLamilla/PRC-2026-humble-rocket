"""Context sets for the set-attention member (``model.kind: setenc``, plan N1, 2026-09-22).

For every departure ``i`` (train and test) with takeoff ``t1``: up to ``k`` other movements at
the same airport and in the same data island (``raw_extra.island_expr``), drawn from

* departures and arrivals with ``MVT_TIME`` in ``[t1 − mvt_before, t1 + mvt_after]``, and
* departures with ``AOBT_3`` in ``[t1 − aobt_before, t1]``,

deduplicated, the row itself excluded, **nearest first**. The distance of a candidate is
``|MVT − t1|``, or ``|AOBT_3 − t1|`` when that is smaller and the AOBT window admits it; ties
go to the earlier position in the ``(airport, island, MVT_TIME, MVT_ID)`` order.

What an element carries (all test-visible):

* per movement (``mv_num`` / ``mv_cat``, one row per movement in the sorted order above):
  the NM proxy ``MVT − AOBT_3`` in minutes for **regime-A departures only**
  (``prc.mixture.regimes.assign_regimes``, the ``regime`` block's rule; NaN otherwise), the
  arrival taxi-in ``BLOCK − MVT`` in minutes for **arrivals only** (NaN outside [0, 60] min
  and for departures), and codes for phase, wake category and aircraft type;
* per (departure, slot) (``rel`` / ``flags``): ``MVT − t1`` and ``AOBT_3 − t1`` in minutes
  (NaN when there is no AOBT_3), and three bits: same runway, same stand group
  (``surface_flow.stand_groups``), same operator.

A departure's ``BLOCK_TIME`` and its label are never read: ``BLOCK_TIME`` enters only
through a ``PHASE == ARR`` guard, and ``TAXITIME_SEC_mvt`` is not selected at all (a test
scrambles both and asserts identical arrays). Nothing is fitted on labels, so the sets are
the same in every fold.

The arrays are built once per (movement table, parameters, this module's source) and cached
under ``data/cache/context_sets/<key>/`` as ``.npy`` files, opened memory-mapped: a fold only
gathers its rows. The departures are stored sorted by ``MVT_ID`` (``dep_ids``), and the model
finds its rows through the ``ctx_key`` column of the ``context_sets`` feature block below,
which is nothing but the departure's own ``MVT_ID``.
"""

from __future__ import annotations

import inspect
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from prc import paths
from prc.data import schema as S
from prc.features.raw_extra import island_expr
from prc.features.registry import feature
from prc.features.surface_flow import stand_groups
from prc.features.surface_queue import NO_RUNWAY

log = logging.getLogger("prc.context_sets")

#: The only column the ``context_sets`` block emits: the departure's own ``MVT_ID``, for
#: the ``setenc`` model to find its context set. Useless (and not meant) as a feature.
KEY_COLUMN = "ctx_key"

DEFAULT_PARAMS: dict[str, int] = {
    "k": 64,
    "mvt_before_min": 45,
    "mvt_after_min": 15,
    "aobt_before_min": 60,
}

#: Aircraft types kept as their own code (by count over all movements); the rest are OTHER.
MAX_TYPES = 254
#: Arrival taxi-in outside [0, MAX_TAXI_IN_MIN] is treated as missing.
MAX_TAXI_IN_MIN = 60.0
#: NM proxy outside [0, MAX_PROXY_MIN] is treated as missing (surface_queue's 3 h bound).
MAX_PROXY_MIN = 180.0

FLAG_SAME_RUNWAY = 1
FLAG_SAME_GROUP = 2
FLAG_SAME_OPERATOR = 4

_US = 1_000_000
_MIN_US = 60 * _US
_POS_BITS = 22  # a group holds < 4.19 M movements (the largest airport-island has ~0.55 M)
_CHUNK = 20_000

_ARRAYS = ("dep_ids", "idx", "rel", "flags", "mv_num", "mv_cat")


@feature("context_sets")
def context_sets(df: pl.DataFrame) -> pl.DataFrame:
    """Departure rows only: ``ctx_key`` = the row's own MVT_ID (Float64), the handle the
    ``setenc`` model uses to gather the row's context set from ``data/cache/context_sets``.
    Carries no information of its own; list it only for ``model.kind: setenc`` and keep it
    out of the head with ``mixture.corruption.clf_columns``."""
    return df.filter(pl.col(S.PHASE) == "DEP").select(
        S.MVT_ID, pl.col(S.MVT_ID).cast(pl.Float64).alias(KEY_COLUMN)
    )


# --------------------------------------------------------------------------------------
# building
# --------------------------------------------------------------------------------------

_COLUMNS = [
    S.MVT_ID,
    S.AIRPORT,
    S.PHASE,
    S.MVT_TIME,
    S.AOBT_3,
    S.BLOCK_TIME,
    S.RUNWAY,
    S.STAND,
    S.SPLIT_MONTH,
    S.WAKE_CAT,
    S.AIRCRAFT_TYPE,
    S.OPERATOR,
]


def _codes(s: pl.Series) -> np.ndarray:
    """Dense int codes of a string column (sorted levels), -1 for null."""
    s = s.cast(pl.String)
    levels = s.drop_nulls().unique().sort()
    lut = pl.DataFrame({"_v": levels, "_c": np.arange(levels.len(), dtype=np.int32)})
    out = pl.DataFrame({"_v": s}).join(lut, on="_v", how="left", maintain_order="left")
    return out.get_column("_c").fill_null(-1).to_numpy().astype(np.int32)


def _prepare(mv: pl.DataFrame) -> pl.DataFrame:
    """Every movement, sorted, with the per-movement attributes the sets need."""
    from prc.mixture.regimes import REGIME_COL, assign_regimes

    c = pl.col
    df = mv.select([x for x in _COLUMNS if x in mv.columns])
    is_dep = c(S.PHASE) == "DEP"
    is_arr = c(S.PHASE) == "ARR"
    reg = assign_regimes(df.select(S.MVT_ID, S.AIRPORT, S.MVT_TIME, S.AOBT_3)).select(
        S.MVT_ID, REGIME_COL
    )
    groups = stand_groups(df)
    proxy_min = (c(S.MVT_TIME) - c(S.AOBT_3)).dt.total_seconds().cast(pl.Float64) / 60.0
    # arrivals only: BLOCK_TIME is never read on a departure row
    taxi_in_min = (
        pl.when(is_arr).then(c(S.BLOCK_TIME)).otherwise(None) - c(S.MVT_TIME)
    ).dt.total_seconds().cast(pl.Float64) / 60.0
    types = (
        df.group_by(S.AIRCRAFT_TYPE)
        .len()
        .drop_nulls(S.AIRCRAFT_TYPE)
        .sort(["len", S.AIRCRAFT_TYPE], descending=[True, False])
        .head(MAX_TYPES)
        .with_columns(pl.int_range(2, pl.len() + 2, dtype=pl.Int16).alias("_type"))
        .select(S.AIRCRAFT_TYPE, "_type")
    )
    out = (
        df.join(reg, on=S.MVT_ID, how="left")
        .join(groups, on=[S.AIRPORT, S.STAND], how="left")
        .join(types, on=S.AIRCRAFT_TYPE, how="left")
        .with_columns(
            island_expr().alias("_island"),
            c(S.MVT_TIME).dt.epoch("us").alias("_t"),
            pl.when(is_dep).then(c(S.AOBT_3).dt.epoch("us")).otherwise(None).alias("_a"),
            pl.when(is_dep & (c(REGIME_COL) == "A") & proxy_min.is_between(0, MAX_PROXY_MIN))
            .then(proxy_min)
            .otherwise(None)
            .alias("_proxy"),
            pl.when(taxi_in_min.is_between(0, MAX_TAXI_IN_MIN))
            .then(taxi_in_min)
            .otherwise(None)
            .alias("_taxi_in"),
            is_arr.cast(pl.Int16).alias("_phase"),
            pl.when(c(S.AIRCRAFT_TYPE).is_null())
            .then(pl.lit(0, dtype=pl.Int16))
            .otherwise(c("_type").fill_null(1))
            .alias("_type"),
            pl.when(c(S.RUNWAY).is_null() | (c(S.RUNWAY) == NO_RUNWAY))
            .then(None)
            .otherwise(c(S.RUNWAY))
            .alias("_rwy"),
        )
        .drop_nulls("_t")
        .sort([S.AIRPORT, "_island", "_t", S.MVT_ID])
    )
    wake = _codes(out.get_column(S.WAKE_CAT))
    return out.with_columns(
        pl.Series("_wake", (wake + 1).astype(np.int16)),  # 0 = missing
        pl.Series("_rwy_c", _codes(out.get_column("_rwy"))),
        pl.Series("_grp_c", _codes(out.get_column("sf_group"))),
        pl.Series("_op_c", _codes(out.get_column(S.OPERATOR))),
    )


def _group_sets(
    t: np.ndarray,
    a: np.ndarray,
    has_a: np.ndarray,
    dep_local: np.ndarray,
    p: dict,
) -> np.ndarray:
    """Local candidate indices (n_dep, k), -1 padded, nearest first, for one airport-island.

    ``t`` is sorted. ``a`` / ``has_a``: AOBT_3 (us) of each movement and whether it counts
    for the AOBT window (departures with an AOBT_3)."""
    k = int(p["k"])
    b1, f1 = int(p["mvt_before_min"]) * _MIN_US, int(p["mvt_after_min"]) * _MIN_US
    b2 = int(p["aobt_before_min"]) * _MIN_US
    n = t.shape[0]
    assert n < (1 << _POS_BITS), "airport-island too large for the tie-break key"
    a_pos = np.flatnonzero(has_a)
    order = np.argsort(a[a_pos], kind="stable")
    a_pos = a_pos[order]
    a_sorted = a[a_pos]
    big = np.iinfo(np.int64).max
    out = np.full((dep_local.shape[0], k), -1, dtype=np.int64)
    for s in range(0, dep_local.shape[0], _CHUNK):
        di = dep_local[s : s + _CHUNK]
        ti = t[di]
        lo1 = np.searchsorted(t, ti - b1, side="left")
        hi1 = np.searchsorted(t, ti + f1, side="right")
        lo2 = np.searchsorted(a_sorted, ti - b2, side="left")
        hi2 = np.searchsorted(a_sorted, ti, side="right")
        w1 = max(int((hi1 - lo1).max(initial=0)), 1)
        w2 = max(int((hi2 - lo2).max(initial=0)), 1)
        # MVT window
        c1 = lo1[:, None] + np.arange(w1)[None, :]
        v1 = c1 < hi1[:, None]
        c1 = np.minimum(c1, n - 1)
        d1 = np.abs(t[c1] - ti[:, None])
        a1 = a[c1]
        in_a = has_a[c1] & (a1 >= (ti - b2)[:, None]) & (a1 <= ti[:, None])
        d1 = np.where(in_a, np.minimum(d1, np.abs(a1 - ti[:, None])), d1)
        # AOBT window (departures only); a candidate whose MVT is in the MVT window is
        # already in c1
        j2 = lo2[:, None] + np.arange(w2)[None, :]
        v2 = j2 < hi2[:, None]
        if a_pos.shape[0]:
            c2 = a_pos[np.minimum(j2, a_pos.shape[0] - 1)]
        else:
            c2 = np.zeros_like(j2)
            v2[:] = False
        t2 = t[c2]
        v2 &= ~((t2 >= (ti - b1)[:, None]) & (t2 <= (ti + f1)[:, None]))
        d2 = np.abs(a[c2] - ti[:, None])
        cand = np.concatenate([c1, c2], axis=1)
        valid = np.concatenate([v1, v2], axis=1) & (cand != di[:, None])
        key = np.where(valid, (np.concatenate([d1, d2], axis=1) << _POS_BITS) | cand, big)
        kk = min(k, key.shape[1])
        part = np.argpartition(key, kk - 1, axis=1)[:, :kk] if key.shape[1] > kk else None
        if part is not None:
            key_k = np.take_along_axis(key, part, axis=1)
        else:
            key_k = key
        key_k = np.sort(key_k, axis=1)
        sel = np.where(key_k == big, -1, key_k & ((1 << _POS_BITS) - 1))
        out[s : s + _CHUNK, :kk] = sel
    return out


def build_sets(mv: pl.DataFrame, params: dict | None = None) -> dict[str, np.ndarray]:
    """All arrays for ``mv`` (the full movement table, or any table with its columns).

    ``dep_ids`` (n_dep,) float64 sorted; ``idx`` (n_dep, k) int32 row of ``mv_*`` (-1 pad);
    ``rel`` (n_dep, k, 2) float16 [MVT − t1, AOBT_3 − t1] minutes; ``flags`` (n_dep, k)
    uint8; ``mv_num`` (n_mv, 2) float16 [proxy, taxi-in] minutes; ``mv_cat`` (n_mv, 3)
    int16 [phase, wake, type]."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    k = int(p["k"])
    d = _prepare(mv)
    n_mv = d.height
    t = d.get_column("_t").to_numpy()
    a_raw = d.get_column("_a")
    has_a = a_raw.is_not_null().to_numpy()
    a = a_raw.fill_null(0).to_numpy()
    is_dep = (d.get_column(S.PHASE) == "DEP").to_numpy()
    rwy = d.get_column("_rwy_c").to_numpy()
    grp = d.get_column("_grp_c").to_numpy()
    op = d.get_column("_op_c").to_numpy()
    ids = d.get_column(S.MVT_ID).cast(pl.Float64).to_numpy()
    bounds = (
        d.with_row_index("_r")
        .group_by([S.AIRPORT, "_island"], maintain_order=True)
        .agg(pl.col("_r").min().alias("lo"), pl.col("_r").max().alias("hi"))
    )
    dep_rows_all = np.flatnonzero(is_dep)
    idx = np.full((dep_rows_all.shape[0], k), -1, dtype=np.int32)
    wpos = 0
    for lo, hi in zip(
        bounds.get_column("lo").to_list(), bounds.get_column("hi").to_list(), strict=True
    ):
        sl = slice(lo, hi + 1)
        dep_local = np.flatnonzero(is_dep[sl])
        if not dep_local.shape[0]:
            continue
        loc = _group_sets(t[sl], a[sl], has_a[sl] & is_dep[sl], dep_local, p)
        idx[wpos : wpos + loc.shape[0]] = np.where(loc >= 0, loc + lo, -1).astype(np.int32)
        wpos += loc.shape[0]
    assert wpos == dep_rows_all.shape[0]
    # relational features, (n_dep, k), in row chunks so the float64 temporaries stay small
    n_dep = dep_rows_all.shape[0]
    rel = np.empty((n_dep, k, 2), dtype=np.float16)
    flags = np.zeros((n_dep, k), dtype=np.uint8)
    for s in range(0, n_dep, 10 * _CHUNK):
        ci = idx[s : s + 10 * _CHUNK]
        ok = ci >= 0
        j = np.where(ok, ci, 0)
        own = dep_rows_all[s : s + 10 * _CHUNK, None]
        rel[s : s + 10 * _CHUNK, :, 0] = np.where(ok, (t[j] - t[own]) / _MIN_US, np.nan)
        rel[s : s + 10 * _CHUNK, :, 1] = np.where(
            ok & has_a[j] & is_dep[j], (a[j] - t[own]) / _MIN_US, np.nan
        )
        for code, bit in (
            (rwy, FLAG_SAME_RUNWAY),
            (grp, FLAG_SAME_GROUP),
            (op, FLAG_SAME_OPERATOR),
        ):
            same = ok & (code[j] >= 0) & (code[j] == code[own])
            flags[s : s + 10 * _CHUNK] |= (same * bit).astype(np.uint8)
    mv_num = np.stack(
        [
            d.get_column("_proxy").fill_null(np.nan).to_numpy(),
            d.get_column("_taxi_in").fill_null(np.nan).to_numpy(),
        ],
        axis=1,
    ).astype(np.float16)
    mv_cat = np.stack(
        [
            d.get_column("_phase").to_numpy(),
            d.get_column("_wake").to_numpy(),
            d.get_column("_type").to_numpy(),
        ],
        axis=1,
    ).astype(np.int16)
    dep_ids = ids[dep_rows_all]
    order = np.argsort(dep_ids, kind="stable")
    log.info("context sets: %d departures, %d movements, k=%d", dep_ids.shape[0], n_mv, k)
    return {
        "dep_ids": dep_ids[order],
        "idx": idx[order],
        "rel": rel[order],
        "flags": flags[order],
        "mv_num": mv_num,
        "mv_cat": mv_cat,
    }


# --------------------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------------------


@dataclass
class ContextSets:
    dep_ids: np.ndarray
    idx: np.ndarray
    rel: np.ndarray
    flags: np.ndarray
    mv_num: np.ndarray
    mv_cat: np.ndarray

    @property
    def k(self) -> int:
        return int(self.idx.shape[1])

    def rows(self, keys: np.ndarray) -> np.ndarray:
        """Positions of the departures ``keys`` (MVT_IDs); -1 where a key is unknown."""
        keys = np.asarray(keys, dtype=np.float64)
        pos = np.searchsorted(self.dep_ids, keys)
        pos = np.minimum(pos, max(self.dep_ids.shape[0] - 1, 0))
        hit = self.dep_ids.shape[0] > 0
        ok = hit & (self.dep_ids[pos] == keys) if hit else np.zeros(keys.shape, dtype=bool)
        return np.where(ok, pos, -1)


def _code_hash() -> str:
    import hashlib

    return hashlib.sha1(inspect.getsource(sys.modules[__name__]).encode()).hexdigest()[:12]


_MEMO: dict[str, ContextSets] = {}


def load_sets(params: dict | None = None, movements: Path | None = None) -> ContextSets:
    """The cached sets for the current movement table, built on a miss (under a lock), and
    opened memory-mapped. Memoised per process, so every fit of a run shares one copy."""
    from prc import cache as cache_mod

    p = {**DEFAULT_PARAMS, **(params or {})}
    mvp = movements or (paths.INTERIM / "movements.parquet")
    fh = cache_mod.file_hash(mvp)
    if fh is None:
        raise FileNotFoundError(f"{mvp} missing; run `prc data prepare`")
    inputs = {"v": 1, "kind": "context_sets", "movements": fh, "params": p, "code": _code_hash()}
    entry = cache_mod.entry("context_sets", inputs)
    if entry.key in _MEMO:
        return _MEMO[entry.key]
    if not entry.ready():
        with cache_mod.locked(entry):
            if not entry.ready():
                t0 = time.perf_counter()
                arrays = build_sets(pl.read_parquet(mvp, columns=_COLUMNS), p)
                for name in _ARRAYS:
                    arr = arrays[name]
                    cache_mod.atomic_write(
                        entry.path(f"{name}.npy"),
                        lambda f, arr=arr: _save_npy(f, arr),
                    )
                cache_mod.write_json(
                    entry,
                    "build.json",
                    {
                        "seconds": round(time.perf_counter() - t0, 1),
                        "n_dep": int(arrays["dep_ids"].shape[0]),
                        "n_mv": int(arrays["mv_num"].shape[0]),
                        "bytes": int(sum(a.nbytes for a in arrays.values())),
                    },
                )
                entry.mark_ready()
                log.info("context sets built in %.1f s -> %s", time.perf_counter() - t0, entry.dir)
                del arrays
    sets = ContextSets(**{n: np.load(entry.path(f"{n}.npy"), mmap_mode="r") for n in _ARRAYS})
    _MEMO[entry.key] = sets
    return sets


def _save_npy(path: Path, arr: np.ndarray) -> None:
    with open(path, "wb") as fh:
        np.save(fh, arr)
