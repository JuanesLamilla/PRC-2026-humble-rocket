"""Surface queue from observable stamps (task T3, backlog Q1).

How many aircraft are already taxiing when a departure pushes back, and how many take off,
push or land while it taxis. Every quantity reads only test-visible stamps of **other** rows:
departures' ``AOBT_3_flt`` and takeoff (``MVT_TIME``), arrivals' landing (``MVT_TIME``).
Nothing reads the target or the airport off-block stamp of a departure.

Definitions (t0 = own ``AOBT_3``, t1 = own takeoff, per airport and data island):

* A departure's **surface interval** is [AOBT_3, MVT). It is *valid* when AOBT_3 exists and
  0 ≤ MVT − AOBT_3 ≤ ``MAX_TAXI_S`` (3 h; 7 of 2.40 M proxies exceed it, 1,181 are
  negative). Only valid intervals are counted as "taxiing" or "pushed"; derived-constant
  AOBT_3 rows (regime B) are kept in every count, because a derived value still marks a
  pushback about its constant before takeoff.
* Own-anchored columns (``sq_taxiing_at_push*``, ``sq_*_during_taxi*``, ``sq_overtaken_by``,
  ``sq_own_minus_nbr_med``) are null unless the row's own interval is valid. The cap is the
  guard against corrupt AOBT_3: an interval longer than 3 h nulls them all.
* ``sq_taxiing_at_push``      #other valid j with a_j ≤ t0 < m_j.
* ``sq_tko_during_taxi``      #other departures with takeoff in (t0, t1) (all departures,
                              AOBT_3 or not).
* ``sq_push_during_taxi``     #other valid j with a_j in (t0, t1).
* ``sq_ldg_during_taxi``      #arrivals landing in (t0, t1).
* ``sq_overtaken_by``         #other valid j, own runway, with a_j > t0 and m_j < t1.
* ``_rwy`` variants restrict both the row and the events to the row's own runway; a row on
  runway ``NA`` gets null runway columns.
* ``sq_nbr_proxy_{med,p90}_prev10`` / ``sq_nbr_proxy_med_next10``: median / p90 of
  MVT − AOBT_3 over the ten nearest **measured** takeoffs before / after the row on its own
  runway (order = takeoff time, then ``MVT_ID``; the row itself excluded). Measured = valid
  interval and not on a derived constant (``prc.mixture.regimes.assign_regimes``, the
  ``regime`` block's rule). Neighbours more than ``NBR_SPAN_S`` (6 h) from the row's takeoff
  are dropped; fewer than ``NBR_MIN`` (3) left → null. These do not depend on the row's own
  AOBT_3 and are computed for every departure (the no-flight-plan rows included);
  ``mask_for_no_nm`` nulls them on no-NM rows when a YAML asks for that.
* ``sq_own_minus_nbr_med``    own MVT − AOBT_3 minus ``sq_nbr_proxy_med_prev10``.
* ``sq_taxiing_at_tm{10,20,30}``: #other valid j with a_j ≤ t1 − c < m_j, c = 10/20/30 min,
  for every departure (takeoff-anchored, needs no own AOBT_3).

Counts are differences of cumulative counts on sorted arrays (``np.searchsorted``), never
as-of joins: N(a ≤ x) − N(m ≤ x) over valid intervals is the number on the surface at x,
because a ≤ m on every valid interval. Self is subtracted by the same rule.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from prc.data import schema as S
from prc.features.raw_extra import island_expr
from prc.features.registry import feature
from prc.mixture.regimes import ON_SPIKE_COL, assign_regimes

#: A surface interval MVT − AOBT_3 outside [0, MAX_TAXI_S] is invalid (corrupt AOBT_3).
MAX_TAXI_S = 3 * 3600
#: Neighbour-proxy window: this many measured takeoffs before / after the row.
NBR_K = 10
#: Neighbours whose takeoff is further than this from the row's takeoff are dropped.
NBR_SPAN_S = 6 * 3600
#: Fewer neighbours than this left in the window → null statistic.
NBR_MIN = 3
#: Takeoff-anchored look-back offsets, minutes.
TM_OFFSETS_MIN: tuple[int, ...] = (10, 20, 30)
#: A runway label that means "unknown".
NO_RUNWAY = "NA"

US = 1_000_000

COUNT_COLS: tuple[str, ...] = (
    "sq_taxiing_at_push",
    "sq_taxiing_at_push_rwy",
    "sq_tko_during_taxi",
    "sq_tko_during_taxi_rwy",
    "sq_push_during_taxi",
    "sq_ldg_during_taxi",
    "sq_ldg_during_taxi_rwy",
    "sq_overtaken_by",
    *(f"sq_taxiing_at_tm{c}" for c in TM_OFFSETS_MIN),
)
FLOAT_COLS: tuple[str, ...] = (
    "sq_nbr_proxy_med_prev10",
    "sq_nbr_proxy_p90_prev10",
    "sq_nbr_proxy_med_next10",
    "sq_own_minus_nbr_med",
)


def _le(sorted_arr: np.ndarray, x: np.ndarray) -> np.ndarray:
    """#elements ≤ x."""
    return np.searchsorted(sorted_arr, x, side="right")


def _lt(sorted_arr: np.ndarray, x: np.ndarray) -> np.ndarray:
    """#elements < x."""
    return np.searchsorted(sorted_arr, x, side="left")


def _on_surface(a_v: np.ndarray, m_v: np.ndarray, x: np.ndarray) -> np.ndarray:
    """#valid intervals [a, m) containing x; ``a_v`` / ``m_v`` sorted independently."""
    return _le(a_v, x) - _le(m_v, x)


def _open_count(sorted_arr: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """#elements in the open interval (lo, hi); 0 where hi ≤ lo."""
    return np.maximum(_lt(sorted_arr, hi) - _le(sorted_arr, lo), 0)


def _overtaken(a: np.ndarray, m: np.ndarray, t0: np.ndarray, t1: np.ndarray) -> np.ndarray:
    """For each query (t0, t1): #valid j with a_j > t0 and m_j < t1. ``a``/``m`` are the valid
    intervals of one runway. Candidates satisfy t0 < a_j < t1 (a ≤ m < t1), a contiguous
    range of the a-sorted array; the loop runs over offsets inside that range."""
    out = np.zeros(t0.shape, dtype=np.int64)
    if a.size == 0 or t0.size == 0:
        return out
    order = np.argsort(a, kind="stable")
    a_s, m_s = a[order], m[order]
    lo = _le(a_s, t0)
    hi = _lt(a_s, t1)
    width = np.maximum(hi - lo, 0)
    active = np.flatnonzero(width > 0)
    k = 0
    while active.size:
        idx = lo[active] + k
        out[active] += m_s[idx] < t1[active]
        k += 1
        active = active[width[active] > k]
    return out


def _quantile_rows(v: np.ndarray, q: float, n_min: int) -> np.ndarray:
    """Row-wise linear-interpolation quantile (numpy's default method) of a 2-D array with
    NaN padding; NaN where fewer than ``n_min`` values."""
    s = np.sort(v, axis=1)  # NaN last
    k = np.sum(~np.isnan(v), axis=1)
    out = np.full(v.shape[0], np.nan)
    ok = k >= n_min
    if not ok.any():
        return out
    s, k = s[ok], k[ok]
    pos = q * (k - 1)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, k - 1)
    rows = np.arange(s.shape[0])
    vlo, vhi = s[rows, lo], s[rows, hi]
    frac = pos - lo
    out[ok] = vlo + (vhi - vlo) * frac
    return out


def _nbr_window(t: np.ndarray, meas: np.ndarray, proxy: np.ndarray, prev: bool) -> np.ndarray:
    """(n, NBR_K) matrix of neighbour proxies (NaN = none) for rows sorted by (t, id)."""
    n = t.size
    mpos = np.flatnonzero(meas)
    p_m, t_m = proxy[mpos], t[mpos]
    before = np.cumsum(meas) - meas  # measured rows strictly before each row
    start = before - NBR_K if prev else before + meas
    idx = start[:, None] + np.arange(NBR_K)[None, :]
    ok = (idx >= 0) & (idx < mpos.size)
    safe = np.clip(idx, 0, max(mpos.size - 1, 0))
    if mpos.size == 0:
        return np.full((n, NBR_K), np.nan)
    vals = np.where(ok, p_m[safe], np.nan)
    near = np.abs(t_m[safe] - t[:, None]) <= NBR_SPAN_S * US
    return np.where(ok & near, vals, np.nan)


def _group(dep: pl.DataFrame, arr: pl.DataFrame) -> dict[str, np.ndarray]:
    """All columns for the departures of one (airport, island). ``dep`` sorted by (t, id)."""
    n = dep.height
    t = dep.get_column("_t").to_numpy()
    a = dep.get_column("_a").to_numpy()  # int64 µs, arbitrary where invalid
    valid = dep.get_column("_valid").to_numpy()
    meas = dep.get_column("_meas").to_numpy().astype(np.int64)
    rwy = dep.get_column(S.RUNWAY).to_numpy()
    ta = arr.get_column("_t").to_numpy() if arr.height else np.zeros(0, dtype=np.int64)
    rwy_a = arr.get_column(S.RUNWAY).to_numpy() if arr.height else np.zeros(0, dtype=object)

    a_v = np.sort(a[valid])
    m_v = np.sort(t[valid])
    t_all = np.sort(t)
    ta_s = np.sort(ta)
    t0 = np.where(valid, a, 0)
    t1 = t
    self_push = valid & (a < t)  # own interval contains t0
    cols: dict[str, np.ndarray] = {}
    cols["sq_taxiing_at_push"] = _on_surface(a_v, m_v, t0) - self_push
    cols["sq_tko_during_taxi"] = _open_count(t_all, t0, t1)
    cols["sq_push_during_taxi"] = _open_count(a_v, t0, t1)
    cols["sq_ldg_during_taxi"] = _open_count(ta_s, t0, t1)
    for c in TM_OFFSETS_MIN:
        x = t1 - c * 60 * US
        self_c = valid & (a <= x)  # x < t1 always
        cols[f"sq_taxiing_at_tm{c}"] = _on_surface(a_v, m_v, x) - self_c

    for name in ("taxiing_at_push", "tko_during_taxi", "ldg_during_taxi"):
        cols[f"sq_{name}_rwy"] = np.zeros(n, dtype=np.int64)
    cols["sq_overtaken_by"] = np.zeros(n, dtype=np.int64)
    rwy_null = np.zeros(n, dtype=bool)
    nbr = {k: np.full(n, np.nan) for k in ("prev_med", "prev_p90", "next_med")}
    proxy = (t - a).astype(np.float64) / US
    for r in np.unique(rwy):
        sel = np.flatnonzero(rwy == r)
        if r == NO_RUNWAY or r is None:
            rwy_null[sel] = True
            continue
        tr, ar, vr = t[sel], a[sel], valid[sel]
        a_rv, m_rv = ar[vr], tr[vr]
        t0r = t0[sel]
        cols["sq_taxiing_at_push_rwy"][sel] = (
            _on_surface(np.sort(a_rv), np.sort(m_rv), t0r) - self_push[sel]
        )
        cols["sq_tko_during_taxi_rwy"][sel] = _open_count(np.sort(tr), t0r, tr)
        cols["sq_ldg_during_taxi_rwy"][sel] = _open_count(np.sort(ta[rwy_a == r]), t0r, tr)
        cols["sq_overtaken_by"][sel] = _overtaken(a_rv, m_rv, t0r, tr)
        # neighbour proxy: sel is in (t, id) order because dep is
        mr, pr = meas[sel], proxy[sel]
        wp = _nbr_window(tr, mr, pr, prev=True)
        wn = _nbr_window(tr, mr, pr, prev=False)
        nbr["prev_med"][sel] = _quantile_rows(wp, 0.5, NBR_MIN)
        nbr["prev_p90"][sel] = _quantile_rows(wp, 0.9, NBR_MIN)
        nbr["next_med"][sel] = _quantile_rows(wn, 0.5, NBR_MIN)

    own_anchored = (
        "sq_taxiing_at_push",
        "sq_taxiing_at_push_rwy",
        "sq_tko_during_taxi",
        "sq_tko_during_taxi_rwy",
        "sq_push_during_taxi",
        "sq_ldg_during_taxi",
        "sq_ldg_during_taxi_rwy",
        "sq_overtaken_by",
    )
    out: dict[str, np.ndarray] = {}
    for name in COUNT_COLS:
        v = cols[name].astype(np.float64)
        if name in own_anchored:
            v[~valid] = np.nan
        if name.endswith("_rwy") or name == "sq_overtaken_by":
            v[rwy_null] = np.nan
        out[name] = v
    out["sq_nbr_proxy_med_prev10"] = nbr["prev_med"]
    out["sq_nbr_proxy_p90_prev10"] = nbr["prev_p90"]
    out["sq_nbr_proxy_med_next10"] = nbr["next_med"]
    out["sq_own_minus_nbr_med"] = np.where(valid, proxy - nbr["prev_med"], np.nan)
    return out


@feature("surface_queue")
def surface_queue(df: pl.DataFrame) -> pl.DataFrame:
    """Departure rows only. Aircraft on the surface at own pushback (airport / own runway),
    takeoffs, pushbacks and landings during own taxi window, own-runway overtakes,
    neighbours' measured MVT − AOBT_3 (median / p90 of the previous ten, median of the next
    ten, own minus previous median), and aircraft taxiing 10 / 20 / 30 min before own takeoff.
    Per (airport, data island); intervals outside [0, 3 h] are invalid. Counts Int16,
    statistics Float32 (seconds). Definitions in the module docstring."""
    c = pl.col
    island = island_expr().alias("_island")
    dep_in = df.filter(c(S.PHASE) == "DEP").select(
        S.MVT_ID, S.AIRPORT, S.RUNWAY, S.MVT_TIME, S.AOBT_3, S.SPLIT_MONTH
    )
    spike = assign_regimes(dep_in).select(S.MVT_ID, ON_SPIKE_COL)
    proxy_s = (c(S.MVT_TIME) - c(S.AOBT_3)).dt.total_seconds()
    valid = proxy_s.is_between(0, MAX_TAXI_S).fill_null(False)
    deps = (
        dep_in.join(spike, on=S.MVT_ID, how="left")
        .select(
            S.MVT_ID,
            S.AIRPORT,
            island,
            c(S.RUNWAY).fill_null(NO_RUNWAY),
            c(S.MVT_TIME).dt.epoch("us").alias("_t"),
            c(S.AOBT_3).dt.epoch("us").fill_null(0).alias("_a"),
            valid.alias("_valid"),
            (valid & (c(ON_SPIKE_COL).fill_null(0) == 0)).alias("_meas"),
        )
        .sort([S.AIRPORT, "_island", "_t", S.MVT_ID])
    )
    arrs = (
        df.filter(c(S.PHASE) == "ARR")
        .select(
            S.AIRPORT,
            island,
            c(S.RUNWAY).fill_null(NO_RUNWAY),
            c(S.MVT_TIME).dt.epoch("us").alias("_t"),
        )
        .drop_nulls("_t")
        .sort([S.AIRPORT, "_island", "_t"])
    )
    arr_groups = arrs.partition_by([S.AIRPORT, "_island"], as_dict=True, maintain_order=True)
    empty_arr = arrs.clear()
    parts = []
    for key, g in deps.partition_by(
        [S.AIRPORT, "_island"], as_dict=True, maintain_order=True
    ).items():
        cols = _group(g, arr_groups.get(key, empty_arr))
        parts.append(pl.DataFrame({S.MVT_ID: g.get_column(S.MVT_ID), **cols}))
    if not parts:
        return pl.DataFrame(
            {S.MVT_ID: []},
            schema={S.MVT_ID: pl.Float64, **{k: pl.Int16 for k in COUNT_COLS}}
            | {k: pl.Float32 for k in FLOAT_COLS},
        )
    out = pl.concat(parts, how="vertical")
    return out.select(
        S.MVT_ID,
        *[c(k).fill_nan(None).cast(pl.Int16) for k in COUNT_COLS],
        *[c(k).fill_nan(None).cast(pl.Float32) for k in FLOAT_COLS],
    )
