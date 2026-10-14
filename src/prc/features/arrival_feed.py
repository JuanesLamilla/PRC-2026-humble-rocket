"""Arrival-side stamping state around a departure (task T7, backlog Q2).

The hidden departure label error is mostly the airport's off-block stamp against NM's
(``BLOCK − AOBT_3``). Arrivals' in-block stamps stay visible in the test months, so how the
airport stamps its *arrivals* in the hours around a departure is a sensor for the state of
its stamping system. Every quantity below reads arrival rows only (landing ``MVT_TIME``,
in-block ``BLOCK_TIME``, ``SCHED_TIME``, NM ``ARVT_3``, flight number); of the departure
row only its takeoff, schedule and flight number are read — never its target or off-block.

What the NM columns mean on an arrival row:
``ARVT_3`` is NM's actual **landing** (median airport landing − ARVT_3 = +4 s, 88 % within
60 s; median in-block − ARVT_3 = +474 s). There is no NM in-block stamp, so the
"airport vs NM" arrival sensor is the landing disagreement ``MVT − ARVT_3``.

Per arrival (validity in brackets; an invalid arrival is left out of that quantity):

* taxi-in      ``BLOCK − MVT``          (0 < x < 7200 s, as ``arrival_taxi_in``)
* ib-sched     ``BLOCK − SCHED``        (|x| ≤ 48 h)
* ldg-nm       ``MVT − ARVT_3``         (|x| ≤ 48 h)
* sched-eq     ``|BLOCK − SCHED| ≤ 10 s`` over arrivals with both stamps

Columns (departure rows only; per airport and data island, windows centred and closed,
[t − w, t + w], t = takeoff unless stated):

* ``af_taxiin_{med,iqr}_{1h,3h}``  median and p75 − p25 of taxi-in over the window's valid
  arrivals; null with fewer than ``QUANT_MIN``. ``arrival_taxi_in`` has only *backward*
  mean / median / p90 windows; the centred windows and the IQR are new.
* ``af_{ib_sched,ldg_nm}_xs_{med,iqr}_{1h,3h}``  the same statistics of in-block − schedule
  and landing − ARVT_3, **minus** the median / IQR of that quantity over all valid arrivals
  of the airport in the departure's calendar month ("excess" over the month level). The raw
  window values drift +0.42 to +0.52 sd (in-block − schedule, 2026-01: arrival delay) and
  +0.30 sd (landing − ARVT_3, 2026-07: EHAM's median moved from −46 s to −7 s, a stamping
  change training never saw); the excess keeps the within-month state and drops the level
  (as ``feed_health``'s ``fh_nofp_excess_*`` does). Drift table in the EDA file.
* ``af_s_taxiin_med_1h``, ``af_s_ib_sched_xs_med_1h``, ``af_s_sched_share_1h``  ±1 h windows
  anchored on the departure's ``SCHED_TIME`` (the planned off-block): the stamp state near
  the off-block for rows with no AOBT_3. Computed for every departure.
* ``af_sched_share_3h``  share of sched-eq arrivals in ±3 h (``feed_health`` has ±1 h / ±6 h).
* ``af_sched_cp_3h``     Int8, ``|af_sched_share_3h − airport-day share| > CP_DELTA`` (0.3);
  the day share is over the arrivals of the takeoff's UTC day at the airport.
* ``af_sched_run``       length of the run of consecutive sched-eq arrivals (landing order,
  ties by ``MVT_ID``) that contains the last arrival landed at or before the takeoff; 0 when
  that arrival is not sched-eq; null when there is none within ``MAX_GAP_S``.
* ``af_op_day_sched_share`` / ``af_op_day_n``  share of sched-eq arrivals / number of arrivals
  with both stamps, over the arrivals of the departure's own carrier (designator of
  ``FLIGHT_mvt``, present on the no-flight-plan rows too) on the takeoff's UTC day at the
  airport. Null share where n = 0.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from prc.data import schema as S
from prc.features.raw_extra import carrier_expr, island_expr
from prc.features.registry import feature

#: Window half-widths, seconds.
WINDOWS: dict[str, int] = {"1h": 3600, "3h": 3 * 3600}
#: Fewer valid arrivals in a window than this → null median / IQR.
QUANT_MIN = 3
#: In-block "equals the schedule" within this tolerance, seconds.
SCHED_EQ_TOL_S = 10.0
#: Change-point threshold on |3 h share − day share|.
CP_DELTA = 0.3
#: Taxi-in validity (seconds), as ``prc.features.arrivals._taxi_in``.
TAXI_IN_MAX_S = 7200.0
#: Differences above this in absolute value are invalid (a day-level typo).
MAX_ABS_DIFF_S = 48 * 3600.0
#: A gap above this between neighbouring arrivals ends a run / invalidates the as-of match.
MAX_GAP_S = 86_400.0
#: Cap on the padded window matrix per chunk (elements).
CHUNK_ELEMS = 2_000_000

US = 1_000_000
QUANTITIES: tuple[str, ...] = ("taxiin", "ib_sched", "ldg_nm")
#: Quantities reported as excess over their (airport, month) level.
XS_QUANTITIES: tuple[str, ...] = ("ib_sched", "ldg_nm")


def _stem(q: str) -> str:
    return f"af_{q}_xs" if q in XS_QUANTITIES else f"af_{q}"


WIN_COLS: tuple[str, ...] = tuple(
    f"{_stem(q)}_{s}_{w}" for q in QUANTITIES for w in WINDOWS for s in ("med", "iqr")
)
SCHED_COLS: tuple[str, ...] = (
    "af_s_taxiin_med_1h",
    "af_s_ib_sched_xs_med_1h",
    "af_s_sched_share_1h",
)
FLOAT_COLS: tuple[str, ...] = (
    *WIN_COLS,
    *SCHED_COLS,
    "af_sched_share_3h",
    "af_op_day_sched_share",
)
INT_COLS: dict[str, pl.DataType] = {
    "af_sched_cp_3h": pl.Int8(),
    "af_sched_run": pl.Int32(),
    "af_op_day_n": pl.Int16(),
}
ALL_COLS: tuple[str, ...] = (*FLOAT_COLS, *INT_COLS)


def _bounds(t_ev: np.ndarray, t_q: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    """[lo, hi) indexes of the events with time in [t_q − w, t_q + w] (``t_ev`` sorted)."""
    return (
        np.searchsorted(t_ev, t_q - w, side="left"),
        np.searchsorted(t_ev, t_q + w, side="right"),
    )


def _window_med_iqr(
    t_ev: np.ndarray, v_ev: np.ndarray, t_q: np.ndarray, w: int
) -> tuple[np.ndarray, np.ndarray]:
    """Median and p75 − p25 (numpy's linear interpolation) of ``v_ev`` over the events in
    each query's window; NaN where fewer than ``QUANT_MIN`` events or the query is NaN-timed
    (caller passes a sentinel far outside the data for a null query time)."""
    nq = t_q.size
    med = np.full(nq, np.nan)
    iqr = np.full(nq, np.nan)
    if nq == 0 or t_ev.size == 0:
        return med, iqr
    lo, hi = _bounds(t_ev, t_q, w)
    n = hi - lo
    k_max = int(n.max())
    if k_max < QUANT_MIN:
        return med, iqr
    step = max(1, CHUNK_ELEMS // k_max)
    last = t_ev.size - 1
    for s in range(0, nq, step):
        e = min(nq, s + step)
        k = n[s:e]
        kc = int(k.max())
        if kc < QUANT_MIN:
            continue
        ar = np.arange(kc)
        idx = np.minimum(lo[s:e, None] + ar[None, :], last)
        vals = np.where(ar[None, :] < k[:, None], v_ev[idx], np.inf)
        vals.sort(axis=1)
        ok = k >= QUANT_MIN
        rows = np.flatnonzero(ok)
        kk = k[ok]
        sv = vals[ok]
        r = np.arange(rows.size)

        def q_at(q: float, sv=sv, kk=kk, r=r) -> np.ndarray:
            pos = q * (kk - 1)
            f = np.floor(pos).astype(np.int64)
            c = np.minimum(f + 1, kk - 1)
            a, b = sv[r, f], sv[r, c]
            return a + (b - a) * (pos - f)

        med[s + rows] = q_at(0.5)
        iqr[s + rows] = q_at(0.75) - q_at(0.25)
    return med, iqr


def _window_share(t_ev: np.ndarray, flag: np.ndarray, t_q: np.ndarray, w: int) -> np.ndarray:
    """Mean of ``flag`` over the events in each query's window; NaN on an empty window."""
    out = np.full(t_q.size, np.nan)
    if t_ev.size == 0 or t_q.size == 0:
        return out
    cs = np.concatenate([[0.0], np.cumsum(flag, dtype=np.float64)])
    lo, hi = _bounds(t_ev, t_q, w)
    n = hi - lo
    ok = n > 0
    out[ok] = (cs[hi] - cs[lo])[ok] / n[ok]
    return out


def _run_length(flag: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Length of the run of consecutive ``flag`` events containing each event (0 where the
    event's flag is 0); a gap above ``MAX_GAP_S`` between neighbours ends a run."""
    n = flag.size
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    brk = np.ones(n, dtype=bool)
    brk[1:] = (flag[1:] != flag[:-1]) | (np.diff(t) > MAX_GAP_S * US)
    run_id = np.cumsum(brk) - 1
    sizes = np.bincount(run_id)
    return np.where(flag == 1, sizes[run_id], 0).astype(np.int64)


def _secs(a: str, b: str) -> pl.Expr:
    return (pl.col(a) - pl.col(b)).dt.total_seconds().cast(pl.Float64)


def _arrivals(df: pl.DataFrame) -> pl.DataFrame:
    """Arrival rows with the three quantities (NaN where invalid) and the sched-eq flags."""
    c = pl.col
    taxi = _secs(S.BLOCK_TIME, S.MVT_TIME)
    ibs = _secs(S.BLOCK_TIME, S.SCHED_TIME)
    ldg = _secs(S.MVT_TIME, S.ARVT_3)
    return (
        df.filter((c(S.PHASE) == "ARR") & c(S.MVT_TIME).is_not_null())
        .select(
            S.MVT_ID,
            S.AIRPORT,
            S.SPLIT_MONTH,
            island_expr().alias("_island"),
            c(S.MVT_TIME).dt.epoch("us").alias("_t"),
            c(S.MVT_TIME).dt.date().alias("_day"),
            carrier_expr().alias("_carrier"),
            pl.when((taxi > 0) & (taxi < TAXI_IN_MAX_S)).then(taxi).alias("taxiin"),
            pl.when(ibs.abs() <= MAX_ABS_DIFF_S).then(ibs).alias("ib_sched"),
            pl.when(ldg.abs() <= MAX_ABS_DIFF_S).then(ldg).alias("ldg_nm"),
        )
        .with_columns(
            c("ib_sched").is_not_null().alias("_valid"),
            (c("ib_sched").abs() <= SCHED_EQ_TOL_S).fill_null(False).alias("_eq"),
        )
        .sort([S.AIRPORT, "_island", "_t", S.MVT_ID])
    )


def _group(dep: pl.DataFrame, arr: pl.DataFrame) -> dict[str, np.ndarray]:
    """Windowed columns for the departures of one (airport, island)."""
    n = dep.height
    t = dep.get_column("_t").to_numpy()
    ts_null = dep.get_column("_ts").is_null().to_numpy()
    # a null schedule becomes a time no window reaches
    ts = dep.get_column("_ts").fill_null(np.iinfo(np.int64).min // 4).to_numpy()
    out: dict[str, np.ndarray] = {}
    ta = arr.get_column("_t").to_numpy()
    for q in QUANTITIES:
        v = arr.get_column(q).to_numpy()
        keep = np.isfinite(v)
        tq, vq = ta[keep], v[keep]
        for w, w_s in WINDOWS.items():
            med, iqr = _window_med_iqr(tq, vq, t, w_s * US)
            out[f"{_stem(q)}_med_{w}"] = med
            out[f"{_stem(q)}_iqr_{w}"] = iqr
        if q in ("taxiin", "ib_sched"):
            med, _ = _window_med_iqr(tq, vq, ts, WINDOWS["1h"] * US)
            med[ts_null] = np.nan
            out[f"af_s_{_stem(q)[3:]}_med_1h"] = med
    valid = arr.get_column("_valid").to_numpy()
    eq = arr.get_column("_eq").to_numpy()
    tv, ev = ta[valid], eq[valid].astype(np.float64)
    out["af_sched_share_3h"] = _window_share(tv, ev, t, WINDOWS["3h"] * US)
    s1 = _window_share(tv, ev, ts, WINDOWS["1h"] * US)
    s1[ts_null] = np.nan
    out["af_s_sched_share_1h"] = s1
    # run of sched-eq arrivals containing the last arrival landed at or before takeoff
    run = np.full(n, np.nan)
    if ta.size:
        rl = _run_length(eq.astype(np.int8), ta)
        k = np.searchsorted(ta, t, side="right") - 1
        ok = k >= 0
        kk = np.where(ok, k, 0)
        ok &= (t - ta[kk]) <= MAX_GAP_S * US
        run[ok] = rl[kk[ok]]
    out["af_sched_run"] = run
    return out


@feature("arrival_feed")
def arrival_feed(df: pl.DataFrame) -> pl.DataFrame:
    """Departure rows only. Arrival-side stamp state around the takeoff: centred ±1 h / ±3 h
    median and IQR of arrivals' taxi-in, in-block − schedule and landing − NM landing
    (ARVT_3); ±1 h variants anchored on the departure's schedule; the ±3 h share of
    schedule-stamped arrivals and its change-point flag against the airport-day share; the
    current run of schedule-stamped arrivals; the own carrier's schedule-stamped share that
    day. Definitions in the module docstring. Float32 (seconds / shares), Int8 / Int32 /
    Int16 flags and counts."""
    c = pl.col
    arr = _arrivals(df)
    deps = (
        df.filter(c(S.PHASE) == "DEP")
        .select(
            S.MVT_ID,
            S.AIRPORT,
            island_expr().alias("_island"),
            c(S.MVT_TIME).dt.epoch("us").alias("_t"),
            c(S.SCHED_TIME).dt.epoch("us").alias("_ts"),
            c(S.MVT_TIME).dt.date().alias("_day"),
            carrier_expr().alias("_carrier"),
            S.SPLIT_MONTH,
        )
        .sort([S.AIRPORT, "_island", "_t", S.MVT_ID])
    )
    arr_groups = arr.partition_by([S.AIRPORT, "_island"], as_dict=True, maintain_order=True)
    empty = arr.clear()
    parts = []
    for key, g in deps.partition_by(
        [S.AIRPORT, "_island"], as_dict=True, maintain_order=True
    ).items():
        cols = _group(g, arr_groups.get(key, empty))
        parts.append(pl.DataFrame({S.MVT_ID: g.get_column(S.MVT_ID), **cols}))
    if not parts:
        return pl.DataFrame(
            {S.MVT_ID: []},
            schema={S.MVT_ID: df.schema.get(S.MVT_ID, pl.Float64)}
            | {k: pl.Float32 for k in FLOAT_COLS}
            | INT_COLS,
        )
    win = pl.concat(parts, how="vertical").with_columns(
        [c(k).fill_nan(None) for k in (*WIN_COLS, *SCHED_COLS, "af_sched_share_3h", "af_sched_run")]
    )
    valid = arr.filter(c("_valid"))
    day = valid.group_by(S.AIRPORT, "_island", "_day").agg(c("_eq").mean().alias("_day_share"))
    op_day = (
        valid.drop_nulls("_carrier")
        .group_by(S.AIRPORT, "_island", "_day", "_carrier")
        .agg(
            c("_eq").mean().alias("af_op_day_sched_share"),
            pl.len().alias("af_op_day_n"),
        )
    )
    # (airport, month) level of the excess quantities, over that month's valid arrivals
    lvl_exprs = []
    for q in XS_QUANTITIES:
        v = c(q)
        lvl_exprs += [
            v.median().alias(f"_lvl_{q}_med"),
            (
                v.quantile(0.75, interpolation="linear") - v.quantile(0.25, interpolation="linear")
            ).alias(f"_lvl_{q}_iqr"),
        ]
    level = arr.group_by(S.AIRPORT, S.SPLIT_MONTH).agg(lvl_exprs)
    xs = [
        (c(f"{_stem(q)}_{s}_{w}") - c(f"_lvl_{q}_{s}")).alias(f"{_stem(q)}_{s}_{w}")
        for q in XS_QUANTITIES
        for s in ("med", "iqr")
        for w in WINDOWS
    ]
    xs.append(
        (c("af_s_ib_sched_xs_med_1h") - c("_lvl_ib_sched_med")).alias("af_s_ib_sched_xs_med_1h")
    )
    keys = [S.AIRPORT, "_island", "_day"]
    out = (
        deps.select(S.MVT_ID, *keys, S.SPLIT_MONTH, "_carrier")
        .join(win, on=S.MVT_ID, how="left")
        .join(level, on=[S.AIRPORT, S.SPLIT_MONTH], how="left")
        .with_columns(xs)
        .join(day, on=keys, how="left")
        .join(op_day, on=[*keys, "_carrier"], how="left")
        .with_columns(
            ((c("af_sched_share_3h") - c("_day_share")).abs() > CP_DELTA)
            .cast(pl.Int8)
            .alias("af_sched_cp_3h"),
            c("af_op_day_n").fill_null(0),
        )
        .with_columns(
            # no carrier → no verdict (count null, not 0)
            pl.when(c("_carrier").is_null())
            .then(None)
            .otherwise(c("af_op_day_n"))
            .alias("af_op_day_n")
        )
    )
    return out.select(
        S.MVT_ID,
        *[c(k).cast(pl.Float32) for k in FLOAT_COLS],
        *[c(k).cast(dt) for k, dt in INT_COLS.items()],
    ).sort(S.MVT_ID)
