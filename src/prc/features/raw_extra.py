"""Raw-column candidates (task T2).

Four blocks, all row-local or count-based over the movement table; none reads the target or
the off-block stamp of a departure:

* ``fpl_updates``      differences between the flight-plan times that ``fpl_deltas`` only
                       gives relative to takeoff (trees split badly on a difference of two
                       features), plus the NM arrival-time spans.
* ``source_mismatch``  where the airport's movement record and NM's flight record disagree.
* ``identity_freq``    carrier / callsign parses and *rate* encodings of how common a value is
                       inside its (airport, month) — never raw counts, because 2025 has twelve
                       months and each test island one.
* ``feed_health``      how healthy the NM and airport feeds are around the row's takeoff:
                       windowed shares of no-flight-plan departures, derived-AOBT departures
                       and schedule-stamped arrivals, and the no-flight-plan run length.

Every window and run is computed inside one (airport, data island); the islands are 2025,
2026-01 and 2026-07 (:func:`island_expr`).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from prc.data import schema as S
from prc.features.registry import feature
from prc.mixture.regimes import ON_SPIKE_COL, assign_regimes

#: Differences whose absolute value exceeds this are nulled (a day-level typo, not a delay).
MAX_ABS_DIFF_S = 48 * 3600.0
#: See ``prc.features.congestion.MAX_GAP_S``; repeated so the module hash covers it.
MAX_GAP_S = 86_400.0
#: A carrier / callsign prefix seen on fewer departures than this (whole table) becomes
#: ``OTHER``: keeps the categoricals at a few hundred levels instead of ~3,500 (GA
#: registrations parse as their own "carrier").
MIN_LEVEL_COUNT = 200
#: Centred window half-widths of ``feed_health``, seconds.
FEED_WINDOWS: dict[str, int] = {"1h": 3600, "6h": 6 * 3600}
#: An arrival's in-block stamp "equals the schedule" within this tolerance, seconds.
SCHED_EQ_TOL_S = 10.0


def island_expr() -> pl.Expr:
    """Data island of a row: ``"2025"`` for the contiguous 2025 year, else the month
    (``"2026-01"``, ``"2026-07"``). Windows and runs never cross an island edge."""
    m = pl.col(S.SPLIT_MONTH)
    return pl.when(m.str.starts_with("2025")).then(pl.lit("2025")).otherwise(m)


def _cat(expr: pl.Expr) -> pl.Expr:
    return expr.cast(pl.Utf8).fill_null("NA").cast(pl.Categorical)


# --------------------------------------------------------------------------------------
# 1. fpl_updates
# --------------------------------------------------------------------------------------


def _diff(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    """``a − b`` in seconds, Float32, null where either input is null or |diff| > 48 h."""
    d = (a - b).dt.total_seconds().cast(pl.Float64)
    return pl.when(d.abs() <= MAX_ABS_DIFF_S).then(d).otherwise(None).cast(pl.Float32)


@feature("fpl_updates")
def fpl_updates(df: pl.DataFrame) -> pl.DataFrame:
    """Pairwise differences of the flight-plan times (seconds, Float32; null when an input
    is null or |value| > 48 h). ``fpl_deltas`` gives every time relative to takeoff and
    ``lobt_minus_iobt``; none of the columns here is one of those exactly.

    ``nm_planned_block_to_arr_s`` = ARVT_1 − EOBT_1 (NM's planned off-block-to-landing),
    ``actual_tko_to_arr_s`` = ARVT_3 − takeoff (the actual flight time: ARVT_3 is the landing
    time, within 60 s of the twin arrival row's landing on 87 % of twins), and
    ``planned_minus_actual_s`` = their difference = NM's assumed taxi-out + en-route error.
    """
    c = pl.col
    t = c(S.MVT_TIME)
    planned = _diff(c(S.ARVT_1), c(S.EOBT_1))
    actual = _diff(c(S.ARVT_3), t)
    return df.select(
        S.MVT_ID,
        _diff(c(S.EOBT_1), c(S.IOBT)).alias("eobt1_minus_iobt"),
        _diff(c(S.LOBT), c(S.EOBT_1)).alias("lobt_minus_eobt1"),
        _diff(c(S.AOBT_3), c(S.LOBT)).alias("aobt3_minus_lobt"),
        _diff(c(S.AOBT_3), c(S.EOBT_1)).alias("aobt3_minus_eobt1"),
        _diff(c(S.AOBT_3), c(S.SCHED_TIME)).alias("aobt3_minus_sched"),
        _diff(c(S.SCHED_TIME), c(S.IOBT)).alias("sched_minus_iobt"),
        _diff(c(S.ARVT_3), c(S.ARVT_1)).alias("arvt3_minus_arvt1"),
        planned.alias("nm_planned_block_to_arr_s"),
        actual.alias("actual_tko_to_arr_s"),
        (planned - actual).cast(pl.Float32).alias("planned_minus_actual_s"),
    )


# --------------------------------------------------------------------------------------
# 2. source_mismatch
# --------------------------------------------------------------------------------------


def _col_or_null(df: pl.DataFrame, name: str) -> pl.Expr:
    """The column, or an all-null string column where the frame lacks it (the synthetic
    test fixture carries no ``ADEP_flt``; the challenge data always does)."""
    return pl.col(name) if name in df.columns else pl.lit(None, dtype=pl.Utf8).alias(name)


def _neq(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    """Int8 disagreement flag; null when either side is null (no NM record = no verdict)."""
    return (a != b).cast(pl.Int8)


@feature("source_mismatch")
def source_mismatch(df: pl.DataFrame) -> pl.DataFrame:
    """The movement record's flight rule (categorical) and Int8 flags where the airport's
    movement record and NM's flight record disagree: flight rule (NM's ``Y`` = IFR-then-VFR
    and ``Z`` = VFR-then-IFR are compared as their first leg, ``I`` / ``V``), aircraft type,
    destination and origin; ``n_mismatch`` is their sum. Every flag is null on a row without
    an NM record. ``diverted`` (filed vs NM destination) lives in ``fpl_deltas``."""
    c = pl.col
    rule_flt = c(S.FLIGHT_RULE_FLT).replace({"Y": "I", "Z": "V"})
    flags = [
        _neq(c(S.FLIGHT_RULE), rule_flt).alias("rule_mismatch"),
        _neq(c(S.AIRCRAFT_TYPE), c(S.AIRCRAFT_TYPE_FLT)).alias("actype_mismatch"),
        _neq(c(S.ADES), c(S.ADES_FLT)).alias("ades_mismatch"),
        _neq(c(S.ADEP), _col_or_null(df, S.ADEP_FLT)).alias("adep_mismatch"),
    ]
    n = pl.sum_horizontal([f.fill_null(0) for f in flags])
    return df.select(
        S.MVT_ID,
        _cat(c(S.FLIGHT_RULE)).alias("flight_rule_mvt"),
        *flags,
        pl.when(c(S.FLIGHT_ID).is_not_null())
        .then(n)
        .otherwise(None)
        .cast(pl.Int8)
        .alias("n_mismatch"),
    )


# --------------------------------------------------------------------------------------
# 3. identity_freq
# --------------------------------------------------------------------------------------

#: (output stem, source column) of the rate encodings.
FREQ_KEYS: tuple[tuple[str, str], ...] = (
    ("flight", S.FLIGHT_NO),
    ("callsign", S.CALLSIGN),
    ("stand", S.STAND),
    ("operator", S.OPERATOR),
    ("dest", S.ADES),
    ("actype", S.AIRCRAFT_TYPE),
)


def carrier_expr() -> pl.Expr:
    """Airline designator of ``FLIGHT_mvt``: its alphabetic prefix, or the first two
    characters when that prefix is shorter than two (IATA codes with a digit, ``4Y512``)."""
    f = pl.col(S.FLIGHT_NO)
    alpha = f.str.extract(r"^([A-Z]+)", 1)
    return pl.when(alpha.str.len_chars() >= 2).then(alpha).otherwise(f.str.slice(0, 2))


def _collapse_rare(s: pl.Series, min_count: int) -> pl.Series:
    """Values seen fewer than ``min_count`` times become ``OTHER``; null stays null."""
    vc = s.value_counts()
    keep = vc.filter(pl.col("count") >= min_count).get_column(s.name)
    return (
        pl.select(
            pl.when(s.is_null())
            .then(None)
            .when(s.is_in(keep.implode()))
            .then(s)
            .otherwise(pl.lit("OTHER"))
        )
        .to_series()
        .alias(s.name)
    )


@feature("identity_freq")
def identity_freq(df: pl.DataFrame) -> pl.DataFrame:
    """Departure rows only. ``flt_carrier`` (designator of ``FLIGHT_mvt``, present where the
    NM operator is null), ``callsign_prefix`` (first three characters of the NM callsign),
    both categorical with levels under 200 departures collapsed to ``OTHER``;
    ``callsign_is_numeric_suffix``. Then, for flight number, callsign, stand, operator,
    destination and aircraft type: ``freq_<key>_share`` = share of the (airport, month)
    departures carrying the row's value, and ``freq_<key>_days`` = distinct UTC days of that
    month on which the value appears at the airport. Null where the value is null."""
    deps = df.filter(pl.col(S.PHASE) == "DEP").select(
        S.MVT_ID,
        S.AIRPORT,
        S.SPLIT_MONTH,
        S.MVT_TIME,
        *dict.fromkeys(col for _, col in FREQ_KEYS),
    )
    deps = deps.with_columns(
        carrier_expr().alias("_carrier"),
        pl.col(S.CALLSIGN).str.slice(0, 3).alias("_csp"),
        pl.col(S.MVT_TIME).dt.date().alias("_day"),
    )
    carrier = _collapse_rare(deps.get_column("_carrier"), MIN_LEVEL_COUNT)
    csp = _collapse_rare(deps.get_column("_csp"), MIN_LEVEL_COUNT)
    grp = [S.AIRPORT, S.SPLIT_MONTH]
    n_grp = pl.len().over(grp).cast(pl.Float64)
    exprs = []
    for stem, col in FREQ_KEYS:
        v = pl.col(col)
        by = [*grp, col]
        exprs.append(
            pl.when(v.is_null())
            .then(None)
            .otherwise(pl.len().over(by) / n_grp)
            .cast(pl.Float32)
            .alias(f"freq_{stem}_share")
        )
        exprs.append(
            pl.when(v.is_null())
            .then(None)
            .otherwise(pl.col("_day").n_unique().over(by))
            .cast(pl.Int16)
            .alias(f"freq_{stem}_days")
        )
    suffix = pl.col(S.CALLSIGN).str.slice(3)
    return deps.select(
        S.MVT_ID,
        _cat(pl.lit(carrier)).alias("flt_carrier"),
        _cat(pl.lit(csp)).alias("callsign_prefix"),
        pl.when(pl.col(S.CALLSIGN).is_null())
        .then(None)
        .otherwise(suffix.str.contains(r"^[0-9]+$"))
        .cast(pl.Int8)
        .alias("callsign_is_numeric_suffix"),
        *exprs,
    )


# --------------------------------------------------------------------------------------
# 4. feed_health
# --------------------------------------------------------------------------------------


def _window_sum(t_ev: np.ndarray, v_ev: np.ndarray, t_q: np.ndarray, w: int) -> np.ndarray:
    """Sum of ``v_ev`` over events with time in [t_q − w, t_q + w] (``t_ev`` sorted)."""
    cs = np.concatenate([[0.0], np.cumsum(v_ev, dtype=np.float64)])
    lo = np.searchsorted(t_ev, t_q - w, side="left")
    hi = np.searchsorted(t_ev, t_q + w, side="right")
    return cs[hi] - cs[lo]


def _share(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full(num.shape, np.nan, dtype=np.float64)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def _run_length(flag: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Length of the run of consecutive ``flag`` rows containing each row (0 where the row's
    flag is 0). A gap above ``MAX_GAP_S`` between neighbours ends a run."""
    n = flag.size
    if n == 0:
        return np.zeros(0, dtype=np.int32)
    brk = np.ones(n, dtype=bool)
    brk[1:] = (flag[1:] != flag[:-1]) | (np.diff(t) > MAX_GAP_S * 1e6)
    run_id = np.cumsum(brk) - 1
    sizes = np.bincount(run_id)
    return np.where(flag == 1, sizes[run_id], 0).astype(np.int32)


@feature("feed_health")
def feed_health(df: pl.DataFrame) -> pl.DataFrame:
    """Departure rows only. Per (airport, data island), centred windows of ±1 h and ±6 h
    around the takeoff: ``fh_nofp_excess_<w>`` = share of the other departures in the window
    with no NM flight record **minus** the same share over the other departures of the row's
    (airport, month) — the raw window share drifts +0.35 / +0.57 sd in 2026-01 (EHAM's
    match rate fell to 95 %; drift table in the EDA file), the excess measures a local feed
    outage against the month's own level; ``fh_spike_share_<w>`` = share of the other departures with an NM
    off-block whose `MVT − AOBT_3` sits on a derived constant (``regime`` block's rule);
    ``fh_arr_sched_share_<w>`` = share of arrivals (landing in the window) whose in-block
    stamp equals the schedule within 10 s. The row itself never counts in its own shares;
    a share with an empty denominator is null. ``fh_nofp_run`` = length of the run of
    consecutive no-flight-plan departures (takeoff order) that contains the row, 0 when the
    row has a flight record."""
    c = pl.col
    island = island_expr().alias("_island")
    # departures: takeoff, NM match and AOBT_3 only (never their off-block stamp or target)
    dep_in = df.filter(c(S.PHASE) == "DEP").select(
        S.MVT_ID, S.AIRPORT, S.MVT_TIME, S.FLIGHT_ID, S.AOBT_3, S.SPLIT_MONTH
    )
    spike = assign_regimes(dep_in).select(S.MVT_ID, ON_SPIKE_COL)
    deps = (
        dep_in.join(spike, on=S.MVT_ID, how="left")
        .select(
            S.MVT_ID,
            S.AIRPORT,
            island,
            c(S.MVT_TIME).dt.epoch("us").alias("_t"),
            c(S.FLIGHT_ID).is_null().cast(pl.Int8).alias("_nofp"),
            # the (airport, month) no-flight-plan share over the row's *other* departures
            (
                (
                    c(S.FLIGHT_ID).is_null().sum().over(S.AIRPORT, S.SPLIT_MONTH).cast(pl.Float64)
                    - c(S.FLIGHT_ID).is_null().cast(pl.Float64)
                )
                / (pl.len().over(S.AIRPORT, S.SPLIT_MONTH).cast(pl.Float64) - 1.0)
            ).alias("_nofp_month"),
            c(S.AOBT_3).is_not_null().cast(pl.Int8).alias("_has_aobt"),
            c(ON_SPIKE_COL).fill_null(0).cast(pl.Int8).alias("_spike"),
        )
        .sort([S.AIRPORT, "_island", "_t", S.MVT_ID])
    )
    # arrivals: only their in-block vs schedule is read (allowed: arrivals' BLOCK_TIME)
    gap = (c(S.BLOCK_TIME) - c(S.SCHED_TIME)).dt.total_seconds().abs()
    arrs = (
        df.filter(c(S.PHASE) == "ARR")
        .select(
            S.AIRPORT,
            island,
            c(S.MVT_TIME).dt.epoch("us").alias("_t"),
            gap.is_not_null().alias("_valid"),
            (gap <= SCHED_EQ_TOL_S).fill_null(False).alias("_eq"),
        )
        .sort([S.AIRPORT, "_island", "_t"])
    )
    arr_groups = arrs.partition_by([S.AIRPORT, "_island"], as_dict=True, maintain_order=True)
    parts = []
    for key, g in deps.partition_by(
        [S.AIRPORT, "_island"], as_dict=True, maintain_order=True
    ).items():
        t = g.get_column("_t").to_numpy()
        nofp = g.get_column("_nofp").to_numpy().astype(np.float64)
        nofp_month = g.get_column("_nofp_month").to_numpy()
        has = g.get_column("_has_aobt").to_numpy().astype(np.float64)
        spk = g.get_column("_spike").to_numpy().astype(np.float64)
        ones = np.ones_like(nofp)
        cols: dict[str, np.ndarray] = {}
        a = arr_groups.get(key)
        for name, w_s in FEED_WINDOWS.items():
            w = w_s * 1_000_000
            n_dep = _window_sum(t, ones, t, w) - 1.0
            win = _share(_window_sum(t, nofp, t, w) - nofp, n_dep)
            cols[f"fh_nofp_excess_{name}"] = win - nofp_month
            n_has = _window_sum(t, has, t, w) - has
            cols[f"fh_spike_share_{name}"] = _share(_window_sum(t, spk, t, w) - spk, n_has)
            if a is None or a.height == 0:
                cols[f"fh_arr_sched_share_{name}"] = np.full(t.shape, np.nan)
            else:
                ta = a.get_column("_t").to_numpy()
                valid = a.get_column("_valid").to_numpy().astype(np.float64)
                eq = a.get_column("_eq").to_numpy().astype(np.float64)
                cols[f"fh_arr_sched_share_{name}"] = _share(
                    _window_sum(ta, eq, t, w), _window_sum(ta, valid, t, w)
                )
        cols["fh_nofp_run"] = _run_length(g.get_column("_nofp").to_numpy(), t)
        part = pl.DataFrame({S.MVT_ID: g.get_column(S.MVT_ID), **cols})
        parts.append(part)
    share_cols = [
        f"fh_{k}_{w}"
        for w in FEED_WINDOWS
        for k in ("nofp_excess", "spike_share", "arr_sched_share")
    ]
    out = pl.concat(parts, how="vertical") if parts else pl.DataFrame({S.MVT_ID: []})
    return out.with_columns(
        [pl.col(x).fill_nan(None).cast(pl.Float32) for x in share_cols]
        + [pl.col("fh_nofp_run").cast(pl.Int32)]
    ).select(S.MVT_ID, *sorted(share_cols), "fh_nofp_run")
