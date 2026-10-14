"""Fold-aware target statistics per group (smoothed mean, and PRU-style quantiles).

These are the one class of feature that must see the target, so they are NOT
``@feature`` blocks: ``eval/run.py`` computes them inside the CV loop when
``cfg.target_encoding`` lists groups. For fold k the statistics come from that fold's
training rows only; for the test set from all training rows.

Statistics (``cfg.te_stats``):
  mean  smoothed toward the global mean with pseudo-count ``alpha`` (rare stands / operators)
  p10   10th percentile — the EUROCONTROL PRU "unimpeded reference" per (airport, stand, runway)
  p50   median
  n     row count in the group (lets the model learn when the reference is unreliable)
"""

from __future__ import annotations

import polars as pl

from prc.data import schema as S

DEFAULT_GROUPS: dict[str, list[str]] = {
    "te_airport_runway": [S.AIRPORT, S.RUNWAY],
    "te_airport_stand": [S.AIRPORT, S.STAND],
    "te_airport_stand_runway": [S.AIRPORT, S.STAND, S.RUNWAY],
    "te_airport_operator": [S.AIRPORT, S.OPERATOR],
    "te_airport_actype": [S.AIRPORT, S.AIRCRAFT_TYPE],
    "te_airport_stand_runway_month": [S.AIRPORT, S.STAND, S.RUNWAY, S.SPLIT_MONTH],
}

STATS = ("mean", "p10", "p50", "n")

#: ``clean_labels``: a taxi-out above this is an artefact of the reporting system, not a taxi
#: Code review 2026-09-19, finding #6.
CLEAN_MAX_S = 3600.0


def clean_mask() -> pl.Expr:
    """Rows whose label is usable as a *statistic* of how long taxiing takes there.

    Two exclusions, both label-defined — which is legitimate here and nowhere else, because
    this filter only ever narrows the rows a fold's statistics are FITTED on. It never
    touches the rows that are scored, so it cannot select the evaluation population.

    * ``y > 3600 s`` — rollovers (+86,400 s), stamp-early artefacts, and the long tail of
      corrupted off-block stamps.
    * the schedule-default artefact: the airport wrote the *scheduled* off-block into the
      block-time field, so ``y`` is exactly ``MVT_TIME - SCHED_TIME`` and measures lateness,
      not taxiing. Detected here as equality to the second; skipped when the frame does not
      carry both timestamps.
    """
    return pl.col(S.TARGET) <= CLEAN_MAX_S


def _sched_default() -> pl.Expr:
    return (
        (pl.col(S.MVT_TIME) - pl.col(S.SCHED_TIME)).dt.total_seconds() - pl.col(S.TARGET)
    ).abs() < 1.0


def encode(
    fit_on: pl.DataFrame,
    apply_to: pl.DataFrame,
    groups: dict[str, list[str]],
    alpha: float = 20.0,
    stats: tuple[str, ...] = ("mean",),
    clean_labels: bool = False,
) -> pl.DataFrame:
    """Return ``apply_to`` keyed by MVT_ID with one column per (group, stat).

    ``clean_labels`` fits the statistics on :func:`clean_mask` rows only (the fit set, never
    the applied set). The 2026-09-19 code review measured the bias it removes:
    ``te_airport_stand_mean`` moves by +10.1 s on average and **+78.1 s at LIRF**, the airport
    that owns a third of the squared error, and ``te_airport_stand_runway_mean`` — the single
    largest feature in R5 at 26.8 % of gain — moves by up to 1,666 s.
    """
    unknown = set(stats) - set(STATS)
    if unknown:
        raise ValueError(f"unknown stats {unknown}; known {STATS}")
    if clean_labels:
        keep = clean_mask()
        if S.MVT_TIME in fit_on.columns and S.SCHED_TIME in fit_on.columns:
            keep = keep & ~_sched_default()
        filtered = fit_on.filter(keep)
        if filtered.height:  # never hand the encoder an empty fit set
            fit_on = filtered
    g_mean = float(fit_on[S.TARGET].mean() or 0.0)  # type: ignore[arg-type]
    keys = sorted({c for cols in groups.values() for c in cols})
    out = apply_to.select(S.MVT_ID, *keys)
    names: list[str] = []
    for name, cols in groups.items():
        aggs = [pl.col(S.TARGET).sum().alias("_s"), pl.len().alias("_n")]
        if "p10" in stats:
            aggs.append(pl.col(S.TARGET).quantile(0.10).alias(f"{name}_p10"))
        if "p50" in stats:
            aggs.append(pl.col(S.TARGET).quantile(0.50).alias(f"{name}_p50"))
        st = fit_on.group_by(cols).agg(aggs)
        exprs = []
        if "mean" in stats:
            exprs.append(
                ((pl.col("_s") + alpha * g_mean) / (pl.col("_n") + alpha)).alias(f"{name}_mean")
            )
        if "n" in stats:
            exprs.append(pl.col("_n").cast(pl.Int32).alias(f"{name}_n"))
        st = st.with_columns(exprs).drop("_s", "_n")
        new = [c for c in st.columns if c not in cols]
        names += new
        out = out.join(st, on=cols, how="left")
        fills = []
        for c in new:
            fills.append(pl.col(c).fill_null(0 if c.endswith("_n") else g_mean))
        out = out.with_columns(fills)
    return out.select(S.MVT_ID, *names)


def encode_inner_month(
    fit_on: pl.DataFrame,
    groups: dict[str, list[str]],
    alpha: float = 20.0,
    stats: tuple[str, ...] = ("mean",),
    clean_labels: bool = False,
    month_col: str = S.SPLIT_MONTH,
) -> pl.DataFrame:
    """Training-row encodings with served-row semantics (``te_train_mode: inner_month``).

    A validation or test row is encoded from rows of *other* months. :func:`encode` called as
    ``encode(fit_on, fit_on, …)`` instead gives every training row statistics that contain its
    own label (the 2026-09-20 parity audit). Here the rows of month ``m`` are encoded by
    :func:`encode` fitted on the rows of ``fit_on`` whose month is not ``m``, with the same
    ``alpha``, ``stats`` and ``clean_labels``; the global mean used for smoothing and null fill
    is therefore that inner fit's mean. Month rather than random k-fold, because same-day rows
    at one stand share congestion and serving always crosses a month boundary.

    Returns one row per row of ``fit_on``, in its order, keyed by MVT_ID — the same shape as
    ``encode(fit_on, fit_on, …)``.
    """
    for name, cols in groups.items():
        if month_col in cols:
            # every inner fit excludes the row's own month, so this group would be unseen
            # for every training row: a constant column, not a statistic.
            raise ValueError(f"{name}: a month-keyed group cannot be encoded inner_month")
    pos = "__te_inner_pos"
    base = fit_on.with_row_index(pos)
    parts: list[pl.DataFrame] = []
    for m in base.get_column(month_col).unique().sort().to_list():
        is_m = pl.col(month_col).eq_missing(m)
        rows_m, fit_m = base.filter(is_m), base.filter(~is_m)
        enc = encode(fit_m.drop(pos), rows_m.drop(pos), groups, alpha, stats, clean_labels)
        parts.append(enc.with_columns(rows_m.get_column(pos)))
    out = pl.concat(parts, how="vertical").sort(pos)
    if not out.get_column(S.MVT_ID).equals(fit_on.get_column(S.MVT_ID)):
        raise AssertionError("encode_inner_month: MVT_ID order not preserved")
    return out.drop(pos)


def encode_train(
    fit_on: pl.DataFrame,
    groups: dict[str, list[str]],
    alpha: float = 20.0,
    stats: tuple[str, ...] = ("mean",),
    clean_labels: bool = False,
    mode: str = "self",
) -> pl.DataFrame:
    """The training rows' encodings under ``cfg.te_train_mode``.

    ``"self"`` is literally ``encode(fit_on, fit_on, …)`` — the pre-2026-09-21 call, so the
    default path is unchanged byte for byte; ``"inner_month"`` is :func:`encode_inner_month`.
    """
    if mode == "self":
        return encode(fit_on, fit_on, groups, alpha, stats, clean_labels)
    if mode == "inner_month":
        return encode_inner_month(fit_on, groups, alpha, stats, clean_labels)
    raise ValueError(f"unknown te_train_mode {mode!r}")
