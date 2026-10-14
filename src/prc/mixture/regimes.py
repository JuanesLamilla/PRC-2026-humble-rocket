"""Regime assignment for the NM off-block proxy (plan M2 / approach-ideas A1).

`AOBT_3_flt` is *either* the airport's actual off-block received by message, *or*, absent
such a message, `takeoff − a standard taxi time` computed by NM (ADRR metadata, quoted in
internal note 2026-09-17_literature-analysis §2.3). So `MVT − AOBT_3` is a mixture and
every departure falls into one of three regimes:

    A  real NM off-block   AOBT_3 present and `MVT − AOBT_3` NOT on a derived constant
    B  derived constant    AOBT_3 present and `MVT − AOBT_3` within ±tol of one of the
                           airport's spike modes
    C  no NM match         AOBT_3 missing

Nothing here reads the target or the off-block stamp of the row it describes, and both
`MVT_TIME_UTC_mvt` and `AOBT_3_flt` are populated on the ranking rows, so the same rule
assigns regimes at test time.
"""

from __future__ import annotations

import polars as pl

from prc.data import schema as S

#: Per-airport derived-AOBT constants, in seconds of `MVT − AOBT_3`.
#:
#: Provenance: internal note 2026-09-17_arrival-regime §3 (1 s bins of `MVT − AOBT_3` over 2025 + 2026 departures; a bin is a spike when it holds > 8×
#: the local mean of the surrounding ±60 s and > 200 rows). The listed bins cluster tightly
#: (LTFM 1194–1205 and 1495–1505, LIRF 595 and 1495–1505, EDDM 295–305, EGLL 595–605), so each
#: cluster is stored as its centre and matched with ``SPIKE_TOL_S``. Airports with no spike bin
#: (EDDF, EHAM, LEBL, LEMD, LFPG, LSZH) have no derived-AOBT regime and never produce a B row.
#: Re-validate with :func:`detect_spike_modes` / :func:`validate_spike_modes` when the data is
#: reissued, never by hand.
SPIKE_MODES: dict[str, tuple[float, ...]] = {
    "EDDM": (300.0,),
    "EGLL": (600.0,),
    "LIRF": (595.0, 1500.0),
    "LTFM": (1200.0, 1500.0),
}

#: Half-width of the match window around a mode, seconds (the observed clusters are ±5 s wide).
SPIKE_TOL_S: float = 6.0

REGIMES: tuple[str, ...] = ("A", "B", "C")

PROXY_COL = "proxy_s"
REGIME_COL = "regime"
ON_SPIKE_COL = "on_spike_any"
SPIKE_VALUE_COL = "spike_value_s"


def proxy_expr() -> pl.Expr:
    """`MVT − AOBT_3` in seconds: the naive NM taxi-out proxy (null without an NM match)."""
    return (pl.col(S.MVT_TIME) - pl.col(S.AOBT_3)).dt.total_seconds().cast(pl.Float64)


def assign_regimes(
    df: pl.DataFrame,
    modes: dict[str, tuple[float, ...]] | None = None,
    tol_s: float = SPIKE_TOL_S,
) -> pl.DataFrame:
    """One row per ``MVT_ID_mvt``: ``regime`` (A/B/C), ``on_spike_any``, ``spike_value_s``,
    ``proxy_s``.

    Pure function of (airport, takeoff time, AOBT_3) — deterministic, order-independent and
    identical for train and test rows.
    """
    modes = SPIKE_MODES if modes is None else modes
    d = proxy_expr()
    on_spike: pl.Expr = pl.lit(False)
    spike_value: pl.Expr = pl.lit(0.0, dtype=pl.Float64)
    for airport in sorted(modes):
        for mode in sorted(modes[airport]):
            hit = (pl.col(S.AIRPORT) == airport) & ((d - mode).abs() <= tol_s)
            on_spike = on_spike | hit.fill_null(False)
            spike_value = pl.when(hit.fill_null(False)).then(pl.lit(mode)).otherwise(spike_value)
    regime = (
        pl.when(d.is_null())
        .then(pl.lit("C"))
        .when(on_spike)
        .then(pl.lit("B"))
        .otherwise(pl.lit("A"))
    )
    return df.select(
        S.MVT_ID,
        regime.alias(REGIME_COL),
        on_spike.cast(pl.Int8).alias(ON_SPIKE_COL),
        spike_value.cast(pl.Float32).alias(SPIKE_VALUE_COL),
        d.alias(PROXY_COL),
    )


def detect_spike_modes(
    df: pl.DataFrame,
    min_rows: int = 200,
    ratio: float = 8.0,
    window: int = 121,
    max_delta: int = 7200,
    cluster_gap_s: int = 30,
) -> dict[str, tuple[float, ...]]:
    """Recompute the derived-AOBT constants from the data, so :data:`SPIKE_MODES` can be
    validated instead of trusted.

    Same detector as the arrival-regime EDA, §3, (1 s histogram of `MVT − AOBT_3` per
    airport; a bin is a spike when it holds > ``ratio`` × the local mean over ``window`` bins and
    more than ``min_rows`` rows), plus a clustering step: adjacent spike bins closer than
    ``cluster_gap_s`` are one mode, reported at the centre of the cluster.
    """
    dep = df.filter((pl.col(S.PHASE) == "DEP") & pl.col(S.AOBT_3).is_not_null()).with_columns(
        proxy_expr().round(0).cast(pl.Int64).alias("d")
    )
    counts = (
        dep.filter(pl.col("d").is_between(0, max_delta))
        .group_by(S.AIRPORT, "d")
        .agg(pl.len().alias("n"))
    )
    if counts.is_empty():
        return {}
    airports = counts.get_column(S.AIRPORT).unique().sort()
    grid = pl.DataFrame({S.AIRPORT: airports}).join(
        pl.DataFrame({"d": pl.int_range(0, max_delta + 1, eager=True).cast(pl.Int64)}), how="cross"
    )
    hist = (
        grid.join(counts, on=[S.AIRPORT, "d"], how="left")
        .with_columns(pl.col("n").fill_null(0))
        .sort(S.AIRPORT, "d")
        .with_columns(
            pl.col("n").rolling_mean(window_size=window, center=True).over(S.AIRPORT).alias("local")
        )
    )
    spikes = hist.filter((pl.col("n") > ratio * pl.col("local")) & (pl.col("n") > min_rows)).sort(
        S.AIRPORT, "d"
    )
    out: dict[str, tuple[float, ...]] = {}
    for airport in spikes.get_column(S.AIRPORT).unique().sort().to_list():
        ds = spikes.filter(pl.col(S.AIRPORT) == airport).get_column("d").to_list()
        clusters: list[list[int]] = []
        for value in ds:
            if clusters and value - clusters[-1][-1] <= cluster_gap_s:
                clusters[-1].append(value)
            else:
                clusters.append([value])
        out[airport] = tuple(round((c[0] + c[-1]) / 2, 1) for c in clusters)
    return out


def validate_spike_modes(
    df: pl.DataFrame,
    table: dict[str, tuple[float, ...]] | None = None,
    tol_s: float = SPIKE_TOL_S,
    **detect_kwargs,
) -> pl.DataFrame:
    """Detected modes vs the hardcoded table, one row per detected mode.

    ``covered`` is False when a mode found in the data is not within ``tol_s`` of any constant in
    the table — i.e. when :data:`SPIKE_MODES` needs updating.
    """
    table = SPIKE_MODES if table is None else table
    detected = detect_spike_modes(df, **detect_kwargs)
    rows = []
    for airport in sorted(set(detected) | set(table)):
        found = detected.get(airport, ())
        listed = table.get(airport, ())
        for mode in found:
            rows.append(
                {
                    S.AIRPORT: airport,
                    "detected_s": mode,
                    "table_s": min(listed, key=lambda m: abs(m - mode), default=None),
                    "covered": any(abs(m - mode) <= tol_s for m in listed),
                }
            )
        for mode in listed:
            if not any(abs(m - mode) <= tol_s for m in found):
                rows.append(
                    {S.AIRPORT: airport, "detected_s": None, "table_s": mode, "covered": False}
                )
    schema = {
        S.AIRPORT: pl.Utf8,
        "detected_s": pl.Float64,
        "table_s": pl.Float64,
        "covered": pl.Boolean,
    }
    return pl.DataFrame(rows, schema=schema)
