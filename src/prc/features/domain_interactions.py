"""Domain interactions: products and ratios with a physical reading (backlog X2).

The neural members learn interactions from raw columns; the tree member splits one column at a
time, so a ratio it needs (a delay *relative to the operator's usual delay*, a queue *per active
runway*) costs it many splits. This block writes 14 of them down. No brute-force search: each
column below has one reading, fixed before the screen.

Typical values ("the operator's typical delay", "the hour's typical queue") are medians over the
2025 (training-month) departures. They use only test-visible inputs, so no month-holding is
needed; the test rows are left out of the medians so that the 2026 months cannot move them.

| column | reading |
|---|---|
| ``ix_delay_vs_op``        | (AOBT_3 − SCHED) − the airport × operator median of it: is this departure later than the operator usually is |
| ``ix_delay_ratio_op``     | (AOBT_3 − SCHED) / (airport × operator median of |AOBT_3 − SCHED| + 60 s) |
| ``ix_c_vs_op``            | (MVT − SCHED) − its airport × operator median |
| ``ix_eobt_slip_vs_op``    | (AOBT_3 − EOBT_1) − its airport × operator median: an unusual last-minute slip |
| ``ix_p_vs_hour``          | NM proxy (MVT − AOBT_3) − its airport × local-hour median |
| ``ix_p_per_km``           | NM proxy / stand→runway path (s per km): inverse average speed including queueing |
| ``ix_p_vs_nbr``           | NM proxy / the median proxy of the 10 previous measured takeoffs on the runway |
| ``ix_queue_per_rwy``      | aircraft taxiing at push / departure runways active in the hour |
| ``ix_queue_vs_hour``      | aircraft taxiing at push / (airport × local-hour median of it + 1) |
| ``ix_dep_rate_vs_hour``   | departures in ±30 min / (airport × local-hour median of it + 1) |
| ``ix_arr_dep_ratio``      | landings / (takeoffs + 1) in the last 30 min: mixed-mode runway pressure |
| ``ix_backlog``            | scheduled departures in the hour / (takeoffs in the last 60 min + 1) |
| ``ix_tko_rate_in_taxi``   | takeoffs while this aircraft taxied / its NM taxi minutes: the service rate it saw |
| ``ix_overtaken_share``    | departures that pushed later and took off earlier / (takeoffs while it taxied + 1): a hold |

Departure rows only. Dependencies through the registry, folded into the cache key.
"""

from __future__ import annotations

import polars as pl

from prc.data import schema as S
from prc.features.registry import build, feature
from prc.features.screen_common import deps_digest, local_hour, safe_div, secs

DEPS = ("surface_queue", "throughput", "sched_demand", "runway_config", "taxi_distance")
DEP_COLS = (
    "sq_taxiing_at_push",
    "sq_tko_during_taxi",
    "sq_overtaken_by",
    "sq_nbr_proxy_med_prev10",
    "n_dep_tko_30m",
    "n_arr_ldg_30m",
    "n_dep_tko_60m",
    "n_sched_dep_60m",
    "n_dep_rwy_active_60m",
    "td_path_m",
)
OUT = [
    "ix_delay_vs_op",
    "ix_delay_ratio_op",
    "ix_c_vs_op",
    "ix_eobt_slip_vs_op",
    "ix_p_vs_hour",
    "ix_p_per_km",
    "ix_p_vs_nbr",
    "ix_queue_per_rwy",
    "ix_queue_vs_hour",
    "ix_dep_rate_vs_hour",
    "ix_arr_dep_ratio",
    "ix_backlog",
    "ix_tko_rate_in_taxi",
    "ix_overtaken_share",
]


def _ix_external() -> str:
    return deps_digest(__name__, DEPS)


def interactions_from(df: pl.DataFrame, deps: pl.DataFrame) -> pl.DataFrame:
    """Block output for movement table ``df`` given ``deps`` (``MVT_ID`` + ``DEP_COLS``)."""
    c = pl.col
    dep = (
        df.filter(c(S.PHASE) == "DEP")
        .select(
            S.MVT_ID,
            S.AIRPORT,
            S.IS_TEST,
            c(S.OPERATOR).cast(pl.Utf8).fill_null("NA").alias("_op"),
            local_hour().alias("_hour"),
            secs(S.AOBT_3, S.SCHED_TIME).alias("_delay"),
            secs(S.MVT_TIME, S.SCHED_TIME).alias("_c"),
            secs(S.AOBT_3, S.EOBT_1).alias("_slip"),
            secs(S.MVT_TIME, S.AOBT_3).alias("_p"),
        )
        .join(deps.select(S.MVT_ID, *DEP_COLS), on=S.MVT_ID, how="left")
    )
    train = dep.filter(~c(S.IS_TEST))
    by_op = train.group_by(S.AIRPORT, "_op").agg(
        c("_delay").median().alias("_op_delay"),
        c("_delay").abs().median().alias("_op_absdelay"),
        c("_c").median().alias("_op_c"),
        c("_slip").median().alias("_op_slip"),
    )
    by_hour = train.group_by(S.AIRPORT, "_hour").agg(
        c("_p").filter(c("_p").is_between(0, 3 * 3600)).median().alias("_h_p"),
        c("sq_taxiing_at_push").median().alias("_h_queue"),
        c("n_dep_tko_30m").median().alias("_h_dep"),
    )
    dep = dep.join(by_op, on=[S.AIRPORT, "_op"], how="left").join(
        by_hour, on=[S.AIRPORT, "_hour"], how="left"
    )
    p_min = c("_p") / 60.0
    out = dep.select(
        S.MVT_ID,
        (c("_delay") - c("_op_delay")).alias("ix_delay_vs_op"),
        (c("_delay") / (c("_op_absdelay") + 60.0)).alias("ix_delay_ratio_op"),
        (c("_c") - c("_op_c")).alias("ix_c_vs_op"),
        (c("_slip") - c("_op_slip")).alias("ix_eobt_slip_vs_op"),
        (c("_p") - c("_h_p")).alias("ix_p_vs_hour"),
        safe_div(c("_p"), c("td_path_m") / 1000.0).alias("ix_p_per_km"),
        safe_div(c("_p"), c("sq_nbr_proxy_med_prev10")).alias("ix_p_vs_nbr"),
        safe_div(c("sq_taxiing_at_push"), c("n_dep_rwy_active_60m")).alias("ix_queue_per_rwy"),
        (c("sq_taxiing_at_push") / (c("_h_queue") + 1.0)).alias("ix_queue_vs_hour"),
        (c("n_dep_tko_30m") / (c("_h_dep") + 1.0)).alias("ix_dep_rate_vs_hour"),
        (c("n_arr_ldg_30m") / (c("n_dep_tko_30m") + 1.0)).alias("ix_arr_dep_ratio"),
        (c("n_sched_dep_60m") / (c("n_dep_tko_60m") + 1.0)).alias("ix_backlog"),
        pl.when(p_min > 0)
        .then(c("sq_tko_during_taxi") / pl.max_horizontal(p_min, pl.lit(1.0)))
        .otherwise(None)
        .alias("ix_tko_rate_in_taxi"),
        (c("sq_overtaken_by") / (c("sq_tko_during_taxi") + 1.0)).alias("ix_overtaken_share"),
    )
    return out.select(S.MVT_ID, *[c(n).cast(pl.Float32) for n in OUT])


@feature("domain_interactions", external=_ix_external)
def domain_interactions(df: pl.DataFrame) -> pl.DataFrame:
    """Departure rows only. 14 physically-read ratios / differences: delay vs the operator's
    usual, NM proxy vs the hour's usual / the neighbours / the path, queue per active runway /
    vs the hour, departure rate vs the hour, arrival pressure, backlog, service rate seen during
    the taxi, overtaken share."""
    return interactions_from(df, build(df, list(DEPS)))
