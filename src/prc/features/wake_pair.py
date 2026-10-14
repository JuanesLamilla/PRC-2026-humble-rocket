"""Wake-turbulence separation on the departure runway (plan row P2-e).

The minimum separation between two consecutive departures is set by the wake category of
the *leader* and the *follower* (ICAO / RECAT-EU): a Medium behind a Heavy waits about twice
as long as a Medium behind a Medium. The takeoff times of every movement are known post-ops,
so the realised gap to the preceding takeoff on the same runway and the wake pair that
produced it are both observable for train and test rows alike.

This block is deliberately narrow: the gap to the previous / next takeoff on the runway plus
the wake category of the leader and the ``leader|follower`` pair as a categorical. Column
names are prefixed ``wp_`` so the block composes with ``runway_gaps`` and ``discharge``,
which carry similar quantities under their own names.

Never reads the described row's off-block time or target; takeoff times only.
"""

from __future__ import annotations

import polars as pl

from prc.data import schema as S
from prc.features.registry import feature


@feature("wake_pair")
def wake_pair(df: pl.DataFrame) -> pl.DataFrame:
    """Seconds to the previous / next takeoff on the same (airport, runway), the preceding
    aircraft's wake category, and the (leader|follower) wake pair as a categorical."""
    deps = (
        df.filter(pl.col(S.PHASE) == "DEP")
        .select(S.MVT_ID, S.AIRPORT, S.RUNWAY, S.MVT_TIME, S.WAKE_CAT)
        .sort([S.AIRPORT, S.RUNWAY, S.MVT_TIME])
    )
    grp = [S.AIRPORT, S.RUNWAY]
    t = pl.col(S.MVT_TIME)
    own = pl.col(S.WAKE_CAT).cast(pl.Utf8).fill_null("NA")
    lead = pl.col(S.WAKE_CAT).shift(1).over(grp).cast(pl.Utf8).fill_null("NA")
    return deps.select(
        S.MVT_ID,
        (t - t.shift(1).over(grp)).dt.total_seconds().cast(pl.Float32).alias("wp_gap_prev_tko_s"),
        (t.shift(-1).over(grp) - t).dt.total_seconds().cast(pl.Float32).alias("wp_gap_next_tko_s"),
        lead.cast(pl.Categorical).alias("wp_prev_wake"),
        own.cast(pl.Categorical).alias("wp_own_wake"),
        pl.concat_str([lead, pl.lit("|"), own]).cast(pl.Categorical).alias("wp_wake_pair"),
    )
