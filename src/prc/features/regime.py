"""Off-block reporting regime of the NM proxy (plan M2 / approach-ideas A1).

Exposes the rule gate of ``prc.mixture.regimes`` as a feature block: which of the three
regimes a departure is in, whether `MVT − AOBT_3` sits on one of that airport's derived-AOBT
constants, and which constant. Reads only the takeoff time and ``AOBT_3_flt``, both populated
on the ranking rows, so train and test rows are described identically.
"""

from __future__ import annotations

import polars as pl

from prc.data import schema as S
from prc.features.registry import feature
from prc.mixture.regimes import (
    ON_SPIKE_COL,
    REGIME_COL,
    SPIKE_VALUE_COL,
    assign_regimes,
)


@feature("regime")
def regime(df: pl.DataFrame) -> pl.DataFrame:
    """Regime A (real NM off-block) / B (derived constant) / C (no NM match), plus the
    derived-AOBT constant the row sits on (0 when it sits on none)."""
    out = assign_regimes(df)
    return out.select(
        S.MVT_ID,
        pl.col(REGIME_COL).cast(pl.Categorical),
        pl.col(ON_SPIKE_COL),
        pl.col(SPIKE_VALUE_COL),
    )
