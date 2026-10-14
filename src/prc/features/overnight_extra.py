"""Overnight extra block (2026-10-06): features built in ``overnight/newfeat.py`` + ``overnight/fm_v2.py``.

Stand handover (last / second-last arrival in-block at the stand relative to MVT, AOBT_3, LOBT, SCHED; next
occupant), operator x month / day arrival schedule-default shares and their month shift, NM-neighbour median
MVT - AOBT_3, day no-NM share, NM plan residuals (ARVT - EOBT vs route x type median), and EUROCONTROL daily
airport arrival ATFM delay (open data). All test-computable; arrival in-block stamps are allowed inputs.
Reads the newest ``data/external/overnight_extra*/*.parquet`` (MVT_ID_mvt + feature columns).
"""

from __future__ import annotations

import polars as pl

from prc.data import schema as S
from prc.features.registry import archive_files, external_digest, feature


def _files():
    return archive_files("overnight_extra*", "*.parquet")


@feature("overnight_extra", external=lambda: external_digest(_files(), __name__))
def overnight_extra(df: pl.DataFrame) -> pl.DataFrame:
    """New stand-handover / arrival-stamp / NM-neighbour / plan-residual / ATFM features."""
    fs = _files()
    if not fs:
        return df.select(S.MVT_ID)
    x = pl.read_parquet(fs[-1])
    return df.select(S.MVT_ID).join(x, on=S.MVT_ID, how="left")
