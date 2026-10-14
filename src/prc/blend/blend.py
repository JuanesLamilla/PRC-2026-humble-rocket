"""OOF-based blending of several runs.

Given run names, align their ``oof.parquet`` on MVT_ID (they must share the fold
scheme), fit non-negative weights minimizing RMSE (scipy NNLS with a sum-to-one
renormalization), report the blended CV RMSE, and apply the same weights to the
runs' ``test_pred.parquet``.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.optimize import nnls

from prc import paths
from prc.data import schema as S
from prc.eval.metrics import rmse


def _load(run: str, which: str) -> pl.DataFrame:
    return pl.read_parquet(paths.RUNS / run / f"{which}.parquet").select(
        S.MVT_ID, pl.col("pred").alias(run), *([S.TARGET] if which == "oof" else [])
    )


def blend(runs: list[str]) -> tuple[dict[str, float], float, pl.DataFrame]:
    oof = _load(runs[0], "oof")
    for r in runs[1:]:
        oof = oof.join(_load(r, "oof").drop(S.TARGET), on=S.MVT_ID, how="inner")
    P = oof.select(runs).to_numpy()
    y = oof[S.TARGET].to_numpy()
    w, _ = nnls(P, y)
    if w.sum() > 0:
        w = w / w.sum()
    score = rmse(y, P @ w)
    for r in runs:
        print(f"  {r:40s} rmse={rmse(y, oof[r].to_numpy()):.3f} w={w[runs.index(r)]:.3f}")
    print(f"  blend rmse={score:.3f} (n={len(y)})")

    test = _load(runs[0], "test_pred")
    for r in runs[1:]:
        test = test.join(_load(r, "test_pred"), on=S.MVT_ID, how="inner")
    tp = test.select(runs).to_numpy() @ w
    out = test.select(S.MVT_ID).with_columns(pl.Series("pred", tp))
    return dict(zip(runs, map(float, w), strict=True)), score, out


__all__ = ["blend", "np"]
