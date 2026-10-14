from __future__ import annotations

import numpy as np
import polars as pl


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def rmse_breakdown(df: pl.DataFrame, y: str, p: str, by: list[str]) -> pl.DataFrame:
    """RMSE per group (e.g. airport, month) plus the group's share of total SSE."""
    return (
        df.group_by(by)
        .agg(
            pl.len().alias("n"),
            ((pl.col(y) - pl.col(p)) ** 2).mean().sqrt().alias("rmse"),
            ((pl.col(y) - pl.col(p)) ** 2).sum().alias("sse"),
        )
        .with_columns((pl.col("sse") / pl.col("sse").sum()).alias("sse_share"))
        .sort("sse", descending=True)
    )


def bootstrap_rmse_std(y: np.ndarray, p: np.ndarray, n: int = 200, seed: int = 0) -> float:
    """Std of RMSE under row resampling: the *unpaired* noise floor of a single score.
    (Paired comparisons of two models on the same rows resolve ~10x finer.)"""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    rng = np.random.default_rng(seed)
    se = (y - p) ** 2
    idx = rng.integers(0, len(se), size=(n, len(se)))
    return float(np.sqrt(se[idx].mean(axis=1)).std())


def paired_rmse_delta_std(
    y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, n: int = 200, seed: int = 0
) -> float:
    """Std of RMSE(a) − RMSE(b) under paired row resampling: the resolution of an A/B comparison."""
    y = np.asarray(y, dtype=np.float64)
    sa, sb = (y - p_a) ** 2, (y - p_b) ** 2
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n, len(y)))
    d = np.sqrt(sa[idx].mean(axis=1)) - np.sqrt(sb[idx].mean(axis=1))
    return float(d.std())
