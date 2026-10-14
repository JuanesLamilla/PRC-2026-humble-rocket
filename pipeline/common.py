"""Shared helpers for the member pipeline. Paths are env-overridable:

RAW_DIR       challenge movement tables (default data/raw)
FEATURES_DIR  ADS-B ground-track feature files (default data/external)
RANKING_FILE  the evaluation set joined as the test split (default
              ranking.parquet; the fold-0 retrains ran with
              RANKING_FILE=final_ranking.parquet)
RESULTS_DIR   text reports (default results)
TEAM_REPO     root of the team-stack checkout (default ".": this repository)
RUNS_DIR      the team stack's run directories (default TEAM_REPO/outputs/runs)
"""
import os
import numpy as np
import polars as pl

RAW = os.environ.get("RAW_DIR", "data/raw")
FEATURES = os.environ.get("FEATURES_DIR", "data/external")
RANKING = os.environ.get("RANKING_FILE", "ranking.parquet")
RESULTS = os.environ.get("RESULTS_DIR", "results")
TEAM = os.environ.get("TEAM_REPO", ".")
RUNS = os.environ.get("RUNS_DIR", f"{TEAM}/outputs/runs")
os.makedirs("work", exist_ok=True)
W = 3606.0  # |BLOCK - LOBT| <= 3606 s on every 2025 departure with an NM record


def sec(a, b):
    return (pl.col(a) - pl.col(b)).dt.total_seconds().cast(pl.Float64)


def load_movements():
    """All movements (2025 training + 2026 ranking), DEP and ARR, with a split column."""
    import glob

    tr = [pl.read_parquet(f) for f in sorted(glob.glob(f"{RAW}/training_2025-*.parquet"))]
    tr = pl.concat(tr, how="diagonal_relaxed").with_columns(pl.lit("train").alias("split"))
    rk = pl.read_parquet(f"{RAW}/{RANKING}").with_columns(pl.lit("test").alias("split"))
    df = pl.concat([tr, rk], how="diagonal_relaxed")
    return df


def dep_bounds(df):
    """Departures with the label-free LOBT window bounds lo/hi on y (null when no LOBT)."""
    d = df.filter(pl.col("PHASE_mvt") == "DEP").with_columns(
        C=sec("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"),
        ml=sec("MVT_TIME_UTC_mvt", "LOBT_flt"),
        sl=sec("SCHED_TIME_UTC_mvt", "LOBT_flt"),
    )
    d = d.with_columns(
        lo=pl.max_horizontal(pl.lit(0.0), pl.col("ml") - W),
        hi=pl.col("ml") + W,
    )
    return d


def rmse(e):
    e = np.asarray(e, dtype=float)
    return float(np.sqrt(np.mean(e * e))) if len(e) else float("nan")


def block_boot(y, a, b, blocks, n=1000, seed=0):
    """Day-block bootstrap of RMSE(b) - RMSE(a). Returns point, lo, hi (95 %)."""
    y, a, b = (np.asarray(v, float) for v in (y, a, b))
    ea, eb = (a - y) ** 2, (b - y) ** 2
    codes, inv = np.unique(blocks, return_inverse=True)
    sa = np.bincount(inv, ea, len(codes))
    sb = np.bincount(inv, eb, len(codes))
    cn = np.bincount(inv, None, len(codes))
    rng = np.random.default_rng(seed)
    pt = np.sqrt(sb.sum() / cn.sum()) - np.sqrt(sa.sum() / cn.sum())
    out = []
    for _ in range(n):
        k = rng.integers(0, len(codes), len(codes))
        N = cn[k].sum()
        out.append(np.sqrt(sb[k].sum() / N) - np.sqrt(sa[k].sum() / N))
    return float(pt), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))
