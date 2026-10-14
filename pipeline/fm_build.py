"""Build the feature matrix for departures (2025 train + the test split):
raw-derived features (newfeat.py), the ADS-B ground-track columns, and the
LOBT-window and label-structure columns. Writes FM_OUT (default work/fm.parquet).

The test split is whatever RANKING_FILE names (common.py): ranking.parquet for
the primary members, final_ranking.parquet for the fold-0 retrains."""
import glob, os, re
import numpy as np
import polars as pl
from common import RESULTS, sec
from newfeat import build_dep, NEW, key

out = open(f"{RESULTS}/fm_build_report.txt", "w")
def log(*a):
    print(*a); print(*a, file=out, flush=True)

FM_OUT = os.environ.get("FM_OUT", "work/fm.parquet")
dep, acols = build_dep()
dep = dep.with_columns(
    y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64), is_test=(pl.col("split") == "test"),
    lo=pl.max_horizontal(pl.lit(0.0), pl.col("ml") - 3620), hi=pl.col("ml") + 3620,
    dlt=sec("BLOCK_TIME_UTC_mvt", "LOBT_flt"),
    dflt=((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() <= 10),
    me=sec("MVT_TIME_UTC_mvt", "EOBT_1_flt"), mi=sec("MVT_TIME_UTC_mvt", "IOBT_flt"),
    cross=(pl.col("MVT_TIME_UTC_mvt").dt.date() > pl.col("SCHED_TIME_UTC_mvt").dt.date()).cast(pl.Float64),
    sched_infeas=(pl.col("sl").abs() > 3616).cast(pl.Float64),
    day=pl.col("MVT_TIME_UTC_mvt").dt.date().cast(pl.Utf8))
base_cols = [key, "split", "is_test", "ADEP_mvt", "mon", "day", "y", "dlt", "dflt", "C", "ml", "sl", "ma", "me", "mi", "nm",
             "lo", "hi", "cross", "sched_infeas", "hour"] + NEW + acols
fm = dep.select([c for c in base_cols if c in dep.columns])
num = [c for c, t in fm.schema.items() if t in (pl.Float64,)]
fm = fm.with_columns([pl.col(c).cast(pl.Float32) for c in num if c not in ("y", "dlt", "C", "ml", "lo", "hi", key)])
assert fm[key].n_unique() == fm.height, "key not unique"
log("fm shape", fm.shape, "train", fm.filter(~pl.col("is_test")).height, "test", fm.filter(pl.col("is_test")).height)
log("cat columns:", [c for c in fm.columns if c.startswith("cat_")])
fm.write_parquet(FM_OUT)
log("done", FM_OUT)
