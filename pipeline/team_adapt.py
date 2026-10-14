"""Convert a team-stack run (outputs/runs/<run>/{oof,test_pred}.parquet) into a member tag
the class model can read: work/team_<short>/outer_XX.parquet with the prediction column y_<short>.

usage: python pipeline/team_adapt.py <run> <short>     e.g. ... exp9004_setenc_overnight_extra__s1-2-3__month_loo t9004
"""
import sys, os, polars as pl
from common import RUNS

run, short = sys.argv[1], sys.argv[2]
base = f"{RUNS}/{run}"
key = "MVT_ID_mvt"
oof = pl.read_parquet(f"{base}/oof.parquet"); tp = pl.read_parquet(f"{base}/test_pred.parquet")
mon = pl.read_parquet("work/fm2.parquet", columns=[key, "mon", "y"])
o = oof.select(key, pl.col("pred").alias(f"y_{short}")).join(mon, on=key)
os.makedirs(f"work/team_{short}", exist_ok=True)
for m in range(1, 13):
    o.filter(pl.col("mon") == m).write_parquet(f"work/team_{short}/outer_{m:02d}.parquet")
tp.select(key, pl.col("pred").alias(f"y_{short}")).join(mon.select(key, "mon"), on=key).write_parquet(f"work/team_{short}/outer_00.parquet")
print(run, o.height, tp.height)
