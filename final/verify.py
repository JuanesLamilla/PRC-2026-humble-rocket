"""Validate the submission file against the organizers' template.

usage (repo root): python final/verify.py
"""

import hashlib
import os
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")  # R-arrow ext types in the raw files
import polars as pl

KEY = "MVT_ID_mvt"
TARGET = "TAXITIME_SEC_mvt"
RAW = Path(os.environ.get("RAW_DIR", "data/raw"))
SUBMISSIONS = Path(os.environ.get("SUBMISSIONS_DIR", "submissions"))

path = SUBMISSIONS / "humble-rocket_final.parquet"
out = pl.read_parquet(path)
template = pl.read_parquet(RAW / "final_submitting.parquet")
assert out.height == template.height, (out.height, template.height)
assert set(out.columns) == {KEY, TARGET}
assert out.schema[TARGET] == pl.Int32
assert out[TARGET].null_count() == 0 and (out[TARGET] >= 0).all()
assert set(out[KEY].to_list()) == set(template[KEY].to_list())
print(f"{path.name}: OK ({out.height} rows)")

mon = (pl.scan_parquet(RAW / "final_ranking.parquet").filter(pl.col("PHASE_mvt") == "DEP")
       .select(KEY, pl.col("MVT_TIME_UTC_mvt").dt.month().alias("mon")).collect())
stats = (out.join(mon, on=KEY).group_by("mon")
         .agg(n=pl.len(), mean=pl.col(TARGET).mean(),
              p50=pl.col(TARGET).quantile(0.5), p90=pl.col(TARGET).quantile(0.9)).sort("mon"))
print(stats)
print("  sha256", hashlib.sha256(path.read_bytes()).hexdigest())
