"""Stage 0: check the inputs and build the ATFM day table.

usage (repo root): python pipeline/prepare_data.py

* verifies the challenge movement tables and the ADS-B feature files are in
  place (data/README.md lists where each one comes from);
* converts the EUROCONTROL ATFM workbook to data/raw/apt_arr_atfm.parquet
  (skipped when the parquet is already there).
"""
import glob
import os
import sys
from pathlib import Path

import polars as pl

RAW = Path(os.environ.get("RAW_DIR", "data/raw"))
FEATURES = Path(os.environ.get("FEATURES_DIR", "data/external"))
AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]


def build_atfm_table() -> None:
    out = RAW / "apt_arr_atfm.parquet"
    if out.exists():
        print(f"ok: {out}")
        return
    workbook = FEATURES / "Airport_Arrival_ATFM_Delay.xlsx"
    if not workbook.exists():
        sys.exit(f"missing {out} and no workbook at {workbook} to build it from (see data/README.md)")
    import pandas as pd

    sheet = pd.read_excel(workbook, sheet_name="DATA")
    table = pl.from_pandas(sheet).filter(pl.col("APT_ICAO").is_in(AIRPORTS))
    table = table.with_columns(d=pl.col("FLT_DATE").cast(pl.Utf8).str.slice(0, 10))
    keep = ["APT_ICAO", "d"] + [c for c in table.columns if c.startswith(("FLT_ARR", "DLY_APT_ARR"))]
    table.select(keep).write_parquet(out)
    print(f"built {out} from {workbook}")


def main() -> None:
    required = (
        sorted(glob.glob(str(RAW / "training_2025-*.parquet")))
        or [RAW / "training_2025-01-01_2025-02-01.parquet"]
    )
    required += [RAW / "ranking.parquet", RAW / "final_ranking.parquet", RAW / "final_submitting.parquet",
                 FEATURES / "adsb_ground.parquet", FEATURES / "adsb_ground_v2.parquet"]
    missing = [str(p) for p in required if not Path(p).exists()]
    if len(sorted(glob.glob(str(RAW / "training_2025-*.parquet")))) != 12:
        missing.append(str(RAW / "training_2025-*.parquet") + " (expected 12 monthly files)")
    if missing:
        sys.exit("missing inputs (see data/README.md):\n  " + "\n  ".join(missing))
    print("raw inputs ok")
    build_atfm_table()


if __name__ == "__main__":
    main()
