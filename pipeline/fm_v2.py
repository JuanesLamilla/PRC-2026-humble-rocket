"""fm v2 = fm + EUROCONTROL daily airport arrival ATFM delay features (airport x UTC date; open data,
https://www.eurocontrol.int/performance/data/download/xls/Airport_Arrival_ATFM_Delay.xlsx, fetched 2026-10-05;
null = no regulation -> 0). Writes work/fm2.parquet."""
import os
import polars as pl
RAW = os.environ.get("RAW_DIR", "data/raw")
FM_IN = os.environ.get("FM_IN", "work/fm.parquet")
FM_OUT = os.environ.get("FM_OUT", "work/fm2.parquet")
a = pl.read_parquet(f"{RAW}/apt_arr_atfm.parquet").rename({"APT_ICAO": "ADEP_mvt", "d": "day"})
num = [c for c in a.columns if c not in ("ADEP_mvt", "day")]
a = a.with_columns([pl.col(c).fill_null(0.0) for c in num])
a = a.with_columns(
    atfm_flt=pl.col("FLT_ARR_1"), atfm_dly_pf=pl.col("DLY_APT_ARR_1") / pl.col("FLT_ARR_1").clip(1),
    atfm_wx_pf=pl.col("DLY_APT_ARR_W_1") / pl.col("FLT_ARR_1").clip(1), atfm_cap_pf=pl.col("DLY_APT_ARR_G_1") / pl.col("FLT_ARR_1").clip(1),
    atfm_d15_share=pl.col("FLT_ARR_1_DLY_15") / pl.col("FLT_ARR_1").clip(1),
    atfm_other_pf=(pl.col("DLY_APT_ARR_1") - pl.col("DLY_APT_ARR_W_1") - pl.col("DLY_APT_ARR_G_1")) / pl.col("FLT_ARR_1").clip(1))
a = a.with_columns(atfm_flt_rel=pl.col("atfm_flt") / pl.col("atfm_flt").median().over("ADEP_mvt"))
a = a.select("ADEP_mvt", "day", *[c for c in a.columns if c.startswith("atfm_")]).with_columns(pl.all().exclude("ADEP_mvt", "day").cast(pl.Float32))
fm = pl.read_parquet(FM_IN).join(a, on=["ADEP_mvt", "day"], how="left")
print(fm.select([c for c in fm.columns if c.startswith("atfm_")]).null_count())
assert fm["MVT_ID_mvt"].n_unique() == fm.height
fm.write_parquet(FM_OUT); print(fm.shape)
