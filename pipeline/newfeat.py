"""New feature blocks (session 2/3): stand handover, arrival-default shares, NM-neighbour taxi, NM plan residual,
day no-NM share, ADS-B blocks. build_dep() returns one row per departure (2025 + 2026) with these columns."""
import glob
import numpy as np
import polars as pl
from common import FEATURES, load_movements, sec

key = "MVT_ID_mvt"
NEW = ["h_mvt_ib", "h_mvt_ib2", "h_next", "h_ib_aobt", "h_ib2_aobt", "h_ib_lobt", "h_ib_sched", "h_ib2_sched",
       "arr_opm", "arr_op", "arr_opd", "arr_apd", "arr_shift", "nb_ma", "nb_n", "day_nonm", "plan_res", "plan3_res"]


def build_dep():
    mv = load_movements().with_columns(
        apt=pl.when(pl.col("PHASE_mvt") == "DEP").then(pl.col("ADEP_mvt")).otherwise(pl.col("ADES_mvt")),
        op=pl.col("FLIGHT_mvt").str.slice(0, 3), day=pl.col("MVT_TIME_UTC_mvt").dt.date(),
        ym=pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m"),
        dflt=((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() <= 10).cast(pl.Float64))
    arr = mv.filter(pl.col("PHASE_mvt") == "ARR")
    dep = mv.filter(pl.col("PHASE_mvt") == "DEP").with_columns(
        C=sec("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"), ml=sec("MVT_TIME_UTC_mvt", "LOBT_flt"),
        sl=sec("SCHED_TIME_UTC_mvt", "LOBT_flt"), ma=sec("MVT_TIME_UTC_mvt", "AOBT_3_flt"),
        nm=pl.col("LOBT_flt").is_not_null().cast(pl.Float64), hour=pl.col("MVT_TIME_UTC_mvt").dt.hour(),
        mon=pl.col("MVT_TIME_UTC_mvt").dt.month(), ap=pl.col("ADEP_mvt").cast(pl.Categorical))

    # --- stand handover (as-of joins on in-block time per airport+stand)
    ib = arr.filter(pl.col("STAND_mvt").is_not_null() & pl.col("BLOCK_TIME_UTC_mvt").is_not_null()).select(
        pl.col("apt").alias("ADEP_mvt"), "STAND_mvt", pl.col("BLOCK_TIME_UTC_mvt").alias("ib")).sort("ib")
    ib2 = ib.with_columns(ib_prev=pl.col("ib").shift(1).over("ADEP_mvt", "STAND_mvt"))
    ds = dep.select(key, "ADEP_mvt", "STAND_mvt", "MVT_TIME_UTC_mvt").filter(pl.col("STAND_mvt").is_not_null()).sort("MVT_TIME_UTC_mvt")
    back = ds.join_asof(ib2, left_on="MVT_TIME_UTC_mvt", right_on="ib", by=["ADEP_mvt", "STAND_mvt"], strategy="backward")
    fwd = ds.join_asof(ib.rename({"ib": "ib_next"}), left_on="MVT_TIME_UTC_mvt", right_on="ib_next", by=["ADEP_mvt", "STAND_mvt"], strategy="forward")
    dep = dep.join(back.select(key, "ib", "ib_prev"), on=key, how="left").join(fwd.select(key, "ib_next"), on=key, how="left")
    dep = dep.with_columns(
        h_mvt_ib=sec("MVT_TIME_UTC_mvt", "ib"), h_mvt_ib2=sec("MVT_TIME_UTC_mvt", "ib_prev"), h_next=sec("ib_next", "MVT_TIME_UTC_mvt"),
        h_ib_aobt=sec("ib", "AOBT_3_flt"), h_ib2_aobt=sec("ib_prev", "AOBT_3_flt"), h_ib_lobt=sec("ib", "LOBT_flt"),
        h_ib_sched=sec("ib", "SCHED_TIME_UTC_mvt"), h_ib2_sched=sec("ib_prev", "SCHED_TIME_UTC_mvt"))
    dep = dep.with_columns([pl.when(pl.col(c).abs() > 6 * 3600).then(None).otherwise(pl.col(c)).alias(c)
                            for c in ["h_mvt_ib", "h_mvt_ib2", "h_next", "h_ib_aobt", "h_ib2_aobt", "h_ib_lobt", "h_ib_sched", "h_ib2_sched"]])
    # --- arrival default shares
    K = 30.0
    a_opm = arr.group_by("apt", "op", "ym").agg(an=pl.len(), ak=pl.col("dflt").sum())
    a_ap = arr.group_by("apt").agg(ap_rate=pl.col("dflt").mean())
    a_opm = a_opm.join(a_ap, on="apt").with_columns(arr_opm=(pl.col("ak") + K * pl.col("ap_rate")) / (pl.col("an") + K))
    a_op = arr.filter(pl.col("ym").str.starts_with("2025")).group_by("apt", "op").agg(on=pl.len(), ok=pl.col("dflt").sum()).join(a_ap, on="apt").with_columns(
        arr_op=(pl.col("ok") + K * pl.col("ap_rate")) / (pl.col("on") + K))
    a_opd = arr.group_by("apt", "op", "day").agg(dn=pl.len(), dk=pl.col("dflt").sum())
    a_apd = arr.group_by("apt", "day").agg(arr_apd=pl.col("dflt").mean())
    dep = dep.join(a_opm.select(pl.col("apt").alias("ADEP_mvt"), "op", "ym", "arr_opm"), on=["ADEP_mvt", "op", "ym"], how="left")
    dep = dep.join(a_op.select(pl.col("apt").alias("ADEP_mvt"), "op", "arr_op"), on=["ADEP_mvt", "op"], how="left")
    dep = dep.join(a_opd.select(pl.col("apt").alias("ADEP_mvt"), "op", "day", "dn", "dk"), on=["ADEP_mvt", "op", "day"], how="left")
    dep = dep.join(a_apd.select(pl.col("apt").alias("ADEP_mvt"), "day", "arr_apd"), on=["ADEP_mvt", "day"], how="left")
    dep = dep.with_columns(arr_opd=(pl.col("dk") + 5 * pl.col("arr_opm")) / (pl.col("dn") + 5),
                           arr_shift=pl.col("arr_opm") - pl.col("arr_op"))
    # --- NM planned block-to-block (ARVT_1 - EOBT_1) minus its route x type median (2025 NM rows)
    dep = dep.with_columns(plan=sec("ARVT_1_flt", "EOBT_1_flt"), plan3=sec("ARVT_3_flt", "AOBT_3_flt"))
    pm = dep.filter(pl.col("MVT_TIME_UTC_mvt").dt.year() == 2025).group_by("ADEP_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt").agg(pmed=pl.col("plan").median(), p3med=pl.col("plan3").median(), pn=pl.len())
    dep = dep.join(pm, on=["ADEP_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt"], how="left").with_columns(
        plan_res=pl.when(pl.col("pn") >= 20).then(pl.col("plan") - pl.col("pmed")), plan3_res=pl.when(pl.col("pn") >= 20).then(pl.col("plan3") - pl.col("p3med")))
    # --- NM-neighbour taxi proxy and day no-NM share
    nmd = dep.filter(pl.col("nm") > 0).select("ADEP_mvt", "MVT_TIME_UTC_mvt", "ma").sort("MVT_TIME_UTC_mvt")
    roll = nmd.rolling(index_column="MVT_TIME_UTC_mvt", period="2h", offset="-1h", group_by="ADEP_mvt").agg(nb_ma=pl.col("ma").median(), nb_n=pl.len())
    roll = roll.unique(["ADEP_mvt", "MVT_TIME_UTC_mvt"])
    dep = dep.sort("MVT_TIME_UTC_mvt").join_asof(roll.sort("MVT_TIME_UTC_mvt"), on="MVT_TIME_UTC_mvt", by="ADEP_mvt", strategy="nearest")
    dep = dep.with_columns(day_nonm=(1 - pl.col("nm")).mean().over("ADEP_mvt", "day"))
    # --- ADS-B blocks
    v1 = sorted(f for f in glob.glob(f"{FEATURES}/adsb_ground*.parquet") if "_v2" not in f)
    v2 = sorted(glob.glob(f"{FEATURES}/adsb_ground_v2*.parquet"))
    A = pl.read_parquet(v1[0]).join(
        pl.read_parquet(v2[0]), on=key, how="full", coalesce=True, suffix="_v2")
    acols = [c for c, t in A.schema.items() if c != key and (t.is_numeric() or t == pl.Boolean)]
    A = A.select([key] + [pl.col(c).cast(pl.Float64) for c in acols])
    dep = dep.join(A, on=key, how="left")

    return dep, acols
