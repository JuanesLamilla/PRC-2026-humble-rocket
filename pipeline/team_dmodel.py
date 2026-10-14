"""Month-held LightGBM on the bounded off-block offset d = BLOCK - LOBT (= MVT-LOBT - y), NM rows only,
over the same feature frame as team_corrector.py plus the gap ml - v22. The prediction is ml - d_hat,
clipped to the LOBT join window. Writes work/a17_oof.parquet and work/a17_test.parquet
(column dmodel; read by team_table.py, which chooses the blend weight month-held).

usage: python pipeline/team_dmodel.py
"""
import glob, os
import numpy as np
import polars as pl
import lightgbm as lgb
from common import TEAM, RESULTS, load_movements, sec, rmse, block_boot

os.makedirs(RESULTS, exist_ok=True)
out = open(f"{RESULTS}/team_dmodel_report.txt", "w")
def log(*a):
    print(*a); print(*a, file=out, flush=True)
key = "MVT_ID_mvt"
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
FE = f"{TEAM}/data/features"
A = pl.read_parquet(glob.glob(f"{FE}/adsb_ground__*.parquet")[0]).join(
    pl.read_parquet(glob.glob(f"{FE}/adsb_ground_v2__*.parquet")[0]), on=key, how="full", coalesce=True, suffix="_v2")
acols = [c for c, t in A.schema.items() if c != key and (t.is_numeric() or t == pl.Boolean)]
A = A.select([key] + [pl.col(c).cast(pl.Float64) for c in acols])
dep = dep.join(A, on=key, how="left")

new = ["h_mvt_ib", "h_mvt_ib2", "h_next", "h_ib_aobt", "h_ib2_aobt", "h_ib_lobt", "h_ib_sched", "h_ib2_sched",
       "arr_opm", "arr_op", "arr_opd", "arr_apd", "arr_shift", "nb_ma", "nb_n", "day_nonm", "plan_res", "plan3_res"]
feats = new + acols + ["C", "ml", "sl", "ma", "nm", "hour", "ap", "v22"]
log("feature coverage (non-null share) on 2026 test rows:")
o = pl.read_parquet("work/v22_oof.parquet").join(dep, on=key)
t = pl.read_parquet("work/v22_test.parquet").join(dep, on=key)
log({c: round(1 - t[c].null_count() / t.height, 3) for c in new})
o = o.filter(pl.col("nm") > 0); t = t.filter(pl.col("nm") > 0)
o = o.with_columns(gap=pl.col("ml") - pl.col("v22")); t = t.with_columns(gap=pl.col("ml") - pl.col("v22"))
feats2 = feats + ["gap"]
y = o["y"].to_numpy(); base = o["v22"].to_numpy(); mon = o["mon"].to_numpy(); ro = np.abs(y - 86400) < 3600
ml = o["ml"].to_numpy(); dlt = ml - y  # = BLOCK - LOBT, bounded +-3606
X = o.select(feats2).to_pandas()
prm = dict(objective="regression", learning_rate=0.05, num_leaves=255, min_data_in_leaf=200, feature_fraction=0.7,
           bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=int(os.environ.get("SLURM_CPUS_PER_TASK", 16)), seed=0)
NR = 1200
dh = np.zeros(len(y))
for m in range(1, 13):
    tr = (mon != m) & ~ro; va = mon == m
    dh[va] = lgb.train(prm, lgb.Dataset(X[tr], dlt[tr]), NR).predict(X[va])
    log(f"fold {m}")
cand = np.clip(ml - dh, np.maximum(ml - 3620, 0), ml + 3620)
blocks = (o["ADEP_mvt"] + "_" + o["day"].cast(pl.Utf8)).to_numpy(); C = o["C"].to_numpy()
log(f"NM rows {len(y)}: v22 {rmse(base-y):.2f} d-model {rmse(cand-y):.2f}")
for a in (0.3, 0.5, 0.7, 1.0):
    bl = (1 - a) * base + a * cand
    for nm_, L in [("NM all", np.ones_like(ro)), ("C<1h", C < 3600), ("C>=1h", C >= 3600), ("Jan", mon == 1), ("Jul", mon == 7)]:
        m = L & ~ro
        pt, l, h = block_boot(y[m], base[m], bl[m], blocks[m], n=500)
        dm = [np.sum((bl[m & (mon == k)] - y[m & (mon == k)]) ** 2) - np.sum((base[m & (mon == k)] - y[m & (mon == k)]) ** 2) for k in range(1, 13)]
        log(f"  a {a} {nm_:7s} {rmse(base[m]-y[m]):8.2f} -> {rmse(bl[m]-y[m]):8.2f} delta {pt:+.3f} [{l:+.3f}, {h:+.3f}] months {sum(x<0 for x in dm)}/{sum(x!=0 for x in dm)}")
full = lgb.train(prm, lgb.Dataset(X[~ro], dlt[~ro]), NR)
tml = t["ml"].to_numpy(); tc = np.clip(tml - full.predict(t.select(feats2).to_pandas()), np.maximum(tml - 3620, 0), tml + 3620)
pl.DataFrame({key: t[key], "dmodel": tc}).write_parquet("work/a17_test.parquet")
pl.DataFrame({key: o[key], "dmodel": cand}).write_parquet("work/a17_oof.parquet")
log("done")
