"""Month-held residual corrector on top of the team-stack base (work/v22_oof.parquet, team_base.py)
with NEW information the members do not see directly.

New features (all test-computable; arrival in-block stamps are labelled in 2026 and allowed):
  stand handover : last / second-last ARRIVAL in-block at the same airport+stand before take-off, expressed
                   relative to MVT, AOBT_3, LOBT and SCHED (early-stamp detector)
  next occupant  : first arrival in-block at the stand after take-off, minus MVT
  arrival stamps : operator x month and operator x day arrival default share (in-block == SCHED) at the
                   airport, and the logit shift of the operator's month vs its other months (LIRF signal)
  NM neighbours  : median MVT - AOBT_3 of NM departures at the airport within +-60 min (no-NM anchor)
  day no-NM share, ADS-B ground block columns, plus C, MVT-LOBT, SCHED-LOBT, NM flag, airport, hour, v22.
Target: y - v22 (rollovers excluded from fitting). 12 month-held folds; test = full fit.
Pre-registered adoption, per lane (noNM / NM C<1h / NM C>=1h): lane ex_ro interval < 0 and >= 9/12 months.
Writes work/a14_oof.parquet, work/a14_test.parquet, work/a14_test_corr.parquet (read by team_table.py).

usage: python pipeline/team_corrector.py
"""
import glob, os
import numpy as np
import polars as pl
import lightgbm as lgb
from common import TEAM, RESULTS, load_movements, sec, rmse, block_boot

os.makedirs(RESULTS, exist_ok=True)
out = open(f"{RESULTS}/team_corrector_report.txt", "w")
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
y = o["y"].to_numpy(); base = o["v22"].to_numpy(); mon = o["mon"].to_numpy(); ro = np.abs(y - 86400) < 3600
X = o.select(feats).to_pandas(); r = y - base
prm = dict(objective="regression", learning_rate=0.03, num_leaves=63, min_data_in_leaf=500, feature_fraction=0.7,
           bagging_fraction=0.8, bagging_freq=1, lambda_l2=50.0, verbose=-1, num_threads=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)), seed=0)
NR = 400
corr = np.zeros(len(y))
for m in range(1, 13):
    tr = (mon != m) & ~ro; va = mon == m
    bst = lgb.train(prm, lgb.Dataset(X[tr], r[tr]), NR)
    corr[va] = bst.predict(X[va])
    log(f"fold {m} done")
full = lgb.train(prm, lgb.Dataset(X[~ro], r[~ro]), NR)
imp = sorted(zip(full.feature_name(), full.feature_importance("gain")), key=lambda z: -z[1])[:25]
log("top gain features:", [(n, round(g / 1e9, 2)) for n, g in imp])
ct = full.predict(t.select(feats).to_pandas())
blocks = (o["ADEP_mvt"] + "_" + o["day"].cast(pl.Utf8)).to_numpy()
nmv = o["nm"].to_numpy() > 0; C = o["C"].to_numpy()
lanes = {"noNM": ~nmv, "NM_Clt1h": nmv & (C < 3600), "NM_Cge1h": nmv & (C >= 3600)}
adopt = {}
for alpha in [0.5, 1.0]:
    cand = base + alpha * corr
    log(f"\n== alpha {alpha}")
    for nm_, L in list(lanes.items()) + [("ALL", np.ones_like(ro))]:
        m = L & ~ro
        pt, l, h = block_boot(y[m], base[m], cand[m], blocks[m], n=1000)
        dm = [np.sum((cand[m & (mon == k)] - y[m & (mon == k)]) ** 2) - np.sum((base[m & (mon == k)] - y[m & (mon == k)]) ** 2) for k in range(1, 13)]
        jj = [(rmse(cand[m & (mon == k)] - y[m & (mon == k)]) - rmse(base[m & (mon == k)] - y[m & (mon == k)])) for k in (1, 7)]
        better = sum(x < 0 for x in dm)
        log(f"  {nm_:9s} ex_ro {rmse(base[m]-y[m]):8.2f} -> {rmse(cand[m]-y[m]):8.2f} delta {pt:+.3f} [{l:+.3f}, {h:+.3f}] months {better}/12 Jan {jj[0]:+.2f} Jul {jj[1]:+.2f} dSSE {sum(dm):.3e}")
        if nm_ != "ALL" and h < 0 and better >= 9:
            adopt[(alpha, nm_)] = True
log("adopted lanes:", adopt)
# assemble candidate: alpha 0.5 on adopted lanes (alpha 1.0 if also adopted)
tnm = t["nm"].to_numpy() > 0; tC = t["C"].to_numpy()
tl = {"noNM": ~tnm, "NM_Clt1h": tnm & (tC < 3600), "NM_Cge1h": tnm & (tC >= 3600)}
final_o = base.copy(); final_t = t["v22"].to_numpy().copy()
for nm_ in lanes:
    a = 1.0 if (1.0, nm_) in adopt else (0.5 if (0.5, nm_) in adopt else 0.0)
    final_o[lanes[nm_]] = base[lanes[nm_]] + a * corr[lanes[nm_]]
    final_t[tl[nm_]] = final_t[tl[nm_]] + a * ct[tl[nm_]]
    log(f"lane {nm_}: alpha {a}")
for lane, m in [("all", np.ones_like(ro)), ("ex_ro", ~ro), ("Jan ex_ro", (mon == 1) & ~ro), ("Jul ex_ro", (mon == 7) & ~ro)]:
    pt, l, h = block_boot(y[m], base[m], final_o[m], blocks[m], n=1000)
    log(f"FINAL {lane:10s} {rmse(base[m]-y[m]):.3f} -> {rmse(final_o[m]-y[m]):.3f} delta {pt:+.3f} [{l:+.3f}, {h:+.3f}]")
mvt = final_t - t["v22"].to_numpy()
log(f"TEST move: rows {(mvt != 0).sum()}, rmsd {np.sqrt(np.mean(mvt**2)):.2f}, mean {mvt.mean():+.2f}")
pl.DataFrame({key: t[key], "pred": np.maximum(final_t, 0)}).write_parquet("work/a14_test.parquet")
pl.DataFrame({key: o[key], "y": y, "v22": base, "a14": final_o, "corr": corr}).write_parquet("work/a14_oof.parquet")
pl.DataFrame({key: t[key], "corr": ct}).write_parquet("work/a14_test_corr.parquet")
log("done")
