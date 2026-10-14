"""Label-free projections of the team stack's prediction, the base of the class model's expert table.

Reads the gated team-stack run (outputs/runs/stack_l2_cell_ng4h2G_lt1h, docs/05-reproduction.md)
and applies, step by step, scored on the 2025 month_loo OOF (paired day-block bootstrap) and moved
label-free on test:

S1  clip to the LOBT join window [max(0, MVT-LOBT-W), MVT-LOBT+W], W = 3620 s (2025 max 3606; 2026 arrivals 3615)
S2  rows moved by S1: month-held in-window LightGBM (NM rows only), clipped to the window
S3  rows whose LOBT stayed a day before take-off (MVT-LOBT > 20 h): predict MVT-LOBT (window centre;
    2025 analogue y = MVT-LOBT + 1)
S4  no-NM departures with an NM arrival twin (work/twins.parquet, make_twins.py) and
    b = MVT - twinLOBT - 3606 >= 300: cap at b (failed join => y outside (b, b + 7212); 0 of 812 2025 rows above)
S5  LIRF NM rows where a schedule default is infeasible (|SCHED-LOBT| > 3616): subtract a month-held,
    shrunk mean residual

Writes work/v22_oof.parquet (MVT_ID_mvt, y, v22) and work/v22_test.parquet (MVT_ID_mvt, v22), the
file and column names the downstream scripts (team_corrector.py, team_dmodel.py, team_table.py) read,
plus results/team_base_report.txt.

usage: python pipeline/team_base.py
"""
import os
import numpy as np
import polars as pl
import lightgbm as lgb
from common import RUNS, RESULTS, load_movements, sec, rmse, block_boot

W = 3620.0
os.makedirs(RESULTS, exist_ok=True)
out = open(f"{RESULTS}/team_base_report.txt", "w")
def log(*a):
    print(*a); print(*a, file=out, flush=True)

mv = load_movements()
d = mv.filter(pl.col("PHASE_mvt") == "DEP").with_columns(
    C=sec("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"), ml=sec("MVT_TIME_UTC_mvt", "LOBT_flt"),
    sl=sec("SCHED_TIME_UTC_mvt", "LOBT_flt"), ma=sec("MVT_TIME_UTC_mvt", "AOBT_3_flt"),
    me=sec("MVT_TIME_UTC_mvt", "EOBT_1_flt"), mi=sec("MVT_TIME_UTC_mvt", "IOBT_flt"),
    hour=pl.col("MVT_TIME_UTC_mvt").dt.hour(), month=pl.col("MVT_TIME_UTC_mvt").dt.month(),
    ap=pl.col("ADEP_mvt").cast(pl.Categorical), date=pl.col("MVT_TIME_UTC_mvt").dt.date().cast(pl.Utf8),
).with_columns(lo=pl.max_horizontal(pl.lit(0.0), pl.col("ml") - W), hi=pl.col("ml") + W)
tw = pl.read_parquet("work/twins.parquet").select("MVT_ID_mvt", pl.col("ml").alias("tml"))
d = d.join(tw, on="MVT_ID_mvt", how="left")

run = "stack_l2_cell_ng4h2G_lt1h"
oof = pl.read_parquet(f"{RUNS}/{run}/oof.parquet").select("MVT_ID_mvt", "TAXITIME_SEC_mvt", "pred")
tp = pl.read_parquet(f"{RUNS}/{run}/test_pred.parquet").select("MVT_ID_mvt", "pred")
O = oof.join(d.drop("TAXITIME_SEC_mvt"), on="MVT_ID_mvt", how="left")
T = tp.join(d.filter(pl.col("split") == "test").drop("TAXITIME_SEC_mvt"), on="MVT_ID_mvt", how="left")
assert O["ml"].len() == 2085047 and T.height == 344841 and O["split"].null_count() == 0 and T["split"].null_count() == 0

# in-window LightGBM on NM training rows (month-held OOF + full fit)
feat = ["ml", "C", "sl", "ma", "me", "mi", "hour", "ap"]
nm = d.filter(pl.col("LOBT_flt").is_not_null() & (pl.col("split") == "train"))
X = nm.select(feat).to_pandas(); yv = nm["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy(); mon = nm["month"].to_numpy()
prm = dict(objective="regression", learning_rate=0.1, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.9,
           verbose=-1, num_threads=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)), seed=0)
w_oof = np.zeros(len(yv))
for m in range(1, 13):
    tr, va = mon != m, mon == m
    w_oof[va] = lgb.train(prm, lgb.Dataset(X[tr], yv[tr]), 250).predict(X[va])
full = lgb.train(prm, lgb.Dataset(X, yv), 250)
O = O.join(pl.DataFrame({"MVT_ID_mvt": nm["MVT_ID_mvt"], "w": w_oof}), on="MVT_ID_mvt", how="left")
T = T.with_columns(w=pl.Series(full.predict(T.select(feat).to_pandas())))
T = T.with_columns(w=pl.when(pl.col("LOBT_flt").is_null()).then(None).otherwise(pl.col("w")))


def steps(F, is_oof):
    p0 = F["pred"].to_numpy().astype(float)
    lo = F["lo"].fill_null(-np.inf).to_numpy(); hi = F["hi"].fill_null(np.inf).to_numpy()
    s1 = np.clip(p0, lo, hi)
    moved = s1 != p0
    wv = np.clip(F["w"].fill_null(np.nan).to_numpy(), lo, hi)
    s2 = np.where(moved & np.isfinite(wv), wv, s1)
    ml = F["ml"].fill_null(np.nan).to_numpy()
    s3 = np.where(np.nan_to_num(ml, nan=0) > 72000, ml, s2)
    b = F["tml"].fill_null(np.nan).to_numpy() - 3606
    capm = np.isfinite(b) & (b >= 300) & F["LOBT_flt"].is_null().to_numpy()
    s4 = np.where(capm, np.minimum(s3, np.where(capm, b, np.inf)), s3)
    return {"S0": p0, "S1": s1, "S2": s2, "S3": s3, "S4": s4}


so = steps(O, True); st = steps(T, False)
y = O["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy()
# S5: LIRF infeasible-default offset, month-held + shrink k=200
inf_o = ((O["ADEP_mvt"] == "LIRF") & (O["sl"].abs() > 3616).fill_null(False)).to_numpy()
inf_t = ((T["ADEP_mvt"] == "LIRF") & (T["sl"].abs() > 3616).fill_null(False)).to_numpy()
ro = np.abs(y - 86400) < 3600
mo = O["month"].to_numpy(); r4 = so["S4"] - y
s5 = so["S4"].copy(); k = 200.0
for m in range(1, 13):
    tr = inf_o & (mo != m) & ~ro; va = inf_o & (mo == m)
    off = r4[tr].mean() * tr.sum() / (tr.sum() + k)
    s5[va] = so["S4"][va] - off
tr = inf_o & ~ro
off_full = r4[tr].mean() * tr.sum() / (tr.sum() + k)
so["S5"] = np.maximum(s5, 0)
st["S5"] = np.where(inf_t, np.maximum(st["S4"] - off_full, 0), st["S4"])
log(f"S5 LIRF infeasible rows: OOF {inf_o.sum()}, test {inf_t.sum()}, full offset {off_full:.2f} s")

blocks = (O["ADEP_mvt"] + "_" + O["date"]).to_numpy()
jan, jul = mo == 1, mo == 7
ap = O["ADEP_mvt"].to_numpy()
names = ["S0", "S1", "S2", "S3", "S4", "S5"]
log("\n== OOF (2025 month_loo), RMSE")
log(f"{'step':4s} {'all':>9s} {'ex_ro':>9s} {'Jan_exro':>9s} {'Jul_exro':>9s} {'C>=1h':>9s}")
C = O["C"].to_numpy()
for n in names:
    p = so[n]
    log(f"{n:4s} {rmse(p-y):9.3f} {rmse(p[~ro]-y[~ro]):9.3f} {rmse(p[jan&~ro]-y[jan&~ro]):9.3f} {rmse(p[jul&~ro]-y[jul&~ro]):9.3f} {rmse(p[(C>=3600)&~ro]-y[(C>=3600)&~ro]):9.3f}")
log("\n== paired deltas (day-block bootstrap, 1000 reps): candidate - reference")
for a, b_ in [("S0", "S5"), ("S0", "S4"), ("S0", "S3"), ("S1", "S2"), ("S3", "S4"), ("S4", "S5")]:
    for lane, m in [("all", np.ones_like(ro)), ("ex_ro", ~ro), ("Jan ex_ro", jan & ~ro), ("Jul ex_ro", jul & ~ro)]:
        pt, l, h = block_boot(y[m], so[a][m], so[b_][m], blocks[m], n=1000)
        log(f"  {b_} vs {a} {lane:10s} {pt:+.3f} [{l:+.3f}, {h:+.3f}]")
    dm = [(rmse(so[b_][(mo == m) & ~ro] - y[(mo == m) & ~ro]) - rmse(so[a][(mo == m) & ~ro] - y[(mo == m) & ~ro])) for m in range(1, 13)]
    log(f"  {b_} vs {a} months better (ex_ro): {sum(x < 0 for x in dm)}/12 worse {sum(x > 0 for x in dm)}/12")
log("\n== per airport ex_ro, S0 -> S5")
for a_ in sorted(set(ap)):
    m = (ap == a_) & ~ro
    log(f"  {a_} {rmse(so['S0'][m]-y[m]):8.2f} -> {rmse(so['S5'][m]-y[m]):8.2f}  ({rmse(so['S5'][m]-y[m]) - rmse(so['S0'][m]-y[m]):+.3f})")
log("\n== TEST label-free moves vs the stack (S0)")
for n in names[1:]:
    dv = st[n] - st["S0"]
    log(f"  {n}: rows moved {(dv != 0).sum():6d}  rmsd {np.sqrt(np.mean(dv**2)):8.3f}  sum(move^2)/N {np.mean(dv**2):9.1f}  mean {dv.mean():+.3f}")
for a, b_ in [("S0", "S1"), ("S1", "S2"), ("S2", "S3"), ("S3", "S4"), ("S4", "S5")]:
    dv = st[b_] - st[a]
    log(f"  step {a}->{b_}: rows {(dv != 0).sum()}, sum(move^2)/N {np.mean(dv**2):.1f}")

pl.DataFrame({"MVT_ID_mvt": O["MVT_ID_mvt"], "y": y, "v22": so["S5"]}).write_parquet("work/v22_oof.parquet")
pl.DataFrame({"MVT_ID_mvt": T["MVT_ID_mvt"], "v22": st["S5"]}).write_parquet("work/v22_test.parquet")
log("done")
