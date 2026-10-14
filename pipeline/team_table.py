"""Assemble the class model's expert table: the label-free team-stack base (team_base.py,
column v22 of work/v22_oof.parquet) plus components that pass, each added on top of the previous step:
  K1  residual corrector on NM rows with C < 1 h (alpha 0.5; team_corrector.py)
  K2  LIRF arrival-default drift shift, NM rows only (g_nm = 0.25, g_nonm = 0)
  K3  bounded d-target model blended on NM rows (alpha chosen month-held from {0, .25, .5};
      team_dmodel.py), if present
Rule (pre-set): keep a step only if its paired ex_ro interval vs the previous step is below 0 and >= 9/12 months.
Writes work/v23_oof.parquet (MVT_ID_mvt, y, v23) and work/v23_test.parquet (MVT_ID_mvt, v23),
the table class_mix.py and gbm_eval.py read.

usage: python pipeline/team_table.py
"""
import os
import numpy as np
import polars as pl
from common import RESULTS, load_movements, sec, rmse, block_boot

os.makedirs(RESULTS, exist_ok=True)
out = open(f"{RESULTS}/team_table_report.txt", "w")
def log(*a):
    print(*a); print(*a, file=out, flush=True)
key = "MVT_ID_mvt"
mv = load_movements().with_columns(op=pl.col("FLIGHT_mvt").str.slice(0, 3), ym=pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m"),
    dflt=(pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() <= 10)
dep = mv.filter(pl.col("PHASE_mvt") == "DEP").with_columns(
    C=sec("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"), nm=pl.col("LOBT_flt").is_not_null(), ml=sec("MVT_TIME_UTC_mvt", "LOBT_flt"),
    lsl=sec("SCHED_TIME_UTC_mvt", "LOBT_flt").abs(), mon=pl.col("MVT_TIME_UTC_mvt").dt.month(),
    date=pl.col("MVT_TIME_UTC_mvt").dt.date().cast(pl.Utf8))
# --- A12 shift table (LIRF arrivals)
arr = mv.filter((pl.col("PHASE_mvt") == "ARR") & (pl.col("ADES_mvt") == "LIRF")).group_by("op", "ym").agg(n=pl.len(), k=pl.col("dflt").sum())
ap_rate = float(arr["k"].sum() / arr["n"].sum()); K = 50.0
months25 = [f"2025-{m:02d}" for m in range(1, 13)]
def logit(p):
    p = np.clip(p, 1e-3, 1 - 1e-3); return np.log(p / (1 - p))
def shift_table(ym, refs):
    ref = arr.filter(pl.col("ym").is_in(refs)).group_by("op").agg(nr=pl.col("n").sum(), kr=pl.col("k").sum()).with_columns(rr=(pl.col("kr") + K * ap_rate) / (pl.col("nr") + K))
    cur = arr.filter(pl.col("ym") == ym).join(ref, on="op", how="left").with_columns(rr=pl.col("rr").fill_null(ap_rate))
    cur = cur.with_columns(rc=(pl.col("k") + K * pl.col("rr")) / (pl.col("n") + K))
    return cur.select("op", pl.lit(ym).alias("ym"), d=pl.Series(logit(cur["rc"].to_numpy()) - logit(cur["rr"].to_numpy())))
ST = pl.concat([shift_table(m, [x for x in months25 if x != m]) for m in months25] + [shift_table(m, months25) for m in ["2026-01", "2026-07"]])
dep = dep.join(ST, on=["op", "ym"], how="left").with_columns(pl.col("d").fill_null(0.0))

def a12(F, pred, g=0.25, N0=900.0):
    C = F["C"].to_numpy(); d = F["d"].to_numpy(); nm = F["nm"].to_numpy(); lirf = (F["ADEP_mvt"] == "LIRF").to_numpy()
    feas = ~(nm & (F["lsl"].fill_null(0).to_numpy() > 3616)); span = C - N0
    ok = lirf & nm & feas & (span > 600)
    ph = np.clip((pred - N0) / np.where(ok, span, 1.0), 0.01, 0.99)
    pn = 1 / (1 + np.exp(-(logit(ph) + g * d)))
    return np.where(ok, pred + (pn - ph) * span, pred)

o = pl.read_parquet("work/v22_oof.parquet").join(dep, on=key).join(pl.read_parquet("work/a14_oof.parquet").select(key, "corr"), on=key)
t = pl.read_parquet("work/v22_test.parquet").join(dep, on=key).join(pl.read_parquet("work/a14_test_corr.parquet"), on=key)
has17 = os.path.exists("work/a17_oof.parquet")
if has17:
    o = o.join(pl.read_parquet("work/a17_oof.parquet"), on=key, how="left")
    t = t.join(pl.read_parquet("work/a17_test.parquet"), on=key, how="left")
y = o["y"].to_numpy(); mon = o["mon"].to_numpy(); ro = np.abs(y - 86400) < 3600
blocks = (o["ADEP_mvt"] + "_" + o["date"]).to_numpy()
nm_o = o["nm"].to_numpy(); C_o = o["C"].to_numpy(); nm_t = t["nm"].to_numpy(); C_t = t["C"].to_numpy()
lo_o = np.where(nm_o, np.maximum(o["ml"].fill_null(0).to_numpy() - 3620, 0), -np.inf); hi_o = np.where(nm_o, o["ml"].fill_null(0).to_numpy() + 3620, np.inf)
lo_t = np.where(nm_t, np.maximum(t["ml"].fill_null(0).to_numpy() - 3620, 0), -np.inf); hi_t = np.where(nm_t, t["ml"].fill_null(0).to_numpy() + 3620, np.inf)

def k1(F, pred, nm, C, corr): return np.where(nm & (C < 3600), pred + 0.5 * corr, pred)
steps_o = {"v22": o["v22"].to_numpy()}; steps_t = {"v22": t["v22"].to_numpy()}
cur_o, cur_t, prev = steps_o["v22"], steps_t["v22"], "v22"

def evaluate(name, cand_o, cand_t):
    global cur_o, cur_t, prev
    m = ~ro
    pt, l, h = block_boot(y[m], cur_o[m], cand_o[m], blocks[m], n=1000)
    dm = [np.sum((cand_o[(mon == k) & m] - y[(mon == k) & m]) ** 2) - np.sum((cur_o[(mon == k) & m] - y[(mon == k) & m]) ** 2) for k in range(1, 13)]
    jj = [rmse(cand_o[(mon == k) & m] - y[(mon == k) & m]) - rmse(cur_o[(mon == k) & m] - y[(mon == k) & m]) for k in (1, 7)]
    ok = (h < 0) and sum(x < 0 for x in dm) >= 9
    mvt = cand_t - cur_t
    log(f"{name}: vs {prev} ex_ro {pt:+.3f} [{l:+.3f}, {h:+.3f}] months {sum(x<0 for x in dm)}/12 Jan {jj[0]:+.3f} Jul {jj[1]:+.3f}; test rows moved {(mvt!=0).sum()} rmsd {np.sqrt(np.mean(mvt**2)):.2f} -> {'KEEP' if ok else 'drop'}")
    if ok:
        cur_o, cur_t, prev = np.clip(cand_o, lo_o, hi_o), np.clip(cand_t, lo_t, hi_t), name

evaluate("K1_corrector", k1(o, cur_o, nm_o, C_o, o["corr"].to_numpy()), k1(t, cur_t, nm_t, C_t, t["corr"].to_numpy()))
evaluate("K2_lirf_arr", a12(o, cur_o), a12(t, cur_t))
if has17:
    dmo = o["dmodel"].fill_null(np.nan).to_numpy(); dmt = t["dmodel"].fill_null(np.nan).to_numpy()
    for a in (0.25, 0.5):
        co = np.where(nm_o & np.isfinite(dmo), (1 - a) * cur_o + a * np.nan_to_num(dmo), cur_o)
        ct = np.where(nm_t & np.isfinite(dmt), (1 - a) * cur_t + a * np.nan_to_num(dmt), cur_t)
        evaluate(f"K3_dmodel_a{a}", co, ct)
b = o["v22"].to_numpy()
for lane, m in [("all", np.ones_like(ro)), ("ex_ro", ~ro), ("Jan ex_ro", (mon == 1) & ~ro), ("Jul ex_ro", (mon == 7) & ~ro)]:
    pt, l, h = block_boot(y[m], b[m], cur_o[m], blocks[m], n=1000)
    log(f"v23 vs v22 {lane:10s} {rmse(b[m]-y[m]):.3f} -> {rmse(cur_o[m]-y[m]):.3f} delta {pt:+.3f} [{l:+.3f}, {h:+.3f}]")
mvt = cur_t - t["v22"].to_numpy()
log(f"TEST v23 - v22: rows moved {(mvt != 0).sum()}, rmsd {np.sqrt(np.mean(mvt**2)):.2f}, sum(move^2)/N {np.mean(mvt**2):.1f}, mean {mvt.mean():+.2f}")
log("last kept step:", prev)
pl.DataFrame({key: o[key], "y": y, "v23": cur_o}).write_parquet("work/v23_oof.parquet")
pl.DataFrame({key: t[key], "v23": cur_t}).write_parquet("work/v23_test.parquet")
