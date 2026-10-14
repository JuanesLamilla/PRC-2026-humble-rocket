"""Evaluate nested GBM outputs (work/gbm_<TAG>/outer_XX.parquet) against v23 and stack them with v23.
usage: python gbm_eval.py TAG [extra_tags...]
Stack: per lane (noNM / NM C<1h / NM C>=1h), non-negative weights summing to 1 over [v23, model columns],
fitted month-held (11 months) and applied to the held-out month; test weights from all 12 months.
Every prediction is clipped to the LOBT window before scoring."""
import sys, glob, os
import numpy as np
import polars as pl
from scipy.optimize import nnls
from common import RESULTS, rmse, block_boot

CELLS = "--cells" in sys.argv
CBAND = "--cband" in sys.argv
GUARD = "--lirf-guard" in sys.argv
NONM = "--nonm-stack" in sys.argv
META = "--meta" in sys.argv
SEASON = "--season" in sys.argv
WIN, SUM = {11, 12, 1, 2, 3}, {5, 6, 7, 8, 9}
def season_months(k):
    return WIN if k in WIN else (SUM if k in SUM else set(range(1, 13)))
TAGS = [a for a in sys.argv[1:] if not a.startswith("--")]
import hashlib
TS = '_'.join(TAGS)
if len(TS) > 180:  # file-name limit: long member lists get a short hash name
    TS = f"h{hashlib.md5(TS.encode()).hexdigest()[:10]}_{TAGS[-1]}"
out = open(f"{RESULTS}/gbm_eval_{TS}{'_cells' if CELLS else ''}{'_cband' if CBAND else ''}{'_guard' if GUARD else ''}{'_nonm' if NONM else ''}{'_meta' if META else ''}{'_season' if SEASON else ''}{'R' if os.environ.get('META_RICH','0')=='1' else ''}{'X' if os.environ.get('META_EXTRA') else ''}.txt", "w")
def log(*a):
    print(*a); print(*a, file=out, flush=True)
key = "MVT_ID_mvt"
meta = pl.read_parquet("work/fm.parquet", columns=[key, "ADEP_mvt", "mon", "day", "C", "nm", "lo", "hi", "is_test", "adsb_has_gnd", "hour", "sched_infeas", "ml"])
O = pl.read_parquet("work/v23_oof.parquet").join(meta, on=key)
T = pl.read_parquet("work/v23_test.parquet").join(meta, on=key)
cols = []
for tag in TAGS:
    fs = sorted(glob.glob(f"work/{tag}/outer_*.parquet"))
    oo = pl.concat([pl.read_parquet(f) for f in fs if not f.endswith("_00.parquet")], how="diagonal_relaxed")
    tt = pl.read_parquet(f"work/{tag}/outer_00.parquet") if any(f.endswith("_00.parquet") for f in fs) else None
    mc = [c for c in oo.columns if c.startswith("y_") and c != "y"]
    ren = {c: f"{tag}:{c}" for c in mc}
    O = O.join(oo.select([key] + mc).rename(ren), on=key, how="left")
    if tt is not None:
        T = T.join(tt.select([key] + mc).rename(ren), on=key, how="left")
    cols += list(ren.values())
    log(f"{tag}: months present {sorted(set(oo['mon'].to_list()))}, test {'yes' if tt is not None else 'no'}")
# optional extra meta-correction inputs from member outputs, e.g. META_EXTRA="ft_ftC:pdef_ft+ft_ftC:q10_ft" (+ or , separated)
XCOLS = []
for spec in [x for x in os.environ.get("META_EXTRA", "").replace("+", ",").split(",") if x]:
    tg, cname = spec.split(":")
    fs = sorted(glob.glob(f"work/{tg}/outer_*.parquet"))
    oo = pl.concat([pl.read_parquet(f).select(key, cname) for f in fs if not f.endswith("_00.parquet")])
    O = O.join(oo.rename({cname: f"x:{spec}"}), on=key, how="left")
    T = T.join(pl.read_parquet(f"work/{tg}/outer_00.parquet").select(key, cname).rename({cname: f"x:{spec}"}), on=key, how="left")
    XCOLS.append(f"x:{spec}")
if XCOLS:
    log("meta extra inputs:", XCOLS)
O = O.with_columns([pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(pl.col("v23")).alias(c) for c in cols])
T = T.with_columns([pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(pl.col("v23")).alias(c) for c in cols if c in T.columns])
y = O["y"].to_numpy(); mon = O["mon"].to_numpy(); ro = np.abs(y - 86400) < 3600
lo = O["lo"].fill_null(-np.inf).to_numpy(); hi = O["hi"].fill_null(np.inf).to_numpy()
clip = lambda v, a, b: np.clip(v, a, b)
P = {c: clip(O[c].to_numpy().astype(float), lo, hi) for c in ["v23"] + cols}
blocks = (O["ADEP_mvt"] + "_" + O["day"]).to_numpy()
nmv = O["nm"].to_numpy() > 0; C = O["C"].to_numpy()
lanes = {"noNM": ~nmv, "NM_Clt1h": nmv & (C < 3600), "NM_Cge1h": nmv & (C >= 3600)}
if CBAND:
    lanes = {"noNM": ~nmv, "NM_lt30m": nmv & (C < 1800), "NM_30_60m": nmv & (C >= 1800) & (C < 3600),
             "NM_1_2h": nmv & (C >= 3600) & (C < 7200), "NM_2_4h": nmv & (C >= 7200) & (C < 14400), "NM_gt4h": nmv & (C >= 14400)}
log(f"rows evaluated {len(y)} (months {sorted(set(mon))})")
log(f"{'model':28s} {'all':>9s} {'ex_ro':>9s} {'Jan':>9s} {'Jul':>9s} " + " ".join(f"{k:>9s}" for k in lanes))
for c, p in P.items():
    r = [rmse(p - y), rmse(p[~ro] - y[~ro]), rmse(p[(mon == 1) & ~ro] - y[(mon == 1) & ~ro]), rmse(p[(mon == 7) & ~ro] - y[(mon == 7) & ~ro])]
    r += [rmse(p[L & ~ro] - y[L & ~ro]) for L in lanes.values()]
    log(f"{c:28s} " + " ".join(f"{v:9.2f}" for v in r))


def simplex_nnls(X, t, prior=None, n0=0.0):
    # sum-to-one non-negative least squares via augmentation; optional ridge pull toward a parent vector
    lam = 1e3 * np.sqrt(len(t))
    rows, tg = [X, lam * np.ones((1, X.shape[1]))], [t, [lam]]
    if prior is not None and n0 > 0:
        sc = np.sqrt(n0) * np.sqrt(np.mean(t ** 2))
        rows.append(sc * np.eye(X.shape[1])); tg.append(sc * prior)
    w, _ = nnls(np.vstack(rows), np.concatenate(tg)); return w / w.sum()


AP = O["ADEP_mvt"].to_numpy()
def fit_apply(trm, apply_rows, Mtr_all, Mapply, ap_tr, ap_apply, v23_only=False):
    """lane-level weights, optionally per-airport cells shrunk to the lane vector (n0 = 20,000 pseudo-rows)."""
    if v23_only:
        wl = np.zeros(Mtr_all.shape[1]); wl[0] = 1.0
        return wl, Mapply @ wl
    wl = simplex_nnls(Mtr_all[trm], y[trm])
    out = Mapply @ wl
    if CELLS:
        for a in np.unique(ap_apply):
            ta = trm & (ap_tr == a)
            if ta.sum() < 200: continue
            wa = simplex_nnls(Mtr_all[ta], y[ta], prior=wl, n0=20000.0)
            sel = ap_apply == a
            out[sel] = Mapply[sel] @ wa
    return wl, out


names = ["v23"] + cols
M = np.vstack([P[c] for c in names]).T
stk = P["v23"].copy(); W = {}
for k in range(1, 13):
    va = mon == k
    if va.sum() == 0: continue
    for ln, L in lanes.items():
        tr = (mon != k) & L & ~ro
        if SEASON:
            tr = tr & np.isin(mon, list(season_months(k)))
        w, pred = fit_apply(tr, None, M, M[va & L], AP, AP[va & L], v23_only=(ln == "noNM" and not NONM)); W.setdefault(ln, []).append(w)
        stk[va & L] = pred
stk = clip(stk, lo, hi)
if META:
    import lightgbm as lgb
    apc = pl.Series(O["ADEP_mvt"]).cast(pl.Categorical).to_physical().to_numpy()
    RICH = os.environ.get("META_RICH", "0") == "1"
    def ctx(D):
        return np.column_stack([D["hour"].to_numpy(), D["adsb_has_gnd"].fill_null(-1).to_numpy(), D["sched_infeas"].fill_null(-1).to_numpy(),
                                D["ml"].fill_null(np.nan).to_numpy(), D["lo"].fill_null(np.nan).to_numpy(), D["hi"].fill_null(np.nan).to_numpy()])
    def mfeat(Mx, st, Cx, apx, D=None):
        base = [Mx - st[:, None], st, Cx, apx]
        if RICH and D is not None:
            cx = ctx(D); base += [cx, (st - cx[:, 4])[:, None], (cx[:, 5] - st)[:, None]]
        if XCOLS and D is not None:
            base += [D.select(XCOLS).to_numpy().astype(float)]
        return np.column_stack(base)
    F = mfeat(M, stk, C, apc, O)
    prm = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=2000, lambda_l2=100.0,
               feature_fraction=0.8, verbose=-1, num_threads=8, seed=0)
    corr = np.zeros(len(y)); nmmask = nmv
    for k in range(1, 13):
        tr = (mon != k) & nmmask & ~ro; va = (mon == k) & nmmask
        if va.sum() == 0: continue
        corr[va] = lgb.train(prm, lgb.Dataset(F[tr], (y - stk)[tr]), 300).predict(F[va])
    MA = float(os.environ.get("META_ALPHA", "0.5"))
    for a in (0.5, 1.0):
        cand = clip(stk + a * corr, lo, hi)
        for lane, m in [("ex_ro", ~ro), ("Jan ex_ro", (mon == 1) & ~ro), ("Jul ex_ro", (mon == 7) & ~ro), ("all", np.ones_like(ro))]:
            m = m.astype(bool)
            pt, l, h = block_boot(y[m], stk[m], cand[m], blocks[m], n=500)
            log(f"META alpha {a}: vs linear stack {lane:10s} {pt:+.3f} [{l:+.3f}, {h:+.3f}]")
        dm = [np.sum((cand[(mon == j) & ~ro] - y[(mon == j) & ~ro]) ** 2) - np.sum((stk[(mon == j) & ~ro] - y[(mon == j) & ~ro]) ** 2) for j in range(1, 13)]
        log(f"META alpha {a}: months better {sum(x < 0 for x in dm)}/12")
    meta_full = lgb.train(prm, lgb.Dataset(F[nmmask & ~ro], (y - stk)[nmmask & ~ro]), 300)
    stk = clip(stk + MA * corr, lo, hi)
    log(f"META applied with alpha {MA}")
for ln in lanes:
    log(f"mean month-held weights {ln}: " + ", ".join(f"{n}={v:.3f}" for n, v in zip(names, np.mean(W[ln], axis=0)) if v > 0.001))
base = P["v23"]
for lane, m in [("all", np.ones_like(ro)), ("ex_ro", ~ro), ("Jan ex_ro", (mon == 1) & ~ro), ("Jul ex_ro", (mon == 7) & ~ro)] + [(f"{k} ex_ro", L & ~ro) for k, L in lanes.items()]:
    m = m.astype(bool)
    pt, l, h = block_boot(y[m], base[m], stk[m], blocks[m], n=1000)
    dm = [np.sum((stk[m & (mon == j)] - y[m & (mon == j)]) ** 2) - np.sum((base[m & (mon == j)] - y[m & (mon == j)]) ** 2) for j in range(1, 13)]
    log(f"STACK vs v23 {lane:14s} {rmse(base[m]-y[m]):9.3f} -> {rmse(stk[m]-y[m]):9.3f} delta {pt:+.3f} [{l:+.3f}, {h:+.3f}] months {sum(x < 0 for x in dm)}/{sum(x != 0 for x in dm)}")
if all(c in T.columns for c in cols):
    tlo = T["lo"].fill_null(-np.inf).to_numpy(); thi = T["hi"].fill_null(np.inf).to_numpy()
    MT = np.vstack([clip(T[c].to_numpy().astype(float), tlo, thi) for c in names]).T
    tnm = T["nm"].to_numpy() > 0; tC = T["C"].to_numpy()
    tl = {"noNM": ~tnm, "NM_Clt1h": tnm & (tC < 3600), "NM_Cge1h": tnm & (tC >= 3600)}
    if CBAND:
        tl = {"noNM": ~tnm, "NM_lt30m": tnm & (tC < 1800), "NM_30_60m": tnm & (tC >= 1800) & (tC < 3600),
              "NM_1_2h": tnm & (tC >= 3600) & (tC < 7200), "NM_2_4h": tnm & (tC >= 7200) & (tC < 14400), "NM_gt4h": tnm & (tC >= 14400)}
    ts = T["v23"].to_numpy().astype(float).copy()
    for ln, L in lanes.items():
        if SEASON:
            tmon = T["mon"].to_numpy() if "mon" in T.columns else None
            for sm, tm in [(WIN, 1), (SUM, 7)]:
                sel_t = tl[ln] & (T["MVT_ID_mvt"].is_not_null().to_numpy()) & (tmon == tm)
                w, pred = fit_apply(L & ~ro & np.isin(mon, list(sm)), None, M, MT[sel_t], AP, T["ADEP_mvt"].to_numpy()[sel_t], v23_only=(ln == "noNM" and not NONM)); ts[sel_t] = pred
        else:
            w, pred = fit_apply(L & ~ro, None, M, MT[tl[ln]], AP, T["ADEP_mvt"].to_numpy()[tl[ln]], v23_only=(ln == "noNM" and not NONM)); ts[tl[ln]] = pred
        log(f"full-fit weights {ln}: " + ", ".join(f"{n}={v:.3f}" for n, v in zip(names, w) if v > 0.001))
    if META:
        tapc = pl.Series(T["ADEP_mvt"]).cast(pl.Categorical).to_physical().to_numpy()
        # same categorical coding as OOF: map airport strings via the OOF coding
        code = {a: c for a, c in zip(O["ADEP_mvt"].to_list(), apc)}
        tapc = np.array([code[a] for a in T["ADEP_mvt"].to_list()])
        Ft = mfeat(MT, clip(ts, tlo, thi), tC, tapc, T)
        ct = meta_full.predict(Ft)
        ts = np.where(tnm, ts + MA * ct, ts)
        log(f"META on test: NM rows moved rmsd {np.sqrt(np.mean((MA * ct[tnm]) ** 2)):.2f}")
    if GUARD:
        g = (T["ADEP_mvt"] == "LIRF").to_numpy() & ~(T["adsb_has_gnd"].fill_null(0).to_numpy() > 0)
        ts[g] = T["v23"].to_numpy()[g]
        log(f"LIRF guard: {g.sum()} test rows without a ground track kept on v23")
    ts = clip(ts, tlo, thi)
    mv = ts - T["v23"].to_numpy()
    log(f"TEST stack vs v23: rmsd {np.sqrt(np.mean(mv**2)):.2f}, sum(move^2)/N {np.mean(mv**2):.1f}, mean {mv.mean():+.2f}")
    pl.DataFrame({key: T[key], "pred": np.maximum(ts, 0)}).write_parquet(f"work/stack_{TS}{'_cells' if CELLS else ''}{'_cband' if CBAND else ''}{'_guard' if GUARD else ''}{'_nonm' if NONM else ''}{'_meta' if META else ''}{'_season' if SEASON else ''}{'R' if os.environ.get('META_RICH','0')=='1' else ''}{'X' if os.environ.get('META_EXTRA') else ''}_test.parquet")
if os.environ.get("SAVE_MEMBERS", "0") == "1":
    mo = pl.DataFrame({key: O[key].to_numpy(), "stack": stk}).with_columns([pl.Series(f"m{i}", M[:, i]) for i in range(M.shape[1])])
    mo.write_parquet("work/members_oof.parquet")
    if all(c in T.columns for c in cols):
        mt = pl.DataFrame({key: T[key].to_numpy(), "stack": np.maximum(ts, 0)}).with_columns([pl.Series(f"m{i}", MT[:, i]) for i in range(MT.shape[1])])
        mt.write_parquet("work/members_test.parquet")
    log("saved members:", names)
pl.DataFrame({key: O[key], "y": y, "stack": stk}).write_parquet(f"work/stack_{TS}{'_cells' if CELLS else ''}{'_cband' if CBAND else ''}{'_guard' if GUARD else ''}{'_nonm' if NONM else ''}{'_meta' if META else ''}{'_season' if SEASON else ''}{'R' if os.environ.get('META_RICH','0')=='1' else ''}{'X' if os.environ.get('META_EXTRA') else ''}_oof.parquet")
log("done")
