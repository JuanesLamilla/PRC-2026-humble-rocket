"""Nested two-stage GPU XGBoost on the full feature matrix (work/fm.parquet).

usage: python gbm_nested.py OUTER [--frac F] [--tag T]
  OUTER in 1..12: held-out 2025 month (OOF predictions for that month)
  OUTER = 0     : test mode (fit on all 12 months, predict the 2026 ranking rows)

Stage 1 (off-block offset & stamp classes), fitted with inner month-held folds over the training months so the
stage-2 training rows carry out-of-fold stage-1 features:
  d_hat      : mean of d = BLOCK - LOBT (NM rows; bounded +-3606)
  dq10/50/90 : quantiles of d  -> predicted bounds on y: [ml - dq90, ml - dq10]
  p_def      : P(|BLOCK - SCHED| <= 10 s)
  p_gt1h     : P(y > 1 h and not a schedule default)
Stage 2 (taxi time):
  y_base     : y-model without stage-1 features (reference)
  y_s1       : y-model with stage-1 features
  y_mix      : p_def * C + (1 - p_def) * normal-model(with stage-1 features, trained on non-default rows)
All stage-2 outputs are reported raw; window clipping/stacking happens downstream.
"""
import os, sys, time
import numpy as np
import pandas as pd
import polars as pl
try:
    import xgboost as xgb
except ImportError:
    xgb = None

OUTER = int(sys.argv[1])
FRAC = float(sys.argv[sys.argv.index("--frac") + 1]) if "--frac" in sys.argv else 1.0
TAG = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "v1"
RES = os.environ.get("RESULTS_DIR", "results")
os.makedirs(f"work/gbm_{TAG}", exist_ok=True)
t0 = time.time()
def log(*a):
    print(f"[{time.time() - t0:7.0f}s]", *a, flush=True)

key = "MVT_ID_mvt"
FMF = os.environ.get("FM", "work/fm.parquet")
fm = pl.read_parquet(FMF)
if FRAC < 1:
    fm = fm.filter((pl.col("is_test")) | (pl.col(key).hash(7) % 1000 < int(FRAC * 1000)))
META = {key, "split", "is_test", "ADEP_mvt", "mon", "day", "y", "dlt", "dflt"}
feats = [c for c in fm.columns if c not in META]
NOADSB = "--noadsb" in sys.argv
if NOADSB:
    feats = [c for c in feats if not c.startswith("adsb")]
cats = [c for c in feats if c.startswith("cat_")]
tr_all = fm.filter(~pl.col("is_test"))
if OUTER == 0:
    TR, VA = tr_all, fm.filter(pl.col("is_test"))
else:
    TR, VA = tr_all.filter(pl.col("mon") != OUTER), tr_all.filter(pl.col("mon") == OUTER)
log(f"outer {OUTER}: train {TR.height}, eval {VA.height}, features {len(feats)} ({len(cats)} categorical)")


def to_pd(df, extra=()):
    X = df.select(feats + list(extra)).to_pandas()
    for c in cats:
        X[c] = X[c].fillna(-1).astype("int32").astype("category")
    return X


# consistent category levels across frames
ALL_LEVELS = {c: np.unique(fm[c].fill_null(-1).to_numpy()) for c in cats}
def fix_cats(X):
    for c in cats:
        X[c] = pd.Categorical(X[c].astype("int32"), categories=ALL_LEVELS[c])
    return X

GPU = dict(device="cuda", tree_method="hist", max_bin=256, enable_categorical=True, max_cat_to_onehot=8)
P_D = dict(objective="reg:squarederror", max_depth=10, learning_rate=0.05, subsample=0.8, colsample_bytree=0.6,
           min_child_weight=200, reg_lambda=5.0, **GPU)
P_Q = dict(objective="reg:quantileerror", quantile_alpha=np.array([0.1, 0.5, 0.9]), max_depth=8, learning_rate=0.05,
           subsample=0.8, colsample_bytree=0.6, min_child_weight=200, **GPU)
P_C = dict(objective="binary:logistic", eval_metric="logloss", max_depth=8, learning_rate=0.05, subsample=0.8,
           colsample_bytree=0.6, min_child_weight=50, **GPU)
P_Y = dict(objective="reg:squarederror", max_depth=10, learning_rate=0.03, subsample=0.8, colsample_bytree=0.6,
           min_child_weight=1000, reg_lambda=10.0, **GPU)
NR = dict(d=1500, q=800, c=800, y=2500)


import gc
def fit_predict(params, n, Xtr, ytr, Xte_list, w=None):
    dtr = xgb.QuantileDMatrix(Xtr, label=ytr, weight=w, enable_categorical=True, max_bin=256)
    bst = xgb.train(params, dtr, n)
    del dtr; gc.collect()
    out = []
    for X in Xte_list:
        dte = xgb.DMatrix(X, enable_categorical=True); out.append(bst.predict(dte)); del dte
    del bst; gc.collect()
    return out


def stage1(train_df, pred_dfs):
    """Fit stage-1 models on train_df, predict each frame in pred_dfs. Returns list of dicts of arrays."""
    Xtr = fix_cats(to_pd(train_df)); Xp = [fix_cats(to_pd(p)) for p in pred_dfs]
    nm = train_df["nm"].to_numpy() > 0
    d = train_df["dlt"].to_numpy()
    out = [dict() for _ in pred_dfs]
    m = nm & np.isfinite(d)
    for i, pr in enumerate(fit_predict(P_D, NR["d"], Xtr[m], d[m], Xp)):
        out[i]["d_hat"] = pr
    for i, pr in enumerate(fit_predict(P_Q, NR["q"], Xtr[m], d[m], Xp)):
        pr = np.asarray(pr).reshape(len(Xp[i]), -1)
        out[i]["dq10"], out[i]["dq50"], out[i]["dq90"] = pr[:, 0], pr[:, 1], pr[:, 2]
    yv = train_df["y"].to_numpy(); dfl = train_df["dflt"].to_numpy().astype(float)
    for i, pr in enumerate(fit_predict(P_C, NR["c"], Xtr, dfl, Xp)):
        out[i]["p_def"] = pr
    g1 = ((yv > 3600) & (dfl < 0.5) & (np.abs(yv - 86400) >= 3600)).astype(float)
    for i, pr in enumerate(fit_predict(P_C, NR["c"], Xtr, g1, Xp)):
        out[i]["p_gt1h"] = pr
    del Xtr, Xp; gc.collect()
    return out


S1C = ["d_hat", "dq10", "dq50", "dq90", "p_def", "p_gt1h"]
def add_s1(df, s1):
    df = df.with_columns([pl.Series(k, np.asarray(v, dtype=np.float32)) for k, v in s1.items()])
    nmv = pl.col("nm") > 0
    return df.with_columns(
        y_from_d=pl.when(nmv).then(pl.col("ml") - pl.col("d_hat")),
        b_lo=pl.when(nmv).then(pl.col("ml") - pl.col("dq90")), b_hi=pl.when(nmv).then(pl.col("ml") - pl.col("dq10")),
        mix_c=pl.col("p_def") * pl.col("C"))
EXTRA = S1C + ["y_from_d", "b_lo", "b_hi", "mix_c"]

# ---- stage 1: inner month-held OOF for the training months, full-train fit for the eval frame
months = sorted(set(TR["mon"].to_list()))
S1TAG = os.environ.get("S1TAG", TAG)
c_in, c_out = f"work/gbm_{S1TAG}/s1_inner_{OUTER:02d}.parquet", f"work/gbm_{S1TAG}/s1_outer_{OUTER:02d}.parquet"
if os.path.exists(c_in) and os.path.exists(c_out):
    inner = pl.read_parquet(c_in); so = pl.read_parquet(c_out)
    s_va = {c: VA.select(key).join(so, on=key, how="left")[c].to_numpy() for c in S1C}
    log("stage1 loaded from cache", c_in)
else:
    inner_parts = []
    for k in months:
        a, b = TR.filter(pl.col("mon") != k), TR.filter(pl.col("mon") == k)
        s = stage1(a, [b])[0]
        inner_parts.append(b.select(key).with_columns([pl.Series(c, s[c]) for c in S1C]))
        log(f"stage1 inner month {k} done")
    inner = pl.concat(inner_parts)
    s_va = stage1(tr_all if OUTER == 0 else tr_all.filter(pl.col("mon") != OUTER), [VA])[0]
    os.makedirs(f"work/gbm_{S1TAG}", exist_ok=True)
    inner.write_parquet(c_in); VA.select(key).with_columns([pl.Series(c, s_va[c]) for c in S1C]).write_parquet(c_out)
    log("stage1 outer done (cached)")
TR = TR.join(inner, on=key, how="left")
TR = add_s1(TR.drop(S1C), {c: TR[c].to_numpy() for c in S1C})
VA = add_s1(VA, s_va)

# ---- stage 2
yv = TR["y"].to_numpy(); ro = np.abs(yv - 86400) < 3600; dfl = TR["dflt"].to_numpy()
Xs_tr, Xs_va = fix_cats(to_pd(TR, EXTRA)), fix_cats(to_pd(VA, EXTRA))
Xb_tr, Xb_va = Xs_tr[feats], Xs_va[feats]
res = VA.select(key, "mon")
res = res.with_columns([pl.Series(c, VA[c].to_numpy()) for c in S1C + ["y_from_d", "b_lo", "b_hi"]])
ONLYCAT = "--only-cat" in sys.argv
nd = ~ro & ~dfl
pdv = VA["p_def"].to_numpy(); Cv = VA["C"].to_numpy()
if not ONLYCAT:
    res = res.with_columns(y_base=pl.Series(fit_predict(P_Y, NR["y"], Xb_tr[~ro], yv[~ro], [Xb_va])[0]))
    log("stage2 y_base done")
    res = res.with_columns(y_s1=pl.Series(fit_predict(P_Y, NR["y"], Xs_tr[~ro], yv[~ro], [Xs_va])[0]))
    log("stage2 y_s1 done")
    yn = fit_predict(P_Y, NR["y"], Xs_tr[nd], yv[nd], [Xs_va])[0]
    res = res.with_columns(y_normal=pl.Series(yn), y_mix=pl.Series(pdv * Cv + (1 - pdv) * yn))
    log("stage2 y_mix done")
if ONLYCAT:
    try:
        from catboost import CatBoostRegressor, CatBoostClassifier, Pool
    except ImportError:
        pass
    EX2 = [] if NOADSB else EXTRA
    Xc_tr, Xc_va = to_pd(TR, EX2), to_pd(VA, EX2)
    for c in cats:
        Xc_tr[c] = Xc_tr[c].astype(int); Xc_va[c] = Xc_va[c].astype(int)
    pva = Pool(Xc_va, cat_features=cats)
    CFG = {"cat_d10s1": dict(depth=10, learning_rate=0.04, iterations=8000, random_seed=1, min_data_in_leaf=300),
           "cat_d12": dict(depth=12, learning_rate=0.04, iterations=6000, random_seed=2, min_data_in_leaf=600),
           "cat_d8": dict(depth=8, learning_rate=0.06, iterations=10000, random_seed=3, min_data_in_leaf=200),
           "cat_d10s4": dict(depth=10, learning_rate=0.04, iterations=8000, random_seed=4, min_data_in_leaf=300, l2_leaf_reg=10)}
    for name in [a for a in sys.argv[sys.argv.index("--cfg") + 1].split(",")]:
        if name.startswith("anom"):
            # bad-reading specialist: 4-class stamp classifier (normal / default / near / long) with all features + member
            # predictions, class-specific value models, expectation combine.
            from catboost import CatBoostClassifier as CBC
            mo_ = pl.read_parquet("work/members_oof.parquet"); mt_ = pl.read_parquet("work/members_test.parquet")
            mm_ = pl.concat([mo_, mt_]).unique(key, keep="first")
            mcols = [c for c in mm_.columns if c != key]
            Atr = TR.select(key).join(mm_, on=key, how="left").select(mcols).to_pandas()
            Ava = VA.select(key).join(mm_, on=key, how="left").select(mcols).to_pandas()
            for c in mcols:
                if c != "stack":
                    Atr[c] = Atr[c] - Atr["stack"]; Ava[c] = Ava[c] - Ava["stack"]
            Xa_tr = Xc_tr.copy(); Xa_va = Xc_va.copy()
            for c in mcols:
                Xa_tr["mm_" + c] = Atr[c].to_numpy(); Xa_va["mm_" + c] = Ava[c].to_numpy()
            bs_tr = TR["BLOCK_TIME_UTC_mvt"] if "BLOCK_TIME_UTC_mvt" in TR.columns else None
            Ctr = TR["C"].to_numpy(); bmsr = Ctr - yv  # = BLOCK - SCHED
            cls = np.zeros(len(yv), dtype=int)
            cls[(yv > 3600)] = 3
            cls[(np.abs(bmsr) > 10) & (np.abs(bmsr) <= 900) & (Ctr > 1800)] = 2
            cls[dfl] = 1
            keep = ~ro
            pva_a = Pool(Xa_va, cat_features=cats)
            clf = CBC(loss_function="MultiClass", iterations=4000, learning_rate=0.06, depth=8, l2_leaf_reg=10, task_type="GPU", verbose=500, random_seed=91)
            clf.fit(Pool(Xa_tr[keep], cls[keep], cat_features=cats))
            P = clf.predict_proba(pva_a)
            feas = 1.0 - (VA["sched_infeas"].fill_null(0).to_numpy() * (VA["nm"].to_numpy() > 0))
            nmtr = TR["nm"].to_numpy() > 0; dtr = TR["dlt"].to_numpy(); nmva = VA["nm"].to_numpy() > 0
            base_rg = dict(task_type="GPU", grow_policy="Depthwise", verbose=0, border_count=254)
            m1 = keep & (cls == 0) & nmtr & np.isfinite(dtr)
            r1 = CatBoostRegressor(iterations=8000, learning_rate=0.04, depth=10, min_data_in_leaf=50, l2_leaf_reg=30, random_seed=92, **base_rg)
            r1.fit(Pool(Xa_tr[m1], dtr[m1], cat_features=cats))
            m2 = keep & (cls == 0) & ~nmtr
            r2 = CatBoostRegressor(iterations=3000, learning_rate=0.04, depth=8, min_data_in_leaf=100, l2_leaf_reg=10, random_seed=93, **base_rg)
            r2.fit(Pool(Xa_tr[m2], yv[m2], cat_features=cats))
            v_norm = np.where(nmva, VA["ml"].to_numpy() - r1.predict(pva_a), r2.predict(pva_a))
            m3 = keep & (cls == 2)
            r3 = CatBoostRegressor(iterations=2000, learning_rate=0.04, depth=6, min_data_in_leaf=50, l2_leaf_reg=30, random_seed=94, **base_rg)
            r3.fit(Pool(Xa_tr[m3], (Ctr - yv)[m3], cat_features=cats))
            v_near = Cv - r3.predict(pva_a)
            m4 = keep & (cls == 3)
            r4 = CatBoostRegressor(iterations=2000, learning_rate=0.03, depth=6, min_data_in_leaf=50, l2_leaf_reg=50, random_seed=95, **base_rg)
            r4.fit(Pool(Xa_tr[m4], np.log(yv[m4] - 3600 + 1), cat_features=cats))
            lp = r4.predict(pva_a); s2 = float(np.var(np.log(yv[m4] - 3600 + 1) - r4.predict(Pool(Xa_tr[m4], cat_features=cats))))
            v_long = 3600 + np.exp(lp + s2 / 2) - 1
            p_norm, p_def, p_near, p_long = P[:, 0], P[:, 1] * feas, P[:, 2], P[:, 3]
            tot = p_norm + p_def + p_near + p_long
            yhat = (p_norm * v_norm + p_def * Cv + p_near * v_near + p_long * v_long) / tot
            res = res.with_columns(pl.Series("y_" + name, yhat), pl.Series("pa_def", p_def / tot), pl.Series("pa_near", p_near / tot),
                                   pl.Series("pa_long", p_long / tot), pl.Series("va_long", v_long))
        elif name.startswith("catmixap"):
            import re as _re; _m = _re.search(r"_s(\d+)", name); sd_ = int(_m.group(1)) if _m else 77
            ytgt = name.startswith("catmixapy")
            dep_ = 12 if "_d12" in name else 10
            # per-airport mixture: classifier + bounded-offset regressor (NM) / y regressor (no-NM), one set per airport
            feas = 1.0 - (VA["sched_infeas"].fill_null(0).to_numpy() * (VA["nm"].to_numpy() > 0))
            nmtr = TR["nm"].to_numpy() > 0; dtr = TR["dlt"].to_numpy(); nmva = VA["nm"].to_numpy() > 0
            aptr = TR["ADEP_mvt"].to_numpy(); apva = VA["ADEP_mvt"].to_numpy()
            outp = np.full(VA.height, np.nan)
            for ap in np.unique(apva):
                a_tr = aptr == ap; a_va = apva == ap
                Xa_va = Pool(Xc_va[a_va], cat_features=cats)
                clf = CatBoostClassifier(iterations=2000, learning_rate=0.06, depth=6, l2_leaf_reg=10, task_type="GPU", verbose=0, random_seed=sd_)
                clf.fit(Pool(Xc_tr[a_tr & ~ro], dfl[a_tr & ~ro].astype(int), cat_features=cats))
                pc = clf.predict_proba(Xa_va)[:, 1] * feas[a_va]
                m1 = nd & nmtr & np.isfinite(dtr) & a_tr
                rg1 = CatBoostRegressor(iterations=5000, learning_rate=0.04, depth=dep_, task_type="GPU", grow_policy="Depthwise", min_data_in_leaf=100,
                                        l2_leaf_reg=30, verbose=0, random_seed=sd_ + 1, border_count=254)
                rg1.fit(Pool(Xc_tr[m1], (yv[m1] if ytgt else dtr[m1]), cat_features=cats))
                m2 = nd & ~nmtr & a_tr
                if m2.sum() > 200:
                    rg2 = CatBoostRegressor(iterations=1500, learning_rate=0.04, depth=6, task_type="GPU", verbose=0, random_seed=79)
                    rg2.fit(Pool(Xc_tr[m2], yv[m2], cat_features=cats)); y2 = rg2.predict(Xa_va)
                else:
                    y2 = np.full(a_va.sum(), np.median(yv[nd & a_tr]))
                r1p = rg1.predict(Xa_va)
                ynn = np.where(nmva[a_va], (r1p if ytgt else VA["ml"].to_numpy()[a_va] - r1p), y2)
                outp[a_va] = pc * Cv[a_va] + (1 - pc) * ynn
            res = res.with_columns(pl.Series("y_" + name, outp))
        elif name == "catmixd_json":
            import json
            cfg = json.loads(os.environ["CFG_JSON"])
            feas = 1.0 - (VA["sched_infeas"].fill_null(0).to_numpy() * (VA["nm"].to_numpy() > 0))
            clf = CatBoostClassifier(task_type="GPU", verbose=0, **cfg["clf"])
            clf.fit(Pool(Xc_tr[~ro], dfl[~ro].astype(int), cat_features=cats))
            pc = clf.predict_proba(pva)[:, 1] * feas
            nmtr = TR["nm"].to_numpy() > 0; dtr = TR["dlt"].to_numpy(); nmva = VA["nm"].to_numpy() > 0
            m1 = nd & nmtr & np.isfinite(dtr)
            rg1 = CatBoostRegressor(task_type="GPU", verbose=0, grow_policy="Depthwise", **cfg["reg"])
            rg1.fit(Pool(Xc_tr[m1], dtr[m1], cat_features=cats))
            rg2 = CatBoostRegressor(task_type="GPU", verbose=0, grow_policy="Depthwise", **dict(cfg["reg"], iterations=max(1000, cfg["reg"]["iterations"] // 2)))
            m2 = nd & ~nmtr
            rg2.fit(Pool(Xc_tr[m2], yv[m2], cat_features=cats))
            ynn = np.where(nmva, VA["ml"].to_numpy() - rg1.predict(pva), rg2.predict(pva))
            res = res.with_columns(pl.Series("y_" + cfg.get("name", "hpo"), pc * Cv + (1 - pc) * ynn))
        elif name.startswith("lgbap"):
            import lightgbm as lgb
            sd = int(name.split("_s")[-1]) if "_s" in name else 0
            nth = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
            Xl_tr, Xl_va = fix_cats(to_pd(TR, EX2)), fix_cats(to_pd(VA, EX2))
            feas = 1.0 - (VA["sched_infeas"].fill_null(0).to_numpy() * (VA["nm"].to_numpy() > 0))
            nmtr = TR["nm"].to_numpy() > 0; dtr = TR["dlt"].to_numpy(); nmva = VA["nm"].to_numpy() > 0
            aptr = TR["ADEP_mvt"].to_numpy(); apva = VA["ADEP_mvt"].to_numpy(); outp = np.full(VA.height, np.nan)
            pc_ = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.6, bagging_fraction=0.8,
                       bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=nth, seed=sd, max_cat_to_onehot=8, cat_smooth=50)
            pr_ = dict(pc_, objective="regression", learning_rate=0.03, num_leaves=255, min_data_in_leaf=100, lambda_l2=30.0)
            for ap in np.unique(apva):
                a_tr = aptr == ap; a_va = apva == ap
                clf = lgb.train(pc_, lgb.Dataset(Xl_tr[a_tr & ~ro], dfl[a_tr & ~ro].astype(float)), 800)
                pcl = clf.predict(Xl_va[a_va]) * feas[a_va]
                m1 = nd & nmtr & np.isfinite(dtr) & a_tr
                r1 = lgb.train(pr_, lgb.Dataset(Xl_tr[m1], dtr[m1]), 2500)
                m2 = nd & ~nmtr & a_tr
                y2 = lgb.train(dict(pr_, num_leaves=31), lgb.Dataset(Xl_tr[m2], yv[m2]), 600).predict(Xl_va[a_va]) if m2.sum() > 200 else np.full(a_va.sum(), np.median(yv[nd & a_tr]))
                ynl = np.where(nmva[a_va], VA["ml"].to_numpy()[a_va] - r1.predict(Xl_va[a_va]), y2)
                outp[a_va] = pcl * Cv[a_va] + (1 - pcl) * ynl
            res = res.with_columns(pl.Series("y_" + name, outp))
        elif name.startswith("lgbmix"):
            import lightgbm as lgb
            sd = int(name.split("_s")[-1]) if "_s" in name else 0
            nth = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
            Xl_tr, Xl_va = fix_cats(to_pd(TR, EX2)), fix_cats(to_pd(VA, EX2))
            feas = 1.0 - (VA["sched_infeas"].fill_null(0).to_numpy() * (VA["nm"].to_numpy() > 0))
            pc_ = dict(objective="binary", learning_rate=0.05, num_leaves=255, min_data_in_leaf=200, feature_fraction=0.6,
                       bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=nth, seed=sd, max_cat_to_onehot=8, cat_smooth=50)
            clf = lgb.train(pc_, lgb.Dataset(Xl_tr[~ro], dfl[~ro].astype(float)), 1500)
            pcl = clf.predict(Xl_va) * feas
            pr_ = dict(pc_, objective="regression", learning_rate=0.03, num_leaves=511, min_data_in_leaf=300, lambda_l2=20.0)
            nmtr = TR["nm"].to_numpy() > 0; dtr = TR["dlt"].to_numpy(); nmva = VA["nm"].to_numpy() > 0
            m1 = nd & nmtr & np.isfinite(dtr)
            r1 = lgb.train(pr_, lgb.Dataset(Xl_tr[m1], dtr[m1]), 3000)
            r2 = lgb.train(dict(pr_, num_leaves=127), lgb.Dataset(Xl_tr[nd & ~nmtr], yv[nd & ~nmtr]), 1500)
            ynl = np.where(nmva, VA["ml"].to_numpy() - r1.predict(Xl_va), r2.predict(Xl_va))
            res = res.with_columns(pl.Series("y_" + name, pcl * Cv + (1 - pcl) * ynl))
        elif name == "nonm_mix":
            # mixture trained only on no-NM rows (5 seeds averaged); NaN on NM rows
            nmtr = TR["nm"].to_numpy() > 0; nmva = VA["nm"].to_numpy() > 0
            m_c = ~ro & ~nmtr; m_r = nd & ~nmtr
            preds = []
            for sd in range(5):
                clf = CatBoostClassifier(iterations=1500, learning_rate=0.05, depth=6, task_type="GPU", verbose=0, random_seed=50 + sd)
                clf.fit(Pool(Xc_tr[m_c], dfl[m_c].astype(int), cat_features=cats))
                rg = CatBoostRegressor(iterations=2500, learning_rate=0.04, depth=8, task_type="GPU", grow_policy="Depthwise",
                                       min_data_in_leaf=100, verbose=0, random_seed=60 + sd, border_count=128)
                rg.fit(Pool(Xc_tr[m_r], yv[m_r], cat_features=cats))
                pc = clf.predict_proba(pva)[:, 1]
                preds.append(pc * Cv + (1 - pc) * rg.predict(pva))
            res = res.with_columns(pl.Series("y_nonm_mix", np.where(nmva, np.nan, np.mean(preds, 0))))
        elif name.startswith("catmix") and name != "catmix":
            sd = int(name.split("_s")[-1]) if "_s" in name else 7
            feas = 1.0 - (VA["sched_infeas"].fill_null(0).to_numpy() * (VA["nm"].to_numpy() > 0))
            clf = CatBoostClassifier(iterations=3000, learning_rate=0.06, depth=8, task_type="GPU", verbose=1000, random_seed=sd)
            clf.fit(Pool(Xc_tr[~ro], dfl[~ro].astype(int), cat_features=cats))
            pc = clf.predict_proba(pva)[:, 1] * feas
            base_rg = dict(iterations=8000, learning_rate=0.04, depth=10, task_type="GPU", grow_policy="Depthwise",
                           min_data_in_leaf=300, verbose=1000, random_seed=sd + 100, border_count=128)
            if name.startswith("catmixd"):
                # normal component on NM rows via the bounded offset d = BLOCK - LOBT; no-NM rows via y
                nmtr = TR["nm"].to_numpy() > 0; dtr = TR["dlt"].to_numpy()
                m1 = nd & nmtr & np.isfinite(dtr)
                rg1 = CatBoostRegressor(**base_rg); rg1.fit(Pool(Xc_tr[m1], dtr[m1], cat_features=cats))
                m2 = nd & ~nmtr
                rg2 = CatBoostRegressor(**dict(base_rg, iterations=4000)); rg2.fit(Pool(Xc_tr[m2], yv[m2], cat_features=cats))
                nmva = VA["nm"].to_numpy() > 0
                ynn = np.where(nmva, VA["ml"].to_numpy() - rg1.predict(pva), rg2.predict(pva))
                res = res.with_columns(pl.Series("y_" + name, pc * Cv + (1 - pc) * ynn))
            elif name.startswith("catmix3"):
                long_ = (~dfl) & (yv > 3600) & ~ro
                clf2 = CatBoostClassifier(iterations=3000, learning_rate=0.06, depth=8, task_type="GPU", verbose=1000, random_seed=sd + 1)
                clf2.fit(Pool(Xc_tr[~ro & ~dfl], long_[~ro & ~dfl].astype(int), cat_features=cats))
                pl_ = clf2.predict_proba(pva)[:, 1]
                rg = CatBoostRegressor(**base_rg); mn = nd & (yv <= 3600)
                rg.fit(Pool(Xc_tr[mn], yv[mn], cat_features=cats))
                rgl = CatBoostRegressor(**dict(base_rg, depth=6, min_data_in_leaf=200, iterations=2000, l2_leaf_reg=30))
                rgl.fit(Pool(Xc_tr[long_], yv[long_], cat_features=cats))
                yl = np.maximum(rgl.predict(pva), 3600)
                res = res.with_columns(pl.Series("y_" + name, pc * Cv + (1 - pc) * (pl_ * yl + (1 - pl_) * rg.predict(pva))))
            elif name.startswith("catmixs"):
                # normal component on sqrt(y) with a smearing variance correction
                rg = CatBoostRegressor(**base_rg); sy = np.sqrt(np.maximum(yv[nd], 0))
                rg.fit(Pool(Xc_tr[nd], sy, cat_features=cats))
                s2 = float(np.var(sy - rg.predict(Pool(Xc_tr[nd], cat_features=cats))))
                yn3 = np.maximum(rg.predict(pva), 0) ** 2 + s2
                res = res.with_columns(pl.Series("y_" + name, pc * Cv + (1 - pc) * yn3))
            else:
                rg = CatBoostRegressor(**base_rg); rg.fit(Pool(Xc_tr[nd], yv[nd], cat_features=cats))
                res = res.with_columns(pl.Series("y_" + name, pc * Cv + (1 - pc) * rg.predict(pva)))
        elif name == "catmix":
            clf = CatBoostClassifier(iterations=3000, learning_rate=0.06, depth=8, task_type="GPU", verbose=1000, random_seed=5)
            clf.fit(Pool(Xc_tr[~ro], dfl[~ro].astype(int), cat_features=cats))
            pc = clf.predict_proba(pva)[:, 1]
            rg = CatBoostRegressor(iterations=8000, learning_rate=0.04, depth=10, task_type="GPU", grow_policy="Depthwise",
                                   min_data_in_leaf=300, verbose=1000, random_seed=6, border_count=128)
            rg.fit(Pool(Xc_tr[nd], yv[nd], cat_features=cats))
            yn2 = rg.predict(pva)
            feas = 1.0 - (VA["sched_infeas"].fill_null(0).to_numpy() * (VA["nm"].to_numpy() > 0))
            pc = pc * feas
            res = res.with_columns(y_catmix=pl.Series(pc * Cv + (1 - pc) * yn2), y_catnormal=pl.Series(yn2))
        else:
            cb = CatBoostRegressor(task_type="GPU", grow_policy="Depthwise", loss_function="RMSE", verbose=1000, border_count=128, **CFG[name])
            cb.fit(Pool(Xc_tr[~ro], yv[~ro], cat_features=cats))
            res = res.with_columns(pl.Series("y_" + name, cb.predict(pva)))
        log(f"stage2 {name} done")
if "--tail" in sys.argv:
    # tail-split identity: E[y] = E[min(y,c)] + P(y>c) * E[y-c | y>c], c = 3600 s
    cap = 3600.0
    ycap = np.minimum(yv, cap)
    yc = fit_predict(P_Y, NR["y"], Xs_tr[~ro], ycap[~ro], [Xs_va])[0]
    gt = (yv > cap).astype(float)
    pg = fit_predict(P_C, NR["c"], Xs_tr[~ro], gt[~ro], [Xs_va])[0]
    P_E = dict(P_Y, min_child_weight=300, max_depth=6, reg_lambda=50.0)
    ex = fit_predict(P_E, 800, Xs_tr[~ro & (yv > cap)], (yv - cap)[~ro & (yv > cap)], [Xs_va])[0]
    res = res.with_columns(y_tail=pl.Series(yc + pg * np.maximum(ex, 0)))
    log("stage2 y_tail done")
if "--cat" in sys.argv:
    from catboost import CatBoostRegressor, Pool
    Xc_tr, Xc_va = to_pd(TR, EXTRA), to_pd(VA, EXTRA)
    for c in cats:
        Xc_tr[c] = Xc_tr[c].astype(int); Xc_va[c] = Xc_va[c].astype(int)
    cb = CatBoostRegressor(iterations=6000, learning_rate=0.05, depth=10, task_type="GPU", grow_policy="Depthwise", min_data_in_leaf=300,
                           loss_function="RMSE", verbose=1000, random_seed=0, border_count=128)
    cb.fit(Pool(Xc_tr[~ro], yv[~ro], cat_features=cats))
    res = res.with_columns(y_cat=pl.Series(cb.predict(Pool(Xc_va, cat_features=cats))))
    log("stage2 y_cat done")
if OUTER > 0:
    res = res.with_columns(y=pl.Series(VA["y"].to_numpy()))
res.write_parquet(f"work/gbm_{TAG}/outer_{OUTER:02d}.parquet")
log("written", res.shape)
