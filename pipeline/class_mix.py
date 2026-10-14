"""class_mix.py [--epochs E] [--seeds S] [--tag T] [--more] [--plus] [--wide W] [--lane all|ge1h|nonm]: class-structured supermodel (GPU).

Combines member *components* instead of member final predictions:
  yhat = p_def * C + p_near * (C - 300*softplus(a)) + p_long * va_long * exp(0.3*tanh(b)) + p_norm * sum_j g_j * n_j
  * class probabilities (default masked where infeasible) from a network that sees every member's class signals:
    transformer p_def (ftC3), GBM stage-1 p_def / p_gt1h, anomaly specialist pa_def / pa_near / pa_long, plus context;
  * normal-value experts n_j: GBM normal-only regressors (y_normal, y_catnormal), the transformer's conditional normal
    mean (y_ft - p_def*C)/(1 - p_def), the bounded offset model (MVT - LOBT - d_hat), v23, q50 of the transformer;
  * va_long = anomaly specialist's long-stamp value.
Loss: MSE/1e4 + 0.1 * cross-entropy on the observed recording class (default / near / long / normal).
Leave-one-month-out on 2025 (rollovers out of training), test fitted on all 2025. Writes work/cm_<TAG>/outer_XX.parquet
(y_cm, cm_pdef, cm_pnear, cm_plong).
"""
import glob, os, sys, time
import numpy as np, polars as pl
import torch, torch.nn as nn, torch.nn.functional as F
from common import RESULTS

arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
EPOCHS = int(arg("--epochs", "10")); MORE = "--more" in sys.argv; SEEDS = [int(s) for s in arg("--seeds", "0,1,2").split(",")]; TAG = arg("--tag", "cm1")
PLUS = "--plus" in sys.argv; WIDE = int(arg("--wide", "256")); LANE = arg("--lane", "all")  # all | ge1h | nonm (lane specialist)
os.makedirs(f"work/cm_{TAG}", exist_ok=True)
t0 = time.time()
out = open(f"{RESULTS}/cm_{TAG}.txt", "w")
def log(*a): print(f"[{time.time()-t0:6.0f}s]", *a, flush=True); print(*a, file=out, flush=True)
dev = torch.device("cuda"); K = "MVT_ID_mvt"
meta = pl.read_parquet(__import__("os").environ.get("FM_META", "work/fm.parquet"), columns=[K, "ADEP_mvt", "mon", "C", "nm", "lo", "hi", "is_test", "adsb_has_gnd", "hour", "sched_infeas", "ml", "dflt"])

def load(tag, cols, test):
    fs = sorted(glob.glob(f"work/{tag}/outer_*.parquet"))
    if test:
        return pl.read_parquet(f"work/{tag}/outer_00.parquet").select([K] + cols)
    return pl.concat([pl.read_parquet(f).select([K] + cols) for f in fs if not f.endswith("_00.parquet")])

SRC = [("gbm_v2", ["p_def", "p_gt1h", "d_hat", "y_normal"]), ("gbm_v3a", ["y_catnormal"]), ("gbm_anom1", ["pa_def", "pa_near", "pa_long", "va_long"]),
       ("ft_ftC7", ["y_ft", "pdef_ft", "q50_ft", "q10_ft", "q90_ft"])]
if MORE:  # extra class signals and normal-value experts (conditional normal means of the other distributional members + strong finals)
    SRC += [("ft_ftB3", ["y_ft", "pdef_ft"]), ("ft_ftSC", ["y_ft", "pdef_ft"]), ("tr_trE", ["y_tr", "pdef_tr"]), ("team_t9004", ["y_t9004"]),
            ("gbm_vap3", ["y_catmixapy_s201"]), ("gbm_hf10", ["y_hpo10"]), ("gbm_anom1", ["y_anom_s1"]), ("nn_nn2", ["y_nn"])]
if PLUS:  # 7-seed base transformer and the transformer's predictive entropy
    SRC += [("ft_ftB7", ["y_ft", "pdef_ft"]), ("ft_ftC7", ["ent_ft"])]
CM_FINAL = os.environ.get("CM_FINAL", "") == "1"
FMAP = {"gbm_v2": "gbm_v2F", "gbm_v3a": "gbm_v3aF", "gbm_anom1": "gbm_anom1F", "ft_ftC7": "ft_ftCF",
        "ft_ftB3": "ft_ftB7F", "ft_ftB7": "ft_ftB7F", "ft_ftSC": "ft_ftSCF", "gbm_vap3": "gbm_vap3F"}
if CM_FINAL:
    SRC = [(t, c) for (t, c) in SRC if t in FMAP]
    SRC.append(("gbm_v2", ["y_mix"]))
def build(test):
    D = meta.filter(pl.col("is_test") == test)
    base = pl.read_parquet("work/v23_test.parquet" if test else "work/v23_oof.parquet")
    D = D.join(base.select([K, "v23"] + ([] if test else ["y"])), on=K, how="left" if (CM_FINAL and test) else "inner")
    for tag, cols in SRC:
        rtag = FMAP.get(tag, tag) if (CM_FINAL and test) else tag
        D = D.join(load(rtag, cols, test).rename({c: f"{tag}:{c}" for c in cols}), on=K, how="left")
    return D
if CM_FINAL:
    pass
O = build(False); T = build(True)
if CM_FINAL:
    O = O.with_columns(v23=pl.col("v23").fill_null(pl.col("gbm_v2:y_mix")))
    T = T.with_columns(v23=pl.col("v23").fill_null(pl.col("gbm_v2:y_mix")))
log(f"rows oof {O.height}, test {T.height}")

def arrays(D):
    nm = D["nm"].to_numpy() > 0; C = D["C"].to_numpy().astype(np.float64)
    lo = D["lo"].fill_null(-1e9).to_numpy(); hi = D["hi"].fill_null(1e9).to_numpy()
    g = lambda c: (D[c].cast(pl.Float64).fill_null(np.nan).to_numpy() if c in D.columns else np.full(D.height, np.nan))
    v23 = D["v23"].to_numpy().astype(np.float64)
    pdf = np.clip(np.nan_to_num(g("ft_ftC7:pdef_ft"), nan=0.0), 0, 0.98)
    ft_norm = (np.nan_to_num(g("ft_ftC7:y_ft"), nan=0) - pdf * C) / (1 - pdf)
    from_d = np.where(nm, D["ml"].fill_null(0).to_numpy() - np.nan_to_num(g("gbm_v2:d_hat"), nan=0), np.nan)
    vexp = g("gbm_v2:y_mix") if CM_FINAL else v23
    experts = [g("gbm_v2:y_normal"), g("gbm_v3a:y_catnormal"), ft_norm, from_d, vexp, g("ft_ftC7:q50_ft")]
    probs_more = []
    if MORE:
        def cond(yc, pc):
            p = np.clip(np.nan_to_num(g(pc), nan=0.0), 0, 0.98); return (np.nan_to_num(g(yc), nan=0) - p * C) / (1 - p)
        experts += [cond("ft_ftB3:y_ft", "ft_ftB3:pdef_ft"), cond("ft_ftSC:y_ft", "ft_ftSC:pdef_ft"), cond("tr_trE:y_tr", "tr_trE:pdef_tr"),
                    g("team_t9004:y_t9004"), g("gbm_vap3:y_catmixapy_s201"), g("gbm_hf10:y_hpo10"), g("gbm_anom1:y_anom_s1"), g("nn_nn2:y_nn")]
        probs_more = [g("ft_ftB3:pdef_ft"), g("ft_ftSC:pdef_ft"), g("tr_trE:pdef_tr")]
    if PLUS:
        experts += [cond("ft_ftB7:y_ft", "ft_ftB7:pdef_ft")]; probs_more += [g("ft_ftB7:pdef_ft"), g("ft_ftC7:ent_ft")]
    N = np.column_stack([np.clip(np.where(np.isfinite(e), e, v23), np.maximum(lo, 0), hi) for e in experts]).astype(np.float32)
    probs = [g(c) for c in ("gbm_v2:p_def", "gbm_v2:p_gt1h", "gbm_anom1:pa_def", "gbm_anom1:pa_near", "gbm_anom1:pa_long", "ft_ftC7:pdef_ft")]
    va_long = np.nan_to_num(g("gbm_anom1:va_long"), nan=5000.0)
    q = [g("ft_ftC7:q10_ft"), g("ft_ftC7:q90_ft")]
    ctx = np.column_stack([nm, C / 3600, np.log1p(np.clip(C, 0, None)) / 10, C < 0, D["ml"].fill_null(0).to_numpy() / 3600,
                           np.clip(lo, -1, None) / 3600, np.clip(hi, None, 1e5) / 3600, D["hour"].to_numpy() / 24, D["adsb_has_gnd"].fill_null(-1).to_numpy(),
                           D["sched_infeas"].fill_null(-1).to_numpy(), v23 / 1000, va_long / 10000]
                          + [np.nan_to_num(p, nan=0.0) for p in probs + probs_more] + [np.nan_to_num(x, nan=0.0) / 1000 for x in q]
                          + [np.clip((N - N.mean(1, keepdims=True)) / 300, -10, 10)]).astype(np.float32)
    feas = (~((nm & (D["sched_infeas"].fill_null(0).to_numpy() > 0)) | (C < 0))).astype(np.float32)
    return dict(X=ctx, N=N, C=C.astype(np.float32), VL=va_long.astype(np.float32), feas=feas, lo=lo, hi=hi, nm=nm)
AO = arrays(O); AT = arrays(T)
APS = sorted(O["ADEP_mvt"].unique().to_list()); apo = np.array([APS.index(a) for a in O["ADEP_mvt"]]); apt = np.array([APS.index(a) for a in T["ADEP_mvt"]])
y = O["y"].to_numpy().astype(np.float32); mon = O["mon"].to_numpy(); ro = np.abs(y - 86400) < 3600
WLNG = float(arg("--w-lirfng", "1"))  # training weight of LIRF rows without an ADS-B ground track (the 2026 LIRF July population)
WROW = np.where((O["ADEP_mvt"] == "LIRF").to_numpy() & ~(O["adsb_has_gnd"].fill_null(0).to_numpy() > 0), WLNG, 1.0).astype(np.float32)
Cn = AO["C"]; bms = np.abs(Cn - y); dfl = O["dflt"].fill_null(False).to_numpy() & (AO["feas"] > 0)
cls = np.where(dfl, 0, np.where((bms <= 900) & (Cn > 1800), 1, np.where(y > 3600, 2, 3))).astype(np.int64)
log("class counts (default, near, long, normal):", np.bincount(cls[~ro], minlength=4).tolist())


class CM(nn.Module):
    def __init__(self, dx, ne):
        super().__init__()
        self.ap = nn.Embedding(len(APS), 8)
        self.body = nn.Sequential(nn.Linear(dx + 8, WIDE), nn.SiLU(), nn.Dropout(0.1), nn.Linear(WIDE, WIDE), nn.SiLU())
        self.cls = nn.Linear(WIDE, 4); self.gate = nn.Linear(WIDE, ne); self.aux = nn.Linear(WIDE, 3)
        for m in (self.gate, self.aux):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x, ap, N, C, VL, feas):
        h = self.body(torch.cat([x, self.ap(ap)], 1))
        lg = self.cls(h); lg = torch.cat([lg[:, :1].masked_fill(feas.unsqueeze(1) < 0.5, -1e4), lg[:, 1:]], 1)
        P = torch.softmax(lg, 1); a, b, r = self.aux(h).unbind(1)
        normal = (torch.softmax(self.gate(h), 1) * N).sum(1)
        near = C - 300 * F.softplus(a + 1.0)
        long_ = VL * torch.exp(0.3 * torch.tanh(b))
        return P[:, 0] * C + P[:, 1] * near + P[:, 2] * long_ + P[:, 3] * normal + 100 * torch.tanh(r / 10), lg


def tens(A, idx, ap):
    return [torch.from_numpy(np.ascontiguousarray(v[idx])).to(dev) for v in (A["X"], ap, A["N"], A["C"], A["VL"], A["feas"])]


def fit_predict(tr, A_te, ap_te, te, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = CM(AO["X"].shape[1], AO["N"].shape[1]).to(dev); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    Tr = tens(AO, tr, apo); Y = torch.from_numpy(y[tr]).to(dev); CL = torch.from_numpy(cls[tr]).to(dev); WR = torch.from_numpy(WROW[tr]).to(dev)
    n = len(tr); bs = 4096
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=dev)
        for i in range(n // bs):
            b = perm[i * bs:(i + 1) * bs]
            p, lg = m(*[t[b] for t in Tr])
            loss = (WR[b] * (p - Y[b]) ** 2).sum() / WR[b].sum() / 1e4 + 0.1 * (WR[b] * F.cross_entropy(lg, CL[b], reduction="none")).sum() / WR[b].sum()
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); Te = tens(A_te, te, ap_te); outp, outc = [], []
    with torch.no_grad():
        for i in range(0, len(te), 65536):
            p, lg = m(*[t[i:i + 65536] for t in Te]); outp.append(p.cpu().numpy()); outc.append(torch.softmax(lg, 1).cpu().numpy())
    return np.concatenate(outp), np.concatenate(outc)


def lane_mask(A, train):
    if LANE == "ge1h":
        return A["nm"] & (A["C"] >= (2700 if train else 3600))
    if LANE == "nonm":
        return ~A["nm"]
    return np.ones(len(A["C"]), bool)
lo_tr = lane_mask(AO, True); log(f"lane {LANE}: {int((lo_tr & ~ro).sum())} training rows")
for k in list(range(1, 13)) + [0]:
    if k == 0:
        tr = np.where(~ro & lo_tr)[0]; te = np.arange(T.height); A_te, ap_te, D_te = AT, apt, T
    else:
        tr = np.where((mon != k) & ~ro & lo_tr)[0]; te = np.where(mon == k)[0]; A_te, ap_te, D_te = AO, apo, O
    res = [fit_predict(tr, A_te, ap_te, te, s) for s in SEEDS]
    p = np.clip(np.mean([r[0] for r in res], 0), np.maximum(A_te["lo"][te], 0), A_te["hi"][te]); pc = np.mean([r[1] for r in res], 0)
    p = np.where(lane_mask(A_te, False)[te], p, np.nan)  # outside the specialist's lane: no prediction
    df = pl.DataFrame({K: D_te[K].to_numpy()[te], "mon": np.full(len(te), k, dtype=np.int8), "y_cm": p, "cm_pdef": pc[:, 0], "cm_pnear": pc[:, 1], "cm_plong": pc[:, 2]})
    if k > 0:
        yy = y[te]; mm = ~ro[te]
        mm &= np.isfinite(p)
        df = df.with_columns(y=pl.Series(yy)); log(f"month {k} ex_ro RMSE cm {np.sqrt(np.mean((p[mm] - yy[mm]) ** 2)):.2f} v23 {np.sqrt(np.mean((AO['N'][te, 4][mm] - yy[mm]) ** 2)):.2f}")
    df.write_parquet(f"work/cm_{TAG}/outer_{k:02d}.parquet")
log("done")
