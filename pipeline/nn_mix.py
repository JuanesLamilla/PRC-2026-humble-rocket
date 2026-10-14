"""Window-bounded neural mixture on the full feature matrix (GPU, PyTorch).

usage: python nn_mix.py OUTER [--seeds 0,1,2] [--tag T] [--frac F] [--epochs E]
  OUTER 1..12 = held-out 2025 month; 0 = fit on all 2025, predict the 2026 ranking rows.

Prediction per row:
  p    = sigmoid(a) * feasible        (schedule default; feasible = 0 on NM rows with |SCHED-LOBT| > 3616)
  q    = sigmoid(b) * (1 - p)         (abnormal long stamp, value v_e = C_e-ish: max(C, 0) + softplus(e)*1000)
  n    = NM rows: lo + (hi - lo) * sigmoid(z)   (normal taxi bounded to the LOBT join window)
         no-NM rows: softplus(z) * 1000
  yhat = p * C + q * v_e + (1 - p - q) * n
Loss: MSE(yhat, y)/1e6 on rows without rollovers + 0.1 * BCE(p, BLOCK==SCHED). Embeddings for categoricals.
"""
import os, sys, time
import numpy as np
import polars as pl
import torch, torch.nn as nn, torch.nn.functional as F

OUTER = int(sys.argv[1])
arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
SEEDS = [int(s) for s in arg("--seeds", "0,1,2").split(",")]
TAG = arg("--tag", "nn1"); FRAC = float(arg("--frac", "1")); EPOCHS = int(arg("--epochs", "10"))
os.makedirs(f"work/nn_{TAG}", exist_ok=True)
t0 = time.time()
def log(*a): print(f"[{time.time() - t0:7.0f}s]", *a, flush=True)
dev = torch.device("cuda")
torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
key = "MVT_ID_mvt"
FMF = os.environ.get("FM", "work/fm.parquet")
fm = pl.read_parquet(FMF)
if FRAC < 1:
    fm = fm.filter(pl.col("is_test") | (pl.col(key).hash(7) % 1000 < int(FRAC * 1000)))
S1TAG = arg("--s1", "")
if S1TAG:
    S1C = ["d_hat", "dq10", "dq50", "dq90", "p_def", "p_gt1h"]
    si = pl.read_parquet(f"work/gbm_{S1TAG}/s1_inner_{OUTER:02d}.parquet")
    so = pl.read_parquet(f"work/gbm_{S1TAG}/s1_outer_{OUTER:02d}.parquet")
    s1 = pl.concat([si, so]).unique(key, keep="first")
    fm = fm.join(s1, on=key, how="left")
    nmx = pl.col("nm") > 0
    fm = fm.with_columns(y_from_d=pl.when(nmx).then(pl.col("ml") - pl.col("d_hat")),
                         b_lo=pl.when(nmx).then(pl.col("ml") - pl.col("dq90")), b_hi=pl.when(nmx).then(pl.col("ml") - pl.col("dq10")))
    if OUTER > 0:
        fm = fm.filter(~pl.col("is_test"))
META = {key, "split", "is_test", "ADEP_mvt", "mon", "day", "y", "dlt", "dflt"}
cats = [c for c in fm.columns if c.startswith("cat_")]
nums = [c for c in fm.columns if c not in META and c not in cats]
train_mask = ~fm["is_test"].to_numpy()
# numeric preprocessing: robust clip + standardise on 2025 rows; NaN -> 0 plus missing flags
Xn = fm.select(nums).to_numpy().astype(np.float32)
nanm = np.isnan(Xn)
flag_cols = np.where(nanm[train_mask].mean(0) > 0.01)[0]
q1, q99 = np.nanpercentile(Xn[train_mask], 1, axis=0), np.nanpercentile(Xn[train_mask], 99, axis=0)
Xn = np.clip(Xn, q1, q99)
mu, sd = np.nanmean(Xn[train_mask], 0), np.nanstd(Xn[train_mask], 0) + 1e-6
Xn = np.nan_to_num((Xn - mu) / sd).astype(np.float32)
Xn = np.hstack([Xn, nanm[:, flag_cols].astype(np.float32)])
Xc = np.stack([fm[c].fill_null(-1).to_numpy() for c in cats], 1)
cards = []
for j in range(Xc.shape[1]):
    u, inv = np.unique(Xc[:, j], return_inverse=True); Xc[:, j] = inv; cards.append(len(u))
Xc = Xc.astype(np.int64)
C = fm["C"].to_numpy().astype(np.float32); lo = fm["lo"].to_numpy(); hi = fm["hi"].to_numpy()
nm = (fm["nm"].to_numpy() > 0)
lo = np.where(nm, lo, 0).astype(np.float32); hi = np.where(nm, hi, 0).astype(np.float32)
feas = np.where(nm & (fm["sched_infeas"].fill_null(0).to_numpy() > 0), 0.0, 1.0).astype(np.float32)
y = fm["y"].fill_null(0).to_numpy().astype(np.float32); dfl = fm["dflt"].fill_null(False).to_numpy().astype(np.float32)
mon = fm["mon"].to_numpy(); ro = np.abs(y - 86400) < 3600
dlt_np = fm["dlt"].cast(pl.Float32).fill_null(float("nan")).to_numpy().astype(np.float32)
AUX = "--aux" in sys.argv
log(f"rows {len(y)}, numeric {Xn.shape[1]}, cats {len(cats)}")


class Net(nn.Module):
    def __init__(self, n_num, cards, width=int(arg("--width", "512"))):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(c + 1, min(16, max(2, c // 2))) for c in cards])
        d_in = n_num + sum(e.embedding_dim for e in self.emb)
        self.body = nn.Sequential(nn.Linear(d_in, width), nn.SiLU(), nn.Dropout(0.1), nn.Linear(width, width), nn.SiLU(),
                                  nn.Dropout(0.1), nn.Linear(width, width // 2), nn.SiLU())
        self.head = nn.Linear(width // 2, 6)

    def forward(self, xn, xc, C, lo, hi, nm, feas):
        h = torch.cat([xn] + [e(xc[:, i]) for i, e in enumerate(self.emb)], 1)
        a, b, z, e, dd, _ = self.head(self.body(h)).unbind(1)
        self.d_aux = dd * 1000.0
        p = torch.sigmoid(a) * feas
        q = torch.sigmoid(b - 3.0) * (1 - p)
        n = torch.where(nm, lo + (hi - lo) * torch.sigmoid(z), F.softplus(z) * 1000.0)
        ve = torch.clamp(C, min=0) + F.softplus(e) * 1000.0
        return p * C + q * ve + (1 - p - q) * n, p


def tens(idx):
    return [torch.from_numpy(a[idx]).to(dev) for a in (Xn, Xc, C, lo, hi)] + [torch.from_numpy(nm[idx]).to(dev), torch.from_numpy(feas[idx]).to(dev)]


def train_predict(tr_idx, te_idx, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    net = Net(Xn.shape[1], cards).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-5)
    T = tens(tr_idx); yt = torch.from_numpy(y[tr_idx]).to(dev); dt = torch.from_numpy(dfl[tr_idx]).to(dev)
    dd_t = torch.from_numpy(dlt_np[tr_idx]).to(dev)
    n = len(tr_idx); bs = 4096; steps = EPOCHS * (n // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    net.train(); s = 0
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=dev); tot = 0.0
        for i in range(n // bs):
            b = perm[i * bs:(i + 1) * bs]
            yh, p = net(*[t[b] for t in T])
            loss = F.mse_loss(yh, yt[b]) / 1e6 + 0.1 * F.binary_cross_entropy(p.clamp(1e-6, 1 - 1e-6), dt[b])
            if AUX:
                dm = torch.isfinite(dd_t[b])
                if dm.any():
                    loss = loss + 0.1 * F.mse_loss(net.d_aux[dm], dd_t[b][dm]) / 1e6
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
            tot += loss.item()
        log(f"seed {seed} epoch {ep} loss {tot / (n // bs):.4f}")
    net.eval(); out = []
    TE = tens(te_idx)
    with torch.no_grad():
        for i in range(0, len(te_idx), 65536):
            out.append(net(*[t[i:i + 65536] for t in TE])[0].float().cpu().numpy())
    return np.concatenate(out)


if OUTER == 0:
    tr_idx = np.where(train_mask & ~ro)[0]; te_idx = np.where(~train_mask)[0]
else:
    tr_idx = np.where(train_mask & ~ro & (mon != OUTER))[0]; te_idx = np.where(train_mask & (mon == OUTER))[0]
preds = [train_predict(tr_idx, te_idx, s) for s in SEEDS]
res = pl.DataFrame({key: fm[key].to_numpy()[te_idx], "mon": mon[te_idx], "y_nn": np.mean(preds, 0)})
for s, p in zip(SEEDS, preds):
    res = res.with_columns(pl.Series(f"nn_s{s}", p))
if OUTER > 0:
    res = res.with_columns(y=pl.Series(y[te_idx]))
    yy = y[te_idx]; m = ~ro[te_idx]
    log(f"month {OUTER} ex_ro RMSE nn {np.sqrt(np.mean((res['y_nn'].to_numpy()[m] - yy[m]) ** 2)):.2f}")
res.write_parquet(f"work/nn_{TAG}/outer_{OUTER:02d}.parquet")
log("written", res.shape)
