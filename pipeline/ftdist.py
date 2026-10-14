"""Distributional feature-token transformer (GPU, PyTorch): a new architecture, independent of the GBM stack.

usage: python ftdist.py OUTER [--seeds 0] [--tag T] [--frac F] [--epochs E] [--strikes data/strike_feats.parquet]
  OUTER 1..12 = held-out 2025 month; 0 = fit on all 2025, predict the 2026 ranking rows.

Backbone: FT-Transformer style (Gorishniy et al. 2021). Every numeric feature j becomes a token x_j * W_j + b_j
(a learned "missing" token when NaN), every categorical an embedding, plus a CLS token; pre-norm transformer layers.
Head: one softmax over
  * K taxi-time bins (20 s to 1 h, 120 s to 4 h), masked to the NM join window [MVT-LOBT-3620, MVT-LOBT+3620] on NM rows;
  * one point mass at y = C = MVT - SCHED (schedule default), masked where a default is impossible
    (NM row with |SCHED - LOBT| > 3616, or C < 0).
Target: default rows -> the point mass; other rows -> Gaussian-smoothed histogram (HL-Gauss) of y over the bins.
Prediction = distribution mean (the squared-error optimum); quantiles and p_def are written for comparison.
Loss: cross-entropy + LAM * MSE(mean, y) / 1e6 (--mse, default 10). Rollover rows (|y - 86400| < 3600) are left out of training, as for
every other member. No stage-1 GBM outputs are used, so the model is independent of the stack.
"""
import math, os, sys, time
import numpy as np
import polars as pl
import torch, torch.nn as nn, torch.nn.functional as F

OUTER = int(sys.argv[1])
arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
SEEDS = [int(s) for s in arg("--seeds", "0").split(",")]
TAG = arg("--tag", "ft1"); FRAC = float(arg("--frac", "1")); EPOCHS = int(arg("--epochs", "6"))
D = int(arg("--d", "64")); NL = int(arg("--layers", "3")); BS = int(arg("--bs", "1024")); LR = float(arg("--lr", "5e-4"))
STRIKES = arg("--strikes", ""); LAM = float(arg("--mse", "10")); BW = float(arg("--bw", "20")); S1TAG = arg("--s1", ""); WNONM = float(arg("--w-nonm", "1")); WGE = float(arg("--w-ge", "1"))  # --w-ge: training weight of rows with C >= 1 h
os.makedirs(f"work/ft_{TAG}", exist_ok=True)
t0 = time.time()
def log(*a): print(f"[{time.time() - t0:7.0f}s]", *a, flush=True)
dev = torch.device("cuda")
torch.manual_seed(0)
key = "MVT_ID_mvt"
fm = pl.read_parquet(os.environ.get("FM", "work/fm2.parquet"))
if FRAC < 1:
    fm = fm.filter(pl.col("is_test") | (pl.col(key).hash(7) % 1000 < int(FRAC * 1000)))
if OUTER > 0:
    fm = fm.filter(~pl.col("is_test"))
if STRIKES:
    st = pl.read_parquet(STRIKES)
    fm = fm.join(st, on=key, how="left").with_columns([pl.col(c).fill_null(0) for c in st.columns if c != key])
    log("strike features:", [c for c in st.columns if c != key])
if S1TAG:  # optional: nested stage-1 GBM outputs as extra tokens (same nesting as gbm_nested.py, no leakage)
    S1C = ["d_hat", "dq10", "dq50", "dq90", "p_def", "p_gt1h"]
    s1 = pl.concat([pl.read_parquet(f"work/gbm_{S1TAG}/s1_inner_{OUTER:02d}.parquet"), pl.read_parquet(f"work/gbm_{S1TAG}/s1_outer_{OUTER:02d}.parquet")]).unique(key, keep="first")
    fm = fm.join(s1.select([key] + S1C), on=key, how="left")
    log("stage-1 tokens:", S1C)
META = {key, "split", "is_test", "ADEP_mvt", "mon", "day", "y", "dlt", "dflt"}
cats = [c for c in fm.columns if c.startswith("cat_")] + ["ADEP_mvt"]
nums = [c for c in fm.columns if c not in META and c not in cats]
DROP = [x for x in arg("--drop", "").split(",") if x]  # drop numeric columns whose name contains any of these substrings (or equals "=name")
if DROP:
    dropped = [c for c in nums if any((c == d[1:]) if d.startswith("=") else (d in c) for d in DROP)]
    nums = [c for c in nums if c not in dropped]; log(f"dropped {len(dropped)} columns: {dropped}")
train_mask = ~fm["is_test"].to_numpy()
Xn = fm.select(nums).to_numpy().astype(np.float32)
nan = np.isnan(Xn)
q1, q99 = np.nanpercentile(Xn[train_mask], 1, axis=0), np.nanpercentile(Xn[train_mask], 99, axis=0)
Xn = np.clip(Xn, q1, q99)
mu, sd = np.nanmean(Xn[train_mask], 0), np.nanstd(Xn[train_mask], 0) + 1e-6
Xn = np.where(nan, 0.0, (Xn - mu) / sd).astype(np.float32)
MASKAP = arg("--mask-ap", "")  # e.g. LIRF: every ADS-B column treated as missing on that airport's rows (train and predict)
if MASKAP:
    ac = np.array(["adsb" in c for c in nums]); rows = np.isin(fm["ADEP_mvt"].to_numpy(), MASKAP.split(","))
    nan[np.ix_(rows, ac)] = True; Xn[np.ix_(rows, ac)] = 0.0
    log(f"masked {ac.sum()} ADS-B columns on {rows.sum()} rows of {MASKAP}")
Xc = np.stack([fm[c].cast(pl.Utf8).fill_null("NA").to_numpy() if c == "ADEP_mvt" else fm[c].fill_null(-1).to_numpy() for c in cats], 1)
cards = []
Xci = np.zeros(Xc.shape, dtype=np.int64)
for j in range(Xc.shape[1]):
    u, inv = np.unique(Xc[:, j], return_inverse=True); Xci[:, j] = inv; cards.append(len(u))
C = fm["C"].to_numpy().astype(np.float32)
nm = fm["nm"].to_numpy() > 0
lo = np.where(nm, fm["lo"].fill_null(0).to_numpy(), -1e9).astype(np.float32)
hi = np.where(nm, fm["hi"].fill_null(0).to_numpy(), 1e9).astype(np.float32)
feas = ~((nm & (fm["sched_infeas"].fill_null(0).to_numpy() > 0)) | (C < 0))
y = fm["y"].fill_null(0).to_numpy().astype(np.float32)
dfl = fm["dflt"].fill_null(False).to_numpy() & feas
mon = fm["mon"].to_numpy(); ro = np.abs(y - 86400) < 3600
log(f"rows {len(y)}, numeric tokens {len(nums)}, categorical tokens {len(cats)}")

edges = np.concatenate([np.arange(0, 3600, BW), np.arange(3600, 14400 + 1, 6 * BW)]).astype(np.float32)
centers = (edges[:-1] + edges[1:]) / 2; K = len(centers)
E_lo = torch.tensor(edges[:-1], device=dev); E_hi = torch.tensor(edges[1:], device=dev); CEN = torch.tensor(centers, device=dev)
SIG = torch.where(E_hi <= 3600, torch.tensor(0.75 * BW, device=dev), torch.tensor(4.5 * BW, device=dev))


class FTDist(nn.Module):
    def __init__(self, n_num, cards, d=D, layers=NL, heads=4):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_num, d) * 0.1); self.B = nn.Parameter(torch.zeros(n_num, d))
        self.M = nn.Parameter(torch.zeros(n_num, d))  # missing-value token per feature
        self.emb = nn.ModuleList([nn.Embedding(c + 1, d) for c in cards])
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        lay = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(lay, layers)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, K + 1))

    def forward(self, xn, isnan, xc):
        tok = torch.where(isnan.unsqueeze(-1), self.M, xn.unsqueeze(-1) * self.W + self.B)
        ctok = torch.stack([e(xc[:, i]) for i, e in enumerate(self.emb)], 1)
        h = torch.cat([self.cls.expand(xn.shape[0], -1, -1), tok, ctok], 1)
        return self.head(self.norm(self.enc(h)[:, 0])).float()


def dist(logits, C, lo, hi, feas):
    """Masked softmax -> (bin probs, default prob, mean)."""
    binmask = (E_hi.unsqueeze(0) >= lo.unsqueeze(1)) & (E_lo.unsqueeze(0) <= hi.unsqueeze(1))
    binmask[binmask.sum(1) == 0] = True
    lg = torch.cat([logits[:, :K].masked_fill(~binmask, -1e9), logits[:, K:].masked_fill(~feas.unsqueeze(1), -1e9)], 1)
    P = torch.softmax(lg, 1)
    return P, binmask, (P[:, :K] * CEN).sum(1) + P[:, K] * C


def target(yb, defb, binmask):
    yc = yb.clamp(0, 14399).unsqueeze(1)
    T = 0.5 * (torch.erf((E_hi - yc) / (SIG * math.sqrt(2))) - torch.erf((E_lo - yc) / (SIG * math.sqrt(2))))
    T = T * binmask; T = T / T.sum(1, keepdim=True).clamp_min(1e-9)
    T = torch.where(defb.unsqueeze(1), torch.zeros_like(T), T)
    return torch.cat([T, defb.float().unsqueeze(1)], 1)


def batch(idx):
    g = lambda a: torch.from_numpy(a[idx]).to(dev, non_blocking=True)
    return g(Xn), g(nan), g(Xci), g(C), g(lo), g(hi), g(feas), g(y), g(dfl)


def train_predict(tr_idx, te_idx, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    net = FTDist(len(nums), cards).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-5)
    steps = EPOCHS * (len(tr_idx) // BS)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1)
    net.train()
    for ep in range(EPOCHS):
        perm = np.random.permutation(tr_idx); tot = 0.0; n = 0
        for i in range(len(perm) // BS):
            xn, isn, xc, Cb, lob, hib, fb, yb, db = batch(perm[i * BS:(i + 1) * BS])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = net(xn, isn, xc)
            P, bm, mean = dist(logits, Cb, lob, hib, fb)
            T = target(yb, db, bm)
            wr = torch.where(lob < -1e8, WNONM, 1.0)  # per-row weight: no-NM rows (no window) x WNONM
            wr = wr * torch.where(Cb >= 3600, WGE, 1.0)
            loss = ((-(T * torch.log(P.clamp_min(1e-9))).sum(1) + LAM * (mean - yb) ** 2 / 1e6) * wr).sum() / wr.sum()
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step()
            tot += loss.item(); n += 1
            if i % 500 == 0:
                log(f"seed {seed} epoch {ep} step {i} loss {tot / n:.4f}")
        log(f"seed {seed} epoch {ep} done loss {tot / n:.4f}")
    net.eval(); outs = []
    with torch.no_grad():
        for i in range(0, len(te_idx), 8192):
            xn, isn, xc, Cb, lob, hib, fb, yb, db = batch(te_idx[i:i + 8192])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = net(xn, isn, xc)
            P, bm, mean = dist(logits, Cb, lob, hib, fb)
            # quantiles of the full distribution (bins + default atom at C)
            vals = torch.cat([CEN.expand(len(Cb), -1), Cb.unsqueeze(1)], 1)
            order = vals.argsort(1); cs = torch.gather(P, 1, order).cumsum(1); sv = torch.gather(vals, 1, order)
            qs = [torch.gather(sv, 1, (cs < q).sum(1, keepdim=True).clamp_max(K)).squeeze(1) for q in (0.1, 0.5, 0.9)]
            ent = -(P * torch.log(P.clamp_min(1e-12))).sum(1)
            outs.append(torch.stack([mean, P[:, K], *qs, ent], 1).cpu().numpy())
    return np.concatenate(outs)


if OUTER == 0:
    tr_idx = np.where(train_mask & ~ro)[0]; te_idx = np.where(~train_mask)[0]
else:
    tr_idx = np.where(train_mask & ~ro & (mon != OUTER))[0]; te_idx = np.where(train_mask & (mon == OUTER))[0]
res = [train_predict(tr_idx, te_idx, s) for s in SEEDS]
R = np.mean(res, 0)
out = pl.DataFrame({key: fm[key].to_numpy()[te_idx], "mon": mon[te_idx], "y_ft": R[:, 0], "pdef_ft": R[:, 1],
                    "q10_ft": R[:, 2], "q50_ft": R[:, 3], "q90_ft": R[:, 4], "ent_ft": R[:, 5]})
if OUTER > 0:
    yy = y[te_idx]; m = ~ro[te_idx]
    out = out.with_columns(y=pl.Series(yy))
    log(f"month {OUTER} ex_ro RMSE ft {np.sqrt(np.mean((R[m, 0] - yy[m]) ** 2)):.2f}; 80% interval coverage {np.mean((yy[m] >= R[m, 2]) & (yy[m] <= R[m, 4])):.3f}")
out.write_parquet(f"work/ft_{TAG}/outer_{OUTER:02d}.parquet")
log("written", out.shape)
