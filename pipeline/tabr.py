"""Retrieval-augmented network (TabR-style) on the full feature matrix (GPU, PyTorch). A third independent architecture.

usage: python tabr.py OUTER [--seeds 0] [--tag T] [--frac F] [--epochs E] [--d 256] [--m 64] [--refresh 400] [--strikes P]
  OUTER 1..12 = held-out 2025 month; 0 = fit on all 2025, predict the 2026 ranking rows.

Each departure is encoded (numeric + categorical embeddings -> MLP) into h and a key k = K(h). Its m nearest LABELLED
2025 departures (squared distance in key space, from a candidate-key table refreshed every --refresh steps; the query
itself is excluded; candidates never include the held-out month or test rows,
and in training never the query's own month, mirroring prediction) contribute values
  v_i = LabelEmb([y_i, BLOCK==SCHED_i, log y_i]) + T(k - k_i)
weighted by softmax(-|k - k_i|^2). The retrieved context is added to h and decoded by an MLP into the window-bounded
mixture used by our other neural members:
  yhat = p * C + (1 - p) * n,  p = sigmoid(a) * feasible,  n = lo + (hi - lo) * sigmoid(z) on NM rows, softplus(z)*1000 otherwise.
Loss: MSE/1e6 + 0.1 BCE(p, default). Rollover rows are excluded from queries and candidates. No stage-1 GBM outputs.
"""
import os, sys, time
import numpy as np
import polars as pl
import torch, torch.nn as nn, torch.nn.functional as F

OUTER = int(sys.argv[1])
arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
SEEDS = [int(s) for s in arg("--seeds", "0").split(",")]
TAG = arg("--tag", "tr1"); FRAC = float(arg("--frac", "1")); EPOCHS = int(arg("--epochs", "6"))
D = int(arg("--d", "256")); M = int(arg("--m", "64")); REFRESH = int(arg("--refresh", "400")); BS = int(arg("--bs", "1024"))
LR = float(arg("--lr", "1e-3")); STRIKES = arg("--strikes", "")
EB = int(arg("--eblocks", "1")); PB = int(arg("--pblocks", "1"))
os.makedirs(f"work/tr_{TAG}", exist_ok=True)
t0 = time.time()
def log(*a): print(f"[{time.time() - t0:7.0f}s]", *a, flush=True)
dev = torch.device("cuda")
key = "MVT_ID_mvt"
fm = pl.read_parquet(os.environ.get("FM", "work/fm2.parquet"))
if FRAC < 1:
    fm = fm.filter(pl.col("is_test") | (pl.col(key).hash(7) % 1000 < int(FRAC * 1000)))
if OUTER > 0:
    fm = fm.filter(~pl.col("is_test"))
if STRIKES:
    st = pl.read_parquet(STRIKES)
    fm = fm.join(st, on=key, how="left").with_columns([pl.col(c).fill_null(0) for c in st.columns if c != key])
META = {key, "split", "is_test", "ADEP_mvt", "mon", "day", "y", "dlt", "dflt"}
cats = [c for c in fm.columns if c.startswith("cat_")] + ["ADEP_mvt"]
nums = [c for c in fm.columns if c not in META and c not in cats]
train_mask = ~fm["is_test"].to_numpy()
Xn = fm.select(nums).to_numpy().astype(np.float32)
nanm = np.isnan(Xn)
flag_cols = np.where(nanm[train_mask].mean(0) > 0.01)[0]
q1, q99 = np.nanpercentile(Xn[train_mask], 1, axis=0), np.nanpercentile(Xn[train_mask], 99, axis=0)
Xn = np.clip(Xn, q1, q99)
mu, sd = np.nanmean(Xn[train_mask], 0), np.nanstd(Xn[train_mask], 0) + 1e-6
Xn = np.hstack([np.nan_to_num((Xn - mu) / sd), nanm[:, flag_cols]]).astype(np.float32)
Xc = np.stack([fm[c].cast(pl.Utf8).fill_null("NA").to_numpy() if c == "ADEP_mvt" else fm[c].fill_null(-1).to_numpy() for c in cats], 1)
Xci = np.zeros(Xc.shape, dtype=np.int64); cards = []
for j in range(Xc.shape[1]):
    u, inv = np.unique(Xc[:, j], return_inverse=True); Xci[:, j] = inv; cards.append(len(u))
C = fm["C"].to_numpy().astype(np.float32); nm = fm["nm"].to_numpy() > 0
lo = np.where(nm, fm["lo"].fill_null(0).to_numpy(), 0).astype(np.float32); hi = np.where(nm, fm["hi"].fill_null(0).to_numpy(), 0).astype(np.float32)
feas = np.where(nm & (fm["sched_infeas"].fill_null(0).to_numpy() > 0), 0.0, 1.0).astype(np.float32)
y = fm["y"].fill_null(0).to_numpy().astype(np.float32); dfl = fm["dflt"].fill_null(False).to_numpy().astype(np.float32)
mon = fm["mon"].to_numpy(); ro = np.abs(y - 86400) < 3600
lab = np.stack([np.clip(y, 0, 14400) / 1000.0, dfl, np.log1p(np.clip(y, 0, 2e5)) / 10.0], 1).astype(np.float32)
log(f"rows {len(y)}, numeric {Xn.shape[1]}, cats {len(cats)}")
G = {k: torch.from_numpy(v).to(dev) for k, v in dict(Xn=Xn, Xc=Xci, C=C, lo=lo, hi=hi, feas=feas, y=y, dfl=dfl, lab=lab).items()}
G["nm"] = torch.from_numpy(nm).to(dev)
# retrieval period: 2025 month for training rows, 100 + month for 2026 rows (never a candidate period)
SAMEMONTH = "--allow-same-month" not in sys.argv
PER = torch.from_numpy(np.where(train_mask, mon, 100 + mon).astype(np.int64) if SAMEMONTH else np.arange(len(y), dtype=np.int64) + 1000).to(dev)
del Xn


class Enc(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(c + 1, min(16, max(2, c // 2))) for c in cards])
        d_in = G["Xn"].shape[1] + sum(e.embedding_dim for e in self.emb)
        self.lin = nn.Linear(d_in, D)
        self.blks = nn.ModuleList([nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 2 * D), nn.ReLU(), nn.Dropout(0.1), nn.Linear(2 * D, D)) for _ in range(EB)])

    def forward(self, idx):
        xc = G["Xc"][idx]
        h = self.lin(torch.cat([G["Xn"][idx]] + [e(xc[:, i]) for i, e in enumerate(self.emb)], 1))
        for b in self.blks:
            h = h + b(h)
        return h


class TabR(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Enc(); self.K = nn.Linear(D, D)
        self.labemb = nn.Linear(3, D)
        self.T = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Dropout(0.1), nn.Linear(D, D, bias=False))
        self.blks = nn.ModuleList([nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 2 * D), nn.ReLU(), nn.Dropout(0.1), nn.Linear(2 * D, D)) for _ in range(PB)])
        self.head = nn.Sequential(nn.LayerNorm(D), nn.ReLU(), nn.Linear(D, 2))

    def keys(self, idx):
        return self.K(self.enc(idx))

    def forward(self, q, nb):
        B = len(q)
        hq = self.enc(q); kq = self.K(hq)
        kn = self.K(self.enc(nb.reshape(-1))).view(B, M, D)
        diff = kq.unsqueeze(1) - kn
        att = torch.softmax(-(diff ** 2).sum(-1), 1)
        v = self.labemb(G["lab"][nb]) + self.T(diff)
        x = hq + (att.unsqueeze(-1) * v).sum(1)
        for b in self.blks:
            x = x + b(x)
        a, z = self.head(x).float().unbind(1)
        p = torch.sigmoid(a) * G["feas"][q]
        n = torch.where(G["nm"][q], G["lo"][q] + (G["hi"][q] - G["lo"][q]) * torch.sigmoid(z), F.softplus(z) * 1000.0)
        return p * G["C"][q] + (1 - p) * n, p


@torch.no_grad()
def cand_keys(model, cand):
    model.eval(); out = []
    for i in range(0, len(cand), 65536):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model.keys(cand[i:i + 65536]).half())
    model.train()
    return torch.cat(out)


@torch.no_grad()
def retrieve(kq, CK, ck2, qpos, qper, CP):
    """top-M candidate positions for query keys kq (B,D); qpos = position of the query in the candidate table or -1;
    candidates from the query's own period (2025 month) are excluded so training retrieval matches held-out prediction."""
    kq = kq.half(); best_d = None; best_i = None
    for s in range(0, CK.shape[0], 524288):
        e = min(s + 524288, CK.shape[0])
        d = (kq.float() ** 2).sum(1, keepdim=True) - 2 * (kq @ CK[s:e].T).float() + ck2[s:e].unsqueeze(0)
        d = d.masked_fill(CP[s:e].unsqueeze(0) == qper.unsqueeze(1), float("inf"))
        inr = (qpos >= s) & (qpos < e)
        if inr.any():
            r = torch.where(inr)[0]; d[r, qpos[r] - s] = float("inf")
        dv, di = torch.topk(d, M, dim=1, largest=False); di = di + s
        if best_d is None:
            best_d, best_i = dv, di
        else:
            cd = torch.cat([best_d, dv], 1); ci = torch.cat([best_i, di], 1)
            best_d, o = torch.topk(cd, M, dim=1, largest=False); best_i = torch.gather(ci, 1, o)
    return best_i


def train_predict(cand_np, te_np, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    cand = torch.from_numpy(cand_np).to(dev)
    pos = torch.full((len(y),), -1, dtype=torch.long, device=dev); pos[cand] = torch.arange(len(cand), device=dev)
    CP = PER[cand]
    model = TabR().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    steps = EPOCHS * (len(cand_np) // BS)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1)
    step = 0
    for ep in range(EPOCHS):
        perm = cand[torch.randperm(len(cand), device=dev)]; tot = 0.0; n = 0
        for i in range(len(cand_np) // BS):
            if step % REFRESH == 0:
                CK = cand_keys(model, cand); ck2 = (CK.float() ** 2).sum(1)
            q = perm[i * BS:(i + 1) * BS]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                kq = model.keys(q)
            nb = cand[retrieve(kq, CK, ck2, pos[q], PER[q], CP)]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                yh, p = model(q, nb)
            loss = F.mse_loss(yh, G["y"][q]) / 1e6 + 0.1 * F.binary_cross_entropy(p.clamp(1e-6, 1 - 1e-6), G["dfl"][q])
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
            tot += loss.item(); n += 1; step += 1
        log(f"seed {seed} epoch {ep} loss {tot / n:.4f}")
    model.eval(); CK = cand_keys(model, cand); ck2 = (CK.float() ** 2).sum(1)
    te = torch.from_numpy(te_np).to(dev); out = []
    with torch.no_grad():
        for i in range(0, len(te), 4096):
            q = te[i:i + 4096]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                kq = model.keys(q)
                nb = cand[retrieve(kq, CK, ck2, pos[q], PER[q], CP)]
                yh, p = model(q, nb)
            out.append(torch.stack([yh.float(), p.float()], 1).cpu().numpy())
    return np.concatenate(out)


if OUTER == 0:
    cand_np = np.where(train_mask & ~ro)[0]; te_np = np.where(~train_mask)[0]
else:
    cand_np = np.where(train_mask & ~ro & (mon != OUTER))[0]; te_np = np.where(train_mask & (mon == OUTER))[0]
R = np.mean([train_predict(cand_np, te_np, s) for s in SEEDS], 0)
res = pl.DataFrame({key: fm[key].to_numpy()[te_np], "mon": mon[te_np], "y_tr": R[:, 0], "pdef_tr": R[:, 1]})
if OUTER > 0:
    yy = y[te_np]; m = ~ro[te_np]
    res = res.with_columns(y=pl.Series(yy))
    log(f"month {OUTER} ex_ro RMSE tabr {np.sqrt(np.mean((R[m, 0] - yy[m]) ** 2)):.2f}")
res.write_parquet(f"work/tr_{TAG}/outer_{OUTER:02d}.parquet")
log("written", res.shape)
