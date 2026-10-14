"""``model.kind: modernnca`` — a retrieval member (NN3, 2026-09-23).

ModernNCA (Ye, Yin, Zhan, Chao 2024, "Revisiting Nearest Neighbor for Tabular Data: A Deep
Tabular Baseline Two Decades Later", arXiv 2407.03257), a TabR-lite: a network embeds every
row, and a row's prediction is the softmax(−distance / τ)-weighted mean of the targets of the
**training-fold** rows nearest to it.

* **Encoder.** TabM's front end (:class:`prc.models.tabm._Prep`: quantile transform or the
  ``slog_patterns`` signed log, missing indicators, capped vocabularies; TabM's periodic PLR
  numeric and categorical embeddings), then ``n_blocks`` × (Linear, ReLU, dropout) of width
  ``d_block`` and a final Linear to ``d_emb``.
* **Prediction.** For a query q and candidates c with standardised (sqrt-space) targets y_c
  and row weights w_c (the corruption head's ``w_fit``; ones when absent)::

      ŷ_q = Σ_c exp(−‖e_q − e_c‖ / τ) · w_c · y_c  /  Σ_c exp(−‖e_q − e_c‖ / τ) · w_c

  Rows with ``w = 0`` are never candidates. τ is **fixed** (``temperature``, default 1): the
  distance is homogeneous in the embedding scale, so a learnable τ is the same degree of
  freedom as the scale of the encoder's last layer. ``residual_head: true`` adds a
  zero-initialised Linear(d_emb → 1) of the query embedding (off by default).
* **Training** (stochastic neighbourhood sampling). Each batch of ``batch_size`` queries
  retrieves against ``n_cand`` candidates drawn without replacement from the training fold
  (a rolling permutation, so every step's pool is distinct rows). The query itself is
  excluded from its own candidates (no self-label leak), and so is every candidate that
  shares the query's ``exclude_same_col`` value (default ``month_num``): at validation and
  test the query's month is never among the candidates (``month_*`` folds hold out whole
  months; the test months are 2026), so training must not learn to lean on same-month
  neighbours either. Candidate embeddings carry gradients (``cand_grad``), computed in
  checkpointed chunks of ``cand_chunk`` rows so that activation memory stays at one chunk.
  Loss: ``Σ w·se / Σ w`` on the standardised target, as TabM.
* **Inference** (validation, test). Exact top-``top_k`` retrieval on the device over all
  training-fold candidates — never validation or test rows, whose labels are hidden: fp32
  ``‖q‖² + ‖c‖² − 2 q·c`` in (``eval_batch_size`` × ``bank_chunk``) tiles with a running top-k,
  then the exact distances of the k chosen, then the formula above over those k.

GBDT vocabulary, early stopping and determinism are TabM's (``best_iteration`` = optimizer
steps, ``num_boost_round`` = step cap, ``early_stopping_rounds`` = patience in epochs, the
validation RMSE unweighted in sqrt space; a seed reproduces bit for bit on CPU).
"""

from __future__ import annotations

import copy
import logging
import math
import time

import numpy as np
import pandas as pd

from prc.config import ModelConfig
from prc.models.tabm import _build_net, _Prep

log = logging.getLogger("prc.modernnca")

DEFAULTS: dict = {
    # front end (TabM's)
    "d_num_emb": 16,
    "n_frequencies": 48,
    "frequency_scale": 0.1,
    "d_cat_emb": 16,
    "cat_max_levels": 1000,
    "cat_min_count": 20,
    "n_quantiles": 1024,
    "qt_subsample": 200_000,
    "slog_patterns": (),  # as prc.models.tabm
    "slog_qt": True,
    # encoder
    "n_blocks": 2,
    "d_block": 512,
    "d_emb": 128,
    "dropout": 0.1,
    # retrieval
    "temperature": 1.0,  # fixed; see the module docstring
    "n_cand": 65_536,  # candidates per training step (stochastic neighbourhood sampling)
    "cand_grad": True,  # backpropagate through the candidates' embeddings (ModernNCA)
    "cand_chunk": 16_384,  # rows per checkpointed candidate chunk / per bank-embedding chunk
    "exclude_same_col": "month_num",  # "" = only the query itself is excluded
    "top_k": 128,  # candidates per query at inference
    "bank_chunk": 131_072,  # candidates per retrieval tile at inference
    "residual_head": False,
    # training
    "learning_rate": 1e-3,
    "weight_decay": 3e-4,
    "batch_size": 2048,
    "eval_batch_size": 1024,  # queries per retrieval tile at inference
    "max_epochs": 30,
    "amp": True,
    "verbose": 1,
}


# --------------------------------------------------------------------------------------
# retrieval maths (pure torch; unit-tested)
# --------------------------------------------------------------------------------------


def _sq_dist(q, c):
    """(B, d), (M, d) fp32 -> (B, M) squared L2 distances, clamped at 0."""
    d2 = (q * q).sum(1, keepdim=True) + (c * c).sum(1)[None, :] - 2.0 * (q @ c.T)
    return d2.clamp_min(0.0)


def _nca(dist, y, logw, tau: float, excl=None):
    """Kernel-weighted mean of the candidates' targets.

    ``dist`` (B, M) distances; ``y`` and ``logw`` broadcast to (B, M) (``log w``, −inf for a
    weight-0 candidate); ``excl`` (B, M) bool, True = not a candidate for this query. A query
    left with no candidate predicts 0 (the standardised mean)."""
    import torch

    logit = -dist / tau + logw
    if excl is not None:
        logit = logit.masked_fill(excl, -math.inf)
    mx = logit.amax(dim=1, keepdim=True).detach()
    mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
    e = torch.exp(logit - mx)
    den = e.sum(1)
    return (e * y).sum(1) / den.clamp_min(1e-30)


def _topk(q, bank, k: int, chunk: int):
    """Exact top-k nearest rows of ``bank`` (N, d) for each row of ``q`` (B, d), in tiles of
    ``chunk`` bank rows: (B, k') indices, k' = min(k, N)."""
    import torch

    n = bank.shape[0]
    k = min(k, n)
    qn = (q * q).sum(1, keepdim=True)
    best_d = best_i = None
    for j in range(0, n, chunk):
        c = bank[j : j + chunk]
        d2 = qn + (c * c).sum(1)[None, :] - 2.0 * (q @ c.T)
        kk = min(k, c.shape[0])
        v, i = torch.topk(d2, kk, dim=1, largest=False, sorted=False)
        i = i + j
        if best_d is None:
            best_d, best_i = v, i
        else:
            v, i = torch.cat([best_d, v], 1), torch.cat([best_i, i], 1)
            vv, pos = torch.topk(v, min(k, v.shape[1]), dim=1, largest=False, sorted=False)
            best_d, best_i = vv, torch.gather(i, 1, pos)
    return best_i


def _groups(X: pd.DataFrame, col: str) -> np.ndarray | None:
    """Integer group codes of ``X[col]`` for the same-group exclusion; a missing value gets a
    code of its own (never equal to another row's)."""
    if not col:
        return None
    if col not in X.columns:
        raise ValueError(
            f"modernnca: exclude_same_col {col!r} is not a column of X "
            "(add the calendar block, or set exclude_same_col: '')"
        )
    s = X[col]
    v = (
        s.cat.codes.to_numpy().astype(np.float64)
        if isinstance(s.dtype, pd.CategoricalDtype)
        else s.to_numpy(dtype=np.float64, na_value=np.nan)
    )
    if isinstance(s.dtype, pd.CategoricalDtype):
        v[v < 0] = np.nan
    miss = ~np.isfinite(v)
    _, codes = np.unique(np.where(miss, 0.0, v), return_inverse=True)
    codes = codes.astype(np.int64)
    codes[miss] = -1 - np.flatnonzero(miss)
    return codes


# --------------------------------------------------------------------------------------
# the network
# --------------------------------------------------------------------------------------


def _build(n_num: int, n_miss: int, cards: list[int], p: dict):
    import torch
    from torch import nn

    # TabM's embedding front end; its one-block backbone is built only to take the pieces.
    tabm_p = {
        "k": 1,
        "n_blocks": 1,
        "d_block": 8,
        "dropout": 0.0,
        "arch": "mlp",
        "d_num_emb": p["d_num_emb"],
        "n_frequencies": p["n_frequencies"],
        "frequency_scale": p["frequency_scale"],
        "d_cat_emb": p["d_cat_emb"],
    }
    tabm = _build_net(n_num, n_miss, cards, tabm_p)
    drop = float(p["dropout"])

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.num_emb = tabm.num_emb
            self.cat_emb = tabm.cat_emb
            d = n_num * int(p["d_num_emb"]) + n_miss + len(cards) * int(p["d_cat_emb"])
            d = max(d, 1)
            layers: list = []
            for _ in range(int(p["n_blocks"])):
                layers += [nn.Linear(d, int(p["d_block"])), nn.ReLU(), nn.Dropout(drop)]
                d = int(p["d_block"])
            layers.append(nn.Linear(d, int(p["d_emb"])))
            self.enc = nn.Sequential(*layers)
            self.res = None
            if bool(p["residual_head"]):
                self.res = nn.Linear(int(p["d_emb"]), 1)
                nn.init.zeros_(self.res.weight)
                nn.init.zeros_(self.res.bias)

        def forward(self, xn, xm, xc):  # -> (B, d_emb)
            parts = []
            if self.num_emb is not None:
                parts.append(self.num_emb(xn))
            if xm.shape[1]:
                parts.append(xm)
            for j, emb in enumerate(self.cat_emb):
                parts.append(emb(xc[:, j]))
            x = torch.cat(parts, dim=1) if parts else xn.new_zeros(xn.shape[0], 1)
            return self.enc(x)

    return Net()


# --------------------------------------------------------------------------------------
# the wrapper
# --------------------------------------------------------------------------------------


class ModernNCAModel:
    """Same interface as :class:`prc.models.tabm.TabMModel`. After ``fit`` the model holds the
    candidate bank (the training rows' embeddings, standardised targets and log-weights) on
    its device; ``predict`` retrieves from it."""

    def __init__(
        self,
        cfg: ModelConfig,
        seed: int,
        matrices=None,
        device: str | None = None,
        n_threads: int | None = None,
    ) -> None:
        self.cfg = cfg
        self.seed = int(seed)
        self.p = {**DEFAULTS, **cfg.params}
        self.device = device or cfg.device
        self.n_threads = n_threads
        self.best_iteration: int | None = None
        self.best_epoch: int | None = None
        self.history: list[dict] = []
        self.timings: dict[str, float] = {}
        self.prep: _Prep | None = None
        self.net = None
        self.bank = None  # (emb (N, d) fp32, y (N,), logw (N,))
        self.y_mu = 0.0
        self.y_sd = 1.0
        self.peak_gpu_gb: float | None = None

    def _t(self, name: str, seconds: float) -> None:
        self.timings[name] = self.timings.get(name, 0.0) + seconds

    def _tensors(self, X: pd.DataFrame, dev):
        import torch

        assert self.prep is not None
        z, m, c = self.prep.transform(X)
        return (
            torch.from_numpy(z).to(dev),
            torch.from_numpy(m).to(dev),
            torch.from_numpy(c).to(dev),
        )

    def _autocast(self, dev):
        import contextlib

        import torch

        if dev.type == "cuda" and bool(self.p["amp"]):
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _enc(self, T, rows, dev):
        xn, xm, xc = T
        with self._autocast(dev):
            return self.net(xn[rows], xm[rows], xc[rows]).float()

    def _embed_all(self, T, dev):
        """Every row of ``T`` embedded in eval mode, no gradient, fp32."""
        import torch

        n = T[0].shape[0]
        cc = int(self.p["cand_chunk"])
        self.net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, n, cc):
                out.append(self._enc(T, torch.arange(i, min(i + cc, n), device=dev), dev))
        d = int(self.p["d_emb"])
        return torch.cat(out) if out else torch.zeros(0, d, device=dev)

    def _build_bank(self, T, y_t, logw_t, elig, dev) -> None:
        emb = self._embed_all(T, dev)
        self.bank = (emb[elig].contiguous(), y_t[elig].contiguous(), logw_t[elig].contiguous())

    def _retrieve(self, Q, dev) -> np.ndarray:
        """Standardised predictions for query embeddings ``Q`` (n, d) against the bank."""
        import torch

        assert self.bank is not None
        E, yb, lwb = self.bank
        tau = float(self.p["temperature"])
        qb, bc, k = int(self.p["eval_batch_size"]), int(self.p["bank_chunk"]), int(self.p["top_k"])
        out = []
        with torch.no_grad():
            for i in range(0, Q.shape[0], qb):
                q = Q[i : i + qb]
                if E.shape[0] == 0:
                    out.append(torch.zeros(q.shape[0], device=dev))
                    continue
                idx = _topk(q, E, k, bc)
                dist = (q[:, None, :] - E[idx]).pow(2).sum(-1).sqrt()
                pred = _nca(dist, yb[idx], lwb[idx], tau)
                if self.net.res is not None:
                    pred = pred + self.net.res(q).squeeze(-1)
                out.append(pred)
        pred = torch.cat(out) if out else torch.zeros(0)
        return pred.cpu().numpy().astype(np.float64)

    def _predict_t(self, T, dev) -> np.ndarray:
        return self._retrieve(self._embed_all(T, dev), dev)

    def fit(self, X, y, X_val=None, y_val=None, w=None, w_val=None, key=None):
        import torch
        from torch.utils.checkpoint import checkpoint

        if self.n_threads is not None:
            torch.set_num_threads(int(self.n_threads))
        dev = torch.device(self.device)
        torch.manual_seed(self.seed)
        gen = torch.Generator().manual_seed(self.seed)
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)

        t0 = time.perf_counter()
        self.prep = _Prep(self.p, self.seed).fit(X)
        groups = _groups(X, str(self.p["exclude_same_col"] or ""))
        y = np.asarray(y, dtype=np.float64)
        ww = None if w is None or not np.sum(w) > 0 else np.asarray(w, dtype=np.float64)
        self.y_mu = float(np.average(y, weights=ww))
        self.y_sd = float(np.sqrt(np.average((y - self.y_mu) ** 2, weights=ww))) or 1.0
        Ttr = self._tensors(X, dev)
        ytr = torch.from_numpy(((y - self.y_mu) / self.y_sd).astype(np.float32)).to(dev)
        wnp = np.ones(len(y)) if w is None else np.asarray(w, dtype=np.float64)
        wtr = None if w is None else torch.from_numpy(wnp.astype(np.float32)).to(dev)
        with np.errstate(divide="ignore"):
            logw = torch.from_numpy(np.log(np.clip(wnp, 0.0, None)).astype(np.float32)).to(dev)
        elig_np = np.flatnonzero(wnp > 0)  # weight-0 rows are never candidates
        if not elig_np.size:
            raise ValueError("modernnca: no training row has a positive weight")
        elig = torch.from_numpy(elig_np).to(dev)
        g_t = None if groups is None else torch.from_numpy(groups).to(dev)
        Tva = yva = None
        if X_val is not None:
            Tva = self._tensors(X_val, dev)
            yva = np.asarray(y_val, dtype=np.float64)
        self._t("matrix", time.perf_counter() - t0)

        self.net = _build(
            len(self.prep.num_cols), len(self.prep.miss_cols), self.prep.cardinalities(), self.p
        ).to(dev)
        opt = torch.optim.AdamW(
            self.net.parameters(),
            lr=float(self.p["learning_rate"]),
            weight_decay=float(self.p["weight_decay"]),
        )
        n = int(ytr.shape[0])
        bs = min(int(self.p["batch_size"]), n)
        steps_per_epoch = math.ceil(n / bs)
        cap = int(self.cfg.num_boost_round)
        if X_val is not None:
            cap = min(cap, int(self.p["max_epochs"]) * steps_per_epoch)
        patience = int(self.cfg.early_stopping_rounds)
        verbose = int(self.p["verbose"])
        tau = float(self.p["temperature"])
        n_el = int(elig_np.size)
        m_cand = min(int(self.p["n_cand"]), n_el)
        cc = int(self.p["cand_chunk"])
        cand_grad = bool(self.p["cand_grad"])

        def enc_rows(rows):
            return self._enc(Ttr, rows, dev)

        def embed_cands(cand):
            parts = []
            for j in range(0, cand.shape[0], cc):
                s = cand[j : j + cc]
                if not cand_grad:
                    with torch.no_grad():
                        parts.append(enc_rows(s))
                else:
                    parts.append(checkpoint(enc_rows, s, use_reentrant=False))
            return torch.cat(parts)

        # the rolling candidate permutation: each step takes the next m_cand eligible rows
        pool = {"perm": None, "ptr": 0}

        def next_pool():
            if m_cand >= n_el:
                return elig
            if pool["perm"] is None or pool["ptr"] + m_cand > n_el:
                pool["perm"] = elig[torch.randperm(n_el, generator=gen).to(dev)]
                pool["ptr"] = 0
            c = pool["perm"][pool["ptr"] : pool["ptr"] + m_cand]
            pool["ptr"] += m_cand
            return c

        t0 = time.perf_counter()
        step, epoch = 0, 0
        best, best_state, since = math.inf, None, 0
        while step < cap:
            self.net.train()
            te = time.perf_counter()
            perm = torch.randperm(n, generator=gen).to(dev)
            loss_sum, n_b = 0.0, 0
            for i in range(0, n, bs):
                if step >= cap:
                    break
                b = perm[i : i + bs]
                cand = next_pool()
                qe = enc_rows(b)
                ce = embed_cands(cand)
                excl = b[:, None] == cand[None, :]  # the query itself: no self-label leak
                if g_t is not None:
                    excl |= g_t[b][:, None] == g_t[cand][None, :]
                dist = _sq_dist(qe, ce).clamp_min(1e-12).sqrt()
                pred = _nca(dist, ytr[cand][None, :], logw[cand][None, :], tau, excl)
                if self.net.res is not None:
                    pred = pred + self.net.res(qe).squeeze(-1)
                se = (pred - ytr[b]).pow(2)
                if wtr is None:
                    loss = se.mean()
                else:
                    wb = wtr[b]
                    loss = (wb * se).sum() / wb.sum().clamp_min(1e-12)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                step += 1
                loss_sum += float(loss.detach())
                n_b += 1
            epoch += 1
            rec = {
                "epoch": epoch,
                "step": step,
                "train_loss": loss_sum / max(n_b, 1),
                "train_s": time.perf_counter() - te,
            }
            if Tva is not None:
                tv = time.perf_counter()
                self._build_bank(Ttr, ytr, logw, elig, dev)
                p_va = self._predict_t(Tva, dev) * self.y_sd + self.y_mu
                rmse = float(np.sqrt(np.mean((p_va - yva) ** 2)))
                rec["val_rmse"] = rmse
                rec["val_s"] = time.perf_counter() - tv
                if rmse < best:
                    best, since = rmse, 0
                    best_state = copy.deepcopy(self.net.state_dict())
                    self.best_iteration, self.best_epoch = step - 1, epoch
                else:
                    since += 1
            self.history.append(rec)
            if verbose:
                log.info(
                    "modernnca seed %d epoch %d step %d loss %.5f train %.1fs (%.3f s/step)%s",
                    self.seed,
                    epoch,
                    step,
                    rec["train_loss"],
                    rec["train_s"],
                    rec["train_s"] / max(n_b, 1),
                    f" val_rmse {rec['val_rmse']:.5f} ({rec['val_s']:.1f}s)"
                    if "val_rmse" in rec
                    else "",
                )
            if Tva is not None and since >= patience:
                break
        if Tva is None:
            self.best_iteration, self.best_epoch = step - 1, epoch
        elif best_state is not None:
            self.net.load_state_dict(best_state)
        self._t("train", time.perf_counter() - t0)
        t0 = time.perf_counter()
        self._build_bank(Ttr, ytr, logw, elig, dev)
        self._t("bank", time.perf_counter() - t0)
        # plain assignments, not ``del``: the closures above share these cells
        Ttr = ytr = wtr = Tva = logw = elig = g_t = pool["perm"] = None
        if dev.type == "cuda":
            self.peak_gpu_gb = torch.cuda.max_memory_allocated(dev) / 2**30
            torch.cuda.empty_cache()
            if verbose:
                log.info(
                    "modernnca seed %d fit done: %d candidates, peak GPU %.2f GB",
                    self.seed,
                    int(self.bank[0].shape[0]),
                    self.peak_gpu_gb,
                )
        return self

    def predict(self, X, key=None):
        import torch

        assert self.net is not None and self.bank is not None, "fit first"
        dev = torch.device(self.device)
        t0 = time.perf_counter()
        T = self._tensors(X, dev)
        out = self._predict_t(T, dev) * self.y_sd + self.y_mu
        del T
        self._t("predict", time.perf_counter() - t0)
        return out

    def importance(self):
        return None
