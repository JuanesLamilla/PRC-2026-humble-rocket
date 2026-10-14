"""``model.kind: setenc`` — a set-attention encoder over nearby movements (plan N1, 2026-09-22).

Builds on :mod:`prc.models.tabm`.

Per departure the network sees

* its own feature row (the same matrix TabM gets), through TabM's preprocessing
  (:class:`prc.models.tabm._Prep`: quantile transform, missing indicators, capped vocabularies)
  and TabM's embeddings (periodic PLR numerics, categorical embeddings), then a shared
  two-layer MLP → the row embedding ``h``;
* its **context set** (:mod:`prc.features.context_sets`, up to ``ctx_k`` other movements at
  the same airport and island around its takeoff), gathered by the row's ``ctx_key``
  column. Each element → a per-element MLP (width ``d_set``) over its numeric attributes
  plus embeddings of phase / wake / type. A learned "null" element is always present, so an
  empty set is well defined; padded slots are masked out of the attention.
* ``n_attn`` pre-norm cross-attention layers (``n_heads`` heads) with the row embedding as
  the single query, each followed by a feed-forward layer → the set summary ``s``.

``[h, s]`` goes to a BatchEnsemble head (TabM's ``LinearBE``, ``k`` members, the first ``r``
initialised to random ±1), and the prediction is the mean over members. Loss, training loop,
early stopping and the GBDT vocabulary (``best_iteration`` = steps, ``num_boost_round`` = step
cap, ``early_stopping_rounds`` = patience in epochs) are exactly TabM's, so the runner's
``test_fit: full`` rule gives the full fit the fold models' median number of epochs.

The ``ctx_key`` column is removed from the row matrix before preprocessing: it is a handle,
not a feature.
"""

from __future__ import annotations

import copy
import logging
import math
import time

import numpy as np
import pandas as pd

from prc.config import ModelConfig
from prc.features.context_sets import (
    DEFAULT_PARAMS as SET_DEFAULTS,
)
from prc.features.context_sets import (
    FLAG_SAME_GROUP,
    FLAG_SAME_OPERATOR,
    FLAG_SAME_RUNWAY,
    KEY_COLUMN,
    MAX_TYPES,
    ContextSets,
    load_sets,
)
from prc.models.tabm import _build_net, _Prep

log = logging.getLogger("prc.setenc")

DEFAULTS: dict = {
    # own row (TabM front end + a shared MLP)
    "d_row": 512,
    "d_num_emb": 16,
    "n_frequencies": 48,
    "frequency_scale": 0.1,
    "d_cat_emb": 16,
    "cat_max_levels": 1000,
    "cat_min_count": 20,
    "n_quantiles": 1024,
    "qt_subsample": 200_000,
    # the set
    "ctx_k": SET_DEFAULTS["k"],
    "ctx_mvt_before_min": SET_DEFAULTS["mvt_before_min"],
    "ctx_mvt_after_min": SET_DEFAULTS["mvt_after_min"],
    "ctx_aobt_before_min": SET_DEFAULTS["aobt_before_min"],
    "d_set": 64,
    "n_attn": 2,
    "n_heads": 4,
    "d_ctx_cat": 8,
    # head
    "k": 16,
    "n_head_blocks": 2,
    "d_head": 256,
    "dropout": 0.1,
    # training
    "learning_rate": 2e-3,
    "weight_decay": 3e-4,
    "batch_size": 4096,
    "eval_batch_size": 16384,
    "max_epochs": 40,
    "amp": True,
    "verbose": 1,
    "slog_patterns": (),  # as prc.models.tabm: signed log on these own-row numerics
    "slog_qt": True,
}

#: Numeric attributes per element (see ``_Ctx.gather``).
N_ELEM_NUM = 10


def set_params(p: dict) -> dict:
    """The :func:`prc.features.context_sets.load_sets` parameters inside the model params."""
    return {
        "k": int(p["ctx_k"]),
        "mvt_before_min": int(p["ctx_mvt_before_min"]),
        "mvt_after_min": int(p["ctx_mvt_after_min"]),
        "aobt_before_min": int(p["ctx_aobt_before_min"]),
    }


def _build(n_num: int, n_miss: int, cards: list[int], p: dict):
    import torch
    from torch import nn

    # TabM's pieces: its embedding front end and its BatchEnsemble layer class. A one-block
    # TabM is built only to take them; its backbone is discarded.
    tabm_p = {
        "k": int(p["k"]),
        "n_blocks": 1,
        "d_block": 8,
        "dropout": 0.0,
        "d_num_emb": p["d_num_emb"],
        "n_frequencies": p["n_frequencies"],
        "frequency_scale": p["frequency_scale"],
        "d_cat_emb": p["d_cat_emb"],
    }
    tabm = _build_net(n_num, n_miss, cards, tabm_p)
    LinearBE = type(tabm.blocks[0])

    d_set, d_row = int(p["d_set"]), int(p["d_row"])
    drop = float(p["dropout"])

    class Attn(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ln_q = nn.LayerNorm(d_set)
            self.ln_kv = nn.LayerNorm(d_set)
            self.mha = nn.MultiheadAttention(
                d_set, int(p["n_heads"]), dropout=0.0, batch_first=True
            )
            self.ln_f = nn.LayerNorm(d_set)
            self.ff = nn.Sequential(
                nn.Linear(d_set, 2 * d_set), nn.ReLU(), nn.Linear(2 * d_set, d_set)
            )

        def forward(self, q, e, pad):  # q (B,1,d), e (B,K+1,d), pad (B,K+1) True = ignore
            kv = self.ln_kv(e)
            a, _ = self.mha(self.ln_q(q), kv, kv, key_padding_mask=pad, need_weights=False)
            q = q + a
            return q + self.ff(self.ln_f(q))

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.num_emb = tabm.num_emb
            self.cat_emb = tabm.cat_emb
            d_in = n_num * int(p["d_num_emb"]) + n_miss + len(cards) * int(p["d_cat_emb"])
            d_in = max(d_in, 1)
            self.row = nn.Sequential(
                nn.Linear(d_in, d_row),
                nn.ReLU(),
                nn.Dropout(drop),
                nn.Linear(d_row, d_row),
                nn.ReLU(),
                nn.Dropout(drop),
            )
            self.q = nn.Linear(d_row, d_set)
            dc = int(p["d_ctx_cat"])
            self.e_phase = nn.Embedding(2, dc)
            self.e_wake = nn.Embedding(8, dc)
            self.e_type = nn.Embedding(MAX_TYPES + 2, dc)
            self.elem = nn.Sequential(
                nn.Linear(N_ELEM_NUM + 3 * dc, d_set), nn.ReLU(), nn.Linear(d_set, d_set)
            )
            self.null = nn.Parameter(torch.zeros(1, 1, d_set))
            self.attn = nn.ModuleList(Attn() for _ in range(int(p["n_attn"])))
            self.ln_s = nn.LayerNorm(d_set)
            k, dh = int(p["k"]), int(p["d_head"])
            blocks = []
            for i in range(int(p["n_head_blocks"])):
                blocks.append(LinearBE(d_row + d_set if i == 0 else dh, dh, k, i == 0))
            self.blocks = nn.ModuleList(blocks)
            self.drop = nn.Dropout(drop)
            bound = 1.0 / math.sqrt(dh)
            self.head_w = nn.Parameter(torch.empty(k, dh).uniform_(-bound, bound))
            self.head_b = nn.Parameter(torch.zeros(k))

        def row_input(self, xn, xm, xc):
            parts = []
            if self.num_emb is not None:
                parts.append(self.num_emb(xn))
            if xm.shape[1]:
                parts.append(xm)
            for j, emb in enumerate(self.cat_emb):
                parts.append(emb(xc[:, j]))
            return torch.cat(parts, dim=1) if parts else xn.new_zeros(xn.shape[0], 1)

        def forward(self, xn, xm, xc, en, ec, valid):
            """``en`` (B,K,N_ELEM_NUM) float, ``ec`` (B,K,3) long, ``valid`` (B,K) bool.
            Padded slots are zeroed here as well as masked, so their content never matters."""
            h = self.row(self.row_input(xn, xm, xc))
            vm = valid[..., None]
            en = torch.where(vm, en, torch.zeros_like(en))
            ec = torch.where(vm, ec, torch.zeros_like(ec))
            e = torch.cat(
                [en, self.e_phase(ec[..., 0]), self.e_wake(ec[..., 1]), self.e_type(ec[..., 2])],
                dim=-1,
            )
            e = self.elem(e)
            B = e.shape[0]
            e = torch.cat([self.null.expand(B, 1, -1).to(e.dtype), e], dim=1)
            pad = torch.cat([valid.new_zeros(B, 1), ~valid], dim=1)
            q = self.q(h)[:, None, :]
            for layer in self.attn:
                q = layer(q, e, pad)
            s = self.ln_s(q[:, 0, :])
            x = torch.cat([h, s.to(h.dtype)], dim=1)
            for blk in self.blocks:
                x = self.drop(torch.relu(blk(x)))
            return (x * self.head_w).sum(-1) + self.head_b

    return Net()


class _Ctx:
    """One fit's context tensors on the device: per-row slots, plus the per-movement tables
    the slots index into. ``gather(b)`` builds a batch's element inputs."""

    def __init__(self, sets: ContextSets, keys: np.ndarray, dev) -> None:
        import torch

        pos = sets.rows(keys)
        n_miss = int((pos < 0).sum())
        if n_miss:
            log.warning("setenc: %d of %d rows have no context set (unknown key)", n_miss, len(pos))
        safe = np.where(pos >= 0, pos, 0)
        idx = np.asarray(sets.idx[safe])
        idx[pos < 0] = -1
        rel = np.asarray(sets.rel[safe])
        flags = np.asarray(sets.flags[safe])
        self.idx = torch.from_numpy(idx.astype(np.int32)).to(dev)
        self.rel = torch.from_numpy(rel).to(dev)
        self.flags = torch.from_numpy(flags).to(dev)
        # per-movement tables, shared by every fit of the process via _TABLES
        self.mv_num, self.mv_cat = _tables(sets, dev)

    def gather(self, b):
        import torch

        I = self.idx[b].long()
        valid = I >= 0
        J = I.clamp_min(0)
        rel = self.rel[b].float()
        mv = self.mv_num[J].float()  # (B,K,2) proxy, taxi-in (min)
        fl = self.flags[b].long()
        dt_m = torch.nan_to_num(rel[..., 0], nan=0.0) / 30.0
        dt_a = rel[..., 1]
        dt_a_miss = torch.isnan(dt_a)
        prox, tin = mv[..., 0], mv[..., 1]
        en = torch.stack(
            [
                dt_m,
                torch.where(dt_a_miss, 0.0, dt_a / 30.0),
                dt_a_miss.float(),
                torch.where(torch.isnan(prox), 0.0, prox / 10.0),
                torch.isnan(prox).float(),
                torch.where(torch.isnan(tin), 0.0, tin / 10.0),
                torch.isnan(tin).float(),
                ((fl & FLAG_SAME_RUNWAY) > 0).float(),
                ((fl & FLAG_SAME_GROUP) > 0).float(),
                ((fl & FLAG_SAME_OPERATOR) > 0).float(),
            ],
            dim=-1,
        )
        ec = self.mv_cat[J]
        return en, ec, valid


_TABLES: dict = {}


def _tables(sets: ContextSets, dev):
    import torch

    key = (id(sets.mv_num), str(dev))
    if key not in _TABLES:
        _TABLES.clear()
        mv_num = torch.from_numpy(np.array(sets.mv_num, dtype=np.float16)).to(dev)
        cat = np.asarray(sets.mv_cat, dtype=np.int64)
        cat[:, 1] = np.clip(cat[:, 1], 0, 7)
        _TABLES[key] = (mv_num, torch.from_numpy(cat).to(dev))
    return _TABLES[key]


class SetEncModel:
    """Same interface as :class:`prc.models.tabm.TabMModel`. ``X`` must carry ``ctx_key``
    (the ``context_sets`` feature block).

    Subclasses (``prc.models.seqenc``) swap the network through :meth:`_make_net`, the
    parameter defaults through ``defaults`` and the log prefix through ``name``; everything
    else (context gathering, preprocessing, the training loop) is shared."""

    defaults: dict = DEFAULTS
    name: str = "setenc"

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
        self.p = {**self.defaults, **cfg.params}
        self.device = device or cfg.device
        self.n_threads = n_threads
        self.best_iteration: int | None = None
        self.best_epoch: int | None = None
        self.history: list[dict] = []
        self.timings: dict[str, float] = {}
        self.prep: _Prep | None = None
        self.net = None
        self.sets: ContextSets | None = None
        self.y_mu = 0.0
        self.y_sd = 1.0

    def _t(self, name: str, seconds: float) -> None:
        self.timings[name] = self.timings.get(name, 0.0) + seconds

    def _make_net(self, n_num: int, n_miss: int, cards: list[int]):
        return _build(n_num, n_miss, cards, self.p)

    @staticmethod
    def _split(X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        if KEY_COLUMN not in X.columns:
            raise ValueError(
                f"setenc / seqenc: X has no {KEY_COLUMN!r} column; add the `context_sets` "
                "feature block"
            )
        keys = X[KEY_COLUMN].to_numpy(dtype=np.float64, na_value=np.nan)
        return X.drop(columns=[KEY_COLUMN]), keys

    def _sets(self) -> ContextSets:
        if self.sets is None:
            self.sets = load_sets(set_params(self.p))
        return self.sets

    def _tensors(self, X: pd.DataFrame, dev):
        import torch

        assert self.prep is not None
        Xr, keys = self._split(X)
        z, m, c = self.prep.transform(Xr)
        T = (
            torch.from_numpy(z).to(dev),
            torch.from_numpy(m).to(dev),
            torch.from_numpy(c).to(dev),
        )
        return T, _Ctx(self._sets(), keys, dev)

    def _autocast(self, dev):
        import contextlib

        import torch

        if dev.type == "cuda" and bool(self.p["amp"]):
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _forward(self, T, C: _Ctx, b):
        en, ec, valid = C.gather(b)
        xn, xm, xc = T
        return self.net(xn[b], xm[b], xc[b], en, ec, valid)

    def _predict_t(self, T, C: _Ctx, dev) -> np.ndarray:
        import torch

        n = T[0].shape[0]
        bs = int(self.p["eval_batch_size"])
        out = []
        self.net.eval()
        with torch.no_grad(), self._autocast(dev):
            for i in range(0, n, bs):
                b = torch.arange(i, min(i + bs, n), device=dev)
                out.append(self._forward(T, C, b).float())
        pred = torch.cat(out).mean(dim=1) if out else torch.zeros(0)
        return pred.cpu().numpy().astype(np.float64)

    def fit(self, X, y, X_val=None, y_val=None, w=None, w_val=None, key=None):
        import torch

        if self.n_threads is not None:
            torch.set_num_threads(int(self.n_threads))
        dev = torch.device(self.device)
        torch.manual_seed(self.seed)
        gen = torch.Generator().manual_seed(self.seed)

        t0 = time.perf_counter()
        self._sets()
        self._t("sets", time.perf_counter() - t0)
        t0 = time.perf_counter()
        self.prep = _Prep(self.p, self.seed).fit(self._split(X)[0])
        y = np.asarray(y, dtype=np.float64)
        ww = None if w is None or not np.sum(w) > 0 else np.asarray(w, dtype=np.float64)
        self.y_mu = float(np.average(y, weights=ww))
        self.y_sd = float(np.sqrt(np.average((y - self.y_mu) ** 2, weights=ww))) or 1.0
        Ttr, Ctr = self._tensors(X, dev)
        ytr = torch.from_numpy(((y - self.y_mu) / self.y_sd).astype(np.float32)).to(dev)
        wtr = None
        if w is not None:
            wtr = torch.from_numpy(np.asarray(w, dtype=np.float32)).to(dev)
        Tva = Cva = yva = None
        if X_val is not None:
            Tva, Cva = self._tensors(X_val, dev)
            yva = np.asarray(y_val, dtype=np.float64)
        self._t("matrix", time.perf_counter() - t0)

        self.net = self._make_net(
            len(self.prep.num_cols), len(self.prep.miss_cols), self.prep.cardinalities()
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
                with self._autocast(dev):
                    pred = self._forward(Ttr, Ctr, b).float()
                se = (pred - ytr[b][:, None]).pow(2).mean(dim=1)
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
                p_va = self._predict_t(Tva, Cva, dev) * self.y_sd + self.y_mu
                rmse = float(np.sqrt(np.mean((p_va - yva) ** 2)))
                rec["val_rmse"] = rmse
                if rmse < best:
                    best, since = rmse, 0
                    best_state = copy.deepcopy(self.net.state_dict())
                    self.best_iteration, self.best_epoch = step - 1, epoch
                else:
                    since += 1
            self.history.append(rec)
            if verbose:
                log.info(
                    "%s seed %d epoch %d step %d loss %.5f train %.1fs (%.3f s/step)%s",
                    self.name,
                    self.seed,
                    epoch,
                    step,
                    rec["train_loss"],
                    rec["train_s"],
                    rec["train_s"] / max(n_b, 1),
                    f" val_rmse {rec['val_rmse']:.5f}" if "val_rmse" in rec else "",
                )
            if Tva is not None and since >= patience:
                break
        if Tva is None:
            self.best_iteration, self.best_epoch = step - 1, epoch
        elif best_state is not None:
            self.net.load_state_dict(best_state)
        self._t("train", time.perf_counter() - t0)
        del Ttr, Ctr, ytr, wtr, Tva, Cva
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        return self

    def predict(self, X, key=None):
        import torch

        assert self.net is not None, "fit first"
        dev = torch.device(self.device)
        t0 = time.perf_counter()
        T, C = self._tensors(X, dev)
        out = self._predict_t(T, C, dev) * self.y_sd + self.y_mu
        del T, C
        self._t("predict", time.perf_counter() - t0)
        return out

    def importance(self):
        return None
