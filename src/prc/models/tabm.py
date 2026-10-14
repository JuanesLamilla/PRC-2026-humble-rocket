"""``model.kind: tabm`` — a TabM neural regressor as a diverse blend member (2026-09-22).

Evidence and intent: internal note 2026-09-22_literature-update item 2 (TabReD,
arXiv 2406.19380; TabM, arXiv 2410.24210). In both earlier PRC editions NN members took zero
blend weight, so this is meant to be *different* from the GBDT members, not better than them.

Implemented here in plain PyTorch rather than through the ``tabm`` / ``rtdl_num_embeddings``
packages: the whole model is ~150 lines, and the cluster then needs nothing beyond ``torch``.

The model (TabM with BatchEnsemble layers, Gorishniy et al. 2025, §3):

* **numerics** — quantile-transformed to a normal (fitted on the fold's training rows only);
  NaN / inf become 0 after the transform plus a 0/1 missing indicator for every column that
  has a missing value in training. Columns that are constant or all-missing in training are
  dropped. Each numeric then gets a **periodic (PLR) embedding**: ``cos/sin(2π c x)`` with
  learned frequencies ``c``, a per-feature linear layer and a ReLU.
* **categoricals** (``category`` dtype) — a learned embedding per column. The vocabulary is fitted
  on training rows: at most ``cat_max_levels`` levels with at least ``cat_min_count`` rows each;
  everything else is OTHER, missing is its own index, unseen levels at prediction are OTHER.
* **backbone** — ``n_blocks`` BatchEnsemble linear layers (a shared weight matrix, per-member
  input / output scalings ``r`` / ``s`` and per-member biases; the first ``r`` initialised with
  random ±1 so the members start apart), ReLU, dropout; ``k`` per-member linear heads. The
  prediction is the mean over the ``k`` members; the loss is the mean of the members' own
  squared errors (so each member is trained as an independent model, as in the paper).
* **training** — AdamW at a constant learning rate, MSE on the standardised (already
  sqrt-transformed, by the runner) target, weighted by the per-row ``w`` (the corruption head's
  ``1 - Σ p``): ``Σ w·se / Σ w`` per batch. bf16 autocast on CUDA when ``amp`` is on.

Variants (2026-09-22; all inert at defaults):
``arch: mlp`` (plain ``nn.Linear`` blocks, ``k: 1``: the MLP-PLR of TabReD, ensembled over the
run's seeds), ``num_emb: ple`` (piecewise-linear embeddings on ``n_bins`` quantile bins of the
transformed training values), ``slog_patterns`` (``sign(x)·log1p(|x|/60)`` on the matching
numerics, then the quantile step, or with ``slog_qt: false`` a standardisation instead).

How the runner's GBDT vocabulary maps onto it:

``best_iteration``         the **optimizer step** count at the best epoch, minus one (0-based,
                           like a tree index). Steps, not epochs, so that
                           ``prc.eval.run._full_fit_rounds``' rule ``median(best+1) · n_all /
                           mean(n_fold)`` gives the full fit the fold models' median number of
                           *epochs* over its (larger) training set — a GBDT's rounds scale with
                           the rows, an NN's steps do too.
``num_boost_round``        a cap on optimizer steps. On the fold path training ends at
                           ``min(num_boost_round, max_epochs · steps_per_epoch)``; on the full
                           path (``X_val=None``) the runner sets it to the exact step count.
``early_stopping_rounds``  the patience, in **epochs**: stop after this many epochs without a
                           lower validation RMSE, and restore the best epoch's weights.
                           The RMSE is in transformed (sqrt) space and unweighted, as the
                           GBDT members' default early stopping is.

Determinism: every random draw (weight init, dropout, batch order, the quantile subsample) is
seeded from the member seed. On CPU a seed reproduces bit for bit; on CUDA it is reproducible
to float noise (cuBLAS / embedding-backward reductions are not bitwise deterministic).
"""

from __future__ import annotations

import copy
import fnmatch
import logging
import math
import time

import numpy as np
import pandas as pd

from prc.config import ModelConfig

log = logging.getLogger("prc.tabm")

DEFAULTS: dict = {
    "k": 32,  # ensemble members
    "n_blocks": 3,
    "d_block": 512,
    "dropout": 0.1,
    "learning_rate": 2e-3,
    "weight_decay": 3e-4,
    "batch_size": 4096,
    "eval_batch_size": 16384,
    "max_epochs": 40,
    "d_num_emb": 16,  # PLR embedding width per numeric feature
    "n_frequencies": 48,
    "frequency_scale": 0.1,  # sigma of the initial frequencies (untuned)
    "d_cat_emb": 16,
    "cat_max_levels": 1000,
    "cat_min_count": 20,
    "n_quantiles": 1024,
    "qt_subsample": 200_000,
    "amp": True,  # bf16 autocast on CUDA; ignored on CPU
    "verbose": 1,  # 0 silent, 1 one log line per epoch
    # Large-batch knobs (2026-09-22). They change
    # results, so they belong in a new YAML, never a runtime switch. At their defaults the
    # optimizer runs at exactly ``learning_rate`` from the first step, as before.
    "lr_scale": "none",  # none | sqrt | linear: lr = learning_rate * f(batch_size / lr_base_batch)
    "lr_base_batch": 4096,  # the batch size ``learning_rate`` was set for
    "warmup_epochs": 0.0,  # linear warmup from lr/W to lr over this many epochs (0 = none)
    # Variants (2026-09-22). Each is inert at its
    # default: no extra random draw, no extra arithmetic, the pre-variant network and inputs.
    "arch": "tabm",  # tabm (BatchEnsemble) | mlp (a plain MLP, needs k: 1; MLP-PLR per seed)
    "num_emb": "periodic",  # periodic (PLR) | ple (piecewise-linear, quantile bins)
    "n_bins": 48,  # ple only: quantile bins per numeric feature (fewer when values repeat)
    "slog_patterns": (),  # fnmatch globs: sign(x)·log1p(|x|/60) on these numerics
    "slog_qt": True,  # True: the quantile step follows the slog; False: slog + standardise only
}

#: Threads for the quantile transform in :meth:`_Prep.transform` (a performance switch,
#: ``runtime.nn_prep_threads`` / ``PRC_NN_PREP_THREADS``, set by the runner and by the spawn
#: fit workers). 0 = one ``QuantileTransformer.transform`` call over the whole matrix, the
#: pre-2026-09-22 path. >0 = the same transform over 32 k-row chunks in a thread pool. Every
#: step of that transform (two ``np.interp``, the bound clamps, ``norm.ppf``, the clip) is
#: elementwise, so the chunked result is bit-identical;
#: ``np.interp`` releases the GIL, and chunking alone is ~1.6x faster (cache locality).
PREP_THREADS: int = 0
_PREP_CHUNK = 32_768


def _slog(a: np.ndarray) -> np.ndarray:
    """Signed log of seconds: ``sign(x) · log1p(|x| / 60)`` (NaN stays NaN)."""
    return np.sign(a) * np.log1p(np.abs(a) / 60.0)


def _ple_edges(v: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile bin edges of one feature's finite training values, duplicates removed."""
    e = np.unique(np.quantile(v, np.linspace(0.0, 1.0, n_bins + 1))) if v.size else np.zeros(1)
    return e if e.size >= 2 else np.array([e[0], e[0] + 1.0])


def _lr_factor(p: dict) -> float | None:
    """The batch-size LR multiplier, or None for ``lr_scale: none`` (no arithmetic at all)."""
    mode = str(p["lr_scale"])
    if mode == "none":
        return None
    r = int(p["batch_size"]) / int(p["lr_base_batch"])
    if mode == "sqrt":
        return math.sqrt(r)
    if mode == "linear":
        return r
    raise ValueError(f"tabm: lr_scale must be none / sqrt / linear, got {mode!r}")


# --------------------------------------------------------------------------------------
# preprocessing (numpy / pandas; fitted on training rows only)
# --------------------------------------------------------------------------------------


class _Prep:
    """Quantile transform + missing indicators for numerics, a capped vocabulary for cats."""

    def __init__(self, p: dict, seed: int) -> None:
        self.p = p
        self.seed = seed
        self.num_cols: list[str] = []
        self.miss_cols: list[str] = []
        self.cat_cols: list[str] = []
        self.vocab: dict[str, dict] = {}
        self.qt = None
        # variants (inert at the defaults; ``p.get`` because setenc passes its own DEFAULTS)
        self.slog_idx: list[int] = []
        self.slog_mu: np.ndarray | None = None  # set only when ``slog_qt`` is False
        self.slog_sd: np.ndarray | None = None
        self.bins: list[np.ndarray] | None = None  # PLE bin edges in the transformed space

    @staticmethod
    def _num(X: pd.DataFrame, cols: list[str]) -> np.ndarray:
        if not cols:
            return np.zeros((len(X), 0), dtype=np.float64)
        a = X[cols].to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
        a[~np.isfinite(a)] = np.nan
        return a

    def fit(self, X: pd.DataFrame) -> _Prep:
        from sklearn.preprocessing import QuantileTransformer

        cats, nums = [], []
        for c in X.columns:
            dt = X[c].dtype
            if isinstance(dt, pd.CategoricalDtype):
                cats.append(c)
            elif pd.api.types.is_bool_dtype(dt) or pd.api.types.is_numeric_dtype(dt):
                nums.append(c)
            else:
                raise TypeError(f"tabm: column {c!r} has unsupported dtype {dt}")
        a = self._num(X, nums)
        finite = np.isfinite(a)
        keep = []
        for j in range(len(nums)):
            v = a[finite[:, j], j]
            if v.size and np.nanmax(v) > np.nanmin(v):
                keep.append(j)
        self.num_cols = [nums[j] for j in keep]
        a = a[:, keep]
        has_miss = ~finite[:, keep].all(axis=0)
        self.miss_cols = [c for c, m in zip(self.num_cols, has_miss, strict=True) if m]
        pats = list(self.p.get("slog_patterns") or ())
        if pats:
            self.slog_idx = [
                j
                for j, c in enumerate(self.num_cols)
                if any(fnmatch.fnmatchcase(c, q) for q in pats)
            ]
            if self.slog_idx:
                a[:, self.slog_idx] = _slog(a[:, self.slog_idx])
                if not bool(self.p.get("slog_qt", True)):
                    s = a[:, self.slog_idx]
                    self.slog_mu = np.nanmean(s, axis=0)
                    self.slog_sd = np.nanstd(s, axis=0)
                    self.slog_sd[~(self.slog_sd > 0)] = 1.0
        if self.num_cols:
            n = a.shape[0]
            self.qt = QuantileTransformer(
                n_quantiles=max(2, min(int(self.p["n_quantiles"]), n)),
                output_distribution="normal",
                subsample=int(self.p["qt_subsample"]),
                random_state=self.seed,
            ).fit(a)
            if self.p.get("num_emb", "periodic") == "ple":
                self._fit_bins(a)
        self.cat_cols = cats
        for c in cats:
            vc = X[c].astype(object).value_counts(dropna=True)
            vc = vc[vc >= int(self.p["cat_min_count"])].head(int(self.p["cat_max_levels"]))
            # 0 = missing, 1 = OTHER (rare or unseen), 2.. = the kept levels
            self.vocab[c] = {lvl: i + 2 for i, lvl in enumerate(vc.index)}
        return self

    def _fit_bins(self, a: np.ndarray) -> None:
        """PLE bins: quantiles of each numeric in the space the network sees (after the quantile
        step), on a seeded subsample of the training rows, missing values excluded."""
        n = a.shape[0]
        m = min(n, int(self.p["qt_subsample"]))
        idx = np.sort(np.random.default_rng(self.seed).choice(n, m, replace=False))
        z = self._z(a[idx])
        nb = int(self.p.get("n_bins", 48))
        self.bins = [_ple_edges(z[np.isfinite(z[:, j]), j], nb) for j in range(z.shape[1])]

    def _z(self, a: np.ndarray) -> np.ndarray:
        """The numeric transform of an already slog-ed matrix (NaN where missing)."""
        z = self._qt_transform(a)
        if self.slog_mu is not None:
            z[:, self.slog_idx] = (a[:, self.slog_idx] - self.slog_mu) / self.slog_sd
        return z

    def _qt_transform(self, a: np.ndarray) -> np.ndarray:
        qt = self.qt
        assert qt is not None
        n_thr = int(PREP_THREADS)
        if n_thr <= 0 or a.shape[0] <= _PREP_CHUNK:
            with np.errstate(invalid="ignore"):
                return qt.transform(a)
        from concurrent.futures import ThreadPoolExecutor

        def one(i: int) -> np.ndarray:
            with np.errstate(invalid="ignore"):  # errstate is per thread
                return qt.transform(a[i : i + _PREP_CHUNK])

        out = np.empty(a.shape, dtype=np.float64)
        starts = range(0, a.shape[0], _PREP_CHUNK)
        with ThreadPoolExecutor(max_workers=n_thr) as ex:
            for i, part in zip(starts, ex.map(one, starts), strict=True):
                out[i : i + part.shape[0]] = part
        return out

    def cardinalities(self) -> list[int]:
        return [len(self.vocab[c]) + 2 for c in self.cat_cols]

    def transform(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a = self._num(X, self.num_cols)
        miss = ~np.isfinite(a)
        if self.slog_idx:
            a[:, self.slog_idx] = _slog(a[:, self.slog_idx])
        if self.qt is not None:
            z = self._z(a)
            z = np.where(miss, 0.0, z)
        else:
            z = a
        idx = [self.num_cols.index(c) for c in self.miss_cols]
        m = miss[:, idx].astype(np.float32)
        codes = np.zeros((len(X), len(self.cat_cols)), dtype=np.int64)
        for j, c in enumerate(self.cat_cols):
            col = X[c].astype(object)
            mapped = col.map(self.vocab[c])
            v = np.where(col.isna().to_numpy(), 0, mapped.fillna(1).to_numpy())
            codes[:, j] = v.astype(np.int64)
        return z.astype(np.float32), m, codes


# --------------------------------------------------------------------------------------
# the network
# --------------------------------------------------------------------------------------


def _build_net(n_num: int, n_miss: int, cards: list[int], p: dict, bins=None):
    import torch
    from torch import nn

    arch = str(p.get("arch", "tabm"))
    num_emb = str(p.get("num_emb", "periodic"))
    if arch not in ("tabm", "mlp"):
        raise ValueError(f"tabm: arch must be tabm / mlp, got {arch!r}")
    if arch == "mlp" and int(p["k"]) != 1:
        raise ValueError("tabm: arch mlp is one plain MLP per seed and needs k: 1")
    if num_emb not in ("periodic", "ple"):
        raise ValueError(f"tabm: num_emb must be periodic / ple, got {num_emb!r}")

    class PeriodicEmb(nn.Module):
        """PLR: ``ReLU(Linear_i([cos(2π c_i x_i), sin(2π c_i x_i)]))`` per feature ``i``."""

        def __init__(self, n: int, n_freq: int, d: int, sigma: float) -> None:
            super().__init__()
            self.freq = nn.Parameter(torch.randn(n, n_freq) * sigma)
            bound = 1.0 / math.sqrt(2 * n_freq)
            self.w = nn.Parameter(torch.empty(n, 2 * n_freq, d).uniform_(-bound, bound))
            self.b = nn.Parameter(torch.empty(n, d).uniform_(-bound, bound))

        def forward(self, x):  # (B, n) -> (B, n * d)
            v = 2 * math.pi * self.freq[None] * x[..., None]
            v = torch.cat([torch.cos(v), torch.sin(v)], dim=-1)
            out = torch.einsum("bnf,nfd->bnd", v, self.w) + self.b
            return torch.relu(out).flatten(1)

    class PiecewiseLinearEmb(nn.Module):
        """PLE (Gorishniy et al. 2022, quantile bins), in the ``rtdl_num_embeddings``
        ``PiecewiseLinearEmbeddings(version="B", activation=False)`` form that TabM uses: a
        per-feature linear embedding of ``x`` plus a zero-initialised per-feature linear map of
        the encoding ``e_t = clamp((x - b_{t-1}) / (b_t - b_{t-1}), 0, 1)``; the first bin is
        not clamped below and the last not above. Features with fewer bins are zero-padded."""

        def __init__(self, edges: list[np.ndarray], d: int) -> None:
            super().__init__()
            n, t = len(edges), max(len(e) - 1 for e in edges)
            lo, width = np.zeros((n, t)), np.ones((n, t))
            mask, lb, ub = np.zeros((n, t)), np.zeros((n, t)), np.ones((n, t))
            for i, e in enumerate(edges):
                ti = len(e) - 1
                lo[i, :ti], width[i, :ti], mask[i, :ti] = e[:-1], np.diff(e), 1.0
                lb[i, 0], ub[i, ti - 1] = -np.inf, np.inf
            for name, v in (("lo", lo), ("width", width), ("mask", mask), ("lb", lb), ("ub", ub)):
                self.register_buffer(name, torch.tensor(v, dtype=torch.float32))
            bound = 1.0 / math.sqrt(d)
            self.w0 = nn.Parameter(torch.empty(n, d).uniform_(-bound, bound))
            self.b0 = nn.Parameter(torch.empty(n, d).uniform_(-bound, bound))
            self.w = nn.Parameter(torch.zeros(n, t, d))

        def forward(self, x):  # (B, n) -> (B, n * d)
            e = (x[..., None] - self.lo) / self.width
            e = torch.maximum(torch.minimum(e, self.ub), self.lb) * self.mask
            out = x[..., None] * self.w0 + self.b0 + torch.einsum("bnt,ntd->bnd", e, self.w)
            return out.flatten(1)

    class LinearBE(nn.Module):
        """BatchEnsemble linear layer: ``((x ⊙ r_m) W) ⊙ s_m + b_m`` for member ``m``."""

        def __init__(self, d_in: int, d_out: int, k: int, random_signs: bool) -> None:
            super().__init__()
            bound = 1.0 / math.sqrt(d_in)
            self.weight = nn.Parameter(torch.empty(d_in, d_out).uniform_(-bound, bound))
            r = torch.ones(k, d_in)
            if random_signs:
                r = torch.where(torch.rand(k, d_in) < 0.5, -1.0, 1.0)
            self.r = nn.Parameter(r)
            self.s = nn.Parameter(torch.ones(k, d_out))
            self.bias = nn.Parameter(torch.empty(k, d_out).uniform_(-bound, bound))

        def forward(self, x):  # (B, d_in) shared or (B, k, d_in) per member -> (B, k, d_out)
            if x.dim() == 2:
                # the first layer, whose input is the same for every member: fold r into k
                # weight matrices and run one (B, d_in) @ (d_in, k·d_out) GEMM, instead of
                # materialising the (B, k, d_in) tensor x ⊙ r (identical maths, ~8x less
                # memory at d_in ≈ 3000)
                k, d_out = self.s.shape
                wk = (self.r.T[:, :, None] * self.weight[:, None, :]).reshape(-1, k * d_out)
                return (x @ wk).view(-1, k, d_out) * self.s + self.bias
            return (x * self.r) @ self.weight * self.s + self.bias

    class TabM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            k = int(p["k"])
            self.k = k
            if not n_num:
                self.num_emb = None
            elif num_emb == "ple":
                assert bins is not None and len(bins) == n_num, "tabm: ple needs fitted bins"
                self.num_emb = PiecewiseLinearEmb(bins, int(p["d_num_emb"]))
            else:
                self.num_emb = PeriodicEmb(
                    n_num, int(p["n_frequencies"]), int(p["d_num_emb"]), float(p["frequency_scale"])
                )
            d_cat = int(p["d_cat_emb"])
            self.cat_emb = nn.ModuleList(nn.Embedding(c, d_cat) for c in cards)
            d_in = n_num * int(p["d_num_emb"]) + n_miss + len(cards) * d_cat
            d_in = max(d_in, 1)
            d = int(p["d_block"])
            layers = []
            for i in range(int(p["n_blocks"])):
                if arch == "mlp":  # plain nn.Linear: the same U(±1/sqrt(d_in)) bias bound
                    layers.append(nn.Linear(d_in if i == 0 else d, d))
                else:
                    layers.append(LinearBE(d_in if i == 0 else d, d, k, random_signs=i == 0))
            self.blocks = nn.ModuleList(layers)
            self.drop = nn.Dropout(float(p["dropout"]))
            bound = 1.0 / math.sqrt(d)
            self.head_w = nn.Parameter(torch.empty(k, d).uniform_(-bound, bound))
            self.head_b = nn.Parameter(torch.zeros(k))

        def forward(self, xn, xm, xc):  # -> (B, k)
            parts = []
            if self.num_emb is not None:
                parts.append(self.num_emb(xn))
            if xm.shape[1]:
                parts.append(xm)
            for j, emb in enumerate(self.cat_emb):
                parts.append(emb(xc[:, j]))
            x = torch.cat(parts, dim=1) if parts else xn.new_zeros(xn.shape[0], 1)
            for blk in self.blocks:  # (B, D) -> (B, k, d) at the first block
                x = self.drop(torch.relu(blk(x)))
            if x.dim() == 2:  # arch mlp: (B, d) -> (B, 1, d), one "member"
                x = x[:, None, :]
            return (x * self.head_w).sum(-1) + self.head_b

    return TabM()


# --------------------------------------------------------------------------------------
# the wrapper
# --------------------------------------------------------------------------------------


class TabMModel:
    """Same interface as :class:`prc.models.xgb.XGBModel`. ``matrices`` / ``key`` are accepted
    and ignored (nothing here is worth caching across seeds: the quantile transform is seeded)."""

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
        self.y_mu = 0.0
        self.y_sd = 1.0

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

    def _predict_t(self, T, dev) -> np.ndarray:
        """Mean over the ``k`` members, in the standardised target space."""
        import torch

        xn, xm, xc = T
        bs = int(self.p["eval_batch_size"])
        out = []
        self.net.eval()
        with torch.no_grad(), self._autocast(dev):
            for i in range(0, xn.shape[0], bs):
                out.append(self.net(xn[i : i + bs], xm[i : i + bs], xc[i : i + bs]).float())
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
        self.prep = _Prep(self.p, self.seed).fit(X)
        y = np.asarray(y, dtype=np.float64)
        # standardise under the same weights as the loss, so a weight-0 row has no influence
        # at all (not even through the scale of the target)
        ww = None if w is None or not np.sum(w) > 0 else np.asarray(w, dtype=np.float64)
        self.y_mu = float(np.average(y, weights=ww))
        self.y_sd = float(np.sqrt(np.average((y - self.y_mu) ** 2, weights=ww))) or 1.0
        Ttr = self._tensors(X, dev)
        ytr = torch.from_numpy(((y - self.y_mu) / self.y_sd).astype(np.float32)).to(dev)
        wtr = None
        if w is not None:
            wtr = torch.from_numpy(np.asarray(w, dtype=np.float32)).to(dev)
        Tva = yva = None
        if X_val is not None:
            Tva = self._tensors(X_val, dev)
            yva = np.asarray(y_val, dtype=np.float64)
        self._t("matrix", time.perf_counter() - t0)

        self.net = _build_net(
            len(self.prep.num_cols),
            len(self.prep.miss_cols),
            self.prep.cardinalities(),
            self.p,
            bins=self.prep.bins,
        ).to(dev)
        lr = float(self.p["learning_rate"])
        factor = _lr_factor(self.p)
        if factor is not None:
            lr = lr * factor
        opt = torch.optim.AdamW(
            self.net.parameters(),
            lr=lr,
            weight_decay=float(self.p["weight_decay"]),
        )
        n = int(ytr.shape[0])
        bs = min(int(self.p["batch_size"]), n)
        steps_per_epoch = math.ceil(n / bs)
        # 0 at the default: the loop below never touches the learning rate
        warmup = math.ceil(float(self.p["warmup_epochs"]) * steps_per_epoch)
        cap = int(self.cfg.num_boost_round)
        if X_val is not None:
            cap = min(cap, int(self.p["max_epochs"]) * steps_per_epoch)
        patience = int(self.cfg.early_stopping_rounds)
        verbose = int(self.p["verbose"])

        t0 = time.perf_counter()
        step, epoch = 0, 0
        best, best_state, since = math.inf, None, 0
        xn, xm, xc = Ttr
        while step < cap:
            self.net.train()
            perm = torch.randperm(n, generator=gen).to(dev)
            loss_sum, n_b = 0.0, 0
            for i in range(0, n, bs):
                if step >= cap:
                    break
                b = perm[i : i + bs]
                with self._autocast(dev):
                    pred = self.net(xn[b], xm[b], xc[b]).float()
                se = (pred - ytr[b][:, None]).pow(2).mean(dim=1)
                if wtr is None:
                    loss = se.mean()
                else:
                    wb = wtr[b]
                    loss = (wb * se).sum() / wb.sum().clamp_min(1e-12)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if step < warmup:
                    for g in opt.param_groups:
                        g["lr"] = lr * (step + 1) / warmup
                opt.step()
                step += 1
                loss_sum += float(loss.detach())
                n_b += 1
            epoch += 1
            rec = {"epoch": epoch, "step": step, "train_loss": loss_sum / max(n_b, 1)}
            if Tva is not None:
                p_va = self._predict_t(Tva, dev) * self.y_sd + self.y_mu
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
                    "tabm seed %d epoch %d step %d loss %.5f%s",
                    self.seed,
                    epoch,
                    step,
                    rec["train_loss"],
                    f" val_rmse {rec['val_rmse']:.5f}" if "val_rmse" in rec else "",
                )
            if Tva is not None and since >= patience:
                break
        if Tva is None:
            self.best_iteration, self.best_epoch = step - 1, epoch
        elif best_state is not None:
            self.net.load_state_dict(best_state)
        self._t("train", time.perf_counter() - t0)
        del Ttr, ytr, wtr, Tva
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        return self

    def predict(self, X, key=None):
        import torch

        assert self.net is not None, "fit first"
        dev = torch.device(self.device)
        t0 = time.perf_counter()
        T = self._tensors(X, dev)
        out = self._predict_t(T, dev) * self.y_sd + self.y_mu
        del T
        self._t("predict", time.perf_counter() - t0)
        return out

    def importance(self):
        return None
