"""``model.kind: realmlp`` — RealMLP-TD as a second, differently built neural blend member.

RealMLP: Holzmüller, Grinsztajn, Steinwart, "Better by Default: Strong Pre-Tuned MLPs and
Boosted Trees on Tabular Data", NeurIPS 2024 (arXiv 2407.04491). The reference implementation is
``pytabkit`` (``DefaultParams.RealMLP_TD_REG``, ``nn_models/{nn,models,categorical,activations}``,
``optim/optimizers.py``, ``training/scheduling.py``). Rebuilt here in plain PyTorch (~300
lines), so the cluster needs nothing beyond ``torch``.

Where it differs from TabM (``prc.models.tabm``) by design: robust scaling + smooth clipping
instead of a quantile transform, PBLD instead of PLR embeddings, one-hot for small categoricals,
a learnable diagonal front scale, NTK parametrisation with data-dependent init, parametric mish,
per-group learning-rate factors, and a scheduled learning rate / weight decay / dropout. One MLP,
not an ensemble of 32.

The recipe (RealMLP-TD for regression, pytabkit's defaults unless marked *PRC*):

* **numerics** — ``x -> (x - median) / (q75 - q25)`` (``(max - min) / 2`` if the IQR is 0),
  then the smooth clip ``z / sqrt(1 + (z / 3)^2)``. Statistics on the fold's training rows
  (*PRC*: on a seeded subsample of at most ``stat_subsample`` rows). NaN / inf become 0 after
  the transform (the median) plus a 0/1 missing indicator per column that has a missing value
  in training (*PRC*; pytabkit imputes upstream). Constant / all-missing columns are dropped.
* **PBLD numeric embeddings** — per feature ``i``: ``[Linear_i(cos(2π w_i x + b_i)), x]``,
  ``w_i ~ N(0, 0.1²)`` (16 frequencies), ``b_i ~ U[-π, π]``, a linear map 16 -> 3, and the
  raw value appended ("densenet"): 4 numbers per numeric.
* **categoricals** — a vocabulary as in TabM (*PRC*: ``cat_min_count`` / ``cat_max_levels``).
  At most ``max_one_hot_cat_size - 1`` (8) levels: one-hot, a single ±1 column when there are
  two levels, missing / rare / unseen = all zeros. More levels: an embedding of width 8
  (``N(0, 1)`` init); index 0 = missing, 1 = OTHER.
* **network** — a learnable diagonal scale on the concatenated input (init 1, lr factor 6),
  then 3 hidden layers of 256: NTK-parametrised linear (``x @ W / sqrt(fan_in)``, ``W`` drawn
  ``N(0, 1)`` and rescaled so each unit has unit std on the data), bias ("he+5": minus a random
  convex combination of 5 rows' pre-activations, lr factor 0.1, no weight decay), a parametric
  activation ``x + α (mish(x) - x)`` (``α`` per unit, init 1, lr factor 0.1), dropout 0.15.
  The output layer is linear with the same init.
* **training** — Adam (β = 0.9, 0.95), lr 0.2 with the ``coslog4`` schedule
  ``½ (1 - cos(2π log2(1 + 15 t)))`` over ``t = step / (max_epochs · steps_per_epoch)``;
  decoupled weight decay 0.02 with the ``flat_cos`` schedule, applied as pytabkit's optimizer
  does: ``p *= 1 - wd·f_wd² · lr·f_lr²`` (its per-parameter factors enter twice); dropout
  also on ``flat_cos``. The loss is MSE on the standardised target, weighted ``Σ w·se / Σ w``;
  predictions are clamped to the training target range (rows with ``w > 0``).

How the runner's GBDT vocabulary maps onto it (identical to TabM):

``best_iteration``         the optimizer step count at the best epoch, minus one.
``num_boost_round``        a cap on optimizer steps; the full fit (``X_val=None``) runs exactly
                           that many. The schedule horizon is always ``max_epochs`` epochs of
                           the rows at hand, so a full fit that runs the fold models' median
                           number of epochs retraces the fold models' schedule up to their best
                           epoch.
``early_stopping_rounds``  the patience in epochs; the best epoch's weights are restored. With
                           coslog4 the validation loss rises within a cycle, so the YAMLs set
                           it to ``max_epochs`` (train the whole horizon, keep the best epoch),
                           which is pytabkit's own best-epoch selection.

Determinism: every draw is seeded from the member seed; on CPU a seed reproduces bit for bit.
"""

from __future__ import annotations

import copy
import fnmatch
import itertools
import logging
import math
import time
import warnings

import numpy as np
import pandas as pd

from prc.config import ModelConfig

log = logging.getLogger("prc.realmlp")

DEFAULTS: dict = {
    "n_hidden_layers": 3,
    "d_hidden": 256,
    "act": "mish",  # mish (regression default) | selu (classification default) | relu
    "use_parametric_act": True,
    "p_drop": 0.15,
    "p_drop_sched": "flat_cos",  # flat_cos | constant
    "learning_rate": 0.2,
    "lr_sched": "coslog4",  # coslog4 | flat_cos | cos | constant
    "weight_decay": 0.02,
    "wd_sched": "flat_cos",  # flat_cos | constant
    "beta1": 0.9,
    "beta2": 0.95,
    "bias_lr_factor": 0.1,
    "act_lr_factor": 0.1,
    "scale_lr_factor": 6.0,
    "plr_lr_factor": 0.1,
    "emb_lr_factor": 1.0,
    "plr_sigma": 0.1,
    "plr_hidden_1": 16,
    "plr_hidden_2": 4,  # output width per numeric, the raw value included
    "max_one_hot_cat_size": 9,  # levels + missing; larger categoricals get an embedding
    "embedding_size": 8,
    "cat_max_levels": 1000,
    "cat_min_count": 20,
    "smooth_clip": 3.0,
    "stat_subsample": 500_000,  # rows for the median / IQR (0 = all)
    "init_rows": 8192,  # rows for the data-dependent init
    "clamp_output": True,
    # PRC scale: pytabkit trains 256 epochs at batch 256 on <= 500 k rows; at ~2 M rows that is
    # out of budget, so the horizon is 32 epochs at batch 2048 (~29 k steps per fold). Untuned.
    "batch_size": 2048,
    "eval_batch_size": 16384,
    "max_epochs": 32,
    "amp": True,  # bf16 autocast on CUDA; ignored on CPU
    "verbose": 1,
    "slog_patterns": (),  # fnmatch patterns of numerics that get sign(x)·log1p(|x|/60) first
    # NM dropout (backlog U2): each epoch a share
    # `nm_dropout` of the training rows WITH an NM record get every NM-derived input set to the
    # state it has on a genuine no-flight-plan row (label and weight unchanged). 0 = off
    # (bit-identical to the member without it). `nm_dropout_cols`: "auto" (the empirical rule
    # of `nm_columns`) or a list of column names; `nm_key_col`: the column whose missing value
    # marks a genuine no-flight-plan row (`fpl_deltas.tko_minus_aobt3` = MVT - AOBT_3_flt, null
    # exactly on the lane rows of `prc.eval.lanes`).
    "nm_dropout": 0.0,
    "nm_dropout_cols": "auto",
    "nm_key_col": "tko_minus_aobt3",
}

# `nm_columns`: a column is NM-derived when one state (missing, one value, or one "=flag"
# suffix) holds on >= NM_STATE_MIN of the genuine no-flight-plan training rows and on
# <= NM_STATE_MAX of the rows with an NM record.
NM_STATE_MIN = 0.99
NM_STATE_MAX = 0.05

_SCHEDS = ("coslog4", "flat_cos", "cos", "constant")


def _sched(name: str, t: float) -> float:
    """pytabkit's schedule shapes on ``t`` in [0, 1]."""
    t = min(max(t, 0.0), 1.0)
    if name == "constant":
        return 1.0
    if name == "coslog4":
        return 0.5 * (1.0 - math.cos(2.0 * math.pi * math.log2(1.0 + 15.0 * t)))
    if name == "flat_cos":
        return 1.0 if t <= 0.5 else 0.5 * (1.0 + math.cos(math.pi * (2.0 * t - 1.0)))
    if name == "cos":
        return 0.5 * (1.0 + math.cos(math.pi * t))
    raise ValueError(f"realmlp: unknown schedule {name!r}; one of {', '.join(_SCHEDS)}")


# --------------------------------------------------------------------------------------
# preprocessing (numpy / pandas; fitted on training rows only)
# --------------------------------------------------------------------------------------


def _slog(a: np.ndarray) -> np.ndarray:
    """Signed log of seconds: ``sign(x) · log1p(|x| / 60)`` (NaN stays NaN)."""
    return np.sign(a) * np.log1p(np.abs(a) / 60.0)


class _Prep:
    """Robust scale + smooth clip + missing indicators; one-hot / embedding codes for cats."""

    def __init__(self, p: dict, seed: int) -> None:
        self.p = p
        self.seed = seed
        self.num_cols: list[str] = []
        self.miss_cols: list[str] = []
        self.center: np.ndarray = np.zeros(0)
        self.factor: np.ndarray = np.zeros(0)
        self.onehot_cols: list[str] = []
        self.emb_cols: list[str] = []
        self.vocab: dict[str, dict] = {}
        self.slog_idx: list[int] = []

    @staticmethod
    def _num(X: pd.DataFrame, cols: list[str]) -> np.ndarray:
        if not cols:
            return np.zeros((len(X), 0), dtype=np.float64)
        a = X[cols].to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
        a[~np.isfinite(a)] = np.nan
        return a

    def fit(self, X: pd.DataFrame) -> _Prep:
        cats, nums = [], []
        for c in X.columns:
            dt = X[c].dtype
            if isinstance(dt, pd.CategoricalDtype):
                cats.append(c)
            elif pd.api.types.is_bool_dtype(dt) or pd.api.types.is_numeric_dtype(dt):
                nums.append(c)
            else:
                raise TypeError(f"realmlp: column {c!r} has unsupported dtype {dt}")
        a = self._num(X, nums)
        finite = np.isfinite(a)
        keep = []
        for j in range(len(nums)):
            v = a[finite[:, j], j]
            if v.size and v.max() > v.min():
                keep.append(j)
        self.num_cols = [nums[j] for j in keep]
        a = a[:, keep]
        has_miss = ~finite[:, keep].all(axis=0)
        self.miss_cols = [c for c, m in zip(self.num_cols, has_miss, strict=True) if m]
        pats = list(self.p.get("slog_patterns") or ())
        self.slog_idx = [
            j for j, c in enumerate(self.num_cols) if any(fnmatch.fnmatchcase(c, q) for q in pats)
        ]
        if self.slog_idx:  # sign(x)·log1p(|x|/60) before the robust scale (as prc.models.tabm)
            a[:, self.slog_idx] = _slog(a[:, self.slog_idx])
        if self.num_cols:
            sub = int(self.p["stat_subsample"])
            if 0 < sub < a.shape[0]:
                rows = np.sort(
                    np.random.default_rng(self.seed).choice(a.shape[0], sub, replace=False)
                )
                s = a[rows]
            else:
                s = a
            with warnings.catch_warnings():  # a column all-missing in the subsample -> scale 0
                warnings.simplefilter("ignore", RuntimeWarning)
                q25, med, q75 = np.nanquantile(s, [0.25, 0.5, 0.75], axis=0)
                diff = q75 - q25
                zero = diff == 0.0
                diff[zero] = 0.5 * (np.nanmax(s, axis=0)[zero] - np.nanmin(s, axis=0)[zero])
            with np.errstate(divide="ignore"):
                factor = np.where(diff > 0.0, 1.0 / (diff + 1e-30), 0.0)
            self.center, self.factor = np.nan_to_num(med), factor
        n_oh = int(self.p["max_one_hot_cat_size"]) - 1
        for c in cats:
            vc = X[c].astype(object).value_counts(dropna=True)
            vc = vc[vc >= int(self.p["cat_min_count"])].head(int(self.p["cat_max_levels"]))
            # 0 = missing, 1 = OTHER (rare or unseen), 2.. = the kept levels
            self.vocab[c] = {lvl: i + 2 for i, lvl in enumerate(vc.index)}
            if len(vc) == 0:
                continue  # only rare / missing levels in training: nothing to learn from
            (self.onehot_cols if len(vc) <= n_oh else self.emb_cols).append(c)
        return self

    def n_onehot(self) -> int:
        return sum(1 if len(self.vocab[c]) <= 2 else len(self.vocab[c]) for c in self.onehot_cols)

    def cardinalities(self) -> list[int]:
        return [len(self.vocab[c]) + 2 for c in self.emb_cols]

    def _codes(self, X: pd.DataFrame, c: str) -> np.ndarray:
        col = X[c].astype(object)
        mapped = col.map(self.vocab[c])
        return np.where(col.isna().to_numpy(), 0, mapped.fillna(1).to_numpy()).astype(np.int64)

    def transform(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(numerics, extras, codes)``: the clipped numerics (PBLD input), the missing
        indicators + one-hot columns (fed to the network as they are), the embedding codes."""
        a = self._num(X, self.num_cols)
        miss = ~np.isfinite(a)
        if self.slog_idx:
            a[:, self.slog_idx] = _slog(a[:, self.slog_idx])
        z = (a - self.center) * self.factor
        m = float(self.p["smooth_clip"])
        z = z / np.sqrt(1.0 + (z / m) ** 2)
        z = np.where(miss, 0.0, z)
        idx = [self.num_cols.index(c) for c in self.miss_cols]
        extras = [miss[:, idx].astype(np.float32)]
        for c in self.onehot_cols:
            k = self._codes(X, c)
            n_lv = len(self.vocab[c])
            if n_lv <= 2:  # one column: first level +1, second -1, missing / other 0
                v = np.zeros((len(X), 1), dtype=np.float32)
                v[k == 2, 0] = 1.0
                v[k == 3, 0] = -1.0
            else:
                v = np.zeros((len(X), n_lv), dtype=np.float32)
                hit = k >= 2
                v[np.nonzero(hit)[0], k[hit] - 2] = 1.0
            extras.append(v)
        codes = np.zeros((len(X), len(self.emb_cols)), dtype=np.int64)
        for j, c in enumerate(self.emb_cols):
            codes[:, j] = self._codes(X, c)
        return z.astype(np.float32), np.concatenate(extras, axis=1), codes


# --------------------------------------------------------------------------------------
# NM dropout (backlog U2)
# --------------------------------------------------------------------------------------


def _col_state(s: pd.Series, r0: np.ndarray, r1: np.ndarray, strict: bool):
    """The state column ``s`` takes on the no-NM rows ``r0``: ``("missing", None)``,
    ``("value", v)`` or ``("flag", f)`` (levels ``<prefix>=<flag>``, e.g. ``airport_nm``'s
    ``LIRF|nm=1``: the prefix is kept, the flag set). ``strict``: the state must also be rare
    (<= NM_STATE_MAX) on the NM rows ``r1``; otherwise None. Not strict (an explicit list):
    the modal state, "missing" when none reaches NM_STATE_MIN."""
    n0 = int(r0.sum())
    if n0 == 0:
        return ("missing", None) if not strict else None
    na = s.isna().to_numpy()
    if na[r0].mean() >= NM_STATE_MIN and (not strict or na[r1].mean() <= NM_STATE_MAX):
        return ("missing", None)
    is_cat = isinstance(s.dtype, pd.CategoricalDtype)
    v0 = s[r0].dropna()
    if is_cat:
        v0 = v0.astype(object)
    if len(v0) >= NM_STATE_MIN * n0:
        if is_cat:  # "=flag" first: a prefix that dominates the no-NM rows (LIRF) is no state
            f0 = v0.astype(str).str.rpartition("=")
            fc = f0[2][f0[1] == "="].value_counts()
            if len(fc) and int(fc.iloc[0]) >= NM_STATE_MIN * n0:
                flag = str(fc.index[0])
                l1 = s[r1].astype(object).dropna().astype(str).str.rpartition("=")
                hit1 = float(((l1[1] == "=") & (l1[2] == flag)).sum()) / max(int(r1.sum()), 1)
                if not strict or hit1 <= NM_STATE_MAX:
                    return ("flag", flag)
        vc = v0.value_counts()
        top, cnt = vc.index[0], int(vc.iloc[0])
        if cnt >= NM_STATE_MIN * n0:
            s1 = s[r1].astype(object) if is_cat else s[r1]
            if not strict or float((s1 == top).mean()) <= NM_STATE_MAX:
                return ("value", top)
    return None if strict else ("missing", None)


def nm_columns(
    X: pd.DataFrame, key_col: str, cols="auto", candidates: list[str] | None = None
) -> list[tuple[str, str, object]]:
    """The NM-derived columns of ``X`` and the state each takes without an NM record,
    ``[(column, "missing" | "value" | "flag", value)]``. Genuine no-flight-plan rows are the
    rows where ``key_col`` is missing. Uses no labels, so it is fold-safe on training rows.
    ``cols="auto"`` applies the rule of NM_STATE_MIN / NM_STATE_MAX to every column (or to
    ``candidates``); a list takes those columns and their modal no-NM state."""
    if key_col not in X.columns:
        raise ValueError(f"realmlp: nm_key_col {key_col!r} is not a model input")
    r0 = X[key_col].isna().to_numpy()
    r1 = ~r0
    if cols == "auto":
        names, strict = list(candidates if candidates is not None else X.columns), True
    else:
        if isinstance(cols, str):
            raise ValueError(f"realmlp: nm_dropout_cols must be 'auto' or a list, got {cols!r}")
        names, strict = list(cols), False
        missing = [c for c in names if c not in X.columns]
        if missing:
            raise ValueError(f"realmlp: nm_dropout_cols not in the inputs: {missing}")
    out = []
    for c in names:
        st = _col_state(X[c], r0, r1, strict)
        if st is not None:
            out.append((c, st[0], st[1]))
    return out


def _apply_state(s: pd.Series, kind: str, value) -> pd.Series:
    """``s`` with every row set to the no-NM state (pandas domain, before the prep)."""
    if kind == "missing":
        if isinstance(s.dtype, pd.CategoricalDtype):
            return pd.Series(pd.Categorical([None] * len(s)), index=s.index)
        return pd.Series(np.nan, index=s.index, dtype=np.float64)
    if kind == "value":
        if isinstance(s.dtype, pd.CategoricalDtype):
            return pd.Series([value] * len(s), index=s.index, dtype=object)
        return pd.Series(float(value), index=s.index, dtype=np.float64)
    o = s.astype(object)  # "flag": <prefix>=<value>
    lv = {
        v: (str(v).rpartition("=")[0] + "=" + str(value) if "=" in str(v) else v)
        for v in o.dropna().unique()
    }
    return o.map(lv)


class _NMDrop:
    """NM dropout on the training rows, in the representation the network sees.

    At construction: the NM-derived columns (``nm_columns`` on the fit's training rows), the
    slots they occupy in ``(numerics, extras, codes)`` and, per training row, the values
    those slots take when the columns are in their no-NM state (the prep's own transform of
    the altered columns, so a missing numeric is 0 with its indicator 1, exactly as on a
    genuine no-flight-plan row). Each epoch :meth:`draw` picks a Bernoulli(``share``) subset
    of the rows with an NM record (own seeded generator, so it does not shift any other
    draw); :meth:`apply` swaps those rows' slots within a batch."""

    CHUNK = 65_536

    def __init__(self, prep: _Prep, X: pd.DataFrame, p: dict, seed: int, dev) -> None:
        import torch

        self.share = float(p["nm_dropout"])
        if not 0.0 < self.share <= 1.0:
            raise ValueError(f"realmlp: nm_dropout must be in [0, 1], got {self.share}")
        kept = set(prep.num_cols) | set(prep.onehot_cols) | set(prep.emb_cols)
        cand = [c for c in X.columns if c in kept]
        spec = nm_columns(X, str(p["nm_key_col"]), p["nm_dropout_cols"], candidates=cand)
        self.spec = [s for s in spec if s[0] in kept]
        self.is_nm = X[str(p["nm_key_col"])].notna().to_numpy()
        self.n_nm, self.n_no_nm = int(self.is_nm.sum()), int((~self.is_nm).sum())
        # slots in (numerics, extras, codes)
        off, oh_off = len(prep.miss_cols), {}
        for c in prep.onehot_cols:
            w = 1 if len(prep.vocab[c]) <= 2 else len(prep.vocab[c])
            oh_off[c] = (off, w)
            off += w
        i_n, i_e, i_c = [], [], []
        for c, _, _ in self.spec:
            if c in prep.num_cols:
                i_n.append(prep.num_cols.index(c))
                if c in prep.miss_cols:
                    i_e.append(prep.miss_cols.index(c))
            elif c in oh_off:
                i_e.extend(range(oh_off[c][0], oh_off[c][0] + oh_off[c][1]))
            else:
                i_c.append(prep.emb_cols.index(c))
        n = len(X)
        rn = np.zeros((n, len(i_n)), dtype=np.float32)
        re_ = np.zeros((n, len(i_e)), dtype=np.float32)
        rc = np.zeros((n, len(i_c)), dtype=np.int64)
        if self.spec:
            for a in range(0, n, self.CHUNK):
                Xc = X.iloc[a : a + self.CHUNK].copy()
                for c, kind, value in self.spec:
                    Xc[c] = _apply_state(Xc[c], kind, value)
                z, e, k = prep.transform(Xc)
                rn[a : a + len(Xc)] = z[:, i_n]
                re_[a : a + len(Xc)] = e[:, i_e]
                rc[a : a + len(Xc)] = k[:, i_c]
        self.slots = {"num": i_n, "extra": i_e, "code": i_c}
        self.idx = [torch.tensor(i, dtype=torch.long, device=dev) for i in (i_n, i_e, i_c)]
        self.rep = [torch.from_numpy(r).to(dev) for r in (rn, re_, rc)]
        self.rng = np.random.default_rng([int(seed), 0x4E4D])  # "NM"
        self.dev = dev

    def describe(self) -> str:
        cols = ", ".join(c if k == "missing" else f"{c}={v!r}" for c, k, v in self.spec)
        return (
            f"nm_dropout {self.share}: {len(self.spec)} NM-derived columns from "
            f"{self.n_no_nm} no-NM / {self.n_nm} NM training rows: {cols}"
        )

    def draw(self):
        """This epoch's rows to alter: Bernoulli(share) among the rows with an NM record."""
        import torch

        d = self.rng.random(len(self.is_nm)) < self.share
        return torch.from_numpy(d & self.is_nm).to(self.dev)

    def apply(self, b, m, xn, xe, xc):
        """Batch tensors (copies, rows ``b``) with the rows flagged in ``m`` swapped to the
        no-NM state; the other rows are returned as they are."""
        import torch

        out = []
        for x, idx, rep in zip((xn, xe, xc), self.idx, self.rep, strict=True):
            if idx.numel():
                x[:, idx] = torch.where(m[:, None], rep[b], x[:, idx])
            out.append(x)
        return out


# --------------------------------------------------------------------------------------
# the network
# --------------------------------------------------------------------------------------


def _build_net(n_num: int, n_extra: int, cards: list[int], p: dict, gen):
    """The RealMLP module. Parameters are tagged with ``(lr_factor, wd_factor)`` in
    ``param_factors``; the data-dependent init is :meth:`init_from_data`."""
    import torch
    import torch.nn.functional as F
    from torch import nn

    def mish(x):
        return x * torch.tanh(F.softplus(x))

    acts = {"mish": mish, "selu": torch.selu, "relu": torch.relu}
    if str(p["act"]) not in acts:
        raise ValueError(f"realmlp: act must be one of {', '.join(acts)}, got {p['act']!r}")
    f_act = acts[str(p["act"])]

    class RealMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            h1, h2 = int(p["plr_hidden_1"]), int(p["plr_hidden_2"]) - 1
            self.n_num = n_num
            if n_num:
                sig = float(p["plr_sigma"])
                bnd = 1.0 / math.sqrt(h1)
                self.plr_w1 = nn.Parameter(sig * torch.randn(n_num, h1, generator=gen))
                self.plr_b1 = nn.Parameter(math.pi * (2 * torch.rand(n_num, h1, generator=gen) - 1))
                self.plr_w2 = nn.Parameter(bnd * (2 * torch.rand(n_num, h1, h2, generator=gen) - 1))
                self.plr_b2 = nn.Parameter(bnd * (2 * torch.rand(n_num, h2, generator=gen) - 1))
            d_emb = int(p["embedding_size"])
            self.emb = nn.ParameterList(
                nn.Parameter(torch.randn(c, d_emb, generator=gen)) for c in cards
            )
            d_in = max(n_num * (h2 + 1) + n_extra + len(cards) * d_emb, 1)
            self.scale = nn.Parameter(torch.ones(d_in))
            sizes = [d_in] + [int(p["d_hidden"])] * int(p["n_hidden_layers"]) + [1]
            self.W = nn.ParameterList(
                nn.Parameter(torch.randn(a, b, generator=gen)) for a, b in itertools.pairwise(sizes)
            )
            self.b = nn.ParameterList(nn.Parameter(torch.zeros(b)) for b in sizes[1:])
            self.gain = [1.0 / math.sqrt(a) for a in sizes[:-1]]
            self.alpha = nn.ParameterList(
                nn.Parameter(torch.ones(b)) for b in sizes[1:-1]
            )  # parametric activations (kept at 1 when use_parametric_act is off)
            if not bool(p["use_parametric_act"]):
                for a in self.alpha:
                    a.requires_grad_(False)
            self.p_drop = float(p["p_drop"])
            self.register_buffer("y_lo", torch.tensor(-math.inf))
            self.register_buffer("y_hi", torch.tensor(math.inf))

        def param_factors(self) -> list[tuple[list, float, float]]:
            """``(params, lr_factor, wd_factor)`` groups, pytabkit's hyper factors."""
            plr = [self.plr_w1, self.plr_b1, self.plr_w2, self.plr_b2] if self.n_num else []
            groups = [
                (list(self.W), 1.0, 1.0),
                (list(self.b), float(p["bias_lr_factor"]), 0.0),
                ([self.scale], float(p["scale_lr_factor"]), 1.0),
                (plr, float(p["plr_lr_factor"]), 1.0),
                (list(self.emb), float(p["emb_lr_factor"]), 1.0),
            ]
            if bool(p["use_parametric_act"]):
                groups.append((list(self.alpha), float(p["act_lr_factor"]), 1.0))
            return [g for g in groups if g[0]]

        def embed(self, xn, xe, xc):  # -> (B, d_in), before the front scale
            parts = []
            if self.n_num:
                v = 2 * math.pi * xn[..., None] * self.plr_w1 + self.plr_b1  # (B, n, h1)
                v = torch.einsum("bnh,nhd->bnd", torch.cos(v), self.plr_w2) + self.plr_b2
                parts += [v.flatten(1), xn]
            if xe.shape[1]:
                parts.append(xe)
            for j, e in enumerate(self.emb):
                parts.append(e[xc[:, j]])
            return torch.cat(parts, dim=1) if parts else xn.new_zeros(xn.shape[0], 1)

        def act(self, i: int, x):
            return x + (f_act(x) - x) * self.alpha[i]

        def forward(self, xn, xe, xc, p_drop: float | None = None):  # -> (B,)
            x = self.embed(xn, xe, xc) * self.scale
            last = len(self.W) - 1
            for i, (w, b) in enumerate(zip(self.W, self.b, strict=True)):
                x = (x @ w) * self.gain[i] + b
                if i < last:
                    x = self.act(i, x)
                    q = self.p_drop if p_drop is None else p_drop
                    if self.training and q > 0:
                        x = F.dropout(x, q, training=True)
            out = x[:, 0]
            if not self.training:
                out = torch.minimum(torch.maximum(out, self.y_lo), self.y_hi)
            return out

        @torch.no_grad()
        def init_from_data(self, xn, xe, xc, gen_) -> None:
            """``weight_init_mode: std`` + ``bias_init_mode: he+5``, layer by layer."""
            x = self.embed(xn, xe, xc) * self.scale
            last = len(self.W) - 1
            for i, (w, b) in enumerate(zip(self.W, self.b, strict=True)):
                sd = (x @ w * self.gain[i]).std(dim=0, correction=0)
                w.div_(torch.where(sd > 0, sd, torch.ones_like(sd)))
                pre = x @ w * self.gain[i]
                n, d = pre.shape
                idx = torch.randint(0, n, (d, 5), generator=gen_)
                sw = -torch.log(torch.rand(d, 5, generator=gen_))  # Exponential(1)
                sw = sw / sw.sum(dim=1, keepdim=True)
                picked = pre.cpu()[idx, torch.arange(d)[:, None]]  # (d, 5)
                b.copy_(-(picked * sw).sum(dim=1).to(b.device))
                x = pre + b
                if i < last:
                    x = self.act(i, x)

    return RealMLP()


# --------------------------------------------------------------------------------------
# the wrapper
# --------------------------------------------------------------------------------------


class RealMLPModel:
    """Same interface as :class:`prc.models.tabm.TabMModel`."""

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
        for key in ("lr_sched", "wd_sched", "p_drop_sched"):
            _sched(str(self.p[key]), 0.0)  # fail at construction on a typo
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
        self.nm_spec: list[tuple[str, str, object]] = []

    def _t(self, name: str, seconds: float) -> None:
        self.timings[name] = self.timings.get(name, 0.0) + seconds

    def _tensors(self, X: pd.DataFrame, dev):
        import torch

        assert self.prep is not None
        z, e, c = self.prep.transform(X)
        return (
            torch.from_numpy(z).to(dev),
            torch.from_numpy(e).to(dev),
            torch.from_numpy(c).to(dev),
        )

    def _autocast(self, dev):
        import contextlib

        import torch

        if dev.type == "cuda" and bool(self.p["amp"]):
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _predict_t(self, T, dev) -> np.ndarray:
        """Standardised-space predictions, clamped to the training range."""
        import torch

        xn, xe, xc = T
        bs = int(self.p["eval_batch_size"])
        out = []
        self.net.eval()
        with torch.no_grad(), self._autocast(dev):
            for i in range(0, xn.shape[0], bs):
                out.append(self.net(xn[i : i + bs], xe[i : i + bs], xc[i : i + bs]).float())
        pred = torch.cat(out) if out else torch.zeros(0)
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
        # standardise under the loss weights, so a weight-0 row has no influence at all
        ww = None if w is None or not np.sum(w) > 0 else np.asarray(w, dtype=np.float64)
        self.y_mu = float(np.average(y, weights=ww))
        self.y_sd = float(np.sqrt(np.average((y - self.y_mu) ** 2, weights=ww))) or 1.0
        z = (y - self.y_mu) / self.y_sd
        z_live = z if ww is None else z[ww > 0]
        Ttr = self._tensors(X, dev)
        ytr = torch.from_numpy(z.astype(np.float32)).to(dev)
        wtr = None
        if w is not None:
            wtr = torch.from_numpy(np.asarray(w, dtype=np.float32)).to(dev)
        Tva = yva = None
        if X_val is not None:
            Tva = self._tensors(X_val, dev)
            yva = np.asarray(y_val, dtype=np.float64)
        self._t("matrix", time.perf_counter() - t0)

        net = _build_net(
            len(self.prep.num_cols),
            int(Ttr[1].shape[1]),
            self.prep.cardinalities(),
            self.p,
            gen,
        ).to(dev)
        if bool(self.p["clamp_output"]) and z_live.size:
            net.y_lo.fill_(float(z_live.min()))
            net.y_hi.fill_(float(z_live.max()))
        n = int(ytr.shape[0])
        n_init = min(int(self.p["init_rows"]), n)
        rows = torch.randperm(n, generator=gen)[:n_init].sort().values.to(dev)
        net.init_from_data(*(t[rows] for t in Ttr), gen)
        self.net = net
        nmd = None
        if float(self.p["nm_dropout"]) > 0.0:
            t1 = time.perf_counter()
            nmd = _NMDrop(self.prep, X, self.p, self.seed, dev)
            self.nm_spec = nmd.spec
            log.info("realmlp seed %d %s", self.seed, nmd.describe())
            self._t("nm_dropout_prep", time.perf_counter() - t1)

        groups = net.param_factors()
        base_lr = float(self.p["learning_rate"])
        opt = torch.optim.Adam(
            [{"params": g[0], "lr": 0.0} for g in groups],
            betas=(float(self.p["beta1"]), float(self.p["beta2"])),
            eps=1e-8,
        )
        wd = float(self.p["weight_decay"])
        bs = min(int(self.p["batch_size"]), n)
        steps_per_epoch = math.ceil(n / bs)
        horizon = int(self.p["max_epochs"]) * steps_per_epoch  # the schedules' t = 1
        cap = int(self.cfg.num_boost_round)
        if X_val is not None:
            cap = min(cap, horizon)
        patience = int(self.cfg.early_stopping_rounds)
        verbose = int(self.p["verbose"])
        s_lr, s_wd, s_dr = (str(self.p[k]) for k in ("lr_sched", "wd_sched", "p_drop_sched"))
        p_drop = float(self.p["p_drop"])

        t0 = time.perf_counter()
        step, epoch = 0, 0
        best, best_state, since = math.inf, None, 0
        xn, xe, xc = Ttr
        while step < cap:
            net.train()
            perm = torch.randperm(n, generator=gen).to(dev)
            drop = nmd.draw() if nmd is not None else None
            loss_sum, n_b = 0.0, 0
            for i in range(0, n, bs):
                if step >= cap:
                    break
                t = step / horizon
                lr_t = base_lr * _sched(s_lr, t)
                wd_t = wd * _sched(s_wd, t)
                b = perm[i : i + bs]
                xb = (xn[b], xe[b], xc[b])
                if drop is not None:
                    xb = nmd.apply(b, drop[b], *xb)
                with self._autocast(dev):
                    pred = net(*xb, p_drop=p_drop * _sched(s_dr, t)).float()
                se = (pred - ytr[b]).pow(2)
                if wtr is None:
                    loss = se.mean()
                else:
                    wb = wtr[b]
                    loss = (wb * se).sum() / wb.sum().clamp_min(1e-12)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                with torch.no_grad():
                    for g, (params, f_lr, f_wd) in zip(opt.param_groups, groups, strict=True):
                        g["lr"] = lr_t * f_lr
                        # pytabkit's decoupled decay, factors squared as its optimizer has them
                        decay = wd_t * f_wd * f_wd * lr_t * f_lr * f_lr
                        if decay:
                            torch._foreach_mul_(params, 1.0 - decay)
                opt.step()
                step += 1
                loss_sum += float(loss.detach())
                n_b += 1
            epoch += 1
            rec = {"epoch": epoch, "step": step, "train_loss": loss_sum / max(n_b, 1)}
            if drop is not None:
                rec["nm_dropped"] = int(drop.sum())
            if Tva is not None:
                p_va = self._predict_t(Tva, dev) * self.y_sd + self.y_mu
                rmse = float(np.sqrt(np.mean((p_va - yva) ** 2)))
                rec["val_rmse"] = rmse
                if rmse < best:
                    best, since = rmse, 0
                    best_state = copy.deepcopy(net.state_dict())
                    self.best_iteration, self.best_epoch = step - 1, epoch
                else:
                    since += 1
            self.history.append(rec)
            if verbose:
                log.info(
                    "realmlp seed %d epoch %d step %d loss %.5f%s",
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
            net.load_state_dict(best_state)
        self._t("train", time.perf_counter() - t0)
        del Ttr, ytr, wtr, Tva, nmd
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
