"""``model.kind: seqenc`` — a sequence encoder over the airport's movement stream (NN5, 2026-09-23).

The set-attention member
(:mod:`prc.models.setenc`) with the context treated as a **time-ordered sequence**:

* the same context sets (:mod:`prc.features.context_sets`, up to ``ctx_k`` movements around the
  departure's takeoff) and the same element attributes, gathered by setenc's ``_Ctx`` — so the
  same legal-inputs-only rule: a neighbour's off-block time and label are never inputs;
* the departure itself is inserted as a **self token** (its row embedding projected to
  ``d_set``) at relative time 0, and the tokens are sorted by movement time relative to the
  departure (``MVT − t1``; stable, so the self token precedes a neighbour at exactly 0);
* a relative-time encoding is added to every token after sorting: the offset itself, the gap
  to the previous token, and ``n_time_freq`` learned-frequency sin / cos pairs of the offset;
* a bidirectional GRU (``n_gru`` layers, ``d_gru`` units per direction) runs over the sorted
  sequence. The backward direction runs over each sequence reversed *within its length*, so
  padding (always sorted to the end) never reaches a valid position in either direction;
* pooling: multi-head attention with the row embedding as the query over the GRU outputs
  (padding masked), concatenated with the GRU output at the self token → Linear → LayerNorm
  → the summary ``s``.

``[h, s]`` goes to the same BatchEnsemble head as setenc. Preprocessing, loss, training loop,
early stopping and step accounting are setenc's (:class:`prc.models.setenc.SetEncModel`), and
the self token means a departure with an empty set is a sequence of length 1. The GRU runs
in fp32 (autocast off) on CUDA.
"""

from __future__ import annotations

import logging
import math

from prc.features.context_sets import MAX_TYPES
from prc.models.setenc import DEFAULTS as SET_DEFAULTS
from prc.models.setenc import N_ELEM_NUM, SetEncModel
from prc.models.tabm import _build_net

log = logging.getLogger("prc.seqenc")

DEFAULTS: dict = {
    **{k: v for k, v in SET_DEFAULTS.items() if k != "n_attn"},
    # the sequence encoder (n_heads: the attention-pooling heads)
    "n_gru": 1,
    "d_gru": 64,
    "n_time_freq": 8,
}


def _build(n_num: int, n_miss: int, cards: list[int], p: dict):
    import torch
    from torch import nn

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

    d_set, d_row, d_gru = int(p["d_set"]), int(p["d_row"]), int(p["d_gru"])
    n_tf = int(p["n_time_freq"])
    drop = float(p["dropout"])

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
            dc = int(p["d_ctx_cat"])
            self.e_phase = nn.Embedding(2, dc)
            self.e_wake = nn.Embedding(8, dc)
            self.e_type = nn.Embedding(MAX_TYPES + 2, dc)
            self.elem = nn.Sequential(
                nn.Linear(N_ELEM_NUM + 3 * dc, d_set), nn.ReLU(), nn.Linear(d_set, d_set)
            )
            self.self_tok = nn.Linear(d_row, d_set)
            # relative time: log-spaced initial periods of 5 .. 120 min (offsets are in 30 min)
            self.freq = nn.Parameter(
                torch.exp(torch.linspace(math.log(30 / 120), math.log(30 / 5), max(n_tf, 1)))[
                    :n_tf
                ].clone()
            )
            self.time = nn.Linear(2 + 2 * n_tf, d_set)
            self.ln_in = nn.LayerNorm(d_set)
            self.gru_f = nn.ModuleList()
            self.gru_b = nn.ModuleList()
            for i in range(int(p["n_gru"])):
                d_i = d_set if i == 0 else 2 * d_gru
                self.gru_f.append(nn.GRU(d_i, d_gru, batch_first=True))
                self.gru_b.append(nn.GRU(d_i, d_gru, batch_first=True))
            self.out = nn.Linear(2 * d_gru, d_set)
            self.q = nn.Linear(d_row, d_set)
            self.pool = nn.MultiheadAttention(
                d_set, int(p["n_heads"]), dropout=0.0, batch_first=True
            )
            self.mix = nn.Linear(2 * d_set, d_set)
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

        def sequence(self, h, en, ec, valid):
            """Sorted tokens (B, K+1, d_set) fp32, their validity and the self position."""
            vm = valid[..., None]
            en = torch.where(vm, en, torch.zeros_like(en))
            ec = torch.where(vm, ec, torch.zeros_like(ec))
            e = torch.cat(
                [en, self.e_phase(ec[..., 0]), self.e_wake(ec[..., 1]), self.e_type(ec[..., 2])],
                dim=-1,
            )
            e = self.elem(e).float()
            B = e.shape[0]
            tok = torch.cat([self.self_tok(h).float()[:, None, :], e], dim=1)
            ok = torch.cat([valid.new_ones(B, 1), valid], dim=1)
            dt = torch.cat([en.new_zeros(B, 1), en[..., 0]], dim=1).float()  # (MVT - t1) / 30
            key = torch.where(ok, dt, torch.full_like(dt, math.inf))
            order = torch.sort(key, dim=1, stable=True).indices
            tok = torch.gather(tok, 1, order[..., None].expand_as(tok))
            ok = torch.gather(ok, 1, order)
            dt = torch.where(ok, torch.gather(dt, 1, order), torch.zeros_like(dt))
            gap = torch.cat([dt.new_zeros(B, 1), dt[:, 1:] - dt[:, :-1]], dim=1)
            gap = torch.where(ok, gap, torch.zeros_like(gap))
            ang = 2 * math.pi * dt[..., None] * self.freq
            tf = torch.cat([dt[..., None], gap[..., None], torch.sin(ang), torch.cos(ang)], -1)
            tok = self.ln_in(tok + self.time(tf))
            tok = torch.where(ok[..., None], tok, torch.zeros_like(tok))
            self_pos = (order == 0).float().argmax(dim=1)
            return tok, ok, self_pos

        def encode(self, tok, ok):
            """The bidirectional GRU; the backward pass sees each sequence reversed within its
            own length, so the trailing padding never reaches a valid position."""
            L = tok.shape[1]
            n = ok.sum(dim=1, keepdim=True)
            t = torch.arange(L, device=tok.device)[None, :]
            rev = torch.where(t < n, n - 1 - t, t)  # an involution
            x = tok
            for gf, gb in zip(self.gru_f, self.gru_b, strict=True):
                f = gf(x)[0]
                xr = torch.gather(x, 1, rev[..., None].expand_as(x))
                b = gb(xr)[0]
                b = torch.gather(b, 1, rev[..., None].expand_as(b))
                x = torch.cat([f, b], dim=-1)
                x = torch.where(ok[..., None], x, torch.zeros_like(x))
            return self.out(x)

        def forward(self, xn, xm, xc, en, ec, valid):
            h = self.row(self.row_input(xn, xm, xc))
            tok, ok, self_pos = self.sequence(h, en, ec, valid)
            with torch.autocast(device_type=tok.device.type, enabled=False):
                z = self.encode(tok, ok)
                q = self.q(h.float())[:, None, :]
                a, _ = self.pool(q, z, z, key_padding_mask=~ok, need_weights=False)
                zs = torch.gather(z, 1, self_pos[:, None, None].expand(-1, 1, z.shape[-1]))
                s = self.ln_s(self.mix(torch.cat([a[:, 0], zs[:, 0]], dim=-1)))
            x = torch.cat([h, s.to(h.dtype)], dim=1)
            for blk in self.blocks:
                x = self.drop(torch.relu(blk(x)))
            return (x * self.head_w).sum(-1) + self.head_b

    return Net()


class SeqEncModel(SetEncModel):
    """Same interface as :class:`prc.models.setenc.SetEncModel` (``X`` must carry
    ``ctx_key``); only the network differs."""

    defaults = DEFAULTS
    name = "seqenc"

    def _make_net(self, n_num: int, n_miss: int, cards: list[int]):
        return _build(n_num, n_miss, cards, self.p)
