"""Context-aware level-2 combiners over member OOF, scored by nested leave-one-month-out.

The current blend (`blend_v7l`) is one global sum-to-one weight vector (``scripts/blend_fit.py``,
``nnls_global`` / ``ex_ro``). The members differ by context (airport, NM record present, the
delay ``C = MVT − SCHED``, hour, the corruption probability, member disagreement), so this script
asks whether weights that move with the context beat the global vector *honestly*.

Protocol, identical to ``blend_fit.py``:

- Members are aligned **on ``MVT_ID_mvt``**; the fitting input is each member's ``pred``.
- Outer loop = the twelve 2025 months. For month ``m`` the combiner is fitted on the other
  eleven months' OOF and applied to month ``m``. Every hyper-parameter is chosen *inside* the
  eleven months (inner leave-one-month-out, an inner split, or inner early stopping), never on
  month ``m``.
- The test prediction uses a combiner fitted on all twelve months, hyper-parameters chosen the
  same way on the twelve.
- Day-rollover rows (``|y − 86,400| < 3,600``) are dropped from every fitting and every inner
  selection set, exactly as ``--emit-mode ex_ro``.
- The per-seed columns ``pred_s<seed>`` are the same fitted combiner applied to each member's
  per-seed predictions (context features and the correction of family ``gbdt`` come from the
  seed-averaged ``pred``), so ``scripts/paired_delta.py`` pairs seeds as for any blend.

Families (``--family``), from low to high capacity:

``nnls``  one global simplex vector (reproduces ``blend_v7l``; the sanity check);
``cell``  simplex weights per cell ``airport × has_NM × delay band`` (C < 30 min / 30–60 min /
          1–2 h / > 2 h), shrunk hierarchically: global → ``has_NM × band`` → cell. Shrinkage
          is a ridge pull toward the parent worth ``n0`` pseudo-rows; ``n0`` (including
          ``inf`` = the global vector) is chosen by inner leave-one-month-out. Cells or parents
          under ``--min-cell-rows`` rows take the parent's weights. ``--cell-keys`` may add
          ``adsb``, the ADS-B state of the row (no ground track / ground track without a seen
          pushback / pushback seen at the stand); the
          hierarchy is then global → ``has_NM × band`` → ``has_NM × band × adsb`` → cell.
          ``nmrev`` (the NM revision band of ``AOBT_3 − EOBT_1``: none / early < −15 min /
          on time / late ≥ +15 min) and ``deice`` (``winter_ops.wo_deicing_likely``) are added
          the same way; with several extra keys the middle level is ``has_NM × band ×`` their
          joint level (2026-09-26);
``gate``  softmax gating: ``w(z) = softmax(W z + b)``, ``pred = Σ w_i(z) x_i``, ``z`` the
          standardized context features; ``b`` starts at ``log`` of the global NNLS weights and
          ``W`` is ridge-penalised; the penalty is chosen on an inner two-way month split
          (both directions) on a row subsample;
``gbdt``  the global NNLS blend plus a depth-3 LightGBM (CPU) fitted to its residual, with the
          member predictions, their spread and the context as inputs; rounds by inner early
          stopping on a quarter of the fitting months, then a refit on all of them.

Output: ``outputs/runs/<out>/{oof,test_pred}.parquet`` shaped like ``blend_fit.py --write-oof``
/ ``apply`` (``oof``: ``MVT_ID_mvt, airport, month, fold, TAXITIME_SEC_mvt, pred, pred_s*``;
``test_pred``: ``MVT_ID_mvt, airport, month, pred``) and ``stack.json`` (members, digests,
settings, per-fold choices, summary). Not an experiment; no registry row.

    uv run python scripts/stack_l2.py <member runs ...> --family cell --out stack_l2_cell \
        --ref blend_v7l
    uv run python scripts/paired_delta.py blend_v7l stack_l2_cell --ex-rollovers --bootstrap 2000
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import resource
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from prc import paths

SPIKE = (1194, 1206)
ROLLOVER_S, ROLLOVER_WINDOW_S = 86_400.0, 3_600.0
#: delay-band edges on ``C = MVT − SCHED`` in seconds: < 30 min, 30–60 min, 1–2 h, > 2 h
BAND_EDGES = (1_800.0, 3_600.0, 7_200.0)
N_BANDS = len(BAND_EDGES) + 1
FAMILIES = ("nnls", "cell", "gate", "gbdt")
CELL_N0_GRID = (0.0, 30.0, 100.0, 300.0, 1e3, 3e3, 1e4, 3e4, 1e5, float("inf"))
GATE_LAMBDA_GRID = (1e-4, 1e-3, 1e-2, 1e-1)
GATE_INNER_MAXITER, GATE_MAXITER = 100, 200
GBDT_PARAMS = {
    "objective": "regression",
    "learning_rate": 0.05,
    "num_leaves": 8,
    "max_depth": 3,
    "min_data_in_leaf": 2000,
    "lambda_l2": 100.0,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "max_bin": 255,
    "seed": 0,
    "verbose": -1,
}
GBDT_MAX_ROUNDS, GBDT_PATIENCE = 800, 50


def _blend_fit():
    """``scripts/blend_fit.py`` as a module: the exact simplex solver has one home."""
    src = Path(__file__).resolve().parent / "blend_fit.py"
    spec = importlib.util.spec_from_file_location("blend_fit", src)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("blend_fit", mod)
    spec.loader.exec_module(mod)
    return mod


BF = _blend_fit()
simplex_lsq = BF.simplex_lsq


def log(*a: object) -> None:
    print(f"[{datetime.now(UTC):%H:%M:%S}]", *a, flush=True)


# --- loading --------------------------------------------------------------------------


@dataclass
class Data:
    ids: np.ndarray
    X: np.ndarray  # (n, k) float64, member `pred`
    seeds: dict[str, np.ndarray]  # seed column -> (n, k) float32
    y: np.ndarray | None
    month: np.ndarray  # int index into month_names
    airport: np.ndarray  # int index into AIRPORTS
    has: np.ndarray
    spike: np.ndarray
    c_sched: np.ndarray  # MVT − SCHED, seconds
    hour: np.ndarray
    p_def: np.ndarray  # mean over members of p_sched_default (0 when absent)
    frame: pl.DataFrame  # key columns for the output file
    adsb: np.ndarray | None = None  # ADS-B state (ADSB_STATES index), only when a key needs it
    nmrev: np.ndarray | None = None  # NM revision band (NMREV_LEVELS index), key ``nmrev`` only
    deice: np.ndarray | None = None  # de-icing likely (0 / 1), key ``deice`` only


def load(
    dirs: list[Path],
    which: str,
    airports: list[str] | None = None,
    adsb: bool = False,
    nmrev: bool = False,
    deice: bool = False,
) -> tuple[Data, list[str], list[str]]:
    """Members joined on ``MVT_ID_mvt`` (inner, left order), plus the context columns
    (``adsb``: also the ADS-B state of every row, read through the feature blocks; ``nmrev``:
    the band of ``AOBT_3 − EOBT_1`` from the movement table; ``deice``: the ``winter_ops``
    de-icing flag, read from its materialised block)."""
    base = ["MVT_ID_mvt", "airport", "month", "fold", "TAXITIME_SEC_mvt"]
    seed_cols: list[str] | None = None
    frames = []
    for i, d in enumerate(dirs):
        lf = pl.scan_parquet(d / f"{which}.parquet")
        have = lf.collect_schema().names()
        s = sorted(c for c in have if c.startswith("pred_s"))
        seed_cols = s if seed_cols is None else [c for c in seed_cols if c in s]
        keep = [pl.col("pred").cast(pl.Float64).alias(f"p__{i}")]
        keep += [pl.col(c).cast(pl.Float32).alias(f"{c}__{i}") for c in s]
        pdef = (
            pl.col("p_sched_default").cast(pl.Float32)
            if "p_sched_default" in have
            else pl.lit(None, dtype=pl.Float32)
        )
        keep.append(pdef.alias(f"pdef__{i}"))
        if i == 0:
            keep = [pl.col(c) for c in base if c in have] + keep
        else:
            keep = [pl.col("MVT_ID_mvt"), *keep]
        frames.append(lf.select(keep))
    lf = frames[0]
    for f in frames[1:]:
        lf = lf.join(f, on="MVT_ID_mvt", how="inner", maintain_order="left")
    mv_cols = ["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt", "SCHED_TIME_UTC_mvt"]
    first = frames[0].collect_schema().names()
    mv_cols += [c for c in ("airport", "month") if c not in first]
    if nmrev:
        mv_cols.append("EOBT_1_flt")
    mv = pl.scan_parquet(paths.INTERIM / "movements.parquet").select(mv_cols)
    if nmrev:
        mv = mv.with_columns(
            (pl.col("AOBT_3_flt") - pl.col("EOBT_1_flt"))
            .dt.total_seconds()
            .cast(pl.Float64)
            .alias("_a_minus_eobt")
        ).drop("EOBT_1_flt")
    k = len(dirs)
    df = (
        lf.join(mv, on="MVT_ID_mvt", how="left", maintain_order="left")
        .with_columns(
            (pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds().alias("d_aobt"),
            (pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt"))
            .dt.total_seconds()
            .cast(pl.Float64)
            .fill_null(0.0)
            .alias("c_sched"),
            pl.col("MVT_TIME_UTC_mvt").dt.hour().cast(pl.Float64).fill_null(12.0).alias("hour"),
            pl.mean_horizontal([pl.col(f"pdef__{i}") for i in range(k)])
            .fill_null(0.0)
            .alias("p_def"),
        )
        .with_columns(
            pl.col("AOBT_3_flt").is_not_null().alias("has_aobt"),
            pl.col("d_aobt").is_between(*SPIKE).fill_null(False).alias("on_spike"),
        )
        .drop([f"pdef__{i}" for i in range(k)], "AOBT_3_flt", "SCHED_TIME_UTC_mvt", "d_aobt")
        .collect()
    )
    if df.height == 0:
        raise SystemExit(f"members share no MVT_ID_mvt in {which}")
    X = np.column_stack([df[f"p__{i}"].to_numpy() for i in range(k)]).astype(np.float64)
    seeds = {
        sc: np.column_stack([df[f"{sc}__{i}"].to_numpy() for i in range(k)]).astype(np.float32)
        for sc in (seed_cols or [])
    }
    month_names = sorted(set(df["month"].to_list()))
    airports = airports or sorted(set(df["airport"].to_list()))
    unknown = set(df["airport"].to_list()) - set(airports)
    if unknown:
        raise SystemExit(f"airports not seen in the OOF: {sorted(unknown)}")
    y = (
        df["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy()
        if "TAXITIME_SEC_mvt" in df.columns
        else None
    )
    keys = [c for c in base if c in df.columns]
    data = Data(
        ids=df["MVT_ID_mvt"].to_numpy(),
        X=X,
        seeds=seeds,
        y=y,
        month=np.searchsorted(np.array(month_names), df["month"].to_numpy()),
        airport=np.searchsorted(np.array(airports), df["airport"].to_numpy()),
        has=df["has_aobt"].to_numpy(),
        spike=df["on_spike"].to_numpy(),
        c_sched=df["c_sched"].to_numpy(),
        hour=df["hour"].to_numpy(),
        p_def=df["p_def"].cast(pl.Float64).to_numpy(),
        frame=df.select(keys),
    )
    if adsb:
        data.adsb = adsb_state_of(data.ids)
    if nmrev:
        data.nmrev = nmrev_band_of(df["_a_minus_eobt"].to_numpy())
    if deice:
        data.deice = deice_flag_of(data.ids)
    return data, month_names, airports


#: ADS-B state codes (``cell_of`` key ``adsb``)
ADSB_STATES = ("no ground track", "ground, no push", "push seen")
N_ADSB = len(ADSB_STATES)
ADSB_STATE_DEF = (
    "push seen = adsb2_saw_stationary > 0 and adsb2_first_move_dist_stand_m < 100; "
    "ground, no push = adsb_has_gnd > 0 otherwise; no ground track = the rest (nulls included)"
)


def adsb_state_of(ids: np.ndarray) -> np.ndarray:
    """ADS-B state per ``MVT_ID_mvt`` from the ``adsb_ground`` / ``adsb_ground_v2`` blocks.

    Label-free and computable on the ranking rows (both blocks anchor on ``MVT_TIME``). Rows the
    archive does not cover (or a missing archive) are "no ground track"."""
    from prc.features.adsb_ground import adsb_ground
    from prc.features.adsb_ground_v2 import adsb_ground_v2

    base = pl.DataFrame({"MVT_ID_mvt": ids})
    a = adsb_ground(base).select("MVT_ID_mvt", "adsb_has_gnd")
    b = adsb_ground_v2(base).select(
        "MVT_ID_mvt", "adsb2_saw_stationary", "adsb2_first_move_dist_stand_m"
    )
    f = base.join(a, on="MVT_ID_mvt", how="left", maintain_order="left").join(
        b, on="MVT_ID_mvt", how="left", maintain_order="left"
    )
    gnd = pl.col("adsb_has_gnd").fill_null(0) > 0
    push = (pl.col("adsb2_saw_stationary").fill_null(0) > 0) & (
        pl.col("adsb2_first_move_dist_stand_m").fill_null(float("inf")) < 100
    )
    state = pl.when(push).then(2).when(gnd).then(1).otherwise(0).cast(pl.Int64)
    return f.select(state.alias("s"))["s"].to_numpy()


#: NM revision band (``cell_of`` key ``nmrev``) of ``AOBT_3 − EOBT_1``, seconds. From the
#: 2026-09-26 ordinary-residual study (§6): within ±15 min is calm at LTFM / LFPG / LIRF / EDDF,
#: 15–60 min late is high-variance (sd ratio 1.4–1.7; 28 / 48 / 33 % of ordinary SSE).
NMREV_EDGES = (-900.0, 900.0)
NMREV_LEVELS = ("none", "early", "on time", "late")
N_NMREV = len(NMREV_LEVELS)
NMREV_DEF = (
    "d = AOBT_3 - EOBT_1 (s); none = d null (no NM record or no EOBT_1); early = d < -900; "
    "on time = -900 <= d < 900; late = d >= 900"
)


def nmrev_band_of(a_minus_eobt: np.ndarray) -> np.ndarray:
    """0 none (null), 1 early (< −15 min), 2 on time, 3 late (≥ +15 min)."""
    x = np.asarray(a_minus_eobt, dtype=np.float64)
    band = np.searchsorted(np.array(NMREV_EDGES), x, side="right") + 1
    return np.where(np.isfinite(x), band, 0).astype(np.int64)


#: de-icing flag (``cell_of`` key ``deice``)
DEICE_LEVELS = ("no", "de-icing likely")
N_DEICE = len(DEICE_LEVELS)
DEICE_DEF = (
    "winter_ops.wo_deicing_likely > 0 (METAR: frozen precipitation in the last 3 h, freezing "
    "fog, fog at <= 0 C or rain at <= 3 C, at AOBT_3 else SCHED); null -> 0"
)


def deice_flag_of(ids: np.ndarray) -> np.ndarray:
    """``wo_deicing_likely`` per ``MVT_ID_mvt`` from the materialised ``winter_ops`` block.

    Label-free (METAR at the NM off-block estimate, else the schedule). The parquet is found as
    ``prc.cache.block_inputs`` finds it: ``winter_ops__<current source hash>__*.parquet``,
    exactly one (otherwise build it with ``uv run prc features build winter_ops``). Rows the
    block does not cover count as 0."""
    from prc.features.registry import REGISTRY

    sh = REGISTRY["winter_ops"].source_hash()
    found = sorted(paths.FEATURES.glob(f"winter_ops__{sh}__*.parquet"))
    if len(found) != 1:
        raise SystemExit(
            f"deice: need exactly one winter_ops__{sh}__*.parquet in {paths.FEATURES}, found "
            f"{[f.name for f in found]}; run `uv run prc features build winter_ops`"
        )
    blk = pl.read_parquet(found[0], columns=["MVT_ID_mvt", "wo_deicing_likely"])
    f = pl.DataFrame({"MVT_ID_mvt": ids}).join(
        blk, on="MVT_ID_mvt", how="left", maintain_order="left"
    )
    return (f["wo_deicing_likely"].fill_null(0) > 0).cast(pl.Int64).to_numpy()


# --- context --------------------------------------------------------------------------


def band_of(c_sched: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.array(BAND_EDGES), c_sched, side="right")


CELL_KEYS = ("airport", "has", "band")
#: opt-in keys beyond the default, in encoding order, with their number of levels
EXTRA_CELL_KEYS: dict[str, int] = {"adsb": N_ADSB, "nmrev": N_NMREV, "deice": N_DEICE}
#: every key ``--cell-keys`` accepts; the extra keys are opt-in and not part of the default
KNOWN_CELL_KEYS = (*CELL_KEYS, *EXTRA_CELL_KEYS)


def n_adsb(keys: tuple[str, ...]) -> int:
    return N_ADSB if "adsb" in keys else 1


def n_extra(keys: tuple[str, ...]) -> int:
    """Joint number of levels of the extra keys in ``keys`` (1 without any)."""
    return int(np.prod([v for k, v in EXTRA_CELL_KEYS.items() if k in keys], dtype=np.int64))


def cell_of(d: Data, keys: tuple[str, ...] = CELL_KEYS) -> np.ndarray:
    """``(airport * 2 + has) * 4 + band``, then for each extra key present, in the order of
    ``EXTRA_CELL_KEYS`` (adsb, nmrev, deice), ``cell * levels + value``. The default is
    ``airport * 8 + has * 4 + band`` and its parent ``has_NM × band`` is ``cell % 8``; with
    ``adsb`` alone it is ``cell * 3 + adsb`` as before the other keys existed.

    A key left out of ``keys`` is held at 0, which collapses the cells along it (an attribution
    diagnostic: ``has,band`` = no airport split, ``airport`` = no lane / delay split)."""
    air = d.airport if "airport" in keys else np.zeros_like(d.airport)
    has = d.has.astype(np.int64) if "has" in keys else np.zeros(len(d.has), dtype=np.int64)
    band = band_of(d.c_sched) if "band" in keys else np.zeros(len(d.has), dtype=np.int64)
    cell = (air * 2 + has) * N_BANDS + band
    for k, levels in EXTRA_CELL_KEYS.items():
        if k in keys:
            v = getattr(d, k)
            assert v is not None, f"load(..., {k}=True) is needed for the {k} key"
            cell = cell * levels + np.asarray(v).astype(np.int64)
    return cell


def cell_levels(keys: tuple[str, ...], n_cells: int) -> list[np.ndarray]:
    """The shrinkage levels between the global vector and the cells, coarse to fine, each as a
    map cell index → group index at that level.

    Default: ``[has_NM × band]`` (``cell % 8``). With extra keys (``adsb``, ``nmrev``,
    ``deice``): ``[has_NM × band, has_NM × band × extra]`` where ``extra`` is the joint level of
    the extra keys present, so an airport's cell is pulled toward the same extra level pooled
    over airports, which is pulled toward its lane / delay parent."""
    idx = np.arange(n_cells)
    a = n_extra(keys)
    lane_x_extra = idx % (2 * N_BANDS * a)
    if a == 1:
        return [lane_x_extra]
    return [lane_x_extra // a, lane_x_extra]


def gate_features(d: Data, n_airports: int) -> tuple[np.ndarray, list[str]]:
    band = band_of(d.c_sched)
    has = d.has.astype(np.float32)
    cols, names = [], []
    for a in range(n_airports):
        cols.append((d.airport == a).astype(np.float32))
        names.append(f"ap{a}")
    cols.append(has)
    names.append("has_nm")
    for b in range(N_BANDS):
        cols.append((band == b).astype(np.float32))
        names.append(f"band{b}")
        cols.append(((band == b) & ~d.has).astype(np.float32))
        names.append(f"band{b}_no_nm")
    cols.append(np.log1p(np.clip(d.c_sched, 0, None) / 60.0).astype(np.float32))
    names.append("log_c_min")
    ang = 2 * np.pi * d.hour / 24.0
    cols += [np.sin(ang).astype(np.float32), np.cos(ang).astype(np.float32)]
    names += ["hour_sin", "hour_cos"]
    cols.append(d.p_def.astype(np.float32))
    names.append("p_sched_default")
    cols.append(np.log1p(d.X.std(axis=1)).astype(np.float32))
    names.append("log_spread")
    cols.append(d.spike.astype(np.float32))
    names.append("on_spike")
    return np.column_stack(cols), names


def gbdt_features(d: Data, base: np.ndarray) -> tuple[np.ndarray, list[str]]:
    k = d.X.shape[1]
    cols = [d.X[:, i] for i in range(k)] + [
        base,
        d.X.std(axis=1),
        d.X.max(axis=1) - d.X.min(axis=1),
        d.airport.astype(np.float64),
        d.has.astype(np.float64),
        d.c_sched,
        d.hour,
        d.p_def,
        d.spike.astype(np.float64),
    ]
    names = [f"m{i}" for i in range(k)] + [
        "base",
        "spread_sd",
        "spread_range",
        "airport",
        "has_nm",
        "c_sched",
        "hour",
        "p_sched_default",
        "on_spike",
    ]
    return np.column_stack(cols).astype(np.float32), names


# --- cell statistics (nnls / cell) ----------------------------------------------------


@dataclass
class CellStats:
    S: np.ndarray  # (n_months, n_cells, k, k)
    t: np.ndarray  # (n_months, n_cells, k)
    yty: np.ndarray  # (n_months, n_cells)
    n: np.ndarray  # (n_months, n_cells)


def cell_stats(
    X: np.ndarray,
    y: np.ndarray,
    month: np.ndarray,
    cell: np.ndarray,
    n_months: int,
    n_cells: int,
    center: float,
) -> CellStats:
    """Gram statistics of the centered members against the centered target per (month, cell).

    With sum-to-one weights ``(X − c)w − (y − c) = Xw − y``, so centering changes nothing but
    the conditioning. ``center`` must not read a label (the mean member prediction is used):
    a label mean would let the held-out month's labels move its own fit in the last bits."""
    Xc, yc = X - center, y - center
    key = month * n_cells + cell
    m = n_months * n_cells
    k = X.shape[1]
    S = np.zeros((m, k, k))
    for i in range(k):
        for j in range(i, k):
            v = np.bincount(key, weights=Xc[:, i] * Xc[:, j], minlength=m)
            S[:, i, j] = v
            S[:, j, i] = v
    t = np.stack([np.bincount(key, weights=Xc[:, i] * yc, minlength=m) for i in range(k)], axis=1)
    yty = np.bincount(key, weights=yc * yc, minlength=m)
    n = np.bincount(key, minlength=m).astype(float)
    return CellStats(
        S.reshape(n_months, n_cells, k, k),
        t.reshape(n_months, n_cells, k),
        yty.reshape(n_months, n_cells),
        n.reshape(n_months, n_cells),
    )


def shrunk_simplex(
    G: np.ndarray, b: np.ndarray, n: float, prior: np.ndarray, n0: float, min_rows: int
) -> np.ndarray:
    """``argmin w'Gw − 2b'w + λ‖w − prior‖²`` on the simplex, ``λ`` worth ``n0`` rows.

    ``λ = n0 · tr(G) / (k · n)``: the prior carries the weight of ``n0`` average rows of this
    cell, so a delayed-flight cell (large per-row variance) and an ordinary one are shrunk on
    the same scale. ``n0 = inf`` or ``n < min_rows`` returns the prior itself."""
    if not np.isfinite(n0) or n < min_rows:
        return prior
    k = len(b)
    lam = n0 * float(np.trace(G)) / (k * n)
    return simplex_lsq(G + lam * np.eye(k), b + lam * prior)


def cell_weights(
    st: CellStats,
    use: np.ndarray,
    n0: float,
    min_rows: int,
    levels: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(w_global, w_parent (8, k), w_cell (n_cells, k))`` fitted on the months in ``use``.

    Each level of ``levels`` (``cell_levels``; default: the ``has_NM × band`` parent alone) is
    shrunk toward the level above it, the first toward the global vector, and the cells toward
    the last level. ``w_parent`` is the first level (``has_NM × band``)."""
    S, t, n = st.S[use].sum(0), st.t[use].sum(0), st.n[use].sum(0)
    n_cells, k = t.shape
    wg = simplex_lsq(S.sum(0), t.sum(0))
    if levels is None:
        levels = [np.arange(n_cells) % (2 * N_BANDS)]
    prior_of_group = [wg]  # the prior of each group of the current level, via its first cell
    prior_map = np.zeros(n_cells, dtype=np.int64)  # cell -> index into prior_of_group
    found = []
    for par_of in levels:
        n_par = int(par_of.max()) + 1
        wl = np.zeros((n_par, k))
        for p in range(n_par):
            sel = par_of == p
            prior = prior_of_group[prior_map[int(np.argmax(sel))]]
            wl[p] = shrunk_simplex(S[sel].sum(0), t[sel].sum(0), n[sel].sum(), prior, n0, min_rows)
        found.append(wl)
        prior_of_group = list(wl)
        prior_map = par_of
    wc = np.zeros((n_cells, k))
    for c in range(n_cells):
        wc[c] = shrunk_simplex(S[c], t[c], n[c], prior_of_group[prior_map[c]], n0, min_rows)
    return wg, found[0], wc


def cell_sse(st: CellStats, m: int, wc: np.ndarray) -> float:
    S, t, yty = st.S[m], st.t[m], st.yty[m]
    return float(np.einsum("ci,cij,cj->", wc, S, wc) - 2 * np.einsum("ci,ci->", wc, t) + yty.sum())


def choose_n0(
    st: CellStats,
    fit_months: np.ndarray,
    min_rows: int,
    levels: list[np.ndarray] | None = None,
) -> tuple[float, dict]:
    """Inner leave-one-month-out over ``fit_months`` only."""
    n_months = st.n.shape[0]
    scores = {}
    for n0 in CELL_N0_GRID:
        tot = 0.0
        for j in fit_months:
            use = np.zeros(n_months, dtype=bool)
            use[fit_months] = True
            use[j] = False
            _, _, wc = cell_weights(st, use, n0, min_rows, levels=levels)
            tot += cell_sse(st, j, wc)
        scores[n0] = tot
    best = min(scores, key=lambda v: scores[v])
    return best, {str(k): round(v, 1) for k, v in scores.items()}


# --- gating ---------------------------------------------------------------------------


def gate_fit(
    Z: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    w0: np.ndarray,
    lam: float,
    maxiter: int = GATE_MAXITER,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """``(W (d, k), b (k,), info)`` minimising ``mean((Σ softmax(zW + b)_i x_i − y)²) / var(y)
    + lam · ‖W‖²`` by L-BFGS; ``b`` starts at ``log(w0)``, ``W`` at zero."""
    from scipy.optimize import minimize

    n, d = Z.shape
    k = X.shape[1]
    scale = float(y.var()) * n or 1.0

    def f(theta: np.ndarray) -> tuple[float, np.ndarray]:
        W = theta[: d * k].reshape(d, k)
        b = theta[d * k :]
        U = Z @ W.astype(np.float32) + b
        U = U - U.max(axis=1, keepdims=True)
        E = np.exp(U, dtype=np.float64)
        Sm = E / E.sum(axis=1, keepdims=True)
        p = (Sm * X).sum(axis=1)
        r = p - y
        loss = float(r @ r) / scale + lam * float((W * W).sum())
        G = Sm * (X - p[:, None]) * (2 * r / scale)[:, None]
        gW = (Z.T @ G.astype(np.float32)).astype(np.float64) + 2 * lam * W
        gb = G.sum(axis=0)
        return loss, np.concatenate([gW.ravel(), gb])

    b0 = np.log(np.clip(w0, 1e-3, None))
    theta0 = np.concatenate([np.zeros(d * k), b0 - b0.mean()])
    res = minimize(
        f, theta0, jac=True, method="L-BFGS-B", options={"maxiter": maxiter, "gtol": 1e-10}
    )
    W = res.x[: d * k].reshape(d, k)
    return W, res.x[d * k :], {"nit": int(res.nit), "loss": float(res.fun)}


def gate_weights(Z: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    U = Z @ W.astype(np.float32) + b
    U = U - U.max(axis=1, keepdims=True)
    E = np.exp(U, dtype=np.float64)
    return E / E.sum(axis=1, keepdims=True)


def standardize(Z: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = Z[rows].mean(axis=0)
    sd = Z[rows].std(axis=0)
    sd[sd < 1e-6] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)


# --- the combiner, fitted on a set of months ------------------------------------------


@dataclass
class Setup:
    family: str
    X: np.ndarray
    y: np.ndarray
    month: np.ndarray
    fit_ok: np.ndarray  # not a rollover
    cell: np.ndarray
    n_cells: int
    min_rows: int
    Z: np.ndarray | None = None
    st: CellStats | None = None
    gate_subsample: float = 0.2
    threads: int = 8
    feat_names: list[str] = field(default_factory=list)
    levels: list[np.ndarray] | None = None  # cell_levels; None = the default has_NM x band


def fit_combiner(s: Setup, fit_months: np.ndarray) -> dict:
    """Everything needed to predict any row, fitted on ``fit_months`` (rollovers excluded)."""
    n_months = int(s.month.max()) + 1
    use = np.zeros(n_months, dtype=bool)
    use[fit_months] = True
    assert s.st is not None
    wg, _, _ = cell_weights(s.st, use, float("inf"), s.min_rows)
    out: dict = {"w_global": wg}
    if s.family == "nnls":
        return out
    if s.family == "cell":
        n0, scores = choose_n0(s.st, fit_months, s.min_rows, levels=s.levels)
        _, wp, wc = cell_weights(s.st, use, n0, s.min_rows, levels=s.levels)
        return out | {"n0": n0, "inner_sse": scores, "w_parent": wp, "w_cell": wc}
    rows = np.isin(s.month, fit_months) & s.fit_ok
    if s.family == "gate":
        assert s.Z is not None
        mu, sd = standardize(s.Z, rows)
        rng = np.random.default_rng(0)
        sub = rows & (rng.random(len(s.y)) < s.gate_subsample)
        half_a, half_b = fit_months[0::2], fit_months[1::2]
        scores = {}
        for lam in GATE_LAMBDA_GRID:
            tot = 0.0
            for tr_m, va_m in ((half_a, half_b), (half_b, half_a)):
                tr = sub & np.isin(s.month, tr_m)
                va = sub & np.isin(s.month, va_m)
                use_tr = np.zeros(n_months, dtype=bool)
                use_tr[tr_m] = True
                w0 = cell_weights(s.st, use_tr, float("inf"), s.min_rows)[0]
                Zt = (s.Z[tr] - mu) / sd
                W, b, _ = gate_fit(Zt, s.X[tr], s.y[tr], w0, lam, maxiter=GATE_INNER_MAXITER)
                Wv = gate_weights((s.Z[va] - mu) / sd, W, b)
                tot += float((((Wv * s.X[va]).sum(axis=1) - s.y[va]) ** 2).sum())
            scores[lam] = tot
        lam = min(scores, key=lambda v: scores[v])
        Zt = (s.Z[rows] - mu) / sd
        W, b, info = gate_fit(Zt, s.X[rows], s.y[rows], wg, lam)
        return out | {
            "lambda": lam,
            "inner_sse": {str(k): round(v, 1) for k, v in scores.items()},
            "mu": mu,
            "sd": sd,
            "W": W,
            "b": b,
            "opt": info,
        }
    if s.family == "gbdt":
        import lightgbm as lgb

        assert s.Z is not None  # the gbdt feature matrix
        base = s.X @ wg
        resid = s.y - base
        val_m = fit_months[1::4]
        tr = rows & ~np.isin(s.month, val_m)
        va = rows & np.isin(s.month, val_m)
        params = GBDT_PARAMS | {"num_threads": s.threads}
        cat = [s.Z.shape[1] - 6]  # the airport column (see gbdt_features)
        dtr = lgb.Dataset(s.Z[tr], resid[tr], categorical_feature=cat, free_raw_data=True)
        dva = lgb.Dataset(s.Z[va], resid[va], reference=dtr, categorical_feature=cat)
        es = lgb.train(
            params,
            dtr,
            num_boost_round=GBDT_MAX_ROUNDS,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(GBDT_PATIENCE, verbose=False)],
        )
        best = max(int(es.best_iteration or 0), 1)
        # the inner early-stopping run already beat "no correction"? if not, keep none
        base_va = float((resid[va] ** 2).sum())
        es_va = float(((resid[va] - es.predict(s.Z[va], num_iteration=best)) ** 2).sum())
        del dtr, dva, es
        n_rounds = round(best * len(fit_months) / max(len(fit_months) - len(val_m), 1))
        if es_va >= base_va:
            return out | {"booster": None, "rounds": 0, "inner_gain": es_va - base_va}
        dall = lgb.Dataset(s.Z[rows], resid[rows], categorical_feature=cat)
        booster = lgb.train(params, dall, num_boost_round=n_rounds)
        return out | {
            "booster": booster,
            "rounds": n_rounds,
            "es_best": best,
            "inner_gain": es_va - base_va,
            "val_months": [int(v) for v in val_m],
        }
    raise ValueError(s.family)


def row_weights(fit: dict, family: str, cell: np.ndarray, Zstd: np.ndarray | None) -> np.ndarray:
    """Per-row member weights ``(n, k)`` of a linear family (``nnls`` / ``cell`` / ``gate``)."""
    if family in ("nnls", "gbdt"):
        return np.broadcast_to(fit["w_global"], (len(cell), len(fit["w_global"])))
    if family == "cell":
        return fit["w_cell"][cell]
    if family == "gate":
        assert Zstd is not None
        return gate_weights(Zstd, fit["W"], fit["b"])
    raise ValueError(family)


def predict(
    fit: dict,
    family: str,
    X: np.ndarray,
    seeds: dict[str, np.ndarray],
    cell: np.ndarray,
    Z: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """``(pred, {seed column: pred})`` for rows given as member matrices."""
    Zstd = None
    if family == "gate":
        assert Z is not None
        Zstd = (Z - fit["mu"]) / fit["sd"]
    w = row_weights(fit, family, cell, Zstd)
    pred = (w * X).sum(axis=1)
    per_seed = {sc: (w * Xs).sum(axis=1) for sc, Xs in seeds.items()}
    if family == "gbdt" and fit.get("booster") is not None:
        assert Z is not None
        corr = fit["booster"].predict(Z)
        pred = pred + corr
        per_seed = {sc: v + corr for sc, v in per_seed.items()}
    return pred, per_seed


# --- reporting ------------------------------------------------------------------------


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2))) if len(a) else float("nan")


def jsonable(fit: dict) -> dict:
    out = {}
    for k, v in fit.items():
        if k == "booster":
            if v is not None:
                imp = v.feature_importance(importance_type="gain")
                out["gain_importance"] = [round(float(x), 1) for x in imp]
            continue
        if isinstance(v, np.ndarray):
            out[k] = np.round(v, 5).tolist()
        elif isinstance(v, float) and not np.isfinite(v):
            out[k] = "inf"
        else:
            out[k] = v
    return out


def summary_tables(fit: dict, family: str, members: list[str], airports: list[str]) -> list[str]:
    """Readable weights of the all-months fit (printed; also stored in ``stack.json``)."""
    short = [m.split("_")[0] for m in members]
    lines = [
        "global: " + ", ".join(f"{s}={w:.3f}" for s, w in zip(short, fit["w_global"], strict=True))
    ]
    if family == "cell":
        lines.append(f"n0 = {fit['n0']}")
        for p in range(2 * N_BANDS):
            has, band = divmod(p, N_BANDS)
            w = fit["w_parent"][p]
            lines.append(
                f"has_nm={has} band={band}: "
                + ", ".join(f"{s}={x:.3f}" for s, x in zip(short, w, strict=True))
            )
    if family == "gate":
        lines.append(f"lambda = {fit['lambda']}  opt {fit['opt']}")
    if family == "gbdt":
        lines.append(f"rounds = {fit.get('rounds')}  inner gain {fit.get('inner_gain')}")
    return lines


# --- main -----------------------------------------------------------------------------


def nested(
    a: argparse.Namespace,
    s: Setup,
    oof: Data,
    cell: np.ndarray,
    ok: np.ndarray,
    st: CellStats,
    month_names: list[str],
    out: Path,
) -> tuple[dict, dict]:
    """The twelve outer fits; writes ``oof.parquet``, returns ``(per-fold choices, summary)``."""
    assert oof.y is not None
    n_months = len(month_names)
    pred = np.zeros(len(oof.y))
    per_seed = {sc: np.zeros(len(oof.y)) for sc in oof.seeds}
    folds = {}
    for m in range(n_months):
        fit_months = np.array([j for j in range(n_months) if j != m])
        if a.family == "gbdt":
            # the NNLS base of this outer fold, then its residual features
            use = np.isin(np.arange(n_months), fit_months)
            wg = cell_weights(st, use, float("inf"), a.min_cell_rows)[0]
            s.Z, s.feat_names = gbdt_features(oof, oof.X @ wg)
        fit = fit_combiner(s, fit_months)
        rows = oof.month == m
        p, ps = predict(
            fit,
            a.family,
            oof.X[rows],
            {sc: v[rows] for sc, v in oof.seeds.items()},
            cell[rows],
            None if s.Z is None else s.Z[rows],
        )
        pred[rows] = p
        for sc, col in per_seed.items():
            col[rows] = ps[sc]
        folds[month_names[m]] = jsonable(fit)
        log(
            f"{month_names[m]}: rmse_ex_ro {rmse(oof.y[rows & ok], p[ok[rows]]):.2f}  "
            + ", ".join(
                f"{k}={folds[month_names[m]][k]}"
                for k in ("n0", "lambda", "rounds")
                if k in folds[month_names[m]]
            )
        )

    oof.frame.with_columns(
        pl.Series("pred", pred), *[pl.Series(k, v) for k, v in per_seed.items()]
    ).write_parquet(out / "oof.parquet")
    log("wrote", out / "oof.parquet")

    jj = np.isin(np.array(month_names)[oof.month], ["2025-01", "2025-07"])
    summary: dict = {
        "rmse": round(rmse(oof.y, pred), 3),
        "rmse_ex_ro": round(rmse(oof.y[ok], pred[ok]), 3),
        "jj_ex_ro": round(rmse(oof.y[ok & jj], pred[ok & jj]), 3),
    }
    if a.ref:
        rd = BF.run_dir(a.ref)
        r = pl.read_parquet(rd / "oof.parquet", columns=["MVT_ID_mvt", "pred"])
        ref_oof = (
            pl.DataFrame({"MVT_ID_mvt": oof.ids})
            .join(r, on="MVT_ID_mvt", how="left", maintain_order="left")["pred"]
            .to_numpy()
        )
        if np.isnan(ref_oof).any():
            raise SystemExit(f"{a.ref}: OOF rows missing against the members")
        summary["ref_rmse_ex_ro"] = round(rmse(oof.y[ok], ref_oof[ok]), 3)
        summary["oof_rmsd_vs_ref"] = round(rmse(pred, ref_oof), 3)
        summary["oof_mean_diff_vs_ref"] = round(float(np.mean(pred - ref_oof)), 3)
        summary["oof_max_abs_diff_vs_ref"] = round(float(np.abs(pred - ref_oof).max()), 3)

    return folds, summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__ and __doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", help="member runs (outputs/runs/<name> or a path)")
    ap.add_argument("--family", choices=FAMILIES, required=True)
    ap.add_argument("--out", required=True, help="output run name under outputs/runs/")
    ap.add_argument("--ref", default=None, help="reference blend run for the rmsd report")
    ap.add_argument("--min-cell-rows", type=int, default=1000)
    ap.add_argument(
        "--cell-keys",
        default=",".join(CELL_KEYS),
        help="family cell: the keys that split cells (subset of airport,has,band,adsb,nmrev,"
        "deice; adsb = the ADS-B state, read through the adsb_ground / adsb_ground_v2 blocks; "
        "nmrev = band of AOBT_3 - EOBT_1; deice = winter_ops.wo_deicing_likely)",
    )
    ap.add_argument("--gate-subsample", type=float, default=0.2)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument(
        "--stage",
        choices=("all", "nested", "test"),
        default="all",
        help="nested = the twelve outer fits and oof.parquet; test = the all-months fit and "
        "test_pred.parquet (merged into an existing stack.json); all = both",
    )
    a = ap.parse_args(argv)

    t0 = time.time()
    dirs = [BF.run_dir(r) for r in a.runs]
    members = [d.name for d in dirs]
    cell_keys = tuple(k for k in a.cell_keys.split(",") if k)
    if not set(cell_keys) <= set(KNOWN_CELL_KEYS):
        raise SystemExit(f"--cell-keys must be a subset of {KNOWN_CELL_KEYS}")
    use_adsb = "adsb" in cell_keys
    extra_keys = [k for k in cell_keys if k in EXTRA_CELL_KEYS]
    if extra_keys and a.family != "cell":
        raise SystemExit(f"--cell-keys {','.join(extra_keys)} is for --family cell only")
    use = {k: k in cell_keys for k in EXTRA_CELL_KEYS}
    oof, month_names, airports = load(dirs, "oof", **use)
    assert oof.y is not None
    n_months, n_air = len(month_names), len(airports)
    n_cells = n_air * 2 * N_BANDS * n_extra(cell_keys)
    levels = cell_levels(cell_keys, n_cells) if extra_keys else None
    ro = np.abs(oof.y - ROLLOVER_S) < ROLLOVER_WINDOW_S
    cell = cell_of(oof, cell_keys)
    log(f"oof rows {len(oof.y):,}  members {len(members)}  seeds {list(oof.seeds)}  ro {ro.sum()}")
    ok = ~ro
    center = float(oof.X.mean())
    st = cell_stats(oof.X[ok], oof.y[ok], oof.month[ok], cell[ok], n_months, n_cells, center)
    Z = None
    feat_names: list[str] = []
    if a.family == "gate":
        Z, feat_names = gate_features(oof, n_air)
    s = Setup(
        family=a.family,
        X=oof.X,
        y=oof.y,
        month=oof.month,
        fit_ok=ok,
        cell=cell,
        n_cells=n_cells,
        min_rows=a.min_cell_rows,
        Z=Z,
        st=st,
        gate_subsample=a.gate_subsample,
        threads=a.threads,
        feat_names=feat_names,
        levels=levels,
    )

    out = paths.RUNS / a.out
    out.mkdir(parents=True, exist_ok=True)
    prev = json.loads((out / "stack.json").read_text()) if (out / "stack.json").is_file() else {}
    if a.stage == "test" and prev.get("family") != a.family:
        raise SystemExit(f"--stage test: {out}/stack.json is missing or of another family")
    summary: dict = dict(prev.get("summary", {}))
    folds = prev.get("nested_folds", {})
    fit_full_doc = prev.get("full_fit")
    if a.stage != "test":
        folds, nested_summary = nested(a, s, oof, cell, ok, st, month_names, out)
        summary |= nested_summary
    if a.stage != "nested":
        all_months = np.arange(n_months)
        if a.family == "gbdt":
            wg = cell_weights(st, np.ones(n_months, dtype=bool), float("inf"), a.min_cell_rows)[0]
            s.Z, s.feat_names = gbdt_features(oof, oof.X @ wg)
        fit_full = fit_combiner(s, all_months)
        fit_full_doc = jsonable(fit_full)
        s.Z = None
        del oof
        test, _, _ = load(dirs, "test_pred", airports, **use)
        tcell = cell_of(test, cell_keys)
        Zt = None
        if a.family == "gate":
            Zt, _ = gate_features(test, n_air)
        if a.family == "gbdt":
            Zt, _ = gbdt_features(test, test.X @ fit_full["w_global"])
        tp, _ = predict(fit_full, a.family, test.X, {}, tcell, Zt)
        test.frame.with_columns(pl.Series("pred", tp)).write_parquet(out / "test_pred.parquet")
        log("wrote", out / "test_pred.parquet", f"rows {len(tp):,}")
        summary["test_rows"] = len(tp)
        summary["test_mean"] = round(float(tp.mean()), 3)
        if a.ref:
            rt = BF.run_dir(a.ref) / "test_pred.parquet"
            r = pl.read_parquet(rt, columns=["MVT_ID_mvt", "pred"])
            ref_test = (
                pl.DataFrame({"MVT_ID_mvt": test.ids})
                .join(r, on="MVT_ID_mvt", how="left", maintain_order="left")["pred"]
                .to_numpy()
            )
            summary["test_rmsd_vs_ref"] = round(rmse(tp, ref_test), 3)
            summary["test_mean_diff_vs_ref"] = round(float(np.mean(tp - ref_test)), 3)
            summary["test_max_abs_diff_vs_ref"] = round(float(np.abs(tp - ref_test).max()), 3)
        for line in summary_tables(fit_full, a.family, members, airports):
            print("  " + line)

    doc = {
        "tool": "scripts/stack_l2.py",
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "family": a.family,
        "members": [BF.member_provenance(d, digests=True) for d in dirs],
        "month_names": month_names,
        "airports": airports,
        "band_edges_s": BAND_EDGES,
        "settings": {
            "min_cell_rows": a.min_cell_rows,
            "cell_keys": list(cell_keys),
            **({"adsb_state": ADSB_STATE_DEF, "adsb_states": ADSB_STATES} if use_adsb else {}),
            **({"nmrev": NMREV_DEF, "nmrev_levels": NMREV_LEVELS} if use["nmrev"] else {}),
            **({"deice": DEICE_DEF, "deice_levels": DEICE_LEVELS} if use["deice"] else {}),
            "cell_n0_grid": [str(v) for v in CELL_N0_GRID],
            "gate_lambda_grid": GATE_LAMBDA_GRID,
            "gate_subsample": a.gate_subsample,
            "gbdt_params": GBDT_PARAMS,
            "gbdt_max_rounds": GBDT_MAX_ROUNDS,
            "features": s.feat_names,
            "fit": "ex_ro (|y - 86400| < 3600 dropped from every fit and inner selection)",
        },
        "summary": summary,
        "nested_folds": folds,
        "full_fit": fit_full_doc,
        "seconds": round(time.time() - t0, 1),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 0),
    }
    (out / "stack.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    log("summary", json.dumps(summary), f"peak RSS {doc['peak_rss_mb']:.0f} MB")
    log("wrote", out / "stack.json")


if __name__ == "__main__":
    main()
