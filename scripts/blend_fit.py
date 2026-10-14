"""Honest blending of several full-year OOF runs (validation-protocol ADR 2026-09-17).

Members are runs with a ``month_loo`` ``oof.parquet`` (each 2025 month held out once,
seeds merged). Rows are aligned **on ``MVT_ID_mvt``, never on row order** — a past blend
check that trusted row order reported RMSE 882.

Every family is scored by *nested* leave-one-month-out: the weights for month ``m`` are
fitted on the other eleven months' OOF and applied to month ``m``. The in-sample number
(weights fitted on all twelve) is reported too, and is always optimistic; only the nested
number is quoted.

Families (``--families``):

``mean``          equal weights, nothing fitted;
``nnls_global``   non-negative weights summing to one, one vector for the whole year;
``nnls_aobt``     the same, separately for rows with and without an NM off-block;
``nnls_spike``    the same, split on the derived-AOBT spike;
``nnls_airport``  the same, per airport;
``ridge_global``  ridge with an intercept, weights unconstrained, the penalty chosen by an
                  inner leave-one-month-out inside the eleven fitting months;
``gbdt``          (``--gbdt``) a small LightGBM stacker on ``[member preds, has_aobt,
                  airport]``. It can express anything, so it is the family to distrust.

**Group keys may only be test-observable.** ``has_aobt``, ``airport`` and the derived-AOBT
spike are properties of the unlabelled 2026 rows; ``ordinary`` and ``gt1h`` are defined
*through the label* and are therefore diagnostics only, never a grouping — a previous
analysis leaked exactly this way. :data:`GROUPINGS` is the whitelist and nothing else is
accepted.

Weights are fitted twice: ``--fit-mode pooled`` uses every training row, ``ex_ro`` drops the
day-rollover rows (``|y − 86,400| < 3,600``) from the *fitting* set. Both are scored on the
full set, pooled and ex-rollovers, because 15 rows of the year carry a large share of the
squared error and make the pooled number a lottery. Dropping rollovers from a fit is not a
lane mask: it changes constants fitted on labelled 2025 rows, not what is done to a 2026 row.

Intervals are the day-block bootstrap of ``scripts/paired_delta.py`` (airport x UTC date,
days resampled inside airport-month), imported from there, not copied.

**JJ (2026-09-23).** Every report carries ``jj`` and ``jj|ex_ro``: January and July 2025
pooled over rows, the population that tracks the board (review 2026-09-23 B, finding 3). It
is a scoring slice of the nested predictions, bootstrapped like every other row.

**Composition-weighted fits (``--fit-weights``, 2026-09-23).** The default ``none`` fits on
the OOF rows in their natural 2025 composition, exactly as before. The other options give
every *fitting* row a weight, constant within a cell ``(month, airport, has_aobt, on_spike)``
— the observable cells of ``scripts/lane_report.py --reweight``:

``jj``       1 on January / July 2025 rows, ``--fit-other-weight`` (default 0 = drop) on the
             other ten months;
``mix2026``  on January / July 2025 rows, the cell's share of the 2026 ranking rows (DEP,
             ``is_test``; 2025-01 -> 2026-01, 2025-07 -> 2026-07) divided by its share of the
             2025 Jan + Jul OOF rows, capped at ``--fit-weight-cap`` (default 5); the other
             months get ``--fit-other-weight``.

The weights read only the cell keys of 2025 and 2026 rows, never a label, so the nested
evaluation stays honest: month ``m``'s weights are still fitted on the other eleven months'
labels only (with the default ``--fit-other-weight 0`` that is the other one of January /
July). Scoring is unweighted, as always.

``apply --test-source foldavg`` blends the members' ``test_pred_foldavg.parquet`` (the
fold-model average a ``test_fit: full`` run writes next to its full-fit ``test_pred``) and
writes ``test_pred_foldavg.parquet`` into the output run.

    uv run python scripts/blend_fit.py fit exp0068_... exp0060_... [--gbdt] \
        --write report.md --emit-weights outputs/blend/w.json
    uv run python scripts/blend_fit.py apply outputs/blend/w.json --which oof \
        --out blend_r5_cat --which test
    uv run python scripts/blend_fit.py fit <members> --fit-weights mix2026 \
        --emit-family nnls_global --emit-mode ex_ro --write-oof blend_x_mix
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import itertools
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from prc import paths

SPIKE = (1194, 1206)
ROLLOVER_S, ROLLOVER_WINDOW_S = 86_400.0, 3_600.0
MIN_GROUP_ROWS = 5_000  # below this a group falls back to the global fit of the same mode
FIT_MODES = ("pooled", "ex_ro")
#: `--fit-weights`: `none` is the pre-2026-09-23 fit, byte for byte.
FIT_WEIGHTS = ("none", "jj", "mix2026")
JJ_MONTHS = ("2025-01", "2025-07")
#: The 2025 gate month whose 2026 ranking month it stands in for (as `lane_report.py`).
GATE_TO_RANK = {"2025-01": "2026-01", "2025-07": "2026-07"}
TEST_SOURCES = {"test_pred": "test_pred", "foldavg": "test_pred_foldavg"}

#: The only admissible grouping keys: every one is observable on an unlabelled 2026 row.
#: ``ordinary`` / ``gt1h`` are label-defined and are deliberately absent.
GROUPINGS = ("global", "has_aobt", "on_spike", "airport")


@dataclass(frozen=True)
class Family:
    name: str
    kind: str  # mean | simplex | ridge | gbdt
    grouping: str
    complexity: int  # ordering for "simplest family that is not beaten"


FAMILIES: tuple[Family, ...] = (
    Family("mean", "mean", "global", 0),
    Family("nnls_global", "simplex", "global", 1),
    Family("nnls_aobt", "simplex", "has_aobt", 2),
    Family("nnls_spike", "simplex", "on_spike", 2),
    Family("nnls_airport", "simplex", "airport", 3),
    Family("ridge_global", "ridge", "global", 4),
    Family("gbdt", "gbdt", "global", 9),
)
BY_NAME = {f.name: f for f in FAMILIES}
DEFAULT_FAMILIES = tuple(f.name for f in FAMILIES if f.name != "gbdt")
RIDGE_ALPHAS = (0.0, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


# --- plumbing -------------------------------------------------------------------------


def paired_delta_module():
    """``scripts/paired_delta.py`` as a module, so the day-block logic has one home."""
    src = Path(__file__).resolve().parent / "paired_delta.py"
    spec = importlib.util.spec_from_file_location("paired_delta", src)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_dir(name: str) -> Path:
    p = Path(name)
    if p.is_dir():
        return p.resolve()
    q = paths.RUNS / name
    if q.is_dir():
        return q.resolve()
    raise SystemExit(f"no such run: {name}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def member_provenance(d: Path, digests: bool) -> dict:
    out: dict = {"run": d.name, "dir": str(d)}
    mj = d / "metrics.json"
    if mj.is_file():
        m = json.loads(mj.read_text())
        out |= {
            k: m.get(k) for k in ("config_digest", "cv_scheme", "seeds", "n_oof", "git", "cv_rmse")
        }
    if digests:
        for which in ("oof", "test_pred", "test_pred_foldavg"):
            f = d / f"{which}.parquet"
            if f.is_file():
                out[f"{which}_sha256"] = sha256(f)
    return out


# --- loading --------------------------------------------------------------------------


def load_frame(
    dirs: list[Path], which: str, seeds: bool
) -> tuple[pl.DataFrame, list[str], list[str]]:
    """Members joined **on ``MVT_ID_mvt``** plus the lane columns from the movement table.

    Returns the frame, the per-member prediction columns (``p__<run>``) and the per-seed
    column stems the members have in common."""
    names = [d.name for d in dirs]
    base_cols = ["MVT_ID_mvt", "airport", "month", "fold", "TAXITIME_SEC_mvt"]
    frames, seed_cols = [], None
    for d, name in zip(dirs, names, strict=True):
        lf = pl.scan_parquet(d / f"{which}.parquet")
        have = lf.collect_schema().names()
        s = sorted(c for c in have if c.startswith("pred_s"))
        seed_cols = s if seed_cols is None else [c for c in seed_cols if c in s]
        keep = [pl.col("pred").alias(f"p__{name}")]
        if seeds:
            keep += [pl.col(c).alias(f"{c}__{name}") for c in s]
        if d is dirs[0]:
            keep = [pl.col(c) for c in base_cols if c in have] + keep
        else:
            keep = [pl.col("MVT_ID_mvt"), *keep]
        frames.append(lf.select(keep))
    d0 = frames[0]
    for f in frames[1:]:
        d0 = d0.join(f, on="MVT_ID_mvt", how="inner", maintain_order="left")
    df = d0.collect()
    if df.height == 0:
        raise SystemExit("members share no MVT_ID_mvt — wrong scheme or wrong files")
    mv_cols = ["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt"]
    # `test_pred.parquet` carries neither airport nor month; the movement table does, and it
    # is the same table the lane definitions come from.
    mv_cols += [c for c in ("airport", "month") if c not in df.columns]
    mv = pl.scan_parquet(paths.INTERIM / "movements.parquet").select(mv_cols)
    df = (
        df.lazy()
        .join(mv, on="MVT_ID_mvt", how="left", maintain_order="left")
        .with_columns(
            (pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds().alias("d_aobt")
        )
        .with_columns(
            pl.col("AOBT_3_flt").is_not_null().alias("has_aobt"),
            pl.col("d_aobt").is_between(*SPIKE).fill_null(False).alias("on_spike"),
        )
        .collect()
    )
    return df, [f"p__{n}" for n in names], seed_cols or []


# --- sufficient statistics ------------------------------------------------------------


@dataclass
class Cells:
    """Per-cell Gram statistics of ``A = [centered preds, 1]`` against the centered target.

    A cell is ``(month, airport, has_aobt, on_spike, rollover)``; every fit in this script is
    a sum of cells, so twelve nested fits x seven families cost one pass over the rows."""

    S: np.ndarray  # (n_cells, k+1, k+1)
    t: np.ndarray  # (n_cells, k+1)
    yty: np.ndarray  # (n_cells,)
    n: np.ndarray  # (n_cells,)
    c_month: np.ndarray
    c_group: dict[str, np.ndarray]
    c_ro: np.ndarray


def build_cells(A: np.ndarray, yc: np.ndarray, keys: dict[str, np.ndarray]) -> tuple[Cells, dict]:
    months = keys["month"]
    airports = keys["airport"]
    has = keys["has_aobt"].astype(np.int64)
    spike = keys["on_spike"].astype(np.int64)
    ro = keys["rollover"].astype(np.int64)
    n_m, n_a = int(months.max()) + 1, int(airports.max()) + 1
    cell = ((((months * n_a + airports) * 2 + has) * 2 + spike) * 2 + ro).astype(np.int64)
    n_cells = n_m * n_a * 8
    k1 = A.shape[1]
    S = np.zeros((n_cells, k1, k1))
    for i in range(k1):
        for j in range(i, k1):
            v = np.bincount(cell, weights=A[:, i] * A[:, j], minlength=n_cells)
            S[:, i, j] = v
            S[:, j, i] = v
    t = np.stack(
        [np.bincount(cell, weights=A[:, i] * yc, minlength=n_cells) for i in range(k1)], axis=1
    )
    yty = np.bincount(cell, weights=yc * yc, minlength=n_cells)
    n = np.bincount(cell, minlength=n_cells).astype(float)
    idx = np.arange(n_cells)
    c_ro = idx % 2
    c_spike = (idx // 2) % 2
    c_has = (idx // 4) % 2
    c_air = (idx // 8) % n_a
    c_month = idx // (8 * n_a)
    c_group = {
        "global": np.zeros(n_cells, dtype=np.int64),
        "has_aobt": c_has,
        "on_spike": c_spike,
        "airport": c_air,
    }
    return Cells(S, t, yty, n, c_month, c_group, c_ro), {
        "n_months": n_m,
        "n_airports": n_a,
        "row_cell": cell,
    }


def weighted_cells(c: Cells, cw: np.ndarray) -> Cells:
    """The same cells with every row of cell ``i`` carrying weight ``cw[i]``.

    The Gram statistics are sums over rows, so a weight that is constant within a cell is a
    multiply per cell. ``n`` stays a row count (of the cells that keep a positive weight): it
    decides the ``MIN_GROUP_ROWS`` fallback and which months exist, not the solution."""
    cw = np.asarray(cw, dtype=float)
    return Cells(
        S=c.S * cw[:, None, None],
        t=c.t * cw[:, None],
        yty=c.yty * cw,
        n=c.n * (cw > 0),
        c_month=c.c_month,
        c_group=c.c_group,
        c_ro=c.c_ro,
    )


def summed(c: Cells, sel: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    return c.S[sel].sum(0), c.t[sel].sum(0), float(c.yty[sel].sum()), float(c.n[sel].sum())


# --- fitters --------------------------------------------------------------------------


def simplex_lsq(G: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Exact ``argmin_w w'Gw - 2b'w`` over ``{w >= 0, sum w = 1}`` by active-set enumeration.

    ``k`` is the number of blend members (single digits), so all ``2^k - 1`` supports are
    enumerated and the best feasible stationary point wins. No optimizer, no seed, and no
    renormalisation of an unconstrained NNLS solution (which is *not* the constrained
    optimum). ``G`` is scaled to unit diagonal before the KKT system is formed: on 2 M rows
    of taxi seconds its entries are ~1e11 against a constraint row of ones, and an unscaled
    solve silently returns a minimum-norm answer whose weights do not sum to one."""
    k = len(b)
    scale = float(np.trace(G)) / max(k, 1)
    if not np.isfinite(scale) or scale <= 0:
        return np.full(k, 1.0 / k)
    G, b = G / scale, b / scale
    best_w, best_obj = np.full(k, 1.0 / k), np.inf
    for r in range(1, k + 1):
        for sup in itertools.combinations(range(k), r):
            s = list(sup)
            m = len(s)
            kkt = np.zeros((m + 1, m + 1))
            kkt[:m, :m] = G[np.ix_(s, s)]
            kkt[:m, m] = 1.0
            kkt[m, :m] = 1.0
            rhs = np.concatenate([b[s], [1.0]])
            try:
                sol = np.linalg.solve(kkt, rhs)
            except np.linalg.LinAlgError:
                sol, *_ = np.linalg.lstsq(kkt, rhs, rcond=None)
            ws = sol[:m]
            if ws.min() < -1e-9 or abs(ws.sum() - 1.0) > 1e-7:
                continue
            w = np.zeros(k)
            w[s] = np.clip(ws, 0.0, None)
            w /= w.sum()
            obj = float(w @ G @ w - 2 * b @ w)
            if obj < best_obj - 1e-12:
                best_w, best_obj = w, obj
    return best_w


def ridge_solve(S: np.ndarray, t: np.ndarray, lam: float) -> np.ndarray:
    """``[w, intercept]`` of a ridge that penalises the member weights but not the intercept."""
    k1 = S.shape[0]
    D = np.eye(k1)
    D[-1, -1] = 0.0
    # the intercept column contributes `n` where the member columns contribute `n * var`;
    # dividing through by `n` leaves the solution alone and keeps the solve well scaled
    n = float(S[-1, -1]) or 1.0
    sol, *_ = np.linalg.lstsq(S / n + (lam / n) * D, t / n, rcond=None)
    return sol


def sse_of(v: np.ndarray, S: np.ndarray, t: np.ndarray, yty: float) -> float:
    return float(v @ S @ v - 2 * v @ t + yty)


def choose_alpha(per_month: list[tuple], k: int) -> float:
    """Inner leave-one-month-out over the fitting months only (never the held-out month)."""
    St = sum(p[0] for p in per_month)
    tt = sum(p[1] for p in per_month)
    scale = float(np.trace(St[:k, :k])) / max(k, 1)
    best, best_sse = 0.0, np.inf
    for a in RIDGE_ALPHAS:
        lam = a * scale
        tot = 0.0
        for S_m, t_m, yty_m, n_m in per_month:
            if n_m <= 0:
                continue
            v = ridge_solve(St - S_m, tt - t_m, lam)
            tot += sse_of(v, S_m, t_m, yty_m)
        if tot < best_sse - 1e-9:
            best, best_sse = lam, tot
    return best


def fit_group(kind: str, c: Cells, sel: np.ndarray, k: int) -> tuple[np.ndarray, float, float]:
    """``(weights, intercept_centered, n_rows)`` for one family on one selection of cells."""
    S, t, _yty, n = summed(c, sel)
    if kind == "mean":
        return np.full(k, 1.0 / k), 0.0, n
    if kind == "simplex":
        return simplex_lsq(S[:k, :k], t[:k]), 0.0, n
    if kind == "ridge":
        months = np.unique(c.c_month[sel & (c.n > 0)])
        per_month = [summed(c, sel & (c.c_month == m)) for m in months]
        lam = choose_alpha(per_month, k)
        v = ridge_solve(S, t, lam)
        return v[:k], float(v[k]), n
    raise ValueError(kind)


# --- nested evaluation ----------------------------------------------------------------


@dataclass
class Ctx:
    X: np.ndarray  # raw member predictions (n, k)
    y: np.ndarray
    center: float
    cells: Cells
    row_month: np.ndarray
    row_group: dict[str, np.ndarray]
    rollover: np.ndarray
    month_names: list[str]
    airport_names: list[str]
    members: list[str]
    has_aobt: np.ndarray
    on_spike: np.ndarray
    row_cell: np.ndarray | None = None
    #: per-row fitting weight (`--fit-weights`), None = unweighted (the default)
    row_weight: np.ndarray | None = None


def group_names(grouping: str, ctx: Ctx) -> list[str]:
    return {
        "global": ["__all__"],
        "has_aobt": ["no_aobt", "with_aobt"],
        "on_spike": ["off_spike", "on_spike"],
        "airport": ctx.airport_names,
    }[grouping]


def fit_all(ctx: Ctx, fam: Family, mode: str, months: np.ndarray | None) -> dict:
    """Fit one family on every month except those in ``months`` (``None`` = fit on all)."""
    c, k = ctx.cells, ctx.X.shape[1]
    base = c.n > 0
    if months is not None:
        base = base & ~np.isin(c.c_month, months)
    if mode == "ex_ro":
        base = base & (c.c_ro == 0)
    cg = c.c_group[fam.grouping]
    gw, gb = fit_group(fam.kind, c, base & (c.c_group["global"] == 0), k)[:2]
    out = {"__fallback__": {"w": gw, "b": gb, "n": float(c.n[base].sum()), "fallback": False}}
    for gi, gname in enumerate(group_names(fam.grouping, ctx)):
        sel = base & (cg == gi)
        n = float(c.n[sel].sum())
        if n < MIN_GROUP_ROWS:
            out[gname] = {"w": gw, "b": gb, "n": n, "fallback": True}
            continue
        w, b, _ = fit_group(fam.kind, c, sel, k)
        out[gname] = {"w": w, "b": b, "n": n, "fallback": False}
    return out


def apply_fit(
    ctx: Ctx, fam: Family, fit: dict, rows: np.ndarray, X: np.ndarray | None = None
) -> np.ndarray:
    """Centered predictions for ``rows`` under a fitted family.

    ``X`` overrides the member matrix, which is how the per-seed columns of a nested blend
    are produced: the same weights, a different set of member predictions."""
    rg = ctx.row_group[fam.grouping]
    Xc = (ctx.X if X is None else X) - ctx.center
    out = np.zeros(rows.sum())
    names = group_names(fam.grouping, ctx)
    sub_rg = rg[rows]
    for gi, gname in enumerate(names):
        m = sub_rg == gi
        if not m.any():
            continue
        f = fit[gname]
        out[m] = Xc[rows][m] @ f["w"] + f["b"]
    return out


def nested_gbdt(ctx: Ctx, mode: str, params: dict) -> tuple[np.ndarray, list[dict]]:
    """LightGBM stacker on ``[member preds, has_aobt, airport]``, offset by the member mean.

    The offset (`init_score`) matters: a tree ensemble started from the global mean has to
    spend its whole budget rebuilding a near-identity function and underfits the level by
    tens of seconds, which would make the family look bad for the wrong reason. Started from
    the equal-weight blend it only has to learn a correction, which is the version worth
    distrusting."""
    import lightgbm as lgb

    feats = np.column_stack(
        [ctx.X, ctx.has_aobt.astype(np.float32), ctx.row_group["airport"].astype(np.float32)]
    ).astype(np.float32)
    cat = [feats.shape[1] - 1]
    init = ctx.X.mean(axis=1)
    pred = np.zeros(len(ctx.y))
    infos = []
    for mi in range(len(ctx.month_names)):
        te = ctx.row_month == mi
        tr = ~te
        if mode == "ex_ro":
            tr = tr & ~ctx.rollover
        kw = {}
        if ctx.row_weight is not None:
            tr = tr & (ctx.row_weight > 0)
            kw["sample_weight"] = ctx.row_weight[tr]
        m = lgb.LGBMRegressor(**params)
        m.fit(feats[tr], ctx.y[tr], init_score=init[tr], categorical_feature=cat, **kw)
        pred[te] = m.predict(feats[te]) + init[te]
        infos.append({"month": ctx.month_names[mi], "n_fit": int(tr.sum())})
    return pred, infos


def nested_predict(ctx: Ctx, fam: Family, mode: str, gbdt_params: dict) -> dict:
    if fam.kind == "gbdt":
        pred, _ = nested_gbdt(ctx, mode, gbdt_params)
        return {"pred": pred, "per_month": {}, "in_sample": None}
    pred = np.zeros(len(ctx.y))
    per_month = {}
    for mi, mname in enumerate(ctx.month_names):
        fit = fit_all(ctx, fam, mode, np.array([mi]))
        rows = ctx.row_month == mi
        pred[rows] = apply_fit(ctx, fam, fit, rows) + ctx.center
        per_month[mname] = fit
    full = fit_all(ctx, fam, mode, None)
    allrows = np.ones(len(ctx.y), dtype=bool)
    in_sample = apply_fit(ctx, fam, full, allrows) + ctx.center
    return {"pred": pred, "per_month": per_month, "full": full, "in_sample": in_sample}


def nested_apply(ctx: Ctx, fam: Family, per_month: dict, X: np.ndarray) -> np.ndarray:
    """Replay the twelve nested fits on another member matrix (e.g. one seed's columns)."""
    out = np.zeros(len(ctx.y))
    for mi, mname in enumerate(ctx.month_names):
        rows = ctx.row_month == mi
        out[rows] = apply_fit(ctx, fam, per_month[mname], rows, X) + ctx.center
    return out


def write_nested_oof(
    ctx: Ctx, fam: Family, r: dict, dirs: list[Path], df: pl.DataFrame, name: str
) -> Path:
    """The nested blend as a run dir, so ``paired_delta`` / ``lane_report`` read it directly.

    These are the honest predictions — month ``m`` blended with weights fitted on the other
    eleven — not the all-months fit that ``--emit-weights`` carries. Per-seed columns are the
    same weights applied to each member's ``pred_s<seed>``, so the paired-seed machinery keeps
    working on the blend."""
    seeded, _cols, seed_cols = load_frame(dirs, "oof", seeds=True)
    if seeded["MVT_ID_mvt"].to_numpy().tolist() != df["MVT_ID_mvt"].to_numpy().tolist():
        raise SystemExit("row order changed between loads; refusing to write a blend")
    out = paths.RUNS / name
    out.mkdir(parents=True, exist_ok=True)
    per_seed = {}
    for sc in seed_cols:
        Xs = np.column_stack([seeded[f"{sc}__{d.name}"].cast(pl.Float64).to_numpy() for d in dirs])
        per_seed[sc] = nested_apply(ctx, fam, r["per_month"], Xs)
    frame = df.select("MVT_ID_mvt", "airport", "month", "fold", "TAXITIME_SEC_mvt").with_columns(
        pl.Series("pred", r["pred"]), *[pl.Series(k, v) for k, v in per_seed.items()]
    )
    frame.write_parquet(out / "oof.parquet")
    return out / "oof.parquet"


# --- lanes and reporting --------------------------------------------------------------


def lane_masks(ctx: Ctx) -> dict[str, np.ndarray]:
    y = ctx.y
    big = y > 3600
    has, spike = ctx.has_aobt, ctx.on_spike
    lanes: dict[str, np.ndarray] = {
        "all": np.ones(len(y), dtype=bool),
        "with_aobt": has,
        "no_aobt": ~has,
        "spike": spike,
        "gt1h": big,
        "ordinary": has & ~spike & ~big,
    }
    for mi, m in enumerate(ctx.month_names):
        lanes[m] = ctx.row_month == mi
    jj = np.isin(np.array(ctx.month_names)[ctx.row_month], JJ_MONTHS)
    if jj.any():
        lanes["jj"] = jj
    for ai, a in enumerate(ctx.airport_names):
        lanes[f"ap:{a}"] = ctx.row_group["airport"] == ai
    return lanes


def rmse(y: np.ndarray, p: np.ndarray, m: np.ndarray) -> float:
    if m.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y[m] - p[m]) ** 2)))


def md_table(rows: list[dict]) -> list[str]:
    """A markdown table whose cells are escaped: lane names such as ``all|ex_ro`` carry a
    literal pipe and would otherwise split the row into the wrong number of columns."""
    if not rows:
        return ["_(empty)_", ""]
    cols = list(rows[0])

    def cell(v: object) -> str:
        return str(v).replace("|", "\\|")

    return [
        "| " + " | ".join(cell(c) for c in cols) + " |",
        "|" + "---|" * len(cols),
        *["| " + " | ".join(cell(r.get(c, "")) for c in cols) + " |" for r in rows],
        "",
    ]


def fmt(x: float, nd: int = 1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


# --- fit command ----------------------------------------------------------------------


def build_ctx(df: pl.DataFrame, pred_cols: list[str], members: list[str]) -> Ctx:
    y = df["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy()
    X = np.column_stack([df[c].cast(pl.Float64).to_numpy() for c in pred_cols])
    month_names = sorted(set(df["month"].to_list()))
    airport_names = sorted(set(df["airport"].to_list()))
    row_month = np.searchsorted(np.array(month_names), df["month"].to_numpy())
    row_air = np.searchsorted(np.array(airport_names), df["airport"].to_numpy())
    has = df["has_aobt"].to_numpy()
    spike = df["on_spike"].to_numpy()
    ro = np.abs(y - ROLLOVER_S) < ROLLOVER_WINDOW_S
    center = float(y.mean())
    A = np.column_stack([X - center, np.ones(len(y))])
    cells, info = build_cells(
        A,
        y - center,
        {
            "month": row_month,
            "airport": row_air,
            "has_aobt": has,
            "on_spike": spike,
            "rollover": ro,
        },
    )
    return Ctx(
        X=X,
        y=y,
        center=center,
        cells=cells,
        row_month=row_month,
        row_group={
            "global": np.zeros(len(y), dtype=np.int64),
            "has_aobt": has.astype(np.int64),
            "on_spike": spike.astype(np.int64),
            "airport": row_air,
        },
        rollover=ro,
        month_names=month_names,
        airport_names=airport_names,
        members=members,
        has_aobt=has,
        on_spike=spike,
        row_cell=info["row_cell"],
    )


# --- composition weights (--fit-weights) ----------------------------------------------


def ranking_counts() -> pl.DataFrame:
    """Rows per ``(month, airport, has_aobt, on_spike)`` of the 2026 ranking set, from
    ``scripts/lane_report.py`` (one definition of the cells). Features only, no labels."""
    src = Path(__file__).resolve().parent / "lane_report.py"
    spec = importlib.util.spec_from_file_location("lane_report", src)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ranking_composition()


def row_fit_weights(
    ctx: Ctx,
    spec: str,
    other: float = 0.0,
    cap: float = 5.0,
    comp: pl.DataFrame | None = None,
) -> np.ndarray | None:
    """Per-row fitting weight for ``--fit-weights spec`` (``None`` for ``none``).

    ``jj``: 1 on January / July rows, ``other`` elsewhere. ``mix2026``: on January / July rows
    ``min(cap, share_2026(cell) / share_2025(cell))`` with both shares taken over the pooled
    Jan + Jul rows of their year, so the weighted 2025 rows have the 2026 cell mix (month
    included); a cell absent from 2026 gets 0. ``comp`` is the ranking-set count table
    (:func:`ranking_counts`), injectable for tests. Only cell keys are read, never ``y``."""
    if spec == "none":
        return None
    months = np.array(ctx.month_names)[ctx.row_month]
    jj = np.isin(months, JJ_MONTHS)
    if not jj.any():
        raise SystemExit(f"--fit-weights {spec}: the members hold out neither 2025-01 nor 2025-07")
    w = np.full(len(ctx.y), float(other))
    if spec == "jj":
        w[jj] = 1.0
        return w
    if spec != "mix2026":
        raise SystemExit(f"unknown --fit-weights {spec!r}")
    comp = ranking_counts() if comp is None else comp
    airports = np.array(ctx.airport_names)[ctx.row_group["airport"]]
    gate = (
        pl.DataFrame(
            {
                "month": months[jj],
                "airport": airports[jj],
                "has_aobt": ctx.has_aobt[jj].astype(bool),
                "on_spike": ctx.on_spike[jj].astype(bool),
            }
        )
        .with_row_index("i")
        .with_columns(pl.col("month").replace_strict(GATE_TO_RANK).alias("rank_month"))
    )
    counts = gate.group_by("rank_month", "airport", "has_aobt", "on_spike").agg(
        pl.len().alias("n_gate")
    )
    rank = comp.filter(pl.col("month").cast(pl.Utf8).is_in(list(GATE_TO_RANK.values()))).select(
        pl.col("month").cast(pl.Utf8).alias("rank_month"),
        pl.col("airport").cast(pl.Utf8),
        pl.col("has_aobt").cast(pl.Boolean),
        pl.col("on_spike").cast(pl.Boolean),
        pl.col("n_rank").cast(pl.Float64),
    )
    n_rank_total = float(rank["n_rank"].sum())
    if n_rank_total <= 0:
        raise SystemExit("--fit-weights mix2026: the ranking set has no Jan / Jul rows")
    ratio = (
        counts.join(rank, on=["rank_month", "airport", "has_aobt", "on_spike"], how="left")
        .with_columns(pl.col("n_rank").fill_null(0.0))
        .with_columns(
            (
                (pl.col("n_rank") / n_rank_total)
                / (pl.col("n_gate").cast(pl.Float64) / float(jj.sum()))
            )
            .clip(upper_bound=float(cap))
            .alias("ratio")
        )
    )
    per_row = gate.join(
        ratio.select("rank_month", "airport", "has_aobt", "on_spike", "ratio"),
        on=["rank_month", "airport", "has_aobt", "on_spike"],
        how="left",
    ).sort("i")
    w[jj] = per_row["ratio"].to_numpy()
    return w


def cell_weights(ctx: Ctx, row_w: np.ndarray) -> np.ndarray:
    """Row weights -> one weight per cell (they are constant within a cell by construction)."""
    assert ctx.row_cell is not None
    n_cells = len(ctx.cells.n)
    tot = np.bincount(ctx.row_cell, weights=row_w, minlength=n_cells)
    cnt = np.bincount(ctx.row_cell, minlength=n_cells)
    cw = np.divide(tot, cnt, out=np.zeros(n_cells), where=cnt > 0)
    lo = np.full(n_cells, np.inf)
    hi = np.full(n_cells, -np.inf)
    np.minimum.at(lo, ctx.row_cell, row_w)
    np.maximum.at(hi, ctx.row_cell, row_w)
    ok = cnt > 0
    if not np.allclose(lo[ok], hi[ok]):
        raise AssertionError("fit weights must be constant within a cell")
    return cw


def apply_fit_weights(ctx: Ctx, row_w: np.ndarray | None) -> Ctx:
    """``ctx`` with its cell statistics weighted by ``row_w`` (unchanged for ``None``)."""
    if row_w is None:
        return ctx
    return dataclasses.replace(
        ctx, cells=weighted_cells(ctx.cells, cell_weights(ctx, row_w)), row_weight=row_w
    )


def weights_payload(ctx: Ctx, fam: Family, fit: dict, mode: str) -> dict:
    """Uncentered weights + intercept per group, ready for :func:`apply`."""
    groups = {}
    for gname, f in fit.items():
        w = np.asarray(f["w"], dtype=float)
        b = float(f["b"]) + ctx.center * (1.0 - float(w.sum()))
        groups[gname] = {
            "weights": {m: round(float(x), 8) for m, x in zip(ctx.members, w, strict=True)},
            "intercept": round(b, 6),
            "n_fit": int(f["n"]),
            "fallback": bool(f["fallback"]),
        }
    return {"family": fam.name, "grouping": fam.grouping, "fit_mode": mode, "groups": groups}


def cmd_fit(a: argparse.Namespace) -> None:
    pd = paired_delta_module()
    dirs = [run_dir(r) for r in a.runs]
    members = [d.name for d in dirs]
    if len(set(members)) != len(members):
        raise SystemExit("duplicate member runs")
    fams = [BY_NAME[n] for n in a.families]
    if a.gbdt and "gbdt" not in a.families:
        fams.append(BY_NAME["gbdt"])
    for f in fams:
        if f.grouping not in GROUPINGS:
            raise SystemExit(f"{f.name}: grouping {f.grouping!r} is not test-observable")
    print(f"members ({len(members)}):")
    for m in members:
        print("  ", m)
    df, pred_cols, _ = load_frame(dirs, "oof", seeds=False)
    print(f"joined on MVT_ID_mvt: {df.height:,} rows")
    ctx = build_ctx(df, pred_cols, members)
    lanes = lane_masks(ctx)
    ro = ctx.rollover
    print(f"rollover rows: {int(ro.sum())}  months: {len(ctx.month_names)}")
    fit_w = row_fit_weights(ctx, a.fit_weights, a.fit_other_weight, a.fit_weight_cap)
    fit_w_info = None
    if fit_w is not None:
        ctx = apply_fit_weights(ctx, fit_w)
        pos = fit_w > 0
        fit_w_info = {
            "fit_weights": a.fit_weights,
            "other_weight": a.fit_other_weight,
            "cap": a.fit_weight_cap if a.fit_weights == "mix2026" else None,
            "rows_weighted": int(pos.sum()),
            "weight_sum": round(float(fit_w.sum()), 1),
            "jj_weight_min": round(float(fit_w[lanes["jj"]].min()), 4),
            "jj_weight_max": round(float(fit_w[lanes["jj"]].max()), 4),
            "jj_rows_at_cap": int((fit_w[lanes["jj"]] >= a.fit_weight_cap - 1e-12).sum())
            if a.fit_weights == "mix2026"
            else 0,
            "effective_n": round(float(fit_w.sum() ** 2 / max((fit_w**2).sum(), 1e-12)), 1),
        }
        print("fit weights:", json.dumps(fit_w_info))

    # --- bootstrap scaffolding (day blocks, from paired_delta) ---
    boot = None
    if a.bootstrap:
        block_of_row, groups = pd.block_index(df)
        n_blocks = int(block_of_row.max()) + 1
        w = pd.block_weights(groups, n_blocks, a.bootstrap, a.bootstrap_seed)
        boot = (w, block_of_row, n_blocks)
        print(f"bootstrap: {a.bootstrap} replicates over {n_blocks} airport-days")

    # --- members ---
    member_rows = []
    for i, m in enumerate(members):
        member_rows.append(
            {
                "member": m,
                "rmse": fmt(rmse(ctx.y, ctx.X[:, i], lanes["all"])),
                "rmse_ex_ro": fmt(rmse(ctx.y, ctx.X[:, i], ~ro)),
                "jan": fmt(rmse(ctx.y, ctx.X[:, i], lanes["2025-01"])),
                "jul": fmt(rmse(ctx.y, ctx.X[:, i], lanes["2025-07"])),
                "ordinary": fmt(rmse(ctx.y, ctx.X[:, i], lanes["ordinary"])),
                "gt1h": fmt(rmse(ctx.y, ctx.X[:, i], lanes["gt1h"])),
                "no_aobt_ex_ro": fmt(rmse(ctx.y, ctx.X[:, i], lanes["no_aobt"] & ~ro)),
            }
        )
    ref_i = (
        members.index(a.reference)
        if a.reference
        else int(np.argmin([rmse(ctx.y, ctx.X[:, i], ~ro) for i in range(len(members))]))
    )
    ref_name, ref_pred = members[ref_i], ctx.X[:, ref_i]
    print(f"reference member (best ex-rollovers): {ref_name}")

    # --- fit every family both ways ---
    gbdt_params = {
        "n_estimators": a.gbdt_trees,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 2000,
        "verbose": -1,
        "n_jobs": a.threads,
        "random_state": 0,
    }
    results: dict[tuple[str, str], dict] = {}
    for fam in fams:
        for mode in FIT_MODES:
            r = nested_predict(ctx, fam, mode, gbdt_params)
            results[(fam.name, mode)] = r
            print(
                f"  {fam.name:14s} {mode:6s} nested "
                f"{rmse(ctx.y, r['pred'], lanes['all']):7.2f} / ex-ro "
                f"{rmse(ctx.y, r['pred'], ~ro):7.2f}"
            )

    se_ref = (ctx.y - ref_pred) ** 2
    lane_keys = list(lanes)

    def delta_rows(pred: np.ndarray) -> dict[str, dict]:
        se = (ctx.y - pred) ** 2
        out = {}
        for lk in lane_keys:
            for suffix, mask in (("", lanes[lk]), ("|ex_ro", lanes[lk] & ~ro)):
                key = lk + suffix
                row = {
                    "rows": int(mask.sum()),
                    "ref_rmse": rmse(ctx.y, ref_pred, mask),
                    "rmse": rmse(ctx.y, pred, mask),
                }
                row["delta"] = row["rmse"] - row["ref_rmse"]
                if boot and mask.sum():
                    row |= pd.bootstrap_delta(boot[0], boot[1], boot[2], se_ref, se, mask)
                out[key] = row
        return out

    deltas = {}
    for i, m in enumerate(members):
        if i != ref_i:
            deltas[("member", m)] = delta_rows(ctx.X[:, i])
    for key, r in results.items():
        deltas[key] = delta_rows(r["pred"])

    # --- recommendation: the simplest family that beats the reference member and is not
    # itself beaten by any other family. "Not beaten" is a paired day-block interval against
    # the best family, not a comparison of point estimates: a wide interval must not buy a
    # family its place. ---
    best_key = min(results, key=lambda k: rmse(ctx.y, results[k]["pred"], ~ro))
    se_best = (ctx.y - results[best_key]["pred"]) ** 2
    vs_best = {}
    for key, r in results.items():
        row = {"delta": rmse(ctx.y, r["pred"], ~ro) - rmse(ctx.y, results[best_key]["pred"], ~ro)}
        if boot and key != best_key:
            row |= pd.bootstrap_delta(
                boot[0], boot[1], boot[2], se_best, (ctx.y - r["pred"]) ** 2, ~ro
            )
        vs_best[key] = row
    order = sorted(
        results, key=lambda k: (BY_NAME[k[0]].complexity, rmse(ctx.y, results[k]["pred"], ~ro))
    )
    rec = None
    for key in order:
        beats_ref = deltas[key]["all|ex_ro"].get("boot_hi", 0.0) < 0
        # `delta` is candidate - best, so "beaten" means the whole interval sits above zero
        not_beaten = key == best_key or vs_best[key].get("boot_lo", 1.0) <= 0
        if beats_ref and not_beaten:
            rec = key
            break
    rec = rec or best_key

    # --- weights on ALL months ---
    payloads = {}
    for fam in fams:
        if fam.kind == "gbdt":
            continue
        for mode in FIT_MODES:
            payloads[f"{fam.name}|{mode}"] = weights_payload(
                ctx, fam, results[(fam.name, mode)]["full"], mode
            )

    prov = [member_provenance(d, digests=not a.no_digests) for d in dirs]
    if a.emit_weights:
        sel = a.emit_family or rec[0]
        sel_mode = a.emit_mode or rec[1]
        if BY_NAME[sel].kind == "gbdt":
            raise SystemExit("gbdt has no weight vector; --emit-family must be a linear family")
        payload = payloads[f"{sel}|{sel_mode}"]
        doc = {
            "tool": "scripts/blend_fit.py",
            "created": datetime.now(UTC).isoformat(timespec="seconds"),
            "cv_scheme": prov[0].get("cv_scheme"),
            "n_oof_joined": df.height,
            "members": prov,
            "selected": payload,
            "nested": {
                "family": sel,
                "fit_mode": sel_mode,
                "rmse": round(rmse(ctx.y, results[(sel, sel_mode)]["pred"], lanes["all"]), 3),
                "rmse_ex_ro": round(rmse(ctx.y, results[(sel, sel_mode)]["pred"], ~ro), 3),
                "reference_member": ref_name,
            },
            "all_families": payloads,
        }
        if fit_w_info is not None:  # absent under the default, so the file is unchanged
            doc["fit_weights"] = fit_w_info
        Path(a.emit_weights).parent.mkdir(parents=True, exist_ok=True)
        Path(a.emit_weights).write_text(json.dumps(doc, indent=2) + "\n")
        print("wrote", a.emit_weights)

    if a.write_oof:
        fam = BY_NAME[a.emit_family or rec[0]]
        mode = a.emit_mode or rec[1]
        if fam.kind == "gbdt":
            raise SystemExit("--write-oof needs a linear family")
        dst = write_nested_oof(ctx, fam, results[(fam.name, mode)], dirs, df, a.write_oof)
        print(f"wrote {dst}  (nested {fam.name}/{mode}; NOT an experiment, no registry row)")

    report = build_report(
        a,
        ctx,
        lanes,
        ro,
        members,
        member_rows,
        ref_name,
        results,
        deltas,
        rec,
        vs_best,
        best_key,
        fit_w_info,
    )
    print()
    print("\n".join(report[:200]))
    if a.write:
        Path(a.write).parent.mkdir(parents=True, exist_ok=True)
        Path(a.write).write_text("\n".join(report) + "\n")
        print("wrote", a.write)


def _jj_cols(dd: dict) -> dict:
    """``nested_jj`` / ``nested_jj_ex_ro`` and their deltas, or nothing when the members hold
    out neither January nor July."""
    if "jj" not in dd:
        return {}
    return {
        "nested_jj": fmt(dd["jj"]["rmse"], 2),
        "delta_jj": _d(dd["jj"]),
        "nested_jj_ex_ro": fmt(dd["jj|ex_ro"]["rmse"], 2),
        "delta_jj_ex_ro": _d(dd["jj|ex_ro"]),
    }


def _d(row: dict) -> str:
    s = f"{row['delta']:+.1f}"
    if "boot_lo" in row and np.isfinite(row["boot_lo"]):
        s += f" [{row['boot_lo']:+.1f}, {row['boot_hi']:+.1f}] p={row['boot_p_improve']:.2f}"
    return s


def build_report(
    a,
    ctx,
    lanes,
    ro,
    members,
    member_rows,
    ref_name,
    results,
    deltas,
    rec,
    vs_best,
    best_key,
    fit_w_info=None,
) -> list[str]:
    y = ctx.y
    L: list[str] = [
        f"# Blend over the full year — {len(members)} members, nested leave-one-month-out",
        "",
        "Generated by `scripts/blend_fit.py`. Members are aligned on `MVT_ID_mvt`. Every family",
        "is scored *nested*: the weights applied to a month were fitted on the other eleven",
        "months only. Intervals are the day-block bootstrap of `scripts/paired_delta.py`",
        f"({a.bootstrap} replicates over airport x UTC date, resampled inside airport-month,",
        f"seed {a.bootstrap_seed}); negative delta = better than the reference member",
        (
            f"`{ref_name}`. `ex_ro` = the {int(ro.sum())} day-rollover rows"
            " (`|y − 86,400| < 3,600`) removed."
        ),
        "",
        "Group keys are test-observable only (`has_aobt`, `airport`, derived-AOBT spike).",
        "`ordinary` and `gt1h` are defined through the label and appear as diagnostics, never",
        "as a rule. `fit=ex_ro` drops rollover rows from the *fitting* set (a constant fitted on",
        "labelled 2025 rows), not from anything applied to a 2026 row.",
        "",
    ]
    if fit_w_info is not None:
        L += [
            (
                f"**Fit weights: `{fit_w_info['fit_weights']}`** (`--fit-weights`). Every"
                " fitting row carries a weight that is constant in its (month, airport,"
                " has_aobt, on_spike) cell and is built from cell keys only, never a label;"
                " scoring is unweighted. "
                + ", ".join(f"{k} = {v}" for k, v in fit_w_info.items() if k != "fit_weights")
                + "."
            ),
            "",
        ]
    L += ["## 1. Members", ""]
    L += md_table(member_rows)
    L += ["## 2. Families, nested full-year", ""]
    rows = []
    for (fname, mode), r in results.items():
        d_all = deltas[(fname, mode)]["all"]
        d_ex = deltas[(fname, mode)]["all|ex_ro"]
        ins = r.get("in_sample")
        rows.append(
            {
                "family": fname,
                "fit": mode,
                "nested_rmse": fmt(rmse(y, r["pred"], lanes["all"]), 2),
                "delta_vs_ref": _d(d_all),
                "nested_ex_ro": fmt(rmse(y, r["pred"], ~ro), 2),
                "delta_ex_ro": _d(d_ex),
                "in_sample_ex_ro": fmt(rmse(y, ins, ~ro), 2) if ins is not None else "n/a",
                **_jj_cols(deltas[(fname, mode)]),
            }
        )
    for m, dd in [(k[1], v) for k, v in deltas.items() if k[0] == "member"]:
        i = members.index(m)
        rows.append(
            {
                "family": f"(member) {m}",
                "fit": "-",
                "nested_rmse": fmt(rmse(y, ctx.X[:, i], lanes["all"]), 2),
                "delta_vs_ref": _d(dd["all"]),
                "nested_ex_ro": fmt(rmse(y, ctx.X[:, i], ~ro), 2),
                "delta_ex_ro": _d(dd["all|ex_ro"]),
                "in_sample_ex_ro": "-",
                **_jj_cols(dd),
            }
        )
    L += md_table(rows)
    L += [
        (
            f"Recommended family: **`{rec[0]}`, fit `{rec[1]}`**. The rule is: the simplest"
            f" family that beats `{ref_name}` ex-rollovers with an interval excluding zero and"
            f" is not itself beaten by the best family (`{best_key[0]}` / `{best_key[1]}`,"
            " paired day-block interval below)."
        ),
        "",
        "### 2b. Every family against the best one, ex-rollovers (negative = better)",
        "",
        *md_table(
            [
                {
                    "family": f"{k[0]} / {k[1]}",
                    "nested_ex_ro": fmt(rmse(y, results[k]["pred"], ~ro), 2),
                    "vs_best": "(best)" if k == best_key else _d(vs_best[k]),
                }
                for k in sorted(results, key=lambda k: (BY_NAME[k[0]].complexity, k[1]))
            ]
        ),
        "## 3. Jan and Jul",
        "",
    ]
    rows = []
    for key in [("member", m) for m in members if m != ref_name] + list(results):
        dd = deltas[key]
        rows.append(
            {
                "family": key[1] if key[0] == "member" else f"{key[0]} / {key[1]}",
                "jan": fmt(dd["2025-01"]["rmse"], 2),
                "jan_delta": _d(dd["2025-01"]),
                "jan_ex_ro": fmt(dd["2025-01|ex_ro"]["rmse"], 2),
                "jul": fmt(dd["2025-07"]["rmse"], 2),
                "jul_delta": _d(dd["2025-07"]),
                "jul_ex_ro": fmt(dd["2025-07|ex_ro"]["rmse"], 2),
            }
        )
    L += md_table(rows)

    L += ["## 4. Per month (nested RMSE, ex-rollovers in brackets)", ""]
    first = next(iter(results))
    rows = []
    for mi, m in enumerate(ctx.month_names):
        row = {"month": m, "rows": int((ctx.row_month == mi).sum())}
        row[f"ref {ref_name[:14]}"] = (
            f"{fmt(deltas[first][m]['ref_rmse'], 1)} ({fmt(deltas[first][m + '|ex_ro']['ref_rmse'], 1)})"
        )
        for key in results:
            row[f"{key[0]}/{key[1]}"] = (
                f"{fmt(deltas[key][m]['rmse'], 1)} ({fmt(deltas[key][m + '|ex_ro']['rmse'], 1)})"
            )
        rows.append(row)
    L += md_table(rows)

    L += ["## 5. Lanes (diagnostic — never a rule)", ""]
    rows = []
    for lk in ("with_aobt", "no_aobt", "spike", "gt1h", "ordinary"):
        for suffix in ("", "|ex_ro"):
            k = lk + suffix
            row = {
                "lane": k,
                "rows": deltas[first][k]["rows"],
                "ref": fmt(deltas[first][k]["ref_rmse"], 1),
            }
            for key in results:
                row[f"{key[0]}/{key[1]}"] = _d(deltas[key][k])
            rows.append(row)
    L += md_table(rows)

    L += ["## 6. Per airport (delta vs reference, ex-rollovers)", ""]
    rows = []
    for ap in ctx.airport_names:
        k = f"ap:{ap}|ex_ro"
        row = {
            "airport": ap,
            "rows": deltas[first][k]["rows"],
            "ref": fmt(deltas[first][k]["ref_rmse"], 1),
        }
        for key in results:
            row[f"{key[0]}/{key[1]}"] = _d(deltas[key][k])
        rows.append(row)
    L += md_table(rows)

    L += ["## 7. Weight stability across the twelve nested fits", ""]
    rows = []
    for (fname, mode), r in results.items():
        if not r["per_month"]:
            continue
        fam = BY_NAME[fname]
        for gname in group_names(fam.grouping, ctx):
            W = np.array([r["per_month"][m][gname]["w"] for m in ctx.month_names])
            B = np.array([r["per_month"][m][gname]["b"] for m in ctx.month_names])
            full = r["full"][gname]
            row = {"family": fname, "fit": mode, "group": gname}
            for i, mem in enumerate(members):
                row[mem[:16]] = (
                    f"{W[:, i].mean():.3f}±{W[:, i].std(ddof=1):.3f} "
                    f"(all {float(full['w'][i]):.3f})"
                )
            if fam.kind == "ridge":
                row["intercept"] = f"{B.mean():.1f}±{B.std(ddof=1):.1f}"
            rows.append(row)
    L += md_table(rows)
    L += [
        "## 8. How to re-run",
        "",
        "```bash",
        "uv run python scripts/blend_fit.py fit \\",
        *[f"    {m} \\" for m in members],
        f"    --bootstrap {a.bootstrap} --write <report.md> --emit-weights <weights.json>"
        + (
            ""
            if fit_w_info is None
            else f" \\\n    --fit-weights {a.fit_weights} --fit-other-weight "
            f"{a.fit_other_weight} --fit-weight-cap {a.fit_weight_cap}"
        ),
        "```",
        "",
    ]
    return L


# --- apply command --------------------------------------------------------------------


def group_key_of(grouping: str, df: pl.DataFrame) -> np.ndarray:
    if grouping == "global":
        return np.array(["__all__"] * df.height)
    if grouping == "has_aobt":
        return np.where(df["has_aobt"].to_numpy(), "with_aobt", "no_aobt")
    if grouping == "on_spike":
        return np.where(df["on_spike"].to_numpy(), "on_spike", "off_spike")
    if grouping == "airport":
        return df["airport"].to_numpy().astype(str)
    raise SystemExit(f"grouping {grouping!r} is not test-observable")


def combine(df: pl.DataFrame, cols: list[str], sel: dict, gk: np.ndarray) -> np.ndarray:
    members = list(sel["groups"]["__fallback__"]["weights"])
    out = np.zeros(df.height)
    P = np.column_stack([df[c].cast(pl.Float64).to_numpy() for c in cols])
    for g in np.unique(gk):
        spec = sel["groups"].get(g, sel["groups"]["__fallback__"])
        w = np.array([spec["weights"][m] for m in members])
        m = gk == g
        out[m] = P[m] @ w + spec["intercept"]
    return out


def cmd_apply(a: argparse.Namespace) -> None:
    doc = json.loads(Path(a.weights).read_text())
    sel = doc["selected"]
    if a.family:
        if a.family not in doc.get("all_families", {}):
            raise SystemExit(
                f"{a.family!r} is not in this file: {sorted(doc.get('all_families', {}))}"
            )
        sel = doc["all_families"][a.family]
    members = list(sel["groups"]["__fallback__"]["weights"])
    dirs = [run_dir(r) for r in (a.runs or members)]
    got = [d.name for d in dirs]
    if got != members:
        raise SystemExit(f"member mismatch: weights were fitted on {members}, got {got}")
    stem = TEST_SOURCES[a.test_source]
    for d, p in zip(dirs, doc["members"], strict=True):
        for which in ("oof", "test_pred", "test_pred_foldavg"):
            key, f = f"{which}_sha256", d / f"{which}.parquet"
            if key in p and f.is_file() and sha256(f) != p[key]:
                msg = f"{d.name}/{which}.parquet digest differs from the fitted one"
                if not a.allow_digest_mismatch:
                    raise SystemExit(msg + " (pass --allow-digest-mismatch to override)")
                print("WARNING:", msg)
    out = paths.RUNS / a.out
    out.mkdir(parents=True, exist_ok=True)
    for which in a.which:
        if which == "test_pred":
            which = stem
            missing = [d.name for d in dirs if not (d / f"{which}.parquet").is_file()]
            if missing:
                raise SystemExit(
                    f"--test-source {a.test_source}: no {which}.parquet in {missing} "
                    "(only a `test_fit: full` run written after 2026-09-23 has the fold average)"
                )
        df, cols, seed_cols = load_frame(dirs, which, seeds=True)
        gk = group_key_of(sel["grouping"], df)
        keep = [
            pl.col("MVT_ID_mvt"),
            *[
                pl.col(c)
                for c in ("airport", "month", "fold", "TAXITIME_SEC_mvt")
                if c in df.columns
            ],
        ]
        blended = df.select(keep).with_columns(
            pl.Series("pred", combine(df, cols, sel, gk)),
            *[
                pl.Series(sc, combine(df, [f"{sc}__{m}" for m in members], sel, gk))
                for sc in seed_cols
            ],
        )
        dst = out / f"{which}.parquet"
        blended.write_parquet(dst)
        print(f"wrote {dst}  rows={blended.height:,}")
        if which == "oof":
            y = df["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy()
            p = blended["pred"].to_numpy()
            ro = np.abs(y - ROLLOVER_S) < ROLLOVER_WINDOW_S
            print(
                f"  in-sample-of-the-weights RMSE {np.sqrt(np.mean((y - p) ** 2)):.2f}"
                f"  ex-rollovers {np.sqrt(np.mean((y[~ro] - p[~ro]) ** 2)):.2f}"
            )
    (out / "blend.json").write_text(json.dumps(doc, indent=2) + "\n")
    print("wrote", out / "blend.json")


# --- cli ------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__ and __doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit", help="fit and score every blend family")
    f.add_argument("runs", nargs="+", help="member run names or run dirs")
    f.add_argument("--families", nargs="+", default=list(DEFAULT_FAMILIES), choices=list(BY_NAME))
    f.add_argument("--gbdt", action="store_true", help="also fit the LightGBM stacker")
    f.add_argument("--gbdt-trees", type=int, default=400)
    f.add_argument("--threads", type=int, default=8)
    f.add_argument("--reference", help="member to compare against (default: best ex-rollovers)")
    f.add_argument("--bootstrap", type=int, default=2000, metavar="N")
    f.add_argument("--bootstrap-seed", type=int, default=0)
    f.add_argument("--write", type=Path, help="markdown report")
    f.add_argument("--emit-weights", type=Path, help="weights.json fitted on all twelve months")
    f.add_argument("--emit-family", choices=list(BY_NAME))
    f.add_argument("--emit-mode", choices=list(FIT_MODES))
    f.add_argument(
        "--write-oof",
        metavar="RUN",
        help="write the nested blend OOF to outputs/runs/<RUN>/oof.parquet in the native "
        "schema, so paired_delta / lane_report can read it",
    )
    f.add_argument("--no-digests", action="store_true", help="skip sha256 of member parquets")
    f.add_argument(
        "--fit-weights",
        choices=list(FIT_WEIGHTS),
        default="none",
        help="per-row weights of the FIT set: none (default, unchanged), jj (Jan/Jul rows), "
        "mix2026 (Jan/Jul rows at 2026 cell share / 2025 cell share, capped)",
    )
    f.add_argument(
        "--fit-other-weight",
        type=float,
        default=0.0,
        help="weight of the non-Jan/Jul months under --fit-weights jj|mix2026 (0 = drop them)",
    )
    f.add_argument(
        "--fit-weight-cap", type=float, default=5.0, help="cap of the mix2026 cell ratio"
    )
    f.set_defaults(fn=cmd_fit)

    p = sub.add_parser("apply", help="apply a weights.json to member runs")
    p.add_argument("weights")
    p.add_argument("--runs", nargs="+")
    p.add_argument("--which", nargs="+", default=["oof"], choices=["oof", "test_pred"])
    p.add_argument("--out", required=True, help="run name under outputs/runs/")
    p.add_argument(
        "--family",
        metavar="NAME|MODE",
        help="use one of the file's `all_families` entries instead of `selected`",
    )
    p.add_argument("--allow-digest-mismatch", action="store_true")
    p.add_argument(
        "--test-source",
        choices=list(TEST_SOURCES),
        default="test_pred",
        help="with --which test_pred: blend the members' test_pred.parquet (default) or their "
        "fold-model average test_pred_foldavg.parquet (written as test_pred_foldavg.parquet)",
    )
    p.set_defaults(fn=cmd_apply)
    return ap


def main(argv: list[str] | None = None) -> None:
    a = build_parser().parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main(sys.argv[1:])
