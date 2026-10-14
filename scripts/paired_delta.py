"""Paired seed deltas between two runs (validation-protocol ADR 2026-09-17).

Both runs must share the CV scheme and seeds and carry per-seed OOF columns (``pred_s<seed>``).
Reported pooled, per month, per lane (with / without AOBT, derived-AOBT spike, > 1 h,
ordinary) and per airport.

**One estimand (2026-09-23).** ``ref_rmse`` / ``cand_rmse`` / ``delta`` are the RMSE of the
**seed-averaged** prediction (the mean of the ``pred_s<seed>`` columns, which is what a
submission carries and what the registry scores), so the point estimate and the bootstrap
interval now describe the same quantity. Until 2026-09-23 the printed point was the mean over
seeds of per-seed RMSE deltas while the interval was built on the seed-averaged prediction, and
the point could sit outside its own interval (review 2026-09-23 B, finding 9). The per-seed
numbers are kept as diagnostics: ``delta_seedmean`` (the old ``delta_mean``), ``delta_sd`` and
``seeds_improved`` describe seed-to-seed spread; gates read ``delta``.

**Jan + Jul pooled (2026-09-23).** When the run holds out January or July 2025, a ``jj`` row
(the two months pooled over rows, the population closest to the Jan + Jul 2026 ranking set)
follows the month rows; with ``--ex-rollovers`` it also gets its ``jj|ex_ro`` row. It is
bootstrapped with the same day blocks as every other row.

``--ex-rollovers`` repeats every lane with the day-rollover rows (``|y − 86,400| < 3,600``)
removed, as a second ``<lane>|ex_ro`` row. Those rows are a handful per fold and can own a
run's headline number — 5 of them carry 16-21 % of the gate fold's squared error and the whole
of ``exp0057``'s +10.4 — while the 2026
ranking months are expected to hold ≈ 3, Poisson. The ex-rollover delta is therefore the part
of a head result that will still be there in 2026, and the ``rollovers`` column says how many
rows the pair differs by.

``--bootstrap N`` adds a block bootstrap over **calendar days within airport-month**: the block
is ``airport x UTC date of the movement`` and each replicate redraws each airport-month's days
with replacement (so month sizes stay roughly fixed), then recomputes the candidate - reference
RMSE delta on the seed-averaged predictions. It answers the question the seed sd cannot: 7
day-rollover rows hold 38 % of the gate fold's squared error, so a pooled delta is decided by a
handful of rows and a seed sd of 0.1-0.4 s badly understates how much of the number is luck of
the calendar. ``boot_lo`` / ``boot_hi`` are the 2.5 / 97.5 percentiles of the replicate deltas
and ``boot_p_improve`` the share of replicates in which the candidate wins. Off by default; with
``--bootstrap 0`` the table is the same table without the three bootstrap columns.

    uv run python scripts/paired_delta.py exp0008_xgb_ref3 exp0009_xgb_openmeteo [--write path.md]
    uv run python scripts/paired_delta.py exp0047_xgb_r2_weight_1mp exp0061_xgb_r4_zhead \
        --ex-rollovers --bootstrap 2000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from prc import paths

SPIKE = (1194, 1206)
ROLLOVER_S, ROLLOVER_WINDOW_S = 86_400.0, 3_600.0
#: label-free C = MVT - SCHED per 2025 departure (same source as scripts/gate_runs.py)
C_FILE = paths.OUT / "mixture_v2" / "exp0227_r7c_realmlp_nomask_lane__v2.parquet"
#: The two 2025 months whose pooled rows stand in for the Jan + Jul 2026 ranking set.
JJ_MONTHS = ("2025-01", "2025-07")


def jj_mask(month: np.ndarray) -> np.ndarray | None:
    """Rows of January or July 2025, or ``None`` when the run holds out neither month."""
    m = np.isin(month, JJ_MONTHS)
    return m if m.any() else None


def load(run: str) -> pl.DataFrame:
    oof = pl.read_parquet(paths.RUNS / run / "oof.parquet")
    seeds = [c for c in oof.columns if c.startswith("pred_s")]
    if not seeds:
        raise SystemExit(f"{run}: no per-seed OOF columns; rerun with the 2026-09-18 runner")
    return oof.select("MVT_ID_mvt", "month", "airport", "TAXITIME_SEC_mvt", *seeds)


def block_index(d: pl.DataFrame) -> tuple[np.ndarray, list[np.ndarray]]:
    """``(block_of_row, blocks_of_each_group)`` for the day-block bootstrap.

    A *block* is one airport on one UTC calendar date of the movement; a *group* is one
    airport-month, and days are resampled with replacement inside their own group so every
    replicate keeps each airport-month at (roughly) its real size.
    """
    day = d["MVT_TIME_UTC_mvt"].dt.date().cast(pl.Utf8).fill_null("NaT")
    block_key = (d["airport"].cast(pl.Utf8) + "|" + day).to_numpy()
    group_key = (d["airport"].cast(pl.Utf8) + "|" + d["month"].cast(pl.Utf8)).to_numpy()
    _, block_of_row = np.unique(block_key, return_inverse=True)
    _, group_of_row = np.unique(group_key, return_inverse=True)
    n_blocks = int(block_of_row.max()) + 1
    group_of_block = np.zeros(n_blocks, dtype=np.int64)
    group_of_block[block_of_row] = group_of_row
    n_groups = int(group_of_row.max()) + 1
    return block_of_row, [np.flatnonzero(group_of_block == g) for g in range(n_groups)]


def block_weights(groups: list[np.ndarray], n_blocks: int, n_rep: int, seed: int) -> np.ndarray:
    """``(n_rep, n_blocks)`` matrix of how often each day was drawn in each replicate."""
    rng = np.random.default_rng(seed)
    w = np.zeros((n_rep, n_blocks), dtype=np.float64)
    rep = np.arange(n_rep)[:, None]
    for gb in groups:
        k = len(gb)
        draws = rng.integers(0, k, size=(n_rep, k))
        flat = (rep * k + draws).ravel()
        w[:, gb] = np.bincount(flat, minlength=n_rep * k).reshape(n_rep, k).astype(np.float64)
    return w


def bootstrap_delta(
    w: np.ndarray,
    block_of_row: np.ndarray,
    n_blocks: int,
    se_ref: np.ndarray,
    se_cand: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    """2.5 / 97.5 percentiles and P(candidate better) of the replicate RMSE deltas on one slice.

    Per-block sums of squared error and row counts are formed once, so a replicate is a weighted
    sum: two mat-vecs of ``(n_rep, n_blocks)`` per slice, seconds for N = 2000.
    """
    b = block_of_row[mask]
    sse_r = np.bincount(b, weights=se_ref[mask], minlength=n_blocks)
    sse_c = np.bincount(b, weights=se_cand[mask], minlength=n_blocks)
    n_b = np.bincount(b, minlength=n_blocks).astype(np.float64)
    den = w @ n_b
    ok = den > 0
    delta = np.full(w.shape[0], np.nan)
    if ok.any():
        delta[ok] = np.sqrt((w @ sse_c)[ok] / den[ok]) - np.sqrt((w @ sse_r)[ok] / den[ok])
    delta = delta[np.isfinite(delta)]
    if delta.size == 0:
        return {"boot_lo": float("nan"), "boot_hi": float("nan"), "boot_p_improve": float("nan")}
    lo, hi = np.percentile(delta, [2.5, 97.5])
    return {
        "boot_lo": round(float(lo), 2),
        "boot_hi": round(float(hi), 2),
        "boot_p_improve": round(float((delta < 0).mean()), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("candidate")
    ap.add_argument("--write", type=Path)
    ap.add_argument(
        "--ex-rollovers",
        action="store_true",
        help="also report every lane with the |y - 86400| < 3600 rows removed",
    )
    ap.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        metavar="N",
        help="N block-bootstrap replicates over calendar days within airport-month "
        "(0 = off, the default); adds boot_lo / boot_hi / boot_p_improve",
    )
    ap.add_argument("--bootstrap-seed", type=int, default=0, help="RNG seed for --bootstrap")
    ap.add_argument(
        "--months",
        nargs="+",
        default=None,
        help="score only rows of these months (e.g. the months whose features a data change "
        "left untouched); every lane, including `all`, is then restricted to them",
    )
    ap.add_argument(
        "--c-min",
        type=float,
        default=None,
        help="score only rows with C = MVT - SCHED >= this many seconds (covariate lane; 2026-10-01, for "
        "changes to the C >= 1 h rows, e.g. --c-min 3600); every lane is then restricted to them",
    )
    ap.add_argument("--c-max", type=float, default=None, help="score only rows with C < this many seconds")
    a = ap.parse_args()
    if a.bootstrap < 0:
        raise SystemExit("--bootstrap must be >= 0")
    r, c = load(a.reference), load(a.candidate)
    if a.months:
        r = r.filter(pl.col("month").is_in(a.months))
    seeds = sorted(
        {x for x in r.columns if x.startswith("pred_s")}
        & {x for x in c.columns if x.startswith("pred_s")}
    )
    if not seeds:
        raise SystemExit("no common seeds")
    mv = pl.read_parquet(
        paths.INTERIM / "movements.parquet",
        columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt"],
    )
    d = (
        r.join(
            c.select("MVT_ID_mvt", *[pl.col(s).alias(s + "_c") for s in seeds]),
            on="MVT_ID_mvt",
            how="inner",
        )
        .join(mv, on="MVT_ID_mvt", how="left")
        .with_columns(
            (pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds().alias("d_aobt")
        )
    )
    # C = MVT - SCHED (label-free) for the covariate lanes c_lt1h / c_ge1h; the label-defined `ordinary` and
    # `gt1h` lanes condition on y and favour any change that lowers predictions on uncertain rows (red team 10-01)
    # Without the C file (a fresh checkout before scripts/mixture_v2.py has run, or the synthetic
    # tests) the two C lanes are left out; --c-min / --c-max need the file.
    has_c = C_FILE.is_file()
    if not has_c and (a.c_min is not None or a.c_max is not None):
        raise SystemExit(f"--c-min / --c-max need {C_FILE} (written by scripts/mixture_v2.py)")
    if has_c:
        d = d.join(
            pl.read_parquet(C_FILE, columns=["MVT_ID_mvt", "c_sched"]).unique("MVT_ID_mvt"), on="MVT_ID_mvt", how="left"
        )
    if a.c_min is not None or a.c_max is not None:
        lo = a.c_min if a.c_min is not None else -np.inf
        hi = a.c_max if a.c_max is not None else np.inf
        d = d.filter((pl.col("c_sched") >= lo) & (pl.col("c_sched") < hi))
    c_s = d["c_sched"].to_numpy() if has_c else None
    y = d["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy()
    has = d["AOBT_3_flt"].is_not_null().to_numpy()
    spike = d["d_aobt"].is_between(*SPIKE).fill_null(False).to_numpy()
    big = y > 3600
    month = d["month"].to_numpy()
    airport = d["airport"].to_numpy()
    # one row per month actually held out: two for `month_jan_jul`, twelve for `month_loo`
    # (an assumed Jan/Jul pair would report eleven empty lanes on a LOO run and hide the rest)
    jj = jj_mask(month)
    lanes = {
        "all": np.ones_like(has),
        **{m: month == m for m in sorted(set(month))},
        **({"jj": jj} if jj is not None else {}),
        "with_aobt": has,
        "no_aobt": ~has,
        "spike": spike,
        "gt1h": big,
        "ordinary": has & ~spike & ~big,
        **({"c_lt1h": c_s < 3600, "c_ge1h": c_s >= 3600} if c_s is not None else {}),
        **{
            f"{m}:{lane}": (month == m) & mask
            for m in sorted(set(month))
            for lane, mask in {
                "ordinary": has & ~spike & ~big,
                "no_aobt": ~has,
                "gt1h": big,
            }.items()
        },
        **{f"ap:{ap_}": airport == ap_ for ap_ in sorted(set(airport))},
    }
    # a day rollover is `y ~ 86,400`: a handful of rows that can own the headline number
    rollover = np.abs(y - ROLLOVER_S) < ROLLOVER_WINDOW_S
    if a.ex_rollovers:
        lanes = {
            k: m
            for lane, mask in lanes.items()
            for k, m in ((lane, mask), (f"{lane}|ex_ro", mask & ~rollover))
        }

    p_ref = np.stack([d[s].cast(pl.Float64).to_numpy() for s in seeds])
    p_cand = np.stack([d[s + "_c"].cast(pl.Float64).to_numpy() for s in seeds])

    def rm(p: np.ndarray, m: np.ndarray) -> float:
        return float(np.sqrt(np.mean((y[m] - p[m]) ** 2)))

    # the seed-averaged prediction is what a submission would carry, so it is the estimand of
    # both the printed point and the calendar bootstrap
    avg_ref, avg_cand = p_ref.mean(axis=0), p_cand.mean(axis=0)
    se_ref = (y - avg_ref) ** 2
    se_cand = (y - avg_cand) ** 2
    if a.bootstrap:
        block_of_row, groups = block_index(d)
        n_blocks = int(block_of_row.max()) + 1
        w = block_weights(groups, n_blocks, a.bootstrap, a.bootstrap_seed)

    rows = []
    for lane, m in lanes.items():
        ref = np.array([rm(p, m) for p in p_ref])
        cand = np.array([rm(p, m) for p in p_cand])
        delta = cand - ref
        ref_avg, cand_avg = rm(avg_ref, m), rm(avg_cand, m)
        row = {
            "lane": lane,
            "rows": int(m.sum()),
            "rollovers": int(rollover[m].sum()),
            "ref_rmse": round(ref_avg, 1),
            "cand_rmse": round(cand_avg, 1),
            "delta": round(cand_avg - ref_avg, 2),
            "delta_seedmean": round(float(delta.mean()), 2),
            "delta_sd": round(float(delta.std(ddof=1)) if len(seeds) > 1 else float("nan"), 2),
            "seeds_improved": int((delta < 0).sum()),
            "n_seeds": len(seeds),
        }
        if a.bootstrap:
            row |= bootstrap_delta(w, block_of_row, n_blocks, se_ref, se_cand, m)
        rows.append(row)
    out = pl.DataFrame(rows)
    with pl.Config(tbl_rows=len(rows), tbl_cols=len(out.columns), tbl_width_chars=200):
        print(f"paired delta: {a.candidate} − {a.reference}  (negative = candidate better)")
        print(out)
    if a.write:
        cols = out.columns
        lines = [
            f"# Paired delta — {a.candidate} vs {a.reference}",
            "",
            (
                "Generated by `scripts/paired_delta.py`. Negative delta = candidate better. "
                "`ref_rmse` / `cand_rmse` / `delta` are the seed-averaged prediction's RMSE "
                "(the estimand of the interval); `delta_seedmean` / `delta_sd` / "
                "`seeds_improved` are per-seed diagnostics. `jj` = 2025-01 and 2025-07 pooled. "
                f"Seeds: {', '.join(s.removeprefix('pred_s') for s in seeds)}."
                + (
                    f" Block bootstrap: {a.bootstrap} replicates over calendar days within "
                    f"airport-month, seed {a.bootstrap_seed}; boot_lo/boot_hi are the 2.5/97.5 "
                    "percentiles of the delta on the seed-averaged predictions."
                    if a.bootstrap
                    else ""
                )
            ),
            "",
            "| " + " | ".join(cols) + " |",
            "|" + "---|" * len(cols),
            *["| " + " | ".join(str(v) for v in rr) + " |" for rr in out.iter_rows()],
            "",
        ]
        a.write.write_text("\n".join(lines))
        print("wrote", a.write)


if __name__ == "__main__":
    main()
