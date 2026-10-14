"""Per-lane blend of the trained members, fitted month-held on 2025.

usage (repo root): python final/blend.py

Members are combined per lane -- four row populations with very different error
structure -- by non-negative least squares on the simplex (weights >= 0, sum to
1, lightly shrunk toward equal weights):

  nm_lt1h     NM rows with C < 1 h (the bulk)
  nm_ge1h     NM rows with C >= 1 h (long delays, heavy tail)
  nonm_lirf   LIRF rows without an NM plan (smallest cell, largest per-row error)
  nonm_other  no-NM rows elsewhere

Weights are fitted on the members' month-held 2025 out-of-fold predictions: the
weights scored on month k are fitted on the other eleven months, so every 2025
month gets an honest read, and the rows of the never-evaluated submission
months are leave-that-month-out estimates. Nothing here sees 2026 labels.

Two member generations cover the submission rows (see docs/06-final-model.md):

* the PRIMARY members trained while ranking.parquet (January/July) was the
  released evaluation set; their fold-0 files predict exactly those rows, and
  each member produced its 13 folds (OOF and fold 0) in one nested run, so its
  out-of-fold behaviour is a faithful proxy for its fold-0 behaviour;
* fold-0 RETRAINS made after final_ranking.parquet (all four months) appeared;
  they cover the months the primary members never predicted. The class and
  anomaly models are excluded on those months: the anomaly model's
  member-deviation features exist only for the January/July rows (its means
  collapse on the other months), and the class model consumed it as an expert
  (a 35-65 s depression against its sibling members on exactly those months).

Each submission row is blended with the generation that covers its month,
under weights fitted month-held on 2025 with that generation's member list.

reads   work/<member>/outer_NN.parquet, data/raw/ (movement tables)
writes  work/blend_oof.parquet, work/blend_test.parquet,
        work/blend_weights.json, results/blend_report.txt
"""

import glob
import json
import os
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")  # R-arrow ext types in the raw files
import numpy as np
import polars as pl
from scipy.optimize import nnls

KEY = "MVT_ID_mvt"
TARGET = "TAXITIME_SEC_mvt"
CLIP = 3620.0  # label window: |BLOCK - LOBT| <= 3606 s exactly; clip with margin
WORK = Path(os.environ.get("WORK_DIR", "work"))
RAW = Path(os.environ.get("RAW_DIR", "data/raw"))
OUT = Path(os.environ.get("BLEND_DIR", str(WORK)))
RESULTS = Path(os.environ.get("RESULTS_DIR", "results"))
RESULTS.mkdir(exist_ok=True)

# member -> (primary tag, prediction column); outer_01..12 of the tag are its
# month-held 2025 OOF, outer_00 its prediction of the initial evaluation rows.
MEMBERS = {
    "cm": ("cm_cmA", "y_cm"),
    "cmna": ("cm_cmNA", "y_cm"),
    "gbm2_mix": ("gbm_v2", "y_mix"),
    "gbm2_s1": ("gbm_v2", "y_s1"),
    "gbm3a": ("gbm_v3a", "y_catmix"),
    "vap3": ("gbm_vap3", "y_catmixapy_s201"),
    "anom": ("gbm_anom1", "y_anom_s1"),
    "ftc": ("ft_ftC7", "y_ft"),
    "ftne": ("ft_ftNE", "y_ft"),
    "ftne4": ("ft_ftNE4", "y_ft"),
    "ftb7": ("ft_ftB7", "y_ft"),
    "ftsc": ("ft_ftSC", "y_ft"),
    "ftn10": ("ft_ftN10", "y_ft"),
}
# member -> (fold-0 retrain tag, column): covers the months the primary tags
# never predicted. No retrain exists for cm / cmna / anom / ftne4 (see above).
RETRAIN = {
    "gbm2_mix": ("gbm_v2F", "y_mix"),
    "gbm2_s1": ("gbm_v2F", "y_s1"),
    "gbm3a": ("gbm_v3aF", "y_catmix"),
    "vap3": ("gbm_vap3F", "y_catmixapy_s201"),
    "ftc": ("ft_ftCF", "y_ft"),
    "ftne": ("ft_ftNEFF", "y_ft"),
    "ftb7": ("ft_ftB7F", "y_ft"),
    "ftsc": ("ft_ftSCF", "y_ft"),
    "ftn10": ("ft_ftN10F", "y_ft"),
}

LANE_MEMBERS = {
    "nm_lt1h": ["cm", "gbm2_mix", "gbm2_s1", "gbm3a", "vap3", "anom", "ftb7", "ftsc", "ftc"],
    "nm_ge1h": ["cm", "gbm2_mix", "gbm2_s1", "gbm3a", "vap3", "anom", "ftb7", "ftsc", "ftne", "ftne4", "ftn10"],
    "nonm_other": ["cm", "cmna", "ftc", "ftne", "ftn10", "vap3", "gbm2_mix"],
    "nonm_lirf": ["cm", "cmna", "ftc", "gbm2_mix"],
}
#: lanes restricted to the retrain generation (months without primary coverage)
RETRAIN_LANES = {
    "nm_lt1h": ["gbm2_mix", "gbm2_s1", "gbm3a", "vap3", "ftb7", "ftsc", "ftc"],
    "nm_ge1h": ["gbm2_mix", "gbm2_s1", "gbm3a", "vap3", "ftb7", "ftsc", "ftne", "ftn10"],
    "nonm_other": ["ftc", "ftne", "ftn10", "vap3", "gbm2_mix"],
    "nonm_lirf": ["ftc", "ftne", "ftn10", "gbm2_mix", "gbm3a"],
}


def rmse(e) -> float:
    e = np.asarray(e, dtype=float)
    return float(np.sqrt(np.mean(e * e))) if len(e) else float("nan")


def block_bootstrap(y, pred_a, pred_b, blocks, n_boot=1000, seed=0):
    """Day-block bootstrap of RMSE(pred_b) - RMSE(pred_a); negative = b better."""
    y, pred_a, pred_b = (np.asarray(v, float) for v in (y, pred_a, pred_b))
    sq_a, sq_b = (pred_a - y) ** 2, (pred_b - y) ** 2
    codes, inv = np.unique(blocks, return_inverse=True)
    sa = np.bincount(inv, sq_a, len(codes))
    sb = np.bincount(inv, sq_b, len(codes))
    cn = np.bincount(inv, None, len(codes))
    rng = np.random.default_rng(seed)
    point = np.sqrt(sb.sum() / cn.sum()) - np.sqrt(sa.sum() / cn.sum())
    deltas = []
    for _ in range(n_boot):
        k = rng.integers(0, len(codes), len(codes))
        n = cn[k].sum()
        deltas.append(np.sqrt(sb[k].sum() / n) - np.sqrt(sa[k].sum() / n))
    return float(point), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def simplex_fit(X: np.ndarray, target: np.ndarray) -> np.ndarray:
    """NNLS on the simplex with a light ridge toward equal weights."""
    lam = 1e3 * np.sqrt(len(target))
    w, _ = nnls(np.vstack([X, lam * np.ones((1, X.shape[1]))]), np.concatenate([target, [lam]]))
    return w / w.sum()


def read_pred(tag: str, fold: int, col: str, name: str) -> pl.DataFrame:
    df = pl.read_parquet(WORK / tag / f"outer_{fold:02d}.parquet")
    return df.select(KEY, pl.col(col).cast(pl.Float64).alias(name))


def oof_table() -> pl.DataFrame:
    out = None
    for name, (tag, col) in MEMBERS.items():
        part = pl.concat([read_pred(tag, k, col, name) for k in range(1, 13)])
        out = part if out is None else out.join(part, on=KEY, how="full", coalesce=True)
    return out


def label_shift() -> pl.DataFrame:
    """Per-row label shift between the two releases of the evaluation rows.

    final_ranking.parquet re-published the initial months' flights with the
    take-off stamp dithered by a few seconds and everything else identical, so
    the final-row label equals the initial-row label + (MVT_final - MVT_initial).
    A primary member's prediction maps to the final row by adding that shift."""
    mvt = lambda f: (pl.scan_parquet(RAW / f).filter(pl.col("PHASE_mvt") == "DEP")  # noqa: E731
                     .select(KEY, "MVT_TIME_UTC_mvt").collect())
    return (mvt("ranking.parquet").rename({"MVT_TIME_UTC_mvt": "m0"})
            .join(mvt("final_ranking.parquet"), on=KEY, how="inner", maintain_order="left")
            .select(KEY, (pl.col("MVT_TIME_UTC_mvt") - pl.col("m0")).dt.total_seconds()
                    .cast(pl.Float64).alias("shift")))


def test_table() -> pl.DataFrame:
    """One column per member: primary prediction (+ label shift) where the
    primary fold-0 covers the row, the fold-0 retrain elsewhere."""
    shift = label_shift()
    out = None
    for name, (tag, col) in MEMBERS.items():
        part = (read_pred(tag, 0, col, "p").join(shift, on=KEY, how="left", maintain_order="left")
                .select(KEY, (pl.col("p") + pl.col("shift").fill_null(0.0)).alias(name)))
        if name in RETRAIN:
            rtag, rcol = RETRAIN[name]
            r = read_pred(rtag, 0, rcol, "r")
            part = r.join(part, on=KEY, how="left", maintain_order="left").select(
                KEY, pl.coalesce(name, "r").alias(name))
        out = part if out is None else out.join(part, on=KEY, how="full", coalesce=True)
    return out


def build_meta() -> pl.DataFrame:
    sec = lambda a, b: (pl.col(a) - pl.col(b)).dt.total_seconds()  # noqa: E731
    cols = [KEY, "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "LOBT_flt", TARGET]
    train = pl.concat([
        pl.scan_parquet(f).filter(pl.col("PHASE_mvt") == "DEP").select(cols).collect()
        for f in sorted(glob.glob(str(RAW / "training_2025-*.parquet")))
    ]).with_columns(pl.lit("train").alias("split"))
    test = (pl.scan_parquet(RAW / "final_ranking.parquet")
            .filter(pl.col("PHASE_mvt") == "DEP").select(cols).collect()
            .with_columns(pl.lit("test").alias("split")))
    return pl.concat([train, test]).with_columns(
        C=sec("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"),
        ml=sec("MVT_TIME_UTC_mvt", "LOBT_flt"),
        nm=pl.col("LOBT_flt").is_not_null().cast(pl.Float64),
        mon=pl.col("MVT_TIME_UTC_mvt").dt.month(),
        day=pl.col("MVT_TIME_UTC_mvt").dt.date().cast(pl.Utf8),
        is_test=(pl.col("split") == "test"),
        y=pl.col(TARGET).cast(pl.Float64),
    ).with_columns(
        lo=pl.max_horizontal(pl.lit(0.0), pl.col("ml") - CLIP),
        hi=pl.col("ml") + CLIP,
    ).select(KEY, "ADEP_mvt", "mon", "day", "nm", "C", "ml", "lo", "hi", "y", "is_test")


def lane_of(frame: pl.DataFrame) -> np.ndarray:
    nm = frame["nm"].to_numpy() > 0
    C = frame["C"].to_numpy()
    lirf = (frame["ADEP_mvt"] == "LIRF").to_numpy()
    return np.where(nm & (C < 3600), "nm_lt1h",
                    np.where(nm, "nm_ge1h", np.where(lirf, "nonm_lirf", "nonm_other")))


def design(frame: pl.DataFrame, members: list[str], fallback: str = "gbm2_mix") -> np.ndarray:
    X = np.column_stack([frame[m].cast(pl.Float64).fill_null(np.nan).to_numpy()
                         if m in frame.columns else np.full(frame.height, np.nan)
                         for m in members])
    fb = frame[fallback].cast(pl.Float64).to_numpy()
    return np.where(np.isfinite(X), X, fb[:, None])


def main() -> None:
    meta = build_meta()
    test = meta.filter(pl.col("is_test")).join(test_table(), on=KEY, how="left", maintain_order="left")
    # months without primary coverage use the retrain generation
    cov = test.group_by("mon").agg(pl.col("cm").null_count().alias("nn"), pl.len())
    retrain_months = sorted(cov.filter(pl.col("nn") == pl.col("len"))["mon"].to_list())
    oof = meta.filter(~pl.col("is_test")).join(oof_table(), on=KEY, how="left", maintain_order="left")
    y = oof["y"].to_numpy()
    month = oof["mon"].to_numpy()
    rollover = np.abs(y - 86400) < 3600
    lane = lane_of(oof)
    blocks = (oof["ADEP_mvt"] + "_" + oof["day"]).to_numpy()
    report = open(RESULTS / "blend_report.txt", "w")

    def log(*a):
        print(*a, flush=True)
        print(*a, file=report, flush=True)

    log(f"retrain-generation months (no primary fold-0 coverage): {retrain_months}")

    # delivered-equivalent OOF: retrain months are blended with the retrain
    # lists (what their test rows get), the other months with the full lists.
    blended = np.full(oof.height, np.nan)
    weights = {}
    for lane_name in LANE_MEMBERS:
        in_lane = lane == lane_name
        for variant, members in (("primary", LANE_MEMBERS[lane_name]), ("retrain", RETRAIN_LANES[lane_name])):
            X = design(oof, members)
            if variant == "primary":
                for m, col in zip(members, X.T):
                    e = col[in_lane & ~rollover] - y[in_lane & ~rollover]
                    log(f"  [{lane_name}] {m:10s} RMSE {rmse(e):8.2f}")
            months = retrain_months if variant == "retrain" else [k for k in range(1, 13) if k not in retrain_months]
            for k in months:
                fit_rows = in_lane & ~rollover & (month != k)
                score_rows = in_lane & (month == k)
                if score_rows.any():
                    blended[score_rows] = X[score_rows] @ simplex_fit(X[fit_rows], y[fit_rows])
            w = simplex_fit(X[in_lane & ~rollover], y[in_lane & ~rollover])
            weights[f"{lane_name}_{variant}"] = dict(zip(members, [round(float(x), 4) for x in w]))
            log(f"  [{lane_name}] {variant} weights: " +
                ", ".join(f"{m}={x:.2f}" for m, x in weights[f"{lane_name}_{variant}"].items() if x >= 0.005))

    lo = np.maximum(oof["lo"].fill_null(0).to_numpy(), 0)
    hi = oof["hi"].fill_null(1e18).to_numpy()
    nm = oof["nm"].to_numpy() > 0
    blended = np.where(nm, np.clip(blended, lo, hi), np.maximum(blended, 0))
    baseline = oof["cm"].cast(pl.Float64).to_numpy()
    baseline = np.where(np.isfinite(baseline), baseline, oof["gbm2_mix"].to_numpy())
    baseline = np.where(nm, np.clip(baseline, lo, hi), np.maximum(baseline, 0))
    ex = ~rollover
    log(f"\n2025 month-held blend, ex-rollover: RMSE {rmse(blended[ex] - y[ex]):.3f} "
        f"(class model alone {rmse(baseline[ex] - y[ex]):.3f}); all rows {rmse(blended - y):.3f}")
    point, lo_ci, hi_ci = block_bootstrap(y[ex], baseline[ex], blended[ex], blocks[ex])
    log(f"blend vs class model: {point:+.3f} [{lo_ci:+.3f}, {hi_ci:+.3f}] (day-block bootstrap)")
    for k in retrain_months:
        rows = ex & (month == k)
        log(f"  month {k} (leave-out read of a retrain month) RMSE {rmse(blended[rows] - y[rows]):8.3f} n {int(rows.sum())}")
    for lane_name in LANE_MEMBERS:
        rows = ex & (lane == lane_name)
        log(f"  lane {lane_name:11s} RMSE {rmse(blended[rows] - y[rows]):8.2f} n {int(rows.sum())}")
    for k in range(1, 13):
        rows = ex & (month == k)
        log(f"  month {k:2d} RMSE {rmse(blended[rows] - y[rows]):8.2f}")

    oof.select(KEY, "mon", "ADEP_mvt", "day", "nm", "C").with_columns(
        y=pl.Series(y), blend=pl.Series(blended)).write_parquet(OUT / "blend_oof.parquet")
    with open(OUT / "blend_weights.json", "w") as f:
        json.dump(weights, f, indent=1)

    mon_t = test["mon"].to_numpy()
    on_retrain = np.isin(mon_t, retrain_months)
    lane_t = lane_of(test)
    pred = np.full(test.height, np.nan)
    for lane_name in LANE_MEMBERS:
        in_lane = lane_t == lane_name
        for variant, members, rows in (("primary", LANE_MEMBERS[lane_name], in_lane & ~on_retrain),
                                       ("retrain", RETRAIN_LANES[lane_name], in_lane & on_retrain)):
            X = design(test, members)
            w = np.array([weights[f"{lane_name}_{variant}"][m] for m in members])
            pred[rows] = X[rows] @ w
        log(f"  [{lane_name}] test rows primary/retrain: "
            f"{int((in_lane & ~on_retrain).sum())}/{int((in_lane & on_retrain).sum())}")
    assert np.isfinite(pred).all()
    test.select(KEY, "mon", "ADEP_mvt", "nm", "C", "ml", "lo", "hi").with_columns(
        blend=pl.Series(pred)).write_parquet(OUT / "blend_test.parquet")
    log(f"written {OUT}/blend_oof.parquet, {OUT}/blend_test.parquet")


if __name__ == "__main__":
    main()
