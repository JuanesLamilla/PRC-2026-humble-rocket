"""The no-flight-plan lane model of internal note 2026-09-19_no-nm-block-model, as a
post-processor over a stored run.

The lane is the departures with no Network-Manager record (``AOBT_3_flt`` null, and with it
every ``_flt`` column): 1.08 % of the 2025 training rows, 1.53 % of the ranking rows, and
45.7 % of R5's full-year squared error. On it the study measured a two-branch estimator

    pred = p̂ · C_sched + (1 − p̂) · ŷ_lane ,     C_sched = MVT_TIME − SCHED_TIME

with both parts fitted **on the lane only** and leave-one-month-out, beating R5's own mixture
on the lane. This script is that estimator, kept outside `src/prc` and outside the runner: it
reads a finished run's `oof.parquet` / `test_pred.parquet`, replaces the prediction on the
lane rows, and writes a new run-shaped directory, so `scripts/paired_delta.py` and
`scripts/blend_fit.py` can score it against the original with no change to either.

    # 1. fit the lane model against a month_loo run and cache its predictions
    uv run python scripts/no_nm_lane_model.py fit --run exp0068_..._month_loo

    # 2. write a run-shaped directory with the lane predictions swapped in
    uv run python scripts/no_nm_lane_model.py apply --run exp0068_..._month_loo \
        --lane outputs/lane/exp0068_..._month_loo.parquet --out exp0068_lane_all

    # the LIRF-only variant (`M9` in the report)
    uv run python scripts/no_nm_lane_model.py apply --run ... --lane ... \
        --airports LIRF --out exp0068_lane_lirf

    # 3. score it
    uv run python scripts/paired_delta.py exp0068_..._month_loo exp0068_lane_lirf \
        --ex-rollovers --bootstrap 2000

**Scope and honesty.** `fit` is leave-one-month-out over the source run's own months, so the
OOF side is fold-safe in exactly the sense the validation protocol requires; the ranking side
is fitted on all twelve 2025 months, which is what the runner does for its own test-time head.
The lane model is deterministic (one LightGBM fit per fold, no seed loop), so `apply` writes
the same lane value into every `pred_s<seed>` column — the per-seed spread on the lane rows
collapses to zero and a paired-seed sd over those rows is no longer meaningful. Use the
day-block bootstrap, which is the protocol's interval anyway.

**Restricting by airport.** `--airports LIRF` reproduces the report's `M9`; that restriction
was picked after reading the per-airport table, so it is a fitted choice. `--auto-airports`
is the fold-safe replacement: inside each leave-one-month-out fold the airports are chosen by
comparing the lane model against the source run's own prediction on the **other** months'
lane rows only, and the ranking side uses all twelve months. Nothing about the held-out month
enters its own selection.

**`--stand-bounds` (C3).** `stand_bounds` is in
R5's `mask_for_no_nm`, so `taxi_upper_bound_s` / `taxi_lower_bound_s` are nulled exactly on
the rows where they discriminate. This flag hands the *default classifier* (not the
regressor) the unmasked 24 h-capped bounds from the `stand_bounds_v2` feature parquet, plus
the test-observable form of the violation signal: `C_sched − taxi_upper_bound_s` and its
sign, i.e. "would the schedule-default value break a physically valid bound".

**`--rollover-rule` (C5, §5.1).** Rows with a doubled ICAO prefix (`ITY -> ITYTY`), no NM
record and `C_sched > 43,200 s` are enriched ~3 orders of magnitude in day rollovers (9 of
the year's 15, 21 firing rows). The rule is *not* used as an override — a false positive
costs ~86,400 s — but as a `p_rollover` on the firing rows, estimated leave-one-month-out
from the other months' hits and shrunk toward the lane's own rollover base rate with a
Beta(α) prior (`--rollover-prior`), then mixed in with `prc.mixture.corruption.apply_mixture`
against a rollover expert value of 86,400 s + a *normal* taxi. All three rule inputs are
test-observable; no label-defined mask is read anywhere.

The expert value (fix of 2026-09-21) is set by
`--rollover-value`. `const` (default) is the tail scorecard's C = 86,400 + median(y − 86,400)
over the 2025 rollovers (|y − 86,400| < 3,600) of the source run's OOF — leave-one-month-out
on the OOF side (C from the other months), all months on the ranking side; 87,341 s on
`exp0150`. `y_hat_normal` uses 86,400 + the source run's normal-expert column where the file
carries one (C where it is null). The earlier value, 86,400 + ŷ_lane (and 86,400 + the run's
`y_hat_normal` off the lane), aimed at 106–157 k s on the firing rows: ŷ_lane is 31–71 k s and
`y_hat_normal` 19–24 k s there, neither a normal taxi. Without `--rollover-rule` the output is
unchanged byte for byte.

Never reads ``BLOCK_TIME_UTC_mvt`` / ``TAXITIME_SEC_mvt`` of the row it predicts. The feature
frame, the fold-safe encodings and the leak audit all live in
`scripts/eda_no_nm_block_model.py` and are imported from there, not duplicated.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import eda_no_nm_block_model as A
from prc import paths
from prc.mixture.corruption import apply_mixture

LANE_DIR = paths.OUT / "lane"

ROLLOVER_S = 86_400.0
ROLLOVER_WIN_S = 3_600.0
RULE_C_SCHED_S = 43_200.0
ROLLOVER_VALUE_MODES = ("const", "y_hat_normal")

# C3: the unmasked stand-occupancy bounds and the test-observable violation signal.
C3_FEATURES = [
    "f_ub_taxi",
    "f_lb_taxi",
    "f_c_minus_ub",
    "f_c_minus_lb",
    "f_c_violates_ub",
    "f_c_violates_lb",
]


def _auc(label: np.ndarray, score: np.ndarray) -> float:
    """Rank AUC; nan when one class is empty."""
    ok = np.isfinite(score) & np.isfinite(label)
    y, s = label[ok], score[ok]
    n1, n0 = float((y > 0).sum()), float((y <= 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    # average ranks over ties
    s_sorted = s[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[y > 0].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2))) if len(y) else float("nan")


def rollover_constant(y: np.ndarray) -> float:
    """C = 86,400 + median(y − 86,400) over the rollover rows (|y − 86,400| < 3,600) of `y`.

    The tail-scorecard definition of `rollover_value`; nan if `y` has none.
    """
    y = np.asarray(y, dtype=np.float64)
    ro = y[np.isfinite(y) & (np.abs(y - ROLLOVER_S) < ROLLOVER_WIN_S)]
    return float(ROLLOVER_S + np.median(ro - ROLLOVER_S)) if len(ro) else float("nan")


def rollover_expert_values(
    n: int,
    y_train: np.ndarray,
    month_train: np.ndarray,
    row_month: np.ndarray | None,
    y_hat_normal: np.ndarray | None,
    mode: str = "const",
) -> np.ndarray:
    """Per-row rollover expert value: 86,400 + a normal-taxi estimate.

    ``const``: C from ``rollover_constant``. With ``row_month`` given (the OOF side) the row of
    month ``m`` gets C fitted on the other months' labels only; with ``row_month=None`` (the
    ranking side) every row gets C over all of ``y_train``. ``y_hat_normal``: 86,400 + that
    column where it is finite, the ``const`` value elsewhere.
    """
    if mode not in ROLLOVER_VALUE_MODES:
        raise ValueError(f"rollover value mode {mode!r} not in {ROLLOVER_VALUE_MODES}")
    y_train = np.asarray(y_train, dtype=np.float64)
    month_train = np.asarray(month_train)
    val = np.full(n, rollover_constant(y_train), dtype=np.float64)
    if row_month is not None:
        row_month = np.asarray(row_month)
        for m in set(row_month.tolist()):
            val[row_month == m] = rollover_constant(y_train[month_train != m])
    if mode == "y_hat_normal" and y_hat_normal is not None:
        yn = np.asarray(y_hat_normal, dtype=np.float64)
        val = np.where(np.isfinite(yn), ROLLOVER_S + yn, val)
    return val


def add_stand_bound_features(dep: pl.DataFrame) -> pl.DataFrame:
    """Join the 24 h-capped `stand_bounds_v2` parquet and derive the violation signals.

    The bounds come from ARR in-block stamps only and are present on the ranking rows
    (coverage 90.8 % / 91.2 %, `2026-09-19_no-nm-forensics.md` §3). `C_sched` is the value the
    target takes under the schedule-default hypothesis, so `C_sched − taxi_upper_bound_s > 0`
    is the *test-observable* form of "the label would break the bound" — the quantity §3
    measured at a 59.9 % default rate against 3.1 % elsewhere.
    """
    cands = sorted(paths.FEATURES.glob("stand_bounds_v2__*.parquet"))
    if not cands:
        raise SystemExit("no stand_bounds_v2 parquet in data/features; run `prc features build`")
    sb = pl.scan_parquet(cands[-1]).select("MVT_ID_mvt", "taxi_upper_bound_s", "taxi_lower_bound_s")
    dep = dep.join(sb.collect(), on="MVT_ID_mvt", how="left")
    ub, lb, c = pl.col("taxi_upper_bound_s"), pl.col("taxi_lower_bound_s"), pl.col("c_sched")
    return dep.with_columns(
        ub.cast(pl.Float64).alias("f_ub_taxi"),
        lb.cast(pl.Float64).alias("f_lb_taxi"),
        (c - ub.cast(pl.Float64)).alias("f_c_minus_ub"),
        (c - lb.cast(pl.Float64)).alias("f_c_minus_lb"),
        (c > ub.cast(pl.Float64)).cast(pl.Float64).alias("f_c_violates_ub"),
        (c < lb.cast(pl.Float64)).cast(pl.Float64).alias("f_c_violates_lb"),
    ).drop("taxi_upper_bound_s", "taxi_lower_bound_s")


def add_rule_features(dep: pl.DataFrame) -> pl.DataFrame:
    """The C5 rule mask: doubled ICAO prefix ∧ no NM record ∧ `C_sched > 43,200 s`.

    `FLIGHT_mvt`, `AOBT_3_flt`, `MVT_TIME_UTC_mvt` and `SCHED_TIME_UTC_mvt` are all present on
    the ranking departures, so every input is test-observable.
    """
    pfx = pl.col("FLIGHT_mvt").str.extract(r"^([A-Z]+)", 1)
    dup = (
        ((pfx.str.len_chars() > 3) & pfx.str.slice(0, 3).str.ends_with(pfx.str.slice(3)))
        .fill_null(False)
        .cast(pl.Int8)
    )
    return dep.with_columns(dup.alias("f_fn_dup_prefix")).with_columns(
        (
            (pl.col("f_fn_dup_prefix") == 1)
            & pl.col("no_nm")
            & (pl.col("c_sched") > RULE_C_SCHED_S)
        ).alias("rule_fires")
    )


def build_lane_frames(
    run: str, stand_bounds: bool = False, cache: bool = True
) -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    """``(training lane rows, ranking lane rows, months)`` with the study's feature frame.

    The heavy part is the whole-departure-table feature build (2.43 M rows, ~4.8 GB peak); the
    result on the lane is 27,760 rows, so it is cached under ``outputs/lane/_frames/``. The C3
    bounds are a plain join on ``MVT_ID_mvt`` and are therefore added *after* the cut, which
    makes the two variants bit-identical on the shared columns and costs one build, not two.
    """
    src = paths.INTERIM / "movements.parquet"
    stamp = f"{int(src.stat().st_mtime)}_{src.stat().st_size}"
    cache_dir = LANE_DIR / "_frames"
    cached = cache_dir / f"lane_frames__{stamp}.parquet"
    if cache and cached.is_file():
        both = pl.read_parquet(cached)
    else:
        mv = A.load_movements()
        dep = A.departures(mv)
        dep = A.add_base_features(dep)
        dep = A.add_state_features(dep, mv)
        dep = A.add_rotation_features(dep, mv)
        del mv
        months_all = sorted(dep.filter(~pl.col("is_test"))["month"].unique().to_list())
        dep = A.add_history_features(dep, months_all)
        dep = A.encode_categoricals(dep)
        dep = add_rule_features(dep)
        both = dep.filter(pl.col("no_nm")).sort("MVT_ID_mvt")
        del dep
        cache_dir.mkdir(parents=True, exist_ok=True)
        both.write_parquet(cached)
    if stand_bounds:
        both = add_stand_bound_features(both)
    months = sorted(both.filter(~pl.col("is_test"))["month"].unique().to_list())
    oof_months = set(
        pl.read_parquet(paths.RUNS / run / "oof.parquet", columns=["month"])["month"].unique()
    )
    missing = oof_months - set(months)
    if missing:
        raise SystemExit(f"{run}: oof months {sorted(missing)} are not in the movement table")
    lane = both.filter(~pl.col("is_test"))
    test_lane = both.filter(pl.col("is_test"))
    return lane, test_lane, sorted(oof_months)


def rollover_probs(
    lane: pl.DataFrame,
    test_lane: pl.DataFrame,
    months: list[str],
    prior: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fold-safe `p_rollover` on the firing rows, shrunk toward the lane's own base rate.

    For held-out month `m` the estimate is the rule's precision on the **other** months,
    ``p̂_m = (k_other + α·π0_other) / (n_other + α)``, with ``π0_other`` the lane-wide rollover
    rate of those months (≈ 6e-4) and ``α = prior`` the pseudo-count. 21 firing rows a year is
    a thin estimate, so α is the knob the sensitivity table in the report sweeps. The ranking
    rows use all twelve months. Non-firing rows get 0.
    """
    y = lane["y"].to_numpy()
    is_ro = np.abs(y - ROLLOVER_S) < ROLLOVER_WIN_S
    fires = lane["rule_fires"].to_numpy()
    month = lane["month"].to_numpy()
    p = np.zeros(lane.height, dtype=np.float64)
    per_fold = []
    for m in months:
        te = month == m
        tr = ~te
        k, n = float((fires & tr & is_ro).sum()), float((fires & tr).sum())
        pi0 = float(is_ro[tr].mean())
        est = (k + prior * pi0) / (n + prior) if (n + prior) > 0 else pi0
        p[te & fires] = est
        per_fold.append(
            {
                "month": m,
                "fires_other": int(n),
                "ro_other": int(k),
                "p": est,
                "fires_held_out": int((te & fires).sum()),
            }
        )
    k_all, n_all = float((fires & is_ro).sum()), float(fires.sum())
    pi0_all = float(is_ro.mean())
    p_test = (k_all + prior * pi0_all) / (n_all + prior) if (n_all + prior) > 0 else pi0_all
    fires_te = test_lane["rule_fires"].to_numpy()
    p_te = np.where(fires_te, p_test, 0.0)
    info = {
        "prior_alpha": prior,
        "fires_2025": int(n_all),
        "rollovers_caught": int(k_all),
        "raw_precision": (k_all / n_all) if n_all else float("nan"),
        "p_test": p_test,
        "fires_2026": int(fires_te.sum()),
        "base_rate": pi0_all,
        "per_fold": per_fold,
    }
    return p, p_te, info


def fit(
    run: str,
    out: Path,
    stand_bounds: bool = False,
    rollover_prior: float = 5.0,
) -> Path:
    """Leave-one-month-out lane predictions for the training rows, and all-2025-fitted lane
    predictions for the ranking rows. Writes one parquet; nothing else is changed."""
    import lightgbm as lgb

    lane, test_lane, months = build_lane_frames(run, stand_bounds=stand_bounds)
    reg_feats = A.ALL_FEATURES
    clf_feats = A.ALL_FEATURES + (C3_FEATURES if stand_bounds else [])
    reg_cat = [i for i, f in enumerate(reg_feats) if f in A.CAT_FEATURES]
    clf_cat = [i for i, f in enumerate(clf_feats) if f in A.CAT_FEATURES]
    d = lane["d"].to_numpy()
    lane = lane.with_columns(
        pl.Series("is_default", (np.abs(d) <= A.SCHED_TOL_S).astype(np.float64))
    )

    p_def = A.fit_lomo_clf(lane, clf_feats, "is_default", months)
    y_lane = A.fit_lomo(lane, reg_feats, "y", months)
    c = lane["c_sched"].to_numpy()
    pred = np.maximum(p_def * c + (1 - p_def) * y_lane, 0.0)

    x_tr, x_te = A._matrix(lane, clf_feats), A._matrix(test_lane, clf_feats)
    clf = lgb.LGBMClassifier(
        objective="binary",
        num_leaves=15,
        min_data_in_leaf=60,
        learning_rate=0.05,
        n_estimators=300,
        verbose=-1,
        num_threads=8,
    ).fit(x_tr, lane["is_default"].to_numpy(), categorical_feature=clf_cat)
    reg = lgb.LGBMRegressor(
        objective="regression",
        num_leaves=31,
        min_data_in_leaf=40,
        learning_rate=0.05,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        n_estimators=400,
        verbose=-1,
        num_threads=8,
    ).fit(
        A._matrix(lane, reg_feats),
        lane["y"].to_numpy(),
        categorical_feature=reg_cat,
    )
    p_def_te = clf.predict_proba(x_te)[:, 1]
    y_te = reg.predict(A._matrix(test_lane, reg_feats))
    c_te = test_lane["c_sched"].to_numpy()
    pred_te = np.maximum(p_def_te * c_te + (1 - p_def_te) * y_te, 0.0)

    p_ro, p_ro_te, ro_info = rollover_probs(lane, test_lane, months, rollover_prior)

    frame = pl.concat(
        [
            lane.select("MVT_ID_mvt", "airport", "month", "rule_fires", "c_sched").with_columns(
                pl.Series("lane_pred", pred),
                pl.Series("lane_p_default", p_def),
                pl.Series("lane_y_hat", y_lane),
                pl.Series("lane_p_rollover", p_ro),
                pl.lit(False).alias("is_test"),
            ),
            test_lane.select(
                "MVT_ID_mvt", "airport", "month", "rule_fires", "c_sched"
            ).with_columns(
                pl.Series("lane_pred", pred_te),
                pl.Series("lane_p_default", p_def_te),
                pl.Series("lane_y_hat", y_te),
                pl.Series("lane_p_rollover", p_ro_te),
                pl.lit(True).alias("is_test"),
            ),
        ],
        how="vertical",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(out)

    # diagnostics the report quotes: lane classifier AUC and the lane's own RMSE
    y = lane["y"].to_numpy()
    lab = lane["is_default"].to_numpy()
    ap = lane["airport"].to_numpy()
    exro = np.abs(y - ROLLOVER_S) >= ROLLOVER_WIN_S
    stats = {
        "stand_bounds": stand_bounds,
        "n_clf_features": len(clf_feats),
        "auc_lane": _auc(lab, p_def),
        "auc_LIRF": _auc(lab[ap == "LIRF"], p_def[ap == "LIRF"]),
        "auc_non_LIRF": _auc(lab[ap != "LIRF"], p_def[ap != "LIRF"]),
        "lane_rmse": _rmse(y, pred),
        "lane_exro": _rmse(y[exro], pred[exro]),
        "LIRF_exro": _rmse(y[(ap == "LIRF") & exro], pred[(ap == "LIRF") & exro]),
        "non_LIRF": _rmse(y[ap != "LIRF"], pred[ap != "LIRF"]),
        "rollover": ro_info,
    }
    (out.parent / f"{out.stem}.stats.json").write_text(json.dumps(stats, indent=2, default=float))
    print(
        f"fit: {lane.height:,} training lane rows over {len(months)} months, "
        f"{test_lane.height:,} ranking lane rows -> {out}"
    )
    print(json.dumps({k: v for k, v in stats.items() if k != "rollover"}, indent=2, default=float))
    print(
        "rollover rule: fires {fires_2025} in 2025 ({rollovers_caught} rollovers, raw precision "
        "{raw_precision:.4f}), {fires_2026} ranking rows, p_test={p_test:.4f} "
        "(alpha={prior_alpha})".format(**ro_info)
    )
    return out


def auto_airports(run: str, lane: pl.DataFrame) -> tuple[pl.DataFrame, list[str], pl.DataFrame]:
    """Fold-safe airport selection.

    For held-out month ``m`` an airport is selected when the lane model's squared error beats
    the source run's on that airport's lane rows in the **other** months. The ranking side
    uses all twelve months. Returns ``(per-fold selection, airports selected for the ranking
    rows, the full per-airport table)``.
    """
    oof = pl.read_parquet(
        paths.RUNS / run / "oof.parquet",
        columns=["MVT_ID_mvt", "month", "airport", "TAXITIME_SEC_mvt", "pred"],
    )
    j = (
        lane.filter(~pl.col("is_test"))
        .select("MVT_ID_mvt", "lane_pred")
        .join(oof, on="MVT_ID_mvt", how="inner")
        .with_columns(
            ((pl.col("TAXITIME_SEC_mvt") - pl.col("pred")) ** 2).alias("se_base"),
            ((pl.col("TAXITIME_SEC_mvt") - pl.col("lane_pred")) ** 2).alias("se_lane"),
        )
    )
    per = j.group_by(["month", "airport"]).agg(
        pl.sum("se_base").alias("se_base"), pl.sum("se_lane").alias("se_lane"), pl.len().alias("n")
    )
    tot = per.group_by("airport").agg(
        pl.sum("se_base").alias("tot_base"), pl.sum("se_lane").alias("tot_lane")
    )
    sel = (
        per.join(tot, on="airport", how="left")
        .with_columns(
            (pl.col("tot_base") - pl.col("se_base")).alias("oth_base"),
            (pl.col("tot_lane") - pl.col("se_lane")).alias("oth_lane"),
        )
        .with_columns((pl.col("oth_lane") < pl.col("oth_base")).alias("selected"))
    )
    test_sel = sorted(tot.filter(pl.col("tot_lane") < pl.col("tot_base"))["airport"].to_list())
    return sel.select("month", "airport", "selected"), test_sel, tot


def apply(
    run: str,
    lane_path: Path,
    out_name: str,
    airports: list[str] | None,
    rollover_rule: bool = False,
    auto: bool = False,
    rollover_value: str = "const",
) -> Path:
    """Write ``outputs/runs/<out_name>/{oof,test_pred}.parquet`` — the source run with the
    lane rows' ``pred`` (and every ``pred_s<seed>``) replaced."""
    lane = pl.read_parquet(lane_path)
    if auto and airports:
        raise SystemExit("--auto-airports and --airports are mutually exclusive")
    sel_oof, sel_test = None, None
    if auto:
        sel_oof, sel_test, _tot = auto_airports(run, lane)
        print("auto airport selection (ranking side):", ", ".join(sel_test) or "(none)")
        with pl.Config(tbl_rows=40):
            print(sel_oof.filter(pl.col("selected")).group_by("airport").len().sort("airport"))
    src, dst = paths.RUNS / run, paths.RUNS / out_name
    if not (src / "oof.parquet").is_file():
        raise SystemExit(f"{src}: no oof.parquet")
    if rollover_value not in ROLLOVER_VALUE_MODES:
        raise SystemExit(f"--rollover-value must be one of {ROLLOVER_VALUE_MODES}")
    y_lab, m_lab = None, None
    if rollover_rule:
        lab = pl.read_parquet(src / "oof.parquet", columns=["month", "TAXITIME_SEC_mvt"])
        y_lab = lab["TAXITIME_SEC_mvt"].cast(pl.Float64).fill_null(float("nan")).to_numpy()
        m_lab = lab["month"].to_numpy()
        del lab
    dst.mkdir(parents=True, exist_ok=True)
    counts = {}
    for which, is_test in (("oof", False), ("test_pred", True)):
        path = src / f"{which}.parquet"
        if not path.is_file():
            print(f"apply: {path} missing, skipped")
            continue
        df = pl.read_parquet(path)
        sub = lane.filter(pl.col("is_test") == is_test).select(
            "MVT_ID_mvt",
            "lane_pred",
            "lane_y_hat",
            "lane_p_default",
            "lane_p_rollover",
            "rule_fires",
            "c_sched",
            pl.col("airport").alias("_lane_airport"),
            pl.col("month").alias("_lane_month"),
        )
        merged = df.join(sub, on="MVT_ID_mvt", how="left")
        # which rows get the lane substitution
        in_lane = merged["lane_pred"].is_not_null().to_numpy()
        if airports:
            use = in_lane & merged["_lane_airport"].is_in(airports).fill_null(False).to_numpy()
        elif auto:
            if is_test:
                use = in_lane & merged["_lane_airport"].is_in(sel_test).fill_null(False).to_numpy()
            else:
                key = (
                    merged.select("_lane_month", "_lane_airport")
                    .with_row_index("_i")
                    .join(
                        sel_oof.rename({"month": "_lane_month", "airport": "_lane_airport"}),
                        on=["_lane_month", "_lane_airport"],
                        how="left",
                    )
                    .sort("_i")
                )
                use = in_lane & key["selected"].fill_null(False).to_numpy()
        else:
            use = in_lane
        lane_yhat = np.nan_to_num(merged["lane_y_hat"].to_numpy(), nan=0.0)
        lane_pdef = np.nan_to_num(merged["lane_p_default"].to_numpy(), nan=0.0)
        c_sched = np.nan_to_num(merged["c_sched"].to_numpy(), nan=0.0)
        run_yhat = np.nan_to_num(merged["y_hat_normal"].to_numpy().astype(np.float64), nan=0.0)
        fires = merged["rule_fires"].fill_null(False).to_numpy()
        p_ro = np.nan_to_num(merged["lane_p_rollover"].to_numpy(), nan=0.0) * fires
        # The rollover branch joins the mixture, it is not stacked on top of it. Where the lane
        # model took over, the three-branch mixture is rebuilt from its own normal branch
        # `y_lane` with `p_sched_default` and `p_rollover` side by side (`clip_probs` keeps the
        # pair sub-convex); `p_rollover = 0` there reproduces `lane_pred` exactly. Elsewhere the
        # rule composes onto the run's own prediction with the run's `y_hat_normal`, whose own
        # `p_rollover` is ~7e-6 and is left in place.
        if rollover_rule:
            has_normal = "y_hat_normal" in merged.columns
            ro_val = rollover_expert_values(
                merged.height,
                y_lab,
                m_lab,
                None if is_test else merged["month"].to_numpy(),
                merged["y_hat_normal"].to_numpy().astype(np.float64) if has_normal else None,
                rollover_value,
            )
            ro_val_lane = ro_off = ro_val
        else:
            # unchanged pre-fix values; every one is multiplied by p_rollover = 0
            ro_val_lane, ro_off = ROLLOVER_S + lane_yhat, ROLLOVER_S + run_yhat
        lane_mix = apply_mixture(
            lane_yhat,
            {"sched_default": lane_pdef, "rollover": p_ro if rollover_rule else p_ro * 0.0},
            {"sched_default": c_sched, "rollover": ro_val_lane},
        )
        cols = ["pred", *sorted(c for c in df.columns if c.startswith("pred_s"))]
        new = {}
        for col in cols:
            base = merged[col].to_numpy().astype(np.float64)
            base = np.where(use, lane_mix, base)
            if rollover_rule:
                off = p_ro * ~use
                base = apply_mixture(base, {"rollover": off}, {"rollover": ro_off})
            new[col] = np.maximum(base, 0.0)
        merged = merged.with_columns([pl.Series(c, v) for c, v in new.items()]).drop(
            "lane_pred",
            "lane_y_hat",
            "lane_p_default",
            "lane_p_rollover",
            "rule_fires",
            "c_sched",
            "_lane_airport",
            "_lane_month",
        )
        merged.write_parquet(dst / f"{which}.parquet")
        counts[which] = {
            "rows": df.height,
            "replaced": int(use.sum()),
            "rollover_rule_rows": int((p_ro > 0).sum()) if rollover_rule else 0,
        }
        print(
            f"apply: {which} {df.height:,} rows, {int(use.sum()):,} replaced, "
            f"{counts[which]['rollover_rule_rows']} rollover-rule rows "
            f"-> {dst / f'{which}.parquet'}"
        )
        if rollover_rule and int((p_ro > 0).sum()):
            sel = p_ro > 0
            print(
                f"  rollover rows: p={np.unique(np.round(p_ro[sel], 6)).tolist()} "
                f"expert value {np.round(ro_val[sel], 1).tolist()} "
                f"pred {np.round(new['pred'][sel], 1).tolist()}"
            )
            counts[which]["rollover_value_min"] = float(ro_val[sel].min())
            counts[which]["rollover_value_max"] = float(ro_val[sel].max())
    (dst / "meta.json").write_text(
        json.dumps(
            {
                "source_run": run,
                "lane_predictions": str(lane_path),
                "airports": airports or ("auto:" + ",".join(sel_test) if auto else "all"),
                "rollover_rule": rollover_rule,
                **({"rollover_value": rollover_value} if rollover_rule else {}),
                "counts": counts,
                "written": datetime.now(UTC).isoformat(timespec="seconds"),
                "note": "post-processed no-NM lane predictions; not a registry run",
            },
            indent=2,
        )
        + "\n"
    )
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", help="fit the lane model against a run's months")
    f.add_argument("--run", required=True)
    f.add_argument("--out", type=Path, default=None)
    f.add_argument(
        "--stand-bounds",
        action="store_true",
        help="C3: give the default classifier the unmasked stand_bounds_v2 bounds and the "
        "test-observable violation signals",
    )
    f.add_argument(
        "--rollover-prior",
        type=float,
        default=5.0,
        help="pseudo-count alpha shrinking the C5 rule precision toward the lane base rate",
    )
    a = sub.add_parser("apply", help="write a run-shaped directory with the lane preds swapped in")
    a.add_argument("--run", required=True)
    a.add_argument("--lane", type=Path, required=True)
    a.add_argument("--out", required=True)
    a.add_argument(
        "--airports",
        nargs="*",
        default=None,
        help="restrict the substitution to these airports (LIRF reproduces the report's M9)",
    )
    a.add_argument(
        "--auto-airports",
        action="store_true",
        help="fold-safe airport selection from the other months' OOF",
    )
    a.add_argument(
        "--rollover-rule",
        action="store_true",
        help="C5: mix the rollover expert in on the doubled-prefix / no-NM / C_sched>43200 rows",
    )
    a.add_argument(
        "--rollover-value",
        choices=ROLLOVER_VALUE_MODES,
        default="const",
        help="rollover expert value with --rollover-rule: 'const' = 86,400 + median(y - 86,400) "
        "over the OOF rollovers (leave-one-month-out on the OOF side); 'y_hat_normal' = "
        "86,400 + the run's normal-expert column",
    )
    args = ap.parse_args()
    if args.cmd == "fit":
        fit(
            args.run,
            args.out or LANE_DIR / f"{args.run}.parquet",
            stand_bounds=args.stand_bounds,
            rollover_prior=args.rollover_prior,
        )
    else:
        apply(
            args.run,
            args.lane,
            args.out,
            args.airports,
            rollover_rule=args.rollover_rule,
            auto=args.auto_airports,
            rollover_value=args.rollover_value,
        )


if __name__ == "__main__":
    main()
