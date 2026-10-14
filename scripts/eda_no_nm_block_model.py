"""Model the *block stamp*, not the taxi time, on the no-flight-plan lane.

The scored label on a departure is ``y = MVT − BLOCK``. Both ``MVT`` and ``SCHED`` are
present on every ranking row, so with ``C_sched = MVT − SCHED`` (the corruption head's
schedule-default value) and ``d = BLOCK − SCHED`` (the *reported* off-block delay against
schedule) the label is an identity::

    y = C_sched − d

Predicting ``y`` is therefore exactly predicting ``d``. On the 22,470 training departures
with no NM record (``AOBT_3_flt`` null ⇒ every ``_flt`` column null) R5 scores ~1,956 s and
the lane carries 45.7 % of the full-year squared error, so if ``d`` were predictable from
something the model does not currently use, that is where it would show.

This pass asks four questions, on **training rows only**, with every candidate restricted to
information a Jan/Jul 2026 ranking row also carries (``_mvt`` columns of this row, any column
of *other* rows including arrivals, and fold-safe statistics of other months):

1. what the distribution of ``d`` looks like on the lane, and how much of ``var(y)`` is
   ``var(d)`` rather than variance of a real taxi time (§2);
2. how much of ``d`` is carried by contemporaneous observable delay (other departures' and
   arrivals' stamps in a ±30/60/120 min window), the airport-day level, the stand rotation,
   ``C_sched`` itself, and fold-safe airline/stand/destination history (§4);
3. what a fold-safe lane-scoped model of ``d`` — and of ``y`` — is worth against R5's stored
   ``month_loo`` predictions on the same rows, together with three closed forms (§5);
4. whether any of it is a leak: every feature is recomputed on the 5,290 no-NM ranking rows
   and its null rate compared with 2025 (§6).

Nothing here reads ``BLOCK_TIME_UTC_mvt`` or ``TAXITIME_SEC_mvt`` of the row it describes.
The label-defined groups (``d0_default``, ``rollover``, …) are diagnostics; no rule is
conditioned on them.

    uv run python scripts/eda_no_nm_block_model.py
    uv run python scripts/eda_no_nm_block_model.py --write report.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paired_delta as PD
from prc import paths

R5_LOO = "exp0068_xgb_r4_d12_mcw3000__s1-2-3__month_loo"
R5_GATE = "w1-cap/exp0068_xgb_r4_d12_mcw3000"
SCHED_TOL_S = 10.0
ROLLOVER_S, ROLLOVER_WIN_S = 86_400.0, 3_600.0
GATE_MONTHS = ("2025-01", "2025-07")
WINDOWS = (1800, 3600, 7200)
TE_PRIOR = 20.0

MV_COLS = [
    "MVT_ID_mvt",
    "FLIGHT_mvt",
    "FLIGHT_RULE_mvt",
    "ADES_mvt",
    "PHASE_mvt",
    "MVT_TIME_UTC_mvt",
    "BLOCK_TIME_UTC_mvt",
    "SCHED_TIME_UTC_mvt",
    "AIRCRAFT_TYPE_mvt",
    "RUNWAY_mvt",
    "STAND_mvt",
    "TAXITIME_SEC_mvt",
    "AOBT_3_flt",
    "airport",
    "is_test",
    "month",
]

CAT_FEATURES = ["f_airport", "f_airline", "f_stand", "f_runway", "f_ades", "f_actype"]
BASE_FEATURES = [
    "f_c_sched",
    "f_hour",
    "f_dow",
    "f_month_num",
    "f_sched_moh",
    "f_sched_hour",
    *CAT_FEATURES,
]
STATE_FEATURES = [
    "f_dnb_med_1800",
    "f_dnb_med_3600",
    "f_dnb_med_7200",
    "f_dnb_q75_3600",
    "f_dnb_q25_3600",
    "f_dnb_n_3600",
    "f_bnb_med_3600",
    "f_bnb_n_3600",
    "f_anb_in_med_3600",
    "f_anb_land_med_3600",
    "f_day_med",
    "f_day_q90",
    "f_day_nonm_share",
    "f_c_minus_dnb",
]
ROTATION_FEATURES = ["f_prev_arr_gap", "f_prev_arr_delay", "f_reoccupied", "f_next_arr_gap"]
HISTORY_FEATURES = [
    "f_te_airline_dmed",
    "f_te_airline_default",
    "f_te_airline_taxi",
    "f_te_stand_default",
    "f_te_ades_default",
    "f_te_arh_taxi",
    "f_te_airport_default",
]
ALL_FEATURES = BASE_FEATURES + STATE_FEATURES + ROTATION_FEATURES + HISTORY_FEATURES


# --------------------------------------------------------------------------- formatting


def _fmt(v: object) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        if not np.isfinite(v):
            return ""
        if v == 0:
            return "0"
        return f"{v:,.1f}" if abs(v) >= 1 else f"{v:.4f}"
    if v is None:
        return "null"
    return str(v)


def md(df: pl.DataFrame) -> str:
    cols = df.columns
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols)
    body = "\n".join("| " + " | ".join(_fmt(v) for v in r) + " |" for r in df.iter_rows())
    return head + "\n" + body


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2)))


# --------------------------------------------------------------------------- features


def load_movements() -> pl.DataFrame:
    return pl.scan_parquet(paths.INTERIM / "movements.parquet").select(MV_COLS).collect()


def departures(mv: pl.DataFrame) -> pl.DataFrame:
    """All departures (train + test) with the identity's three quantities.

    ``d`` and ``y`` are null on the ranking rows by construction — they are the hidden
    columns — so anything computed from them is a training-row quantity only.
    """
    return mv.filter(pl.col("PHASE_mvt") == "DEP").with_columns(
        (pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt"))
        .dt.total_seconds()
        .cast(pl.Float64)
        .alias("c_sched"),
        (pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt"))
        .dt.total_seconds()
        .cast(pl.Float64)
        .alias("d"),
        (pl.col("AOBT_3_flt") - pl.col("SCHED_TIME_UTC_mvt"))
        .dt.total_seconds()
        .cast(pl.Float64)
        .alias("obs_delay"),
        pl.col("AOBT_3_flt").is_null().alias("no_nm"),
        pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("y"),
        pl.col("FLIGHT_mvt").str.extract(r"^([A-Z]+)", 1).alias("airline"),
    )


def roll(
    df: pl.DataFrame,
    index: str,
    period: str,
    offset: str,
    group_by: str,
    aggs: list[pl.Expr],
) -> pl.DataFrame:
    """``df.rolling(...).agg(...)`` with the results put back on the rows they belong to.

    ``DataFrame.rolling(group_by=…)`` returns its rows ordered by **(group, index)**, not in
    the input order, so attaching the output positionally to a frame that is sorted by the
    index alone silently permutes every window statistic across airports. Sorting the input by
    ``[group, index]`` first makes the two orders identical; the equality check below states
    that guarantee instead of assuming it. (Found 2026-09-19: the first version of this
    function attached the columns to a time-sorted frame, which scrambled them and produced a
    false "contemporaneous delay carries no signal" result.)
    """
    df = df.sort([group_by, index])
    a = df.rolling(index_column=index, period=period, offset=offset, group_by=group_by).agg(*aggs)
    if not (a[group_by].equals(df[group_by]) and a[index].equals(df[index])):
        raise AssertionError(f"rolling output is misaligned with its input ({index}, {group_by})")
    return df.with_columns([a[c] for c in a.columns if c not in (group_by, index)])


def add_state_features(dep: pl.DataFrame, mv: pl.DataFrame) -> pl.DataFrame:
    """Contemporaneous observable delay at the airport, from *other* rows only.

    Three views, all computable on a ranking row: (a) departures whose NM off-block stamp is
    known, indexed at their scheduled time, in a ±W window around this row's scheduled time;
    (b) the same pool indexed at the stamp itself, in a ±1 h window around this row's takeoff
    ("who actually pushed back while we were out there"); (c) arrivals, whose in-block stamp
    and landing time are both present in the ranking months.

    A no-NM row has ``obs_delay`` null, so it never contributes to its own window.
    """
    for w in WINDOWS:
        dep = roll(
            dep,
            "SCHED_TIME_UTC_mvt",
            f"{2 * w}s",
            f"-{w}s",
            "airport",
            [
                pl.col("obs_delay").median().alias(f"f_dnb_med_{w}"),
                pl.col("obs_delay").quantile(0.75).alias(f"f_dnb_q75_{w}"),
                pl.col("obs_delay").quantile(0.25).alias(f"f_dnb_q25_{w}"),
                pl.col("obs_delay").is_not_null().sum().alias(f"f_dnb_n_{w}"),
            ],
        )

    # (b) pool keyed at the actual stamp, queried at this row's takeoff
    pool = (
        dep.filter(pl.col("obs_delay").is_not_null())
        .select("airport", pl.col("AOBT_3_flt").alias("t"), "obs_delay")
        .with_columns(pl.lit(None, pl.Float64).alias("MVT_ID_mvt"))
    )
    q = dep.select(
        "MVT_ID_mvt",
        "airport",
        pl.col("MVT_TIME_UTC_mvt").alias("t"),
        pl.lit(None, pl.Float64).alias("obs_delay"),
    )
    both = pl.concat([pool.select("MVT_ID_mvt", "airport", "t", "obs_delay"), q], how="vertical")
    both = roll(
        both,
        "t",
        "7200s",
        "-3600s",
        "airport",
        [
            pl.col("obs_delay").median().alias("f_bnb_med_3600"),
            pl.col("obs_delay").is_not_null().sum().alias("f_bnb_n_3600"),
        ],
    )
    dep = dep.join(
        both.filter(pl.col("MVT_ID_mvt").is_not_null()).select(
            "MVT_ID_mvt", "f_bnb_med_3600", "f_bnb_n_3600"
        ),
        on="MVT_ID_mvt",
        how="left",
    )

    # (c) arrivals: in-block and landing lateness, both present on the ranking arrivals
    arr = mv.filter(pl.col("PHASE_mvt") == "ARR").with_columns(
        (pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt"))
        .dt.total_seconds()
        .alias("arr_in_delay"),
        (pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt"))
        .dt.total_seconds()
        .alias("arr_land_delay"),
    )
    arr = roll(
        arr,
        "SCHED_TIME_UTC_mvt",
        "7200s",
        "-3600s",
        "airport",
        [
            pl.col("arr_in_delay").median().alias("f_anb_in_med_3600"),
            pl.col("arr_land_delay").median().alias("f_anb_land_med_3600"),
        ],
    )
    arrk = arr.select(
        "airport", "SCHED_TIME_UTC_mvt", "f_anb_in_med_3600", "f_anb_land_med_3600"
    ).sort("SCHED_TIME_UTC_mvt")
    dep = dep.sort("SCHED_TIME_UTC_mvt").join_asof(
        arrk, on="SCHED_TIME_UTC_mvt", by="airport", strategy="nearest"
    )

    # (d) airport-day level
    dep = dep.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.date().alias("date"))
    day = dep.group_by("airport", "date").agg(
        pl.col("obs_delay").median().alias("f_day_med"),
        pl.col("obs_delay").quantile(0.9).alias("f_day_q90"),
        pl.col("no_nm").mean().alias("f_day_nonm_share"),
    )
    dep = dep.join(day, on=["airport", "date"], how="left")
    return dep.with_columns((pl.col("c_sched") - pl.col("f_dnb_med_3600")).alias("f_c_minus_dnb"))


def add_rotation_features(dep: pl.DataFrame, mv: pl.DataFrame) -> pl.DataFrame:
    """The stand's rotation: the arrival that vacated the stand and the one that takes it.

    Arrival in-block stamps are present on the ranking rows (verified in §3), so the gap
    between the previous arrival's in-block and this departure's takeoff is observable.
    """
    arr = (
        mv.filter(pl.col("PHASE_mvt") == "ARR")
        .filter(pl.col("STAND_mvt").is_not_null() & pl.col("BLOCK_TIME_UTC_mvt").is_not_null())
        .select(
            "airport",
            "STAND_mvt",
            pl.col("BLOCK_TIME_UTC_mvt").alias("arr_in"),
            (pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt"))
            .dt.total_seconds()
            .alias("f_prev_arr_delay"),
        )
        .sort("arr_in")
    )
    dep = dep.sort("MVT_TIME_UTC_mvt")
    prev = dep.join_asof(
        arr, left_on="MVT_TIME_UTC_mvt", right_on="arr_in", by=["airport", "STAND_mvt"]
    ).select(
        "MVT_ID_mvt",
        (pl.col("MVT_TIME_UTC_mvt") - pl.col("arr_in")).dt.total_seconds().alias("f_prev_arr_gap"),
        "f_prev_arr_delay",
        pl.col("arr_in").alias("_prev_in"),
    )
    nxt = dep.join_asof(
        arr.select("airport", "STAND_mvt", "arr_in"),
        left_on="MVT_TIME_UTC_mvt",
        right_on="arr_in",
        by=["airport", "STAND_mvt"],
        strategy="forward",
    ).select(
        "MVT_ID_mvt",
        (pl.col("arr_in") - pl.col("MVT_TIME_UTC_mvt")).dt.total_seconds().alias("f_next_arr_gap"),
    )
    dep = dep.join(prev, on="MVT_ID_mvt", how="left").join(nxt, on="MVT_ID_mvt", how="left")
    return dep.with_columns(
        (pl.col("_prev_in") > pl.col("SCHED_TIME_UTC_mvt"))
        .fill_null(False)
        .cast(pl.Int8)
        .alias("f_reoccupied")
    ).drop("_prev_in")


def add_history_features(dep: pl.DataFrame, months: list[str]) -> pl.DataFrame:
    """Fold-safe leave-one-month-out encodings, fitted on **all** 2025 departures.

    The schedule-default rate of an airline is measured ~100× more precisely on its rows with
    a flight plan than on the no-NM ones, so the
    source population is every training departure. Each statistic is computed from the eleven
    months other than the row's own; the ranking months see all twelve.
    """
    src = dep.filter(~pl.col("is_test")).select(
        "month",
        "airport",
        "airline",
        "STAND_mvt",
        "ADES_mvt",
        "RUNWAY_mvt",
        pl.col("MVT_TIME_UTC_mvt").dt.hour().alias("hour"),
        "d",
        "y",
        ((pl.col("d").abs() <= SCHED_TOL_S).fill_null(False)).cast(pl.Float64).alias("is_default"),
    )
    specs = [
        (["airport", "airline"], "d", "median", "f_te_airline_dmed"),
        (["airport", "airline"], "is_default", "mean", "f_te_airline_default"),
        (["airport", "airline"], "y", "median", "f_te_airline_taxi"),
        (["airport", "STAND_mvt"], "is_default", "mean", "f_te_stand_default"),
        (["airport", "ADES_mvt"], "is_default", "mean", "f_te_ades_default"),
        (["airport", "RUNWAY_mvt", "hour"], "y", "median", "f_te_arh_taxi"),
        (["airport"], "is_default", "mean", "f_te_airport_default"),
    ]
    dep = dep.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.hour().alias("hour"))
    pieces: list[pl.DataFrame] = []
    for held in [*months, "__test__"]:
        s = src if held == "__test__" else src.filter(pl.col("month") != held)
        tgt = (
            dep.filter(pl.col("is_test"))
            if held == "__test__"
            else dep.filter(pl.col("month") == held)
        )
        if tgt.height == 0:
            continue
        out = tgt.select("MVT_ID_mvt")
        for keys, col, how, name in specs:
            agg = pl.col(col).median() if how == "median" else pl.col(col).mean()
            g = s.group_by(keys).agg(agg.alias(name), pl.len().alias("_n"))
            if how == "mean":  # smooth a rate towards the global prior
                prior = float(s[col].mean())
                g = g.with_columns(
                    (
                        (pl.col(name) * pl.col("_n") + prior * TE_PRIOR) / (pl.col("_n") + TE_PRIOR)
                    ).alias(name)
                )
            out = out.join(
                tgt.select("MVT_ID_mvt", *keys).join(g.drop("_n"), on=keys, how="left"),
                on="MVT_ID_mvt",
                how="left",
            ).select(*[c for c in out.columns], name)
        pieces.append(out)
    return dep.join(pl.concat(pieces, how="vertical"), on="MVT_ID_mvt", how="left")


def add_base_features(dep: pl.DataFrame) -> pl.DataFrame:
    return dep.with_columns(
        pl.col("c_sched").alias("f_c_sched"),
        pl.col("MVT_TIME_UTC_mvt").dt.hour().alias("f_hour"),
        pl.col("MVT_TIME_UTC_mvt").dt.weekday().alias("f_dow"),
        pl.col("MVT_TIME_UTC_mvt").dt.month().alias("f_month_num"),
        pl.col("SCHED_TIME_UTC_mvt").dt.minute().alias("f_sched_moh"),
        pl.col("SCHED_TIME_UTC_mvt").dt.hour().alias("f_sched_hour"),
        pl.col("airport").alias("f_airport"),
        pl.col("airline").alias("f_airline"),
        pl.col("STAND_mvt").alias("f_stand"),
        pl.col("RUNWAY_mvt").alias("f_runway"),
        pl.col("ADES_mvt").alias("f_ades"),
        pl.col("AIRCRAFT_TYPE_mvt").alias("f_actype"),
    )


def encode_categoricals(dep: pl.DataFrame) -> pl.DataFrame:
    """Integer codes over train + test (no target is read, so this is not fold-dependent)."""
    return dep.with_columns(
        [pl.col(c).cast(pl.Categorical).to_physical().cast(pl.Int32) for c in CAT_FEATURES]
    )


# --------------------------------------------------------------------------- models


def _matrix(d: pl.DataFrame, feats: list[str]) -> np.ndarray:
    return d.select([pl.col(f).cast(pl.Float64) for f in feats]).to_numpy()


def fit_lomo(
    lane: pl.DataFrame,
    feats: list[str],
    target: str,
    months: list[str],
    params: dict | None = None,
) -> np.ndarray:
    """Leave-one-month-out predictions of ``target`` on the lane, trained on the lane only."""
    import lightgbm as lgb

    p = {
        "objective": "regression",
        "num_leaves": 31,
        "min_data_in_leaf": 40,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "n_estimators": 400,
        "verbose": -1,
        "num_threads": 8,
    }
    p.update(params or {})
    cat_idx = [i for i, f in enumerate(feats) if f in CAT_FEATURES]
    out = np.full(lane.height, np.nan)
    month = lane["month"].to_numpy()
    x = _matrix(lane, feats)
    yv = lane[target].cast(pl.Float64).to_numpy()
    for m in months:
        te = month == m
        tr = ~te
        if te.sum() == 0:
            continue
        model = lgb.LGBMRegressor(**p)
        model.fit(x[tr], yv[tr], categorical_feature=cat_idx)
        out[te] = model.predict(x[te])
    return out


def fit_lomo_clf(lane: pl.DataFrame, feats: list[str], label: str, months: list[str]) -> np.ndarray:
    import lightgbm as lgb

    p = {
        "objective": "binary",
        "num_leaves": 15,
        "min_data_in_leaf": 60,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "verbose": -1,
        "num_threads": 8,
    }
    cat_idx = [i for i, f in enumerate(feats) if f in CAT_FEATURES]
    out = np.full(lane.height, np.nan)
    month = lane["month"].to_numpy()
    x = _matrix(lane, feats)
    yv = lane[label].cast(pl.Float64).to_numpy()
    for m in months:
        te = month == m
        if te.sum() == 0 or yv[~te].sum() == 0:
            continue
        model = lgb.LGBMClassifier(**p)
        model.fit(x[~te], yv[~te], categorical_feature=cat_idx)
        out[te] = model.predict_proba(x[te])[:, 1]
    return out


# --------------------------------------------------------------------------- report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", type=Path, default=None)
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()

    out: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        out.append(s)

    mv = load_movements()
    dep = departures(mv)
    dep = add_base_features(dep)
    dep = add_state_features(dep, mv)
    dep = add_rotation_features(dep, mv)
    months = sorted(dep.filter(~pl.col("is_test"))["month"].unique().to_list())
    dep = add_history_features(dep, months)
    dep = encode_categoricals(dep)

    lane = dep.filter(~pl.col("is_test") & pl.col("no_nm")).sort("MVT_ID_mvt")
    test_lane = dep.filter(pl.col("is_test") & pl.col("no_nm"))
    y = lane["y"].to_numpy()
    d = lane["d"].to_numpy()
    c = lane["c_sched"].to_numpy()
    ro = np.abs(y - ROLLOVER_S) < ROLLOVER_WIN_S
    dflt = np.abs(d) <= SCHED_TOL_S

    say("# No-NM lane: modelling the block stamp instead of the taxi time (2026-09-19)")
    say()
    say(
        "Script: `scripts/eda_no_nm_block_model.py`. Training rows only; every feature is "
        "recomputed on the 2026 ranking rows in §6. Reference: R5 = `exp0068`, `month_loo`, "
        "3 seeds (`outputs/runs/" + R5_LOO + "/oof.parquet`)."
    )
    say()

    # ---------------------------------------------------------------- §1 the lane
    r5 = pl.read_parquet(paths.RUNS / R5_LOO / "oof.parquet").select(
        "MVT_ID_mvt", "pred", "y_hat_normal", "p_sched_default", "p_rollover", "p_stamp_early"
    )
    lane = lane.join(r5, on="MVT_ID_mvt", how="left")
    full = (
        dep.filter(~pl.col("is_test"))
        .select("MVT_ID_mvt", "y", "airport", "month", "MVT_TIME_UTC_mvt")
        .join(r5.select("MVT_ID_mvt", "pred"), on="MVT_ID_mvt", how="inner")
    )
    sse_full = float(np.sum((full["y"].to_numpy() - full["pred"].to_numpy()) ** 2))
    pred5 = lane["pred"].to_numpy()

    say("## 1. The lane, and what R5 does on it")
    say()
    say(
        f"Full-year OOF: {full.height:,} departures, pooled RMSE "
        f"{np.sqrt(sse_full / full.height):.1f}. The lane is {lane.height:,} rows "
        f"({lane.height / full.height:.2%}), RMSE {rmse(y, pred5):.1f}, ex-rollovers "
        f"{rmse(y[~ro], pred5[~ro]):.1f}; it carries "
        f"{np.sum((y - pred5) ** 2) / sse_full:.1%} of the year's squared error "
        f"({np.sum((y[~ro] - pred5[~ro]) ** 2) / sse_full:.1%} ex-rollovers)."
    )
    say()
    rows = []
    for (apt,), sub in lane.group_by("airport"):
        yy = sub["y"].to_numpy()
        pp = sub["pred"].to_numpy()
        rr = np.abs(yy - ROLLOVER_S) < ROLLOVER_WIN_S
        rows.append(
            {
                "airport": apt,
                "rows": sub.height,
                "rollovers": int(rr.sum()),
                "default_share": round(float((sub["d"].abs() <= SCHED_TOL_S).mean()), 3),
                "r5_rmse": round(rmse(yy, pp), 1),
                "r5_rmse_exro": round(rmse(yy[~rr], pp[~rr]), 1),
                "sse_share": round(float(np.sum((yy - pp) ** 2) / sse_full), 4),
                "sse_share_exro": round(float(np.sum((yy[~rr] - pp[~rr]) ** 2) / sse_full), 4),
            }
        )
    say(md(pl.DataFrame(rows).sort("sse_share", descending=True)))
    say()

    # ---------------------------------------------------------------- §2 d
    say("## 2. The distribution of `d = BLOCK − SCHED`")
    say()
    comp = np.where(
        ro,
        "rollover",
        np.where(
            dflt,
            "d0_default",
            np.where(d < -SCHED_TOL_S, "d_negative", np.where(d <= 3600, "d_0_1h", "d_gt1h")),
        ),
    )
    lane = lane.with_columns(pl.Series("comp", comp))
    sse_lane = float(np.sum((y - pred5) ** 2))
    rows = []
    for name in ["rollover", "d0_default", "d_negative", "d_0_1h", "d_gt1h"]:
        m = comp == name
        if m.sum() == 0:
            continue
        rows.append(
            {
                "component": name,
                "rows": int(m.sum()),
                "share": round(float(m.mean()), 4),
                "med_d": round(float(np.median(d[m])), 0),
                "med_y": round(float(np.median(y[m])), 0),
                "med_c_sched": round(float(np.median(c[m])), 0),
                "r5_rmse": round(rmse(y[m], pred5[m]), 1),
                "r5_bias": round(float(np.mean(pred5[m] - y[m])), 1),
                "lane_sse_share": round(float(np.sum((y[m] - pred5[m]) ** 2) / sse_lane), 3),
                "lirf_share": round(float((lane["airport"].to_numpy()[m] == "LIRF").mean()), 3),
            }
        )
    say(md(pl.DataFrame(rows)))
    say()
    qs = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    wn = dep.filter(~pl.col("is_test") & ~pl.col("no_nm"))
    say("Quantiles of `d` (s), no-NM lane against the rest of the training departures:")
    say()
    say(
        md(
            pl.DataFrame(
                [
                    dict(
                        population="no-NM",
                        rows=lane.height,
                        **{f"q{int(q * 100):02d}": round(float(np.quantile(d, q)), 0) for q in qs},
                    ),
                    dict(
                        population="with-NM",
                        rows=wn.height,
                        **{
                            f"q{int(q * 100):02d}": round(float(wn["d"].quantile(q)), 0) for q in qs
                        },
                    ),
                ]
            )
        )
    )
    say()
    rows = []
    for tag, m in [("with rollovers", np.ones_like(ro)), ("ex rollovers", ~ro)]:
        for scope, mm in [
            ("lane", m),
            ("LIRF", m & (lane["airport"].to_numpy() == "LIRF")),
            ("non-LIRF", m & (lane["airport"].to_numpy() != "LIRF")),
        ]:
            rows.append(
                {
                    "scope": scope,
                    "rollovers": tag,
                    "rows": int(mm.sum()),
                    "sd_y": round(float(np.std(y[mm])), 1),
                    "sd_d": round(float(np.std(d[mm])), 1),
                    "sd_c_sched": round(float(np.std(c[mm])), 1),
                    "corr_c_d": round(float(np.corrcoef(c[mm], d[mm])[0, 1]), 3),
                    "var_d_over_var_y": round(float(np.var(d[mm]) / np.var(y[mm])), 2),
                }
            )
    say("Variance decomposition of the identity `y = C_sched − d`:")
    say()
    say(md(pl.DataFrame(rows)))
    say()

    # ---------------------------------------------------------------- §3 observability
    say("## 3. What is observable on a 2026 ranking row")
    say()
    rank = pl.read_parquet(paths.RAW / "ranking.parquet")
    rank = rank.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m").alias("m"))
    rows = []
    for (ph, m), sub in rank.group_by("PHASE_mvt", "m"):
        rows.append(
            {
                "phase": ph,
                "month": m,
                "rows": sub.height,
                "block_time": round(float(sub["BLOCK_TIME_UTC_mvt"].is_not_null().mean()), 4),
                "taxitime": round(float(sub["TAXITIME_SEC_mvt"].is_not_null().mean()), 4),
                "sched_time": round(float(sub["SCHED_TIME_UTC_mvt"].is_not_null().mean()), 4),
                "mvt_time": round(float(sub["MVT_TIME_UTC_mvt"].is_not_null().mean()), 4),
                "aobt3": round(float(sub["AOBT_3_flt"].is_not_null().mean()), 4),
            }
        )
    say(md(pl.DataFrame(rows).sort("phase", "month")))
    say()
    say(
        "The organizers hide `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` on the ranking "
        "**departures only**: on the 344,693 ranking **arrivals** both are 100 % present. "
        "Every arrival-side quantity used below (in-block lateness, the stand rotation) is "
        "therefore a test-time observable, and `MVT_TIME_UTC_mvt` / `SCHED_TIME_UTC_mvt` are "
        "present on 100 % of ranking departures, which is what makes `C_sched` computable."
    )
    say()

    # ---------------------------------------------------------------- §4 signals
    say("## 4. How much of `d` do the candidate observables carry?")
    say()
    rows = []
    for f in [
        "f_c_sched",
        "f_dnb_med_1800",
        "f_dnb_med_3600",
        "f_dnb_med_7200",
        "f_dnb_q75_3600",
        "f_dnb_n_3600",
        "f_bnb_med_3600",
        "f_anb_in_med_3600",
        "f_anb_land_med_3600",
        "f_day_med",
        "f_day_q90",
        "f_day_nonm_share",
        "f_prev_arr_gap",
        "f_prev_arr_delay",
        "f_reoccupied",
        "f_next_arr_gap",
        "f_te_airline_dmed",
        "f_te_airline_default",
        "f_te_airline_taxi",
        "f_te_stand_default",
        "f_te_ades_default",
        "f_te_arh_taxi",
        "f_hour",
    ]:
        v = lane[f].cast(pl.Float64).to_numpy()
        m = np.isfinite(v) & ~ro
        if m.sum() < 100:
            continue
        rows.append(
            {
                "feature": f,
                "non_null": round(float(np.isfinite(v).mean()), 3),
                "corr_d": round(float(np.corrcoef(v[m], d[m])[0, 1]), 3),
                "corr_y": round(float(np.corrcoef(v[m], y[m])[0, 1]), 3),
                "corr_default": round(float(np.corrcoef(v[m], dflt[m].astype(float))[0, 1]), 3),
            }
        )
    say("Ex-rollovers, whole lane (Pearson; `corr_default` is against the 0/1 label):")
    say()
    say(md(pl.DataFrame(rows).sort("corr_d", descending=True)))
    say()
    lm = lane["airport"].to_numpy() == "LIRF"
    rows = []
    for f in ["f_c_sched", "f_dnb_med_3600", "f_bnb_med_3600", "f_day_med", "f_te_airline_dmed"]:
        v = lane[f].cast(pl.Float64).to_numpy()
        for scope, mm in [("LIRF", lm), ("non-LIRF", ~lm)]:
            m = np.isfinite(v) & ~ro & mm
            rows.append(
                {
                    "feature": f,
                    "scope": scope,
                    "rows": int(m.sum()),
                    "corr_d": round(float(np.corrcoef(v[m], d[m])[0, 1]), 3),
                    "corr_y": round(float(np.corrcoef(v[m], y[m])[0, 1]), 3),
                }
            )
    say(md(pl.DataFrame(rows)))
    say()

    # ---------------------------------------------------------------- §5 models
    say("## 5. Fold-safe lane models")
    say()
    lane = lane.with_columns(pl.Series("is_default", dflt.astype(np.float64)))
    feats_base = BASE_FEATURES
    feats_all = ALL_FEATURES

    preds: dict[str, np.ndarray] = {"R5 (exp0068, month_loo)": pred5}
    p_y_base = fit_lomo(lane, feats_base, "y", months)
    preds["M1 lane LGBM on y, base"] = p_y_base
    p_y_all = fit_lomo(lane, feats_all, "y", months)
    preds["M2 lane LGBM on y, +state/rotation/history"] = p_y_all
    p_d_all = fit_lomo(lane, feats_all, "d", months)
    preds["M3 lane LGBM on d, y = C − d̂"] = c - p_d_all
    p_def = fit_lomo_clf(lane, feats_all, "is_default", months)
    lane = lane.with_columns(pl.Series("p_def", p_def))
    # The expert branch is `M2` itself: no label-free "clean rows" mask exists, so the taxi
    # expert is the same lane regression, contaminated by the defaults it is mixed against.
    preds["M4 mixture p̂·C + (1−p̂)·M2"] = p_def * c + (1 - p_def) * p_y_all
    preds["CF1 median taxi (airport,runway,hour)"] = (
        lane["f_te_arh_taxi"]
        .cast(pl.Float64)
        .fill_null(float(np.nanmedian(lane["f_te_arh_taxi"].cast(pl.Float64).to_numpy())))
        .to_numpy()
    )
    cf2 = c - lane["f_dnb_med_3600"].cast(pl.Float64).fill_null(0.0).to_numpy()
    preds["CF2 C − contemporaneous median delay"] = cf2
    preds["CF3 p̂·C + (1−p̂)·CF1"] = (
        p_def * c + (1 - p_def) * preds["CF1 median taxi (airport,runway,hour)"]
    )
    preds["M5 blend 0.5·R5 + 0.5·M2"] = 0.5 * pred5 + 0.5 * p_y_all

    # --- what the runner would actually do: keep R5's mixture, swap only p_sched_default ---
    yhat5 = lane["y_hat_normal"].to_numpy()
    p5 = {
        "sched_default": lane["p_sched_default"].to_numpy(),
        "rollover": lane["p_rollover"].to_numpy(),
        "stamp_early": lane["p_stamp_early"].to_numpy(),
    }
    cvals = {
        "sched_default": np.where(np.isfinite(c), c, yhat5),
        "rollover": ROLLOVER_S + yhat5,
        "stamp_early": yhat5 + 1495.0,
    }
    recon = np.maximum(_mix(yhat5, p5, cvals), 0.0)
    diff = np.abs(recon - pred5)
    say(
        "Reconstruction check: `max(0, ŷ + Σ p_k(C_k − ŷ))` rebuilt from the stored head "
        f"columns reproduces R5's stored `pred` on {float((diff <= 1.0).mean()):.4%} of the "
        f"lane to within 1 s (median |difference| {float(np.median(diff)):.2e} s, max "
        f"{float(np.nanmax(diff)):.1f} s on {int((diff > 1.0).sum())} rows). The exceptions "
        "are rows whose *raw* normal prediction was negative: the runner clips the stored "
        "`y_hat_normal` and the stored `pred` separately (`target.clip_min: 0`, "
        "`src/prc/eval/run.py`), so the un-clipped `ŷ` that entered the mixture is not "
        "recoverable there. Every candidate below is clipped the same way."
    )
    say()
    preds["M6 R5 mixture, lane p̂ for p_sched_default"] = _mix(
        yhat5, {**p5, "sched_default": p_def}, cvals
    )
    preds["M7 p̂·C + (1−p̂)·R5 ŷ_normal"] = p_def * c + (1 - p_def) * yhat5
    preds["M8 R5 p_sched_default with M2 as the expert"] = _mix(
        p_y_all,
        p5,
        {
            "sched_default": np.where(np.isfinite(c), c, p_y_all),
            "rollover": ROLLOVER_S + p_y_all,
            "stamp_early": p_y_all + 1495.0,
        },
    )

    # the runner clips every prediction at 0 (`target.clip_min`), so the candidates are too
    preds = {k: (v if k.startswith("R5") else np.maximum(v, 0.0)) for k, v in preds.items()}

    # LIRF carries 31 % of the year's squared error on its own, and §5.2 shows the lane models
    # buy it at the other nine airports' expense; these two restrict them to LIRF.
    lm0 = lane["airport"].to_numpy() == "LIRF"
    preds["M9 M4 at LIRF only, R5 elsewhere"] = np.where(
        lm0, preds["M4 mixture p̂·C + (1−p̂)·M2"], pred5
    )
    preds["M10 M6 at LIRF only, R5 elsewhere"] = np.where(
        lm0, preds["M6 R5 mixture, lane p̂ for p_sched_default"], pred5
    )

    gate = np.isin(lane["month"].to_numpy(), GATE_MONTHS)
    rows = []
    for name, p in preds.items():
        rows.append(
            {
                "model": name,
                "lane_rmse": round(rmse(y, p), 1),
                "lane_exro": round(rmse(y[~ro], p[~ro]), 1),
                "LIRF": round(rmse(y[lm], p[lm]), 1),
                "LIRF_exro": round(rmse(y[lm & ~ro], p[lm & ~ro]), 1),
                "non_LIRF": round(rmse(y[~lm], p[~lm]), 1),
                "jan_jul": round(rmse(y[gate], p[gate]), 1),
                "jan_jul_exro": round(rmse(y[gate & ~ro], p[gate & ~ro]), 1),
                "defaults": round(rmse(y[dflt], p[dflt]), 1),
                "non_default_exro": round(rmse(y[~dflt & ~ro], p[~dflt & ~ro]), 1),
            }
        )
    say(md(pl.DataFrame(rows)))
    say()
    auc = _auc(dflt[~ro].astype(float), p_def[~ro])
    auc_lirf = _auc(dflt[lm & ~ro].astype(float), p_def[lm & ~ro])
    say(
        f"The lane-scoped default classifier (`M4`/`CF3`, leave-one-month-out) has AUC "
        f"{auc:.3f} on the lane and {auc_lirf:.3f} on LIRF, against R5's shipped head "
        f"{_auc(dflt[~ro].astype(float), lane['p_sched_default'].to_numpy()[~ro]):.3f} / "
        f"{_auc(dflt[lm & ~ro].astype(float), lane['p_sched_default'].to_numpy()[lm & ~ro]):.3f} "
        "on the same rows."
    )
    say()

    say("### 5.1 The classifier, per airport")
    say()
    rows = []
    apt = lane["airport"].to_numpy()
    for a in sorted(set(apt.tolist())):
        m = (apt == a) & ~ro
        rows.append(
            {
                "airport": a,
                "rows": int(m.sum()),
                "defaults": int(dflt[m].sum()),
                "auc_lane_head": round(_auc(dflt[m].astype(float), p_def[m]), 3),
                "auc_r5_head": round(
                    _auc(dflt[m].astype(float), lane["p_sched_default"].to_numpy()[m]), 3
                ),
                "mean_p_lane": round(float(np.mean(p_def[m])), 4),
                "mean_p_r5": round(float(np.mean(lane["p_sched_default"].to_numpy()[m])), 4),
                "rate": round(float(dflt[m].mean()), 4),
            }
        )
    say(md(pl.DataFrame(rows)))
    say()

    say("### 5.2 Lane RMSE per airport and per month, R5 against the two candidates")
    say()
    cand = [
        "M4 mixture p̂·C + (1−p̂)·M2",
        "M6 R5 mixture, lane p̂ for p_sched_default",
        "M8 R5 p_sched_default with M2 as the expert",
        "M9 M4 at LIRF only, R5 elsewhere",
    ]
    rows = []
    for a in sorted(set(apt.tolist())):
        m = apt == a
        mx = m & ~ro
        rows.append(
            dict(
                scope=a,
                rows=int(m.sum()),
                r5=round(rmse(y[m], pred5[m]), 1),
                **{f"c{i + 1}": round(rmse(y[m], preds[k][m]), 1) for i, k in enumerate(cand)},
                r5_exro=round(rmse(y[mx], pred5[mx]), 1),
                **{
                    f"c{i + 1}_exro": round(rmse(y[mx], preds[k][mx]), 1)
                    for i, k in enumerate(cand)
                },
            )
        )
    mon = lane["month"].to_numpy()
    for a in months:
        m = mon == a
        mx = m & ~ro
        rows.append(
            dict(
                scope=a,
                rows=int(m.sum()),
                r5=round(rmse(y[m], pred5[m]), 1),
                **{f"c{i + 1}": round(rmse(y[m], preds[k][m]), 1) for i, k in enumerate(cand)},
                r5_exro=round(rmse(y[mx], pred5[mx]), 1),
                **{
                    f"c{i + 1}_exro": round(rmse(y[mx], preds[k][mx]), 1)
                    for i, k in enumerate(cand)
                },
            )
        )
    say(", ".join(f"`c{i + 1}` = {k}" for i, k in enumerate(cand)) + ".")
    say()
    say(md(pl.DataFrame(rows)))
    say()

    say("### 5.2b Where M9's squared-error reduction sits")
    say()
    m9 = preds["M9 M4 at LIRF only, R5 elsewhere"]
    groups = {
        "LIRF rollovers": lm & ro,
        "LIRF schedule defaults": lm & dflt & ~ro,
        "LIRF other rows": lm & ~dflt & ~ro,
        "non-LIRF (untouched by M9)": ~lm,
    }
    tot = float(np.sum((y - pred5) ** 2) - np.sum((y - m9) ** 2))
    rows = []
    for name, g in groups.items():
        red = float(np.sum((y[g] - pred5[g]) ** 2) - np.sum((y[g] - m9[g]) ** 2))
        rows.append(
            {
                "group": name,
                "rows": int(g.sum()),
                "r5_rmse": round(rmse(y[g], pred5[g]), 1),
                "m9_rmse": round(rmse(y[g], m9[g]), 1),
                "sse_reduction": round(red, 0),
                "share_of_total_reduction": round(red / tot, 3) if tot else float("nan"),
            }
        )
    say(md(pl.DataFrame(rows)))
    say()

    say("### 5.3 Feature-family ablation on the `y` model (lane RMSE ex-rollovers)")
    say()
    rows = []
    fam = {
        "base only": BASE_FEATURES,
        "base + state": BASE_FEATURES + STATE_FEATURES,
        "base + rotation": BASE_FEATURES + ROTATION_FEATURES,
        "base + history": BASE_FEATURES + HISTORY_FEATURES,
        "all": ALL_FEATURES,
        "all − C_sched": [f for f in ALL_FEATURES if f != "f_c_sched"],
        "all − state": [f for f in ALL_FEATURES if f not in STATE_FEATURES],
        "all − history": [f for f in ALL_FEATURES if f not in HISTORY_FEATURES],
    }
    for name, fs in fam.items():
        p = fit_lomo(lane, fs, "y", months)
        rows.append(
            {
                "feature_set": name,
                "n_features": len(fs),
                "lane_rmse": round(rmse(y, p), 1),
                "lane_exro": round(rmse(y[~ro], p[~ro]), 1),
                "LIRF_exro": round(rmse(y[lm & ~ro], p[lm & ~ro]), 1),
                "non_LIRF": round(rmse(y[~lm], p[~lm]), 1),
            }
        )
    say(md(pl.DataFrame(rows)))
    say()

    # ---------------------------------------------------------------- swap-in effect
    say("### 5.4 Pooled effect of swapping the lane predictions into R5")
    say()
    id_to_pos = {int(v): i for i, v in enumerate(lane["MVT_ID_mvt"].to_numpy())}
    pos = np.array([id_to_pos.get(int(v), -1) for v in full["MVT_ID_mvt"].to_numpy()])
    in_lane = pos >= 0
    y_full = full["y"].to_numpy()
    p_ref = full["pred"].to_numpy()
    block_of_row, groups = PD.block_index(full)
    n_blocks = int(block_of_row.max()) + 1
    w = PD.block_weights(groups, n_blocks, args.bootstrap, seed=12345) if args.bootstrap else None
    se_ref = (y_full - p_ref) ** 2
    ro_full = np.abs(y_full - ROLLOVER_S) < ROLLOVER_WIN_S
    rows = []
    for name, p in preds.items():
        if name.startswith("R5"):
            continue
        p_cand = p_ref.copy()
        p_cand[in_lane] = p[pos[in_lane]]
        se_cand = (y_full - p_cand) ** 2
        rec = {
            "swapped_in": name,
            "pooled_ref": round(float(np.sqrt(se_ref.mean())), 1),
            "pooled_cand": round(float(np.sqrt(se_cand.mean())), 1),
            "delta": round(float(np.sqrt(se_cand.mean()) - np.sqrt(se_ref.mean())), 2),
            "delta_exro": round(
                float(np.sqrt(se_cand[~ro_full].mean()) - np.sqrt(se_ref[~ro_full].mean())), 2
            ),
        }
        if w is not None:
            b = PD.bootstrap_delta(
                w, block_of_row, n_blocks, se_ref, se_cand, np.ones(len(y_full), bool)
            )
            bx = PD.bootstrap_delta(w, block_of_row, n_blocks, se_ref, se_cand, ~ro_full)
            rec.update(
                boot_lo=b["boot_lo"],
                boot_hi=b["boot_hi"],
                boot_lo_exro=bx["boot_lo"],
                boot_hi_exro=bx["boot_hi"],
            )
        rows.append(rec)
    say(md(pl.DataFrame(rows)))
    say()
    say(
        "`delta` is candidate − reference full-year pooled RMSE; the interval is the 95 % "
        "day-block bootstrap of `scripts/paired_delta.py` (blocks = airport × UTC date, "
        f"resampled within airport-month, N = {args.bootstrap}). Negative is better."
    )
    say()

    # ---------------------------------------------------------------- §6 leak audit
    say("## 6. Leak audit — every feature recomputed on the 2026 ranking rows")
    say()
    rows = []
    for f in ALL_FEATURES:
        a = lane[f].cast(pl.Float64).to_numpy()
        b = test_lane[f].cast(pl.Float64).to_numpy()
        rows.append(
            {
                "feature": f,
                "null_2025": round(float(np.mean(~np.isfinite(a))), 4),
                "null_2026": round(float(np.mean(~np.isfinite(b))), 4),
                "med_2025": round(float(np.nanmedian(a)), 1)
                if np.isfinite(a).any()
                else float("nan"),
                "med_2026": round(float(np.nanmedian(b)), 1)
                if np.isfinite(b).any()
                else float("nan"),
            }
        )
    say(md(pl.DataFrame(rows)))
    say()
    say(
        f"2025 no-NM rows {lane.height:,}; 2026 ranking no-NM rows {test_lane.height:,}. "
        "Source columns, family by family, against what a ranking departure carries:"
    )
    say()
    say(
        "- **base** — `MVT_TIME_UTC_mvt`, `SCHED_TIME_UTC_mvt`, `FLIGHT_mvt`, `STAND_mvt`, "
        "`RUNWAY_mvt`, `ADES_mvt`, `AIRCRAFT_TYPE_mvt` of the row itself: all present on "
        "100 % / 99.2–99.6 % / 92.7 % of the ranking departures (§3 and the table above);\n"
        "- **state** — `AOBT_3_flt − SCHED_TIME_UTC_mvt` of *other departures* (`AOBT_3_flt` "
        "is a Network-Manager column, present on 98.4–98.5 % of ranking departures and "
        "explicitly allowed), plus `BLOCK_TIME_UTC_mvt` and `MVT_TIME_UTC_mvt` of "
        "**arrivals**, which are 100 % present in the ranking months. No departure's "
        "`BLOCK_TIME_UTC_mvt` is read anywhere;\n"
        "- **rotation** — arrival in-block stamps only;\n"
        "- **history** — `BLOCK_TIME_UTC_mvt` / `TAXITIME_SEC_mvt` of *2025* departures in "
        "months other than the row's own. 2025 is fully labelled in both the CV and the "
        "submission setting, so this is an ordinary fold-aware target encoding.\n"
        "\n"
        "The `med_2025` / `med_2026` columns are the empirical check that each is built the "
        "same way on both sides. (In the first version of this study the `f_dnb_*` / `f_bnb_*` "
        "medians read 480 s against 90 s, which looked like a leak and was instead the "
        "`rolling(group_by=…)` row-order bug that `roll()` now asserts against: the window "
        "statistics were permuted across airports.)"
    )
    say()

    # ---------------------------------------------------------------- verdict
    say("## 7. Verdict")
    say()
    best = "M9 M4 at LIRF only, R5 elsewhere"
    say(
        f"1. **Study A's premise is refuted.** Modelling `d` directly (`M3`) is "
        f"{rmse(y[~ro], preds['M3 lane LGBM on d, y = C − d̂'][~ro]):.0f} s ex-rollovers "
        f"against R5's {rmse(y[~ro], pred5[~ro]):.0f} and costs "
        "the pooled year; the closed form `C_sched − contemporaneous median delay` (`CF2`) is "
        "an order of magnitude worse still. `d` is not a delay that the airport's state "
        "predicts: on the 65 % of the lane whose `d` exceeds an hour the taxi time is "
        "ordinary (median 977 s) and the whole spread is idiosyncratic flight-level "
        "lateness."
    )
    # `f_c_minus_dnb` is `C_sched` minus a window statistic, so it inherits `C_sched`'s
    # correlation and is not a measurement of the airport's state; the claim is about the
    # neighbour-delay statistics themselves.
    delay_feats = [
        f for f in STATE_FEATURES if f.startswith(("f_dnb_", "f_bnb_", "f_anb_", "f_day_"))
    ]
    st_corr = max(
        abs(float(np.corrcoef(v[np.isfinite(v) & ~ro], d[np.isfinite(v) & ~ro])[0, 1]))
        for v in (lane[f].cast(pl.Float64).to_numpy() for f in delay_feats)
        if np.isfinite(v).sum() > 100
    )
    st_res = float(
        np.std(d[~ro])
        * np.sqrt(
            1
            - max(
                float(
                    np.corrcoef(
                        lane["f_dnb_med_3600"].cast(pl.Float64).fill_null(0.0).to_numpy()[~ro],
                        d[~ro],
                    )[0, 1]
                )
                ** 2,
                0.0,
            )
        )
    )
    say(
        "2. **No observable pins `BLOCK_TIME` on this lane.** §4: the contemporaneous "
        f"departure- and arrival-delay statistics reach |r| = {st_corr:.2f} against `d` at "
        f"best, which leaves a residual sd of {st_res:,.0f} s against `d`'s "
        f"{np.std(d[~ro]):,.0f} — an order of magnitude more than the ~900 s a taxi-time "
        "model already achieves, so the regression route cannot pay. The only columns that "
        "carry real weight are `C_sched` itself (r = 0.87, which the head already uses as "
        "`C`) and the fold-safe identity encodings (|r| ≤ 0.21 against `d`, but r = 0.52–0.57 "
        "against the schedule-default label)."
    )
    n_better = int(
        sum(
            rmse(y[(mon == m) & ~ro], preds[best][(mon == m) & ~ro])
            < rmse(y[(mon == m) & ~ro], pred5[(mon == m) & ~ro])
            for m in months
        )
    )
    say(
        "3. **What does work is a lane-scoped mixture, and it is a classification gain.** "
        f"`{best}` is the strongest candidate in §5.4, and it improves {n_better} of the "
        "twelve months ex-rollovers (§5.2) while leaving all nine other airports bit-identical "
        "to R5. The components separate: the lane-scoped expert (`M8`, R5's own `p` with the "
        "lane regression as the normal branch) carries most of it, the lane-scoped classifier "
        "(`M6`/`M7`, R5's expert with the lane `p`) the rest — but `M6` alone has an interval "
        "that includes zero and makes July worse, so it is not shippable on its own. §5.2b: "
        "46 % of `M9`'s squared-error reduction is **12 rollover rows**, so the ex-rollover "
        "delta is the number that will still be there in 2026."
    )
    say(
        "4. **Lane-wide, the cost lands on the other nine airports.** §5.2: `M4` is worse "
        "than R5 at EDDF (310 → 747), EDDM, EGLL, EHAM, LEBL, LEMD, LFPG and LSZH, because "
        "the lane regression is fitted on 22 k rows dominated by LIRF's substitution while "
        "R5's normal branch is fitted on 2.08 M. Restricting it to LIRF (`M9`) removes that "
        "cost — but the restriction was chosen after reading the per-airport table, so `M9` "
        "is a diagnostic ceiling, not a validated change. The defensible form of the same "
        "idea is a lane expert *gated by the airport's schedule-default rate* "
        "(`f_te_airport_default` already encodes it, and LIRF's 0.181 is 3–16× every other "
        "airport's), which has not been run."
    )
    say(
        "5. **Recommendation.** Ship nothing from this study as a feature block. Run one "
        "experiment: a lane-scoped corruption head (classifier trained on the no-NM rows "
        "only, with the identity encodings of internal note 2026-09-18_lirf-no-nm §9) "
        "together with a lane-scoped normal expert, gated on the day-block interval on both "
        "`Jan` and `Jul` separately and on the per-airport table — the two things `M4` fails "
        "and `M9` only passes by construction. `scripts/no_nm_lane_model.py` implements the "
        "post-processing form so it can be scored against any stored run without touching "
        "the runner."
    )
    say()

    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text("\n".join(out) + "\n")
        print(f"\nwrote {args.write}")


def _mix(
    y_hat: np.ndarray, probs: dict[str, np.ndarray], values: dict[str, np.ndarray]
) -> np.ndarray:
    """``prc.mixture.corruption.apply_mixture`` on stored columns (imported, not re-derived)."""
    from prc.mixture.corruption import apply_mixture

    return apply_mixture(y_hat, probs, values, p_max=1.0)


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    m = np.isfinite(s)
    if len(set(y[m].tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y[m], s[m]))


if __name__ == "__main__":
    main()
