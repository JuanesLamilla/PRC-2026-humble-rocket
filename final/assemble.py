"""Blend -> corrections -> label-free rules -> humble-rocket_final.parquet.

usage (repo root): python final/assemble.py

Steps, in order:

1. read the blended submission rows (final/blend.py);
2. feeder-lateness correction: when the stand's feeding arrival went in-block
   more than 90 min behind schedule, the blend overshoots slightly. Offsets per
   C band are fitted on the 2025 out-of-fold blend with a heavy smoothing prior
   (validated month-held: negative in 8 of 12 months, interval below zero).
   Arrival block times are published for the submission months, so the trigger
   is computable on every 2026 row;
3. LFPG EOBT-drift adjustment: in June and July 2026 the filed EOBT at LFPG
   sits about four minutes later relative to LOBT than in any 2025 month, a
   filing-practice change no 2025-trained model has seen. EOBT-anchored
   features mis-read exactly those rows; the EOBT-free transformer is immune
   by construction, so the mean gap between it and the base transformer over
   the affected cell estimates the bias, label-free. The gap is taken from the
   transformer pair of the generation that feeds each month's blend. A
   diagnostic prints the same gap for every airport-month;
4. label-free rules (proven on 2025):
   - window clip: on NM rows the label provably lies in [ml - 3620, ml + 3620];
   - rollover identity: when ml > 20 h, LOBT kept the previous day's date and
     the label equals ml itself;
   - twin cap: a no-NM departure whose matched arrival "twin" carries an NM
     plan must have fallen outside the join window, so y <= MVT - twin LOBT
     - 3606 (applied when the bound is >= 300 s);
5. write the file and validate it against the organizers' template.

reads   work/blend_{oof,test}.parquet, work/ft_*/outer_00.parquet, data/raw/
writes  submissions/humble-rocket_final.parquet
"""

import glob
import os
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")  # R-arrow ext types in the raw files
import numpy as np
import polars as pl

KEY = "MVT_ID_mvt"
TARGET = "TAXITIME_SEC_mvt"
WINDOW = 3606.0
CLIP = 3620.0
WORK = Path(os.environ.get("WORK_DIR", "work"))
RAW = Path(os.environ.get("RAW_DIR", "data/raw"))
BLEND = Path(os.environ.get("BLEND_DIR", str(WORK)))
SUBMISSIONS = Path(os.environ.get("SUBMISSIONS_DIR", "submissions"))
SUBMISSIONS.mkdir(exist_ok=True)

FEEDER_LATE = 5400.0   # feeder arrival > 90 min behind schedule
FEEDER_PRIOR = 2000.0  # smoothing pseudo-count for the band offsets
C_BANDS = [(0.0, 3600.0), (3600.0, 7200.0), (7200.0, 1e18)]
DRIFT_CELLS = [("LFPG", 6), ("LFPG", 7)]  # airport-months with the EOBT filing drift
DRIFT_LIMIT = 60.0
#: transformer pair (base, EOBT-free) per generation; months listed in
#: blend_report.txt as retrain months use the retrain pair.
DRIFT_PAIR_PRIMARY = ("ft_ftC7", "ft_ftNE")
DRIFT_PAIR_RETRAIN = ("ft_ftCF", "ft_ftNEFF")

SLIM = [KEY, "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "FLIGHT_mvt", "STAND_mvt",
        "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "LOBT_flt"]
sec = lambda a, b: (pl.col(a) - pl.col(b)).dt.total_seconds()  # noqa: E731


def slim_frame(path) -> pl.DataFrame:
    return pl.scan_parquet(path).select(SLIM).collect()


def feeder_lateness(frame: pl.DataFrame, departures: pl.DataFrame) -> pl.DataFrame:
    """Lateness of the last arrival on the departure's stand before its
    scheduled time (null when there is none)."""
    arrivals = (
        frame.filter(pl.col("PHASE_mvt") == "ARR")
        .with_columns(late=sec("BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"))
        .filter(pl.col("BLOCK_TIME_UTC_mvt").is_not_null())
        .select(pl.col("ADES_mvt").alias("apt"), "STAND_mvt",
                pl.col("BLOCK_TIME_UTC_mvt").alias("t_inblock"), "late")
        .sort("t_inblock")
    )
    deps = departures.select(KEY, pl.col("ADEP_mvt").alias("apt"), "STAND_mvt",
                             pl.col("SCHED_TIME_UTC_mvt").alias("t")).sort("t")
    joined = deps.join_asof(arrivals, left_on="t", right_on="t_inblock",
                            by=["apt", "STAND_mvt"], strategy="backward")
    return joined.select(KEY, "late")


def build_twin_caps(movements: pl.DataFrame) -> pl.DataFrame:
    """Upper bounds for no-NM departures from their NM-joined arrival twin
    (same callsign and route, landing at another challenge airport 10 min - 6 h
    after take-off). Label-free."""
    deps = movements.filter(pl.col("PHASE_mvt") == "DEP")
    no_nm = deps.filter(pl.col("LOBT_flt").is_null()).select(
        KEY, "ADEP_mvt", "ADES_mvt", "FLIGHT_mvt", "MVT_TIME_UTC_mvt")
    twins = movements.filter((pl.col("PHASE_mvt") == "ARR") & pl.col("LOBT_flt").is_not_null()).select(
        pl.col("ADEP_mvt").alias("twin_from"), pl.col("ADES_mvt").alias("twin_to"),
        "FLIGHT_mvt", pl.col("MVT_TIME_UTC_mvt").alias("twin_mvt"), pl.col("LOBT_flt").alias("twin_lobt"))
    joined = no_nm.join(twins, left_on=["ADEP_mvt", "ADES_mvt", "FLIGHT_mvt"],
                        right_on=["twin_from", "twin_to", "FLIGHT_mvt"])
    joined = joined.filter(sec("twin_mvt", "MVT_TIME_UTC_mvt").is_between(600, 6 * 3600)).unique(KEY, keep="first")
    return joined.with_columns(cap=sec("MVT_TIME_UTC_mvt", "twin_lobt") - WINDOW).select(KEY, "cap")


def apply_rules(pred: np.ndarray, ml: np.ndarray, has_nm: np.ndarray, caps: np.ndarray) -> np.ndarray:
    """Window clip, rollover identity, twin cap, floor. ml/caps NaN where absent."""
    lo = np.maximum(np.nan_to_num(ml, nan=0.0) - CLIP, 0.0)
    hi = np.nan_to_num(ml, nan=1e18) + CLIP
    pred = np.where(has_nm, np.clip(pred, lo, hi), pred)
    pred = np.where(has_nm & (np.nan_to_num(ml, nan=0.0) > 72000), np.nan_to_num(ml, nan=0.0), pred)
    cap_active = ~has_nm & np.isfinite(caps) & (caps >= 300)
    pred = np.where(cap_active, np.minimum(pred, caps), pred)
    return np.maximum(pred, 0.0)


def main() -> None:
    test = pl.read_parquet(BLEND / "blend_test.parquet")
    pred = test["blend"].to_numpy().astype(float).copy()
    assert np.isfinite(pred).all(), "blend produced non-finite predictions"

    # --- feeder correction, offsets refit on the delivered-equivalent OOF
    oof = pl.read_parquet(BLEND / "blend_oof.parquet")
    oof = oof.filter((pl.col("y") - 86400).abs() >= 3600)  # rollovers out
    train = pl.concat([slim_frame(f) for f in sorted(glob.glob(str(RAW / "training_2025-*.parquet")))])
    lateness = feeder_lateness(train, train.filter(pl.col("PHASE_mvt") == "DEP"))
    oof = oof.join(lateness, on=KEY, how="left")
    y, p = oof["y"].to_numpy(), oof["blend"].to_numpy()
    C = oof["C"].to_numpy()
    late = oof["late"].fill_null(np.nan).to_numpy().astype(float)
    f_late = np.isfinite(late) & (late > FEEDER_LATE)
    offsets = {}
    for lo_b, hi_b in C_BANDS:
        rows = f_late & (C >= lo_b) & (C < hi_b)
        offsets[(lo_b, hi_b)] = float(np.sum(y[rows] - p[rows]) / (rows.sum() + FEEDER_PRIOR))
    print("feeder offsets by C band:", {f"{int(a)}-{int(b)}": round(v, 2) for (a, b), v in offsets.items()}, flush=True)

    final_slim = slim_frame(RAW / "final_ranking.parquet")
    lateness_t = feeder_lateness(final_slim, final_slim.filter(pl.col("PHASE_mvt") == "DEP"))
    jt = test.select(KEY, "C").join(lateness_t, on=KEY, how="left", maintain_order="left")
    late_t = jt["late"].fill_null(np.nan).to_numpy().astype(float)
    C_t = jt["C"].to_numpy().astype(float)
    corr = np.zeros(test.height)
    f_late_t = np.isfinite(late_t) & (late_t > FEEDER_LATE)
    for (lo_b, hi_b), v in offsets.items():
        corr[f_late_t & (C_t >= lo_b) & (C_t < hi_b)] = v
    print(f"feeder correction: {int((corr != 0).sum())} rows, mean {corr[corr != 0].mean():+.2f} s", flush=True)
    pred += corr

    # --- LFPG EOBT-drift adjustment, per-generation transformer pair
    def pair_gap(base_tag, noe_tag, name):
        b = pl.read_parquet(WORK / base_tag / "outer_00.parquet").select(KEY, pl.col("y_ft").alias("b"))
        n = pl.read_parquet(WORK / noe_tag / "outer_00.parquet").select(KEY, pl.col("y_ft").alias("n"))
        return b.join(n, on=KEY, how="inner", maintain_order="left").select(
            KEY, (pl.col("n") - pl.col("b")).alias(name))

    j = (test.select(KEY, "mon", "ADEP_mvt", "nm", "C")
         .join(pair_gap(*DRIFT_PAIR_PRIMARY, "gap_p"), on=KEY, how="left", maintain_order="left")
         .join(pair_gap(*DRIFT_PAIR_RETRAIN, "gap_r"), on=KEY, how="left", maintain_order="left"))
    j = j.with_columns(gap=pl.coalesce("gap_p", "gap_r"))
    assert j["gap"].null_count() == 0
    gap = j["gap"].to_numpy()
    airport, month = j["ADEP_mvt"].to_numpy(), j["mon"].to_numpy()
    in_lane = (j["nm"].to_numpy() > 0) & (j["C"].to_numpy() < 3600)
    test_months = sorted(set(month.tolist()))
    for apt in np.unique(airport):
        cells = [f"{m}:{gap[(airport == apt) & (month == m) & in_lane].mean():+.0f}" for m in test_months]
        print(f"  eobt gap {apt} (NM C<1h, s): " + " ".join(cells), flush=True)
    for apt, mon in DRIFT_CELLS:
        rows = (airport == apt) & (month == mon) & in_lane
        if not rows.any():
            continue
        shift = float(np.clip(gap[rows].mean(), -DRIFT_LIMIT, DRIFT_LIMIT))
        pred[rows] += shift
        print(f"EOBT-drift adjustment {apt} month {mon}: {shift:+.1f} s on {int(rows.sum())} rows", flush=True)

    # --- label-free rules + write
    caps = test.select(KEY).join(build_twin_caps(final_slim), on=KEY, how="left",
                                 maintain_order="left")["cap"].fill_null(np.nan).to_numpy()
    ml = test["ml"].fill_null(np.nan).to_numpy().astype(float)
    has_nm = test["nm"].to_numpy() > 0
    pred = apply_rules(pred, ml, has_nm, caps)

    out = test.select(KEY).with_columns(pl.Series(TARGET, np.round(pred)).cast(pl.Int32))
    template = pl.read_parquet(RAW / "final_submitting.parquet")
    assert out.height == template.height
    assert out[TARGET].null_count() == 0 and (out[TARGET] >= 0).all()
    assert set(out[KEY].to_list()) == set(template[KEY].to_list())
    out.write_parquet(SUBMISSIONS / "humble-rocket_final.parquet")
    print(test.select("mon").with_columns(p=pl.Series(pred)).group_by("mon")
          .agg(n=pl.len(), mean=pl.col("p").mean(), p90=pl.col("p").quantile(0.9)).sort("mon"), flush=True)
    print(f"written {SUBMISSIONS}/humble-rocket_final.parquet", flush=True)


if __name__ == "__main__":
    main()
