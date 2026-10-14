"""EUROCONTROL / PRU airport performance, joined per (airport, calendar day) — plan row P1-d.

"Was it a bad day at this airport?" The movement table answers that only through counts we
derive ourselves; the PRU open CSVs answer it with the network's own view — how much traffic
the airport handled, how many minutes of pre-departure delay its flights took, and what share
of its departures were under an ATFM regulation. Archived by
``scripts/fetch_external.py eurocontrol`` to ``data/external/eurocontrol_perf_<date>/``.

Files and columns actually used (headers verified against the 2026-09-18 pull):

===============================  ==========  ==================================================
file                             grain       columns used
===============================  ==========  ==================================================
``airport_traffic_<y>.csv``      daily       ``FLT_DATE``, ``APT_ICAO``, ``FLT_DEP_1``,
                                             ``FLT_ARR_1``, ``FLT_TOT_1``
``all_pre_departure_delays_<y>`` daily       ``DLY_ALL_PRE_2``, ``FLT_DEP_IFR_2``
``atc_pre_departure_delays_<y>`` daily       ``DLY_ATC_PRE_2``, ``FLT_DEP_IFR_2``
``atfm_slot_adherence_<y>.csv``  daily       ``FLT_DEP_1``, ``FLT_DEP_REG_1``,
                                             ``FLT_DEP_OUT_EARLY_1``, ``FLT_DEP_IN_1``,
                                             ``FLT_DEP_OUT_LATE_1``
``asma_additional_time_<y>.csv`` monthly     ``TOTAL_ADD_TIME_MIN``, ``TOTAL_REF_NB_FL``
===============================  ==========  ==================================================

``taxi_out_additional_time_<y>.csv`` is deliberately **not** used here: it is the published
aggregate of the quantity we are predicting, and it belongs to the P1-g post-processor.

**Coverage of the 2026-09-18 pull** (byte-identical to the 2026-09-04 one — the site has not
republished). The daily files run 2026-01-01 → 2026-07-31, so both scored months have rows,
but three of the ten columns are not consolidated yet for July 2026:

* ``ec_flt_*`` and ``ec_atfm_*`` — 0 % null in 2026-01 and 2026-07. Safe.
* ``ec_dly_all_pre_per_dep`` — 0 % null in 2026-01; in 2026-07 only EGLL and EHAM report,
  the other eight airports are null (78 % of the July test rows).
* ``ec_dly_atc_pre_per_dep`` — structurally absent at EGLL / EHAM / LFPG / LTFM in every
  month (55 % null in training) and absent at *all* airports in 2026-07.
* ``ec_asma_add_min_per_arr`` — monthly file stops at 2026-06, so 100 % null in 2026-07.

The last three are therefore expected to fail the plan's feature-drift check on July. They
are kept deliberately: the ablation is what reveals whether the block is worth anything on
January, and the PRU publishes ~2 months in arrears, so a re-fetch before the 2026-10-11
deadline should fill July. If the drift check fails, drop those three columns, not the block.

Returns all-null columns when the archive is absent, so tests and fresh clones still run.

**2026-09-19.** The drift check anticipated above has now been run: the three columns did fail it, and a classifier
given this block alone separates 2025 from the 2026 ranking rows with **AUC 1.00** purely on
their null pattern. ``eurocontrol_daily_stable`` (bottom of this module) is the same block
with those three columns dropped; it is what ``exp0093`` and later configs use. Keep the full
block for the pre-deadline re-fetch: if the PRU has published July 2026 by then, the full
block becomes usable again and the audit script will say so.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from prc import paths
from prc.data import schema as S
from prc.features.registry import archive_files, external_digest, feature

OUT = [
    "ec_flt_dep",
    "ec_flt_arr",
    "ec_flt_tot",
    "ec_flt_tot_rel",
    "ec_dly_all_pre_per_dep",
    "ec_dly_atc_pre_per_dep",
    "ec_atfm_reg_share",
    "ec_atfm_slot_late_share",
    "ec_atfm_slot_early_share",
    "ec_asma_add_min_per_arr",
]


def _read(d: Path, dataset: str, cols: list[str]) -> pl.DataFrame | None:
    files = sorted(d.glob(f"{dataset}_*.csv"))
    if not files:
        return None
    frames = [
        pl.read_csv(
            f,
            columns=cols,
            schema_overrides={c: pl.Float64 for c in cols if c not in ("FLT_DATE", "APT_ICAO")},
            ignore_errors=True,
        ).filter(pl.col("APT_ICAO").is_in(list(S.AIRPORTS)))
        for f in files
    ]
    return pl.concat(frames) if frames else None


def _daily(d: Path) -> pl.DataFrame | None:
    """One row per (airport, date) with the derived airport-day indicators."""
    traffic = _read(
        d, "airport_traffic", ["FLT_DATE", "APT_ICAO", "FLT_DEP_1", "FLT_ARR_1", "FLT_TOT_1"]
    )
    allpre = _read(
        d, "all_pre_departure_delays", ["FLT_DATE", "APT_ICAO", "FLT_DEP_IFR_2", "DLY_ALL_PRE_2"]
    )
    atcpre = _read(
        d, "atc_pre_departure_delays", ["FLT_DATE", "APT_ICAO", "FLT_DEP_IFR_2", "DLY_ATC_PRE_2"]
    )
    atfm = _read(
        d,
        "atfm_slot_adherence",
        [
            "FLT_DATE",
            "APT_ICAO",
            "FLT_DEP_1",
            "FLT_DEP_REG_1",
            "FLT_DEP_OUT_EARLY_1",
            "FLT_DEP_IN_1",
            "FLT_DEP_OUT_LATE_1",
        ],
    )
    if traffic is None:
        return None

    def key(f: pl.DataFrame) -> pl.DataFrame:
        return f.with_columns(
            pl.col("APT_ICAO").alias(S.AIRPORT),
            pl.col("FLT_DATE").str.to_date("%Y-%m-%d").alias("d"),
        ).drop("APT_ICAO", "FLT_DATE")

    out = key(traffic).select(
        S.AIRPORT,
        "d",
        pl.col("FLT_DEP_1").cast(pl.Float32).alias("ec_flt_dep"),
        pl.col("FLT_ARR_1").cast(pl.Float32).alias("ec_flt_arr"),
        pl.col("FLT_TOT_1").cast(pl.Float32).alias("ec_flt_tot"),
    )
    out = out.with_columns(
        (pl.col("ec_flt_tot") / pl.col("ec_flt_tot").median().over(S.AIRPORT))
        .cast(pl.Float32)
        .alias("ec_flt_tot_rel")
    )
    if allpre is not None:
        out = out.join(
            key(allpre).select(
                S.AIRPORT,
                "d",
                (pl.col("DLY_ALL_PRE_2") / pl.col("FLT_DEP_IFR_2"))
                .cast(pl.Float32)
                .alias("ec_dly_all_pre_per_dep"),
            ),
            on=[S.AIRPORT, "d"],
            how="left",
        )
    if atcpre is not None:
        out = out.join(
            key(atcpre).select(
                S.AIRPORT,
                "d",
                (pl.col("DLY_ATC_PRE_2") / pl.col("FLT_DEP_IFR_2"))
                .cast(pl.Float32)
                .alias("ec_dly_atc_pre_per_dep"),
            ),
            on=[S.AIRPORT, "d"],
            how="left",
        )
    if atfm is not None:
        reg = pl.col("FLT_DEP_REG_1")
        out = out.join(
            key(atfm).select(
                S.AIRPORT,
                "d",
                (reg / pl.col("FLT_DEP_1").clip(1.0, None))
                .cast(pl.Float32)
                .alias("ec_atfm_reg_share"),
                (pl.col("FLT_DEP_OUT_LATE_1") / reg.clip(1.0, None))
                .cast(pl.Float32)
                .alias("ec_atfm_slot_late_share"),
                (pl.col("FLT_DEP_OUT_EARLY_1") / reg.clip(1.0, None))
                .cast(pl.Float32)
                .alias("ec_atfm_slot_early_share"),
            ),
            on=[S.AIRPORT, "d"],
            how="left",
        )
    return out.unique(subset=[S.AIRPORT, "d"], keep="first")


def _monthly_asma(d: Path) -> pl.DataFrame | None:
    """ASMA additional time per arrival, per (airport, YYYY-MM)."""
    asma = _read(
        d,
        "asma_additional_time",
        ["YEAR", "MONTH_NUM", "APT_ICAO", "TOTAL_ADD_TIME_MIN", "TOTAL_REF_NB_FL"],
    )
    if asma is None:
        return None
    return asma.select(
        pl.col("APT_ICAO").alias(S.AIRPORT),
        (
            pl.col("YEAR").cast(pl.Int32).cast(pl.Utf8)
            + pl.lit("-")
            + pl.col("MONTH_NUM").cast(pl.Int32).cast(pl.Utf8).str.zfill(2)
        ).alias("ym"),
        (pl.col("TOTAL_ADD_TIME_MIN") / pl.col("TOTAL_REF_NB_FL").clip(1.0, None))
        .cast(pl.Float32)
        .alias("ec_asma_add_min_per_arr"),
    ).unique(subset=[S.AIRPORT, "ym"], keep="first")


def _ec_external() -> str:
    """Hash of the EUROCONTROL CSVs this block actually reads (see the registry docstring)."""
    files = [
        f
        for f in archive_files("eurocontrol_perf_*", "*.csv")
        if f.name.startswith(
            (
                "airport_traffic",
                "all_pre_departure_delays",
                "atc_pre_departure_delays",
                "atfm_slot_adherence",
                "asma_additional_time",
            )
        )
    ]
    return external_digest(files, __name__)


@feature("eurocontrol_daily", external=_ec_external)
def eurocontrol_daily(df: pl.DataFrame) -> pl.DataFrame:
    """Airport-day traffic, pre-departure delay per departure, ATFM-regulated and
    slot-adherence shares, plus monthly ASMA additional time per arrival."""
    dirs = sorted(paths.EXTERNAL.glob("eurocontrol_perf_*"))
    daily = _daily(dirs[-1]) if dirs else None
    base = df.select(
        S.MVT_ID,
        S.AIRPORT,
        pl.col(S.MVT_TIME).dt.date().alias("d"),
        pl.col(S.MVT_TIME).dt.strftime("%Y-%m").alias("ym"),
    )
    if daily is None:
        return base.select(S.MVT_ID).with_columns(
            [pl.lit(None, dtype=pl.Float32).alias(c) for c in OUT]
        )
    out = base.join(daily, on=[S.AIRPORT, "d"], how="left")
    asma = _monthly_asma(dirs[-1])
    if asma is not None:
        out = out.join(asma, on=[S.AIRPORT, "ym"], how="left")
    missing = [c for c in OUT if c not in out.columns]
    return out.select(
        S.MVT_ID,
        *[pl.col(c).cast(pl.Float32) for c in OUT if c not in missing],
        *[pl.lit(None, dtype=pl.Float32).alias(c) for c in missing],
    )


#: The seven columns whose null rate in Jan 2026 and Jul 2026 matches the same month of 2025
#: The other three are dropped by
#: ``eurocontrol_daily_stable``: they are null for most or all of July 2026 because the PRU
#: publishes in arrears, which makes them a perfect "is this a 2026 row?" detector
#: (adversarial AUC 1.00 for the full block) rather than a description of the airport-day.
STABLE = [
    "ec_flt_dep",
    "ec_flt_arr",
    "ec_flt_tot",
    "ec_flt_tot_rel",
    "ec_atfm_reg_share",
    "ec_atfm_slot_late_share",
    "ec_atfm_slot_early_share",
]
DRIFTING = [c for c in OUT if c not in STABLE]


@feature("eurocontrol_daily_stable", external=_ec_external)
def eurocontrol_daily_stable(df: pl.DataFrame) -> pl.DataFrame:
    """``eurocontrol_daily`` minus the three columns that are missing for most of July 2026
    (pre-departure delay per departure, ATC pre-departure delay per departure, ASMA additional
    time per arrival): airport-day traffic, its level relative to the airport's median day, and
    the ATFM-regulated / slot-adherence shares."""
    out = eurocontrol_daily(df)
    return out.select(S.MVT_ID, *STABLE)
