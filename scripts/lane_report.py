"""Lane breakdown of a run's OOF predictions (validation-protocol ADR 2026-09-17).

Lanes: with / without NM off-block (``AOBT_3_flt``), the derived-AOBT spike
(``MVT − AOBT_3`` in [1194, 1206] s), target > 1 h, and "ordinary" (has AOBT, not on the
spike, ≤ 1 h). Reports rows, RMSE and share of the run's squared error per lane and per
month, so a gain can be located before it is believed.

``--reweight`` adds the composition-matched number the ADR asks for: the same per-cell
squared errors, re-averaged under the *ranking set's* lane composition instead of the gate
fold's. Cells are (airport, month, has_aobt, on_spike) — all observable on the unlabelled
2026 rows. The ``> 1 h`` share is **not** observable without labels, so it is not part of
the reweighting; a cell keeps its gate-fold mix of ordinary and catastrophic rows.

    uv run python scripts/lane_report.py exp0001_lgbm_baseline [exp0003_nm_naive ...]
    uv run python scripts/lane_report.py exp0001_lgbm_baseline --write report.md
    uv run python scripts/lane_report.py exp0018_xgb_cap_d12_m1000 --reweight
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from prc import paths
from prc.eval.lanes import SPIKE, lanes  # noqa: F401  (SPIKE re-exported)
from prc.eval.lanes import tag as _tag

# Gate fold month -> the ranking month whose composition it has to stand in for.
GATE_TO_RANK = {"2025-01": "2026-01", "2025-07": "2026-07"}


def ranking_composition() -> pl.DataFrame:
    """Rows per (month, airport, has_aobt, on_spike) cell on the ranking set (2026 DEP rows).

    These three lane keys are observable without labels, which is the whole point: the
    weights come from the scored set itself, not from a model of it."""
    # a lazy scan with the filter pushed down: only the ranking rows are ever materialised
    # (2026-09-23; the same rows and counts as the eager read it replaces)
    mv = (
        pl.scan_parquet(paths.INTERIM / "movements.parquet")
        .select(
            "MVT_ID_mvt",
            "PHASE_mvt",
            "is_test",
            "month",
            "airport",
            "MVT_TIME_UTC_mvt",
            "AOBT_3_flt",
        )
        .filter((pl.col("PHASE_mvt") == "DEP") & pl.col("is_test"))
        .collect()
    )
    return (
        _tag(mv)
        .group_by("month", "airport", "has_aobt", "on_spike")
        .len()
        .rename({"len": "n_rank"})
    )


def gate_cells(run: str) -> pl.DataFrame:
    """Rows and squared error per (month, airport, has_aobt, on_spike) cell on the gate fold."""
    oof = pl.read_parquet(paths.RUNS / run / "oof.parquet").select(
        "MVT_ID_mvt", "month", "airport", "TAXITIME_SEC_mvt", "pred"
    )
    mv = pl.read_parquet(
        paths.INTERIM / "movements.parquet",
        columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt"],
    )
    d = _tag(oof.join(mv, on="MVT_ID_mvt", how="left"))
    return (
        d.with_columns(
            ((pl.col("pred") - pl.col("TAXITIME_SEC_mvt").cast(pl.Float64)) ** 2).alias("se")
        )
        .group_by("month", "airport", "has_aobt", "on_spike")
        .agg(pl.len().alias("n_gate"), pl.col("se").sum().alias("sse"))
        .with_columns((pl.col("sse") / pl.col("n_gate")).alias("mse"))
    )


def reweighted(run: str, comp: pl.DataFrame) -> pl.DataFrame:
    """Pooled gate-fold RMSE beside the RMSE the same per-cell errors give under the
    ranking set's lane composition. Unmatched ranking cells (no gate-fold rows) are dropped
    and their mass is reported as ``rank_cov``."""
    g = gate_cells(run).with_columns(
        pl.col("month").replace_strict(GATE_TO_RANK, default=None).alias("rank_month")
    )
    j = g.join(
        comp,
        left_on=["rank_month", "airport", "has_aobt", "on_spike"],
        right_on=["month", "airport", "has_aobt", "on_spike"],
        how="left",
    ).with_columns(pl.col("n_rank").fill_null(0))
    rows = []
    for scope in ["all", *sorted(GATE_TO_RANK)]:
        sub = j if scope == "all" else j.filter(pl.col("month") == scope)
        rank_total = (
            comp["n_rank"].sum()
            if scope == "all"
            else comp.filter(pl.col("month") == GATE_TO_RANK[scope])["n_rank"].sum()
        )
        n_gate = int(sub["n_gate"].sum())
        w = sub["n_rank"].cast(pl.Float64).to_numpy()
        mse = sub["mse"].to_numpy()
        pooled = float(np.sqrt(sub["sse"].sum() / n_gate))
        rw = float(np.sqrt(np.sum(w * mse) / w.sum())) if w.sum() > 0 else float("nan")
        rows.append(
            {
                "run": run,
                "scope": scope,
                "rank_scope": "2026-01 + 2026-07" if scope == "all" else GATE_TO_RANK[scope],
                "gate_rows": n_gate,
                "rank_rows": int(rank_total),
                "cells": int(sub.height),
                "rank_cov": round(float(w.sum()) / float(rank_total), 4),
                "pooled_rmse": round(pooled, 1),
                "reweighted_rmse": round(rw, 1),
                "delta": round(rw - pooled, 1),
            }
        )
    return pl.DataFrame(rows)


def composition(comp: pl.DataFrame, run: str) -> pl.DataFrame:
    """Gate-fold vs ranking-set share of each observable lane, per month — the reason the
    reweighted number differs from the pooled one."""
    lane = (
        pl.when(~pl.col("has_aobt"))
        .then(pl.lit("no_aobt"))
        .when(pl.col("on_spike"))
        .then(pl.lit("spike"))
        .otherwise(pl.lit("with_aobt_nospike"))
        .alias("lane")
    )
    g = (
        gate_cells(run)
        .with_columns(lane)
        .group_by("month", "lane")
        .agg(pl.col("n_gate").sum(), pl.col("sse").sum())
        .with_columns(
            (pl.col("n_gate") / pl.col("n_gate").sum().over("month")).round(4).alias("gate_share"),
            (pl.col("sse") / pl.col("n_gate")).sqrt().round(1).alias("lane_rmse"),
        )
        .with_columns(pl.col("month").replace_strict(GATE_TO_RANK, default=None).alias("rm"))
    )
    r = (
        comp.with_columns(lane)
        .group_by("month", "lane")
        .agg(pl.col("n_rank").sum())
        .with_columns(
            (pl.col("n_rank") / pl.col("n_rank").sum().over("month")).round(4).alias("rank_share")
        )
    )
    return (
        g.join(r, left_on=["rm", "lane"], right_on=["month", "lane"], how="left")
        .select("month", "lane", "lane_rmse", "n_gate", "gate_share", "n_rank", "rank_share")
        .with_columns((pl.col("rank_share") - pl.col("gate_share")).round(4).alias("share_delta"))
        .sort("month", "lane")
    )


def _md_table(df: pl.DataFrame) -> list[str]:
    cols = df.columns
    return [
        "| " + " | ".join(cols) + " |",
        "|" + "---|" * len(cols),
        *["| " + " | ".join(str(v) for v in r) + " |" for r in df.iter_rows()],
    ]


REWEIGHT_NOTE = (
    "Composition-matched RMSE (`--reweight`). `pooled_rmse` is the gate fold's own average; "
    "`reweighted_rmse` re-averages the **same per-cell squared errors** under the ranking "
    "set's cell shares. Cells are (airport, month, has_aobt, on_spike) — the lane keys that "
    "are observable on the unlabelled 2026 rows. The `> 1 h` share is **not** observable "
    "without labels and is therefore *not* reweighted: each cell keeps its gate-fold mix of "
    "ordinary and catastrophic rows, so this corrects lane *composition*, not the tail rate "
    "inside a lane. `rank_cov` = share of ranking rows in cells that exist on the gate fold."
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--write", type=Path, help="also write a markdown file")
    ap.add_argument(
        "--reweight",
        action="store_true",
        help="also report the RMSE under the ranking set's lane composition",
    )
    a = ap.parse_args()
    out = pl.concat([lanes(r) for r in a.runs])
    with pl.Config(tbl_rows=200, tbl_cols=10):
        print(out)
    rw = cmp_tbl = None
    if a.reweight:
        comp = ranking_composition()
        rw = pl.concat([reweighted(r, comp) for r in a.runs])
        cmp_tbl = composition(comp, a.runs[0])
        print()
        with pl.Config(tbl_rows=200, tbl_cols=12):
            print(rw)
            print(cmp_tbl)
    if a.write:
        lines = [
            "# Lane report — " + ", ".join(a.runs),
            "",
            (
                "Generated by `scripts/lane_report.py`. `sse_share` = share of the run's total "
                "squared error on the gate fold. Lanes per ADR 2026-09-17-validation-protocol."
            ),
            "",
            *_md_table(out),
        ]
        if rw is not None:
            lines += ["", "## Lane-reweighted (composition-matched) RMSE", "", REWEIGHT_NOTE, ""]
            lines += _md_table(rw)
        if cmp_tbl is not None:
            lines += [
                "",
                "### Why they differ — observable lane composition, gate fold vs ranking set",
                "",
                f"`lane_rmse` is `{a.runs[0]}`'s gate-fold RMSE on that lane.",
                "",
            ]
            lines += _md_table(cmp_tbl)
        a.write.write_text("\n".join(lines) + "\n")
        print("wrote", a.write)


if __name__ == "__main__":
    main()
