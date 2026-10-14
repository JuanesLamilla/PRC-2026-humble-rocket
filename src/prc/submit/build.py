"""Build, validate and log a submission file.

Rules (ranking.html): use ``submitting.parquet`` as the template, fill
``TAXITIME_SEC_mvt``, keep exactly the template's MVT_ID set, name the file
``<team-name>_v<int>.parquet``. Every file written here is logged to
``submissions/log.csv`` with its CV RMSE so leaderboard RMSE can be compared later.
"""

from __future__ import annotations

import csv
import os
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from prc import paths
from prc.data import schema as S
from prc.data.load import load_submitting_template

LOG_COLS = ["created_at", "file", "version", "source", "cv_rmse", "lb_rmse", "note"]


def next_version(team: str) -> int:
    existing = sorted(paths.SUBMISSIONS.glob(f"{team}_v*.parquet"))
    nums = []
    for p in existing:
        try:
            nums.append(int(p.stem.rsplit("_v", 1)[1]))
        except ValueError:
            pass
    return (max(nums) + 1) if nums else 1


def build_submission(
    preds: pl.DataFrame,
    *,
    source: str,
    cv_rmse: float | None,
    note: str = "",
    team: str | None = None,
) -> Path:
    team = team or os.environ.get("PRC_TEAM_NAME")
    if not team:
        raise ValueError("set PRC_TEAM_NAME (env) or pass team=")
    template = load_submitting_template()
    tmpl = template.select(S.MVT_ID)
    p = preds.select(S.MVT_ID, pl.col("pred").cast(pl.Float64).alias(S.TARGET))
    # keep the template's row order and its exact dtype for the target (int32 in the
    # organizers' file: whole seconds; rounding costs < 0.1 s of RMSE)
    sub = tmpl.join(p, on=S.MVT_ID, how="left", maintain_order="left")
    target_dtype = template.schema[S.TARGET]
    if target_dtype.is_integer():
        sub = sub.with_columns(pl.col(S.TARGET).round(0).cast(target_dtype))
    validate(sub, tmpl)
    paths.SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    v = next_version(team)
    path = paths.SUBMISSIONS / f"{team}_v{v}.parquet"
    sub.write_parquet(path)
    _log(path, v, source, cv_rmse, note)
    return path


def validate(sub: pl.DataFrame, tmpl: pl.DataFrame) -> None:
    if sub.columns != [S.MVT_ID, S.TARGET]:
        raise ValueError(f"columns must be exactly [{S.MVT_ID}, {S.TARGET}], got {sub.columns}")
    if sub[S.MVT_ID].n_unique() != sub.height:
        raise ValueError("duplicate MVT_ID rows")
    if sub.height != tmpl.height:
        raise ValueError(f"row count {sub.height} != template {tmpl.height}")
    if sub[S.TARGET].null_count():
        raise ValueError(f"{sub[S.TARGET].null_count()} null predictions")
    if not sub[S.TARGET].is_finite().all():
        raise ValueError("non-finite predictions")
    if (sub[S.TARGET] < 0).any():
        raise ValueError("negative taxi times")
    if set(sub[S.MVT_ID].to_list()) != set(tmpl[S.MVT_ID].to_list()):
        raise ValueError("MVT_ID set differs from template")


def _log(path: Path, v: int, source: str, cv_rmse: float | None, note: str) -> None:
    logp = paths.SUBMISSIONS / "log.csv"
    new = not logp.exists()
    with open(logp, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLS)
        if new:
            w.writeheader()
        w.writerow(
            {
                "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "file": path.name,
                "version": v,
                "source": source,
                "cv_rmse": "" if cv_rmse is None else round(cv_rmse, 4),
                "lb_rmse": "",
                "note": note,
            }
        )
