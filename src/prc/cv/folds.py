"""Temporal cross-validation schemes.

Test = January + July 2026, trained on all of 2025. Validation must therefore
mimic *month-level extrapolation with seasonality*. Fold assignment is a function
of the ``month`` column only, so every teammate gets byte-identical folds and OOF
predictions are directly blendable.

A scheme is a list of folds; each fold names its validation months and, optionally,
an explicit list of training months (default: every 2025 month not in validation).
The fold table has one row per train DEP MVT_ID with ``fold`` = validation fold index
(-1 = never validated). Frozen to ``data/folds/<scheme>.parquet`` on first use.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from prc import paths
from prc.data import schema as S

MONTHS_2025 = [f"2025-{m:02d}" for m in range(1, 13)]


@dataclass(frozen=True)
class Fold:
    val: list[str]
    train: list[str] | None = None  # None = all months not in val

    def train_months(self) -> list[str]:
        return (
            self.train if self.train is not None else [m for m in MONTHS_2025 if m not in self.val]
        )


@dataclass(frozen=True)
class Scheme:
    name: str
    folds: list[Fold]
    doc: str = ""
    tags: list[str] = field(default_factory=list)


def _m(*ms: int) -> list[str]:
    return [f"2025-{m:02d}" for m in ms]


SCHEMES: dict[str, Scheme] = {
    s.name: s
    for s in [
        Scheme(
            "month_jan_jul",
            [Fold(_m(1)), Fold(_m(7))],
            "PRIMARY gate. Validate on Jan 2025 and Jul 2025, train on the other 11 months each time. "
            "Mirrors the test's seasonal composition.",
        ),
        Scheme(
            "month_q4",
            [Fold(_m(1)), Fold(_m(4)), Fold(_m(7)), Fold(_m(10))],
            "Four seasonal folds; noisier-but-broader sanity check.",
        ),
        Scheme(
            "month_loo",
            [Fold([m]) for m in MONTHS_2025],
            "Leave-one-month-out, 12 folds. Pre-final only.",
        ),
        Scheme(
            "tail_nov_dec",
            [Fold(_m(11, 12), train=_m(*range(1, 11)))],
            "Forward chaining: train Jan-Oct, validate Nov+Dec (pure extrapolation).",
        ),
        # The gap ladder: how fast does RMSE degrade with distance from the training window?
        Scheme(
            "gap0_jul", [Fold(_m(7), train=_m(*range(1, 7)))], "Train Jan-Jun -> Jul (0-month gap)."
        ),
        Scheme(
            "gap1_jul", [Fold(_m(7), train=_m(*range(1, 6)))], "Train Jan-May -> Jul (1-month gap)."
        ),
        Scheme("gap3_jul", [Fold(_m(7), train=_m(1, 2, 3))], "Train Jan-Mar -> Jul (3-month gap)."),
        Scheme("gap5_jul", [Fold(_m(7), train=_m(1, 2))], "Train Jan-Feb -> Jul (5-month gap)."),
    ]
}


def assign_folds(df: pl.DataFrame, scheme: str) -> pl.DataFrame:
    """One row per train DEP: MVT_ID, fold (validation fold index or -1)."""
    sch = _get(scheme)
    month_to_fold: dict[str, int] = {}
    for k, f in enumerate(sch.folds):
        for m in f.val:
            if m in month_to_fold:
                raise ValueError(f"scheme {scheme}: month {m} validated in two folds")
            month_to_fold[m] = k
    deps = df.filter((pl.col(S.PHASE) == "DEP") & (~pl.col(S.IS_TEST)))
    return deps.select(
        S.MVT_ID,
        pl.col(S.SPLIT_MONTH)
        .replace_strict(month_to_fold, default=-1, return_dtype=pl.Int8)
        .alias("fold"),
    )


def train_mask(months: pl.Series, scheme: str, k: int) -> pl.Series:
    """Boolean mask over rows: is this row in fold k's training set?"""
    return months.is_in(_get(scheme).folds[k].train_months())


def get_folds(df: pl.DataFrame, scheme: str, freeze: bool = True) -> pl.DataFrame:
    path = paths.FOLDS / f"{scheme}.parquet"
    if path.exists():
        return pl.read_parquet(path)
    folds = assign_folds(df, scheme)
    if freeze:
        paths.FOLDS.mkdir(parents=True, exist_ok=True)
        folds.write_parquet(path)
    return folds


def n_folds(scheme: str) -> int:
    return len(_get(scheme).folds)


def _get(scheme: str) -> Scheme:
    if scheme not in SCHEMES:
        raise KeyError(f"unknown cv scheme {scheme!r}; known: {sorted(SCHEMES)}")
    return SCHEMES[scheme]
