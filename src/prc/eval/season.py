"""Season-targeted training weights for the normal regressor (lever 1, 2026-09-23).

The test set covers January and July 2026 only, and every earlier decision optimised the
12-month average. ``season_weight: {tau_months, floor}`` multiplies the normal regressor's
row weights (``w_fit``) by a similarity between the row's calendar month and the **target**
month of the fit::

    w_row *= max(floor, exp(-d / tau_months)),   d = circular month distance (0..6)

The target of a fit is the calendar month(s) it predicts: a fold's validation months (one
under ``month_loo`` / ``month_jan_jul``; November and December together under
``tail_nov_dec``, where ``d`` is the distance to the nearer of the two), and under
``test_fit: full`` one full fit per test month, each predicting only its own month's rows
(``prc.eval.run._run_full_fit``). The corruption head, its nested weights and the target
encodings are not weighted. Pure functions; no I/O.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def month_of_year(months: Iterable[str] | np.ndarray) -> np.ndarray:
    """``"YYYY-MM"`` strings -> calendar month 1..12 (int array)."""
    arr = np.asarray(list(months) if not isinstance(months, np.ndarray) else months, dtype=str)
    return np.array([int(m[5:7]) for m in arr], dtype=np.int64) if arr.size else np.zeros(0, int)


def circular_distance(a: np.ndarray, b: int) -> np.ndarray:
    """Distance in months on the 12-month circle (December to January is 1)."""
    d = np.abs(np.asarray(a, dtype=np.int64) - int(b)) % 12
    return np.minimum(d, 12 - d)


def season_weights(
    row_months: Iterable[str] | np.ndarray,
    target_months: Iterable[str] | Iterable[int],
    tau_months: float,
    floor: float,
) -> np.ndarray:
    """``max(floor, exp(-d / tau))`` per row, ``d`` the distance to the nearest target month.

    ``target_months`` are ``"YYYY-MM"`` strings or calendar months 1..12.
    """
    moy = month_of_year(row_months)
    targets = [int(t) if not isinstance(t, str) else int(t[5:7]) for t in target_months]
    if not targets:
        raise ValueError("season_weights needs at least one target month")
    d = np.min(np.stack([circular_distance(moy, t) for t in targets]), axis=0)
    return np.maximum(float(floor), np.exp(-d / float(tau_months)))


def combine(w: np.ndarray | None, season: np.ndarray) -> np.ndarray:
    """The existing weights (``None`` = all ones) times the seasonal factor."""
    return season.copy() if w is None else np.asarray(w, dtype=np.float64) * season
