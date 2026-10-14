"""Physical taxi-out baseline: unimpeded taxi time + expected queue wait (backlog X1).

A departure's taxi-out is, physically, the time to roll the stand → runway path at an
unimpeded speed plus the time spent waiting behind the departures ahead of it. The members
learn both implicitly (stand × runway categories, queue counts). This block writes the
physical reading down so that a rare stand–runway pair, or an airport without ADS-B, gets it
from geometry and counts instead of from a thin category:

* ``phys_service_s``     runway service time = 3600 / (departures taking off on the same
  airport + runway within ±30 min of this takeoff, this one included). Label-free (takeoffs).
* ``phys_unimpeded_s``   ``a_ap + td_path_m / v_ap``: per airport, a straight line through the
  25th percentile of the NM proxy ``P = MVT − AOBT_3`` in ten path-length deciles, fitted on
  2025 (training-month) regime-A departures with ``60 ≤ P ≤ 3600`` s and a light queue at
  push (``sq_taxiing_at_push`` ≤ the airport's 20th percentile). ``a_ap`` absorbs push-back and
  engine start, ``v_ap`` is the typical unimpeded taxi speed. The fit reads ``AOBT_3`` and
  ``MVT`` of *other* rows (the NM proxy, visible on test rows too), never a label.
* ``phys_expected_s``    a deterministic single-server queue: the n departures already
  taxiing to the same runway at push (``sq_taxiing_at_push_rwy``) leave one per service time,
  so the runway is free for this one after (n + 1) × ``phys_service_s``; it cannot be there
  before its unimpeded time. Expected taxi = max(unimpeded, (n + 1) × service).
* ``phys_queue_wait_s``  ``phys_expected_s − phys_unimpeded_s`` (≥ 0): the queueing part.
* ``phys_gap_s``         ``P − phys_expected_s``: how far the NM proxy sits from the physics.
* ``phys_ratio``         ``P / phys_expected_s``.
* ``phys_expected_tm10_s`` the same with n = aircraft on the surface 10 min before takeoff
  (``sq_taxiing_at_tm10``, all runways): a version that needs no own ``AOBT_3`` (no-NM rows).

(Adding n × service to the unimpeded time instead double-counts: by Little's law the aircraft
on the surface at push are mostly still rolling, and n × service alone is already about the
median NM proxy. Changed to the max form before the first screen run.)

Dependencies (``taxi_distance``, ``surface_queue``, ``regime``) are taken through the registry
(``build``, cache-backed); ``external=`` folds their hashes in, so a change to any of them
changes this block's cache key. Departure rows only.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from prc.data import schema as S
from prc.features.registry import build, feature
from prc.features.screen_common import deps_digest, safe_div, secs

DEPS = ("taxi_distance", "surface_queue", "regime")
DEP_COLS = (
    "td_path_m",
    "sq_taxiing_at_push",
    "sq_taxiing_at_push_rwy",
    "sq_taxiing_at_tm10",
    "regime",
)
OUT = [
    "phys_service_s",
    "phys_unimpeded_s",
    "phys_queue_wait_s",
    "phys_expected_s",
    "phys_gap_s",
    "phys_ratio",
    "phys_expected_tm10_s",
]
#: half-width of the runway-throughput window, seconds
SERVICE_HALF_S = 1800
#: quantile of P per path decile that stands for "unimpeded"
UNIMPEDED_Q = 0.25
#: light-queue filter: sq_taxiing_at_push at or below this airport quantile
LIGHT_QUEUE_Q = 0.20
N_BINS = 10
MIN_FIT_ROWS = 200


def _phys_external() -> str:
    return deps_digest(__name__, DEPS)


def service_time(dep: pl.DataFrame) -> np.ndarray:
    """3600 / (takeoffs on the same airport + runway within ±30 min, self included), per row of
    ``dep`` (in ``dep``'s order). Null runway → its own group (all unknown-runway takeoffs)."""
    t = dep[S.MVT_TIME].dt.epoch("s").to_numpy()
    key = (
        dep[S.AIRPORT].cast(pl.Utf8) + "|" + dep[S.RUNWAY].cast(pl.Utf8).fill_null("NA")
    ).to_numpy()
    out = np.full(len(t), np.nan)
    for k in np.unique(key):
        idx = np.flatnonzero(key == k)
        tk = t[idx]
        s = np.sort(tk)
        n = np.searchsorted(s, tk + SERVICE_HALF_S, side="right") - np.searchsorted(
            s, tk - SERVICE_HALF_S, side="left"
        )
        out[idx] = 3600.0 / np.maximum(n, 1)
    return out


def fit_unimpeded(fit: pl.DataFrame) -> dict[str, tuple[float, float]]:
    """Per airport ``(a_s, b_s_per_m)`` of ``P_q25 ≈ a + b · path`` over path deciles.

    ``fit`` holds ``airport``, ``td_path_m``, ``P`` of the qualifying rows. A non-positive
    slope or too few rows falls back to ``a = 0`` and ``b = median(P_q25 / path)``."""
    params: dict[str, tuple[float, float]] = {}
    for ap in fit[S.AIRPORT].unique().sort().to_list():
        x = fit.filter(pl.col(S.AIRPORT) == ap)
        if x.height < MIN_FIT_ROWS:
            continue
        path = x["td_path_m"].to_numpy().astype(float)
        p = x["P"].to_numpy().astype(float)
        edges = np.unique(np.quantile(path, np.linspace(0, 1, N_BINS + 1)))
        b_idx = np.clip(np.searchsorted(edges, path, side="right") - 1, 0, len(edges) - 2)
        xs, ys, ws = [], [], []
        for b in np.unique(b_idx):
            m = b_idx == b
            if m.sum() < 20:
                continue
            xs.append(np.median(path[m]))
            ys.append(np.quantile(p[m], UNIMPEDED_Q))
            ws.append(m.sum())
        xs_a, ys_a, ws_a = np.array(xs), np.array(ys), np.array(ws, dtype=float)
        slope, icpt = (np.nan, np.nan)
        if len(xs_a) >= 3 and np.ptp(xs_a) > 0:
            slope, icpt = np.polyfit(xs_a, ys_a, 1, w=np.sqrt(ws_a))
        if not np.isfinite(slope) or slope <= 0 or icpt < 0:
            icpt, slope = 0.0, float(np.median(np.quantile(p, UNIMPEDED_Q) / np.maximum(path, 1)))
        params[ap] = (float(icpt), float(slope))
    return params


def physics_from(df: pl.DataFrame, deps: pl.DataFrame) -> pl.DataFrame:
    """Block output for movement table ``df`` given ``deps`` (``MVT_ID`` + ``DEP_COLS``)."""
    dep = (
        df.filter(pl.col(S.PHASE) == "DEP")
        .select(
            S.MVT_ID,
            S.AIRPORT,
            S.RUNWAY,
            S.MVT_TIME,
            S.IS_TEST,
            secs(S.MVT_TIME, S.AOBT_3).alias("P"),
        )
        .join(deps.select(S.MVT_ID, *DEP_COLS), on=S.MVT_ID, how="left")
    )
    dep = dep.with_columns(pl.Series("phys_service_s", service_time(dep), dtype=pl.Float64))
    light = (
        dep.filter(~pl.col(S.IS_TEST))
        .group_by(S.AIRPORT)
        .agg(pl.col("sq_taxiing_at_push").quantile(LIGHT_QUEUE_Q).alias("_q_light"))
    )
    fit = (
        dep.join(light, on=S.AIRPORT, how="left")
        .filter(
            ~pl.col(S.IS_TEST),
            pl.col("regime").cast(pl.Utf8) == "A",
            pl.col("P").is_between(60, 3600),
            pl.col("td_path_m").is_not_null(),
            pl.col("sq_taxiing_at_push") <= pl.col("_q_light"),
        )
        .select(S.AIRPORT, "td_path_m", "P")
    )
    params = fit_unimpeded(fit)
    ptab = pl.DataFrame(
        {
            S.AIRPORT: list(params),
            "_a": [v[0] for v in params.values()],
            "_b": [v[1] for v in params.values()],
        },
        schema={S.AIRPORT: pl.Utf8, "_a": pl.Float64, "_b": pl.Float64},
    )
    dep = dep.join(ptab, on=S.AIRPORT, how="left")
    unimp = pl.col("_a") + pl.col("td_path_m") * pl.col("_b")
    dep = dep.with_columns(unimp.alias("phys_unimpeded_s"))
    u, svc = pl.col("phys_unimpeded_s"), pl.col("phys_service_s")

    def queue(n: str) -> pl.Expr:
        # null when either input is null (max_horizontal would silently drop a null)
        runway_free = (pl.col(n) + 1.0) * svc
        return (
            pl.when(u.is_null() | runway_free.is_null())
            .then(None)
            .otherwise(pl.max_horizontal(u, runway_free))
        )

    dep = dep.with_columns(
        queue("sq_taxiing_at_push_rwy").alias("phys_expected_s"),
        queue("sq_taxiing_at_tm10").alias("phys_expected_tm10_s"),
    ).with_columns(
        (pl.col("phys_expected_s") - u).alias("phys_queue_wait_s"),
        (pl.col("P") - pl.col("phys_expected_s")).alias("phys_gap_s"),
        safe_div(pl.col("P"), pl.col("phys_expected_s")).alias("phys_ratio"),
    )
    return dep.select(S.MVT_ID, *[pl.col(c).cast(pl.Float32) for c in OUT])


@feature("taxi_physics", external=_phys_external)
def taxi_physics(df: pl.DataFrame) -> pl.DataFrame:
    """Departure rows only. Runway service time, unimpeded taxi (OSM path ÷ per-airport speed
    fitted on the NM proxy of 2025 light-queue rows), expected taxi of a deterministic queue
    max(unimpeded, (departures ahead + 1) × service) and its queueing part, the NM proxy's gap /
    ratio to it; a takeoff-anchored variant."""
    return physics_from(df, build(df, list(DEPS)))
