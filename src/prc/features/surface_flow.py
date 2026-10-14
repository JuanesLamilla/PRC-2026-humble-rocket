"""Surface flow by route (literature update 2026-09-22, item 1).

``surface_queue`` counts traffic airport-wide and on the own runway. Lim et al. 2025 (GNN
taxi-out) and Yuan et al. 2024 get their gain over a GBM from counts of other movements on
the departure's *own route*: flows split by stand group and runway. This block splits the
same interval counts by **route = (stand group, runway)**.

**Stand group** is read from the stand identifier (``STAND_mvt``), per airport:

* a stand starting with letters → the letter prefix (``A14`` → ``A``, ``V108`` → ``V``,
  ``GA6`` → ``GA``): the pier / apron letter at EDDF, EHAM, LFPG, LSZH, LTFM, LEMD;
* a stand of three or more digits → ``#`` + its first digit (``511`` → ``#5``): the
  terminal / apron hundreds at EDDM, EGLL, LEBL, LIRF, LEMD;
* one or two digits → ``d`` + its first digit (LEMD / LEBL remote stands ``70``-``74``,
  ``95``, ``96``);
* anything else (null, ``UNKNOWN``) → no group.

A raw group is kept when it holds at least ``MIN_GROUP_SHARE`` of the airport's movements
(departures and arrivals) in at least one data island and ranks among the ``MAX_GROUPS − 1``
largest by that share; the rest are merged into ``OTHER``. The per-island maximum keeps a
pier that opened in 2026 (EDDF H / J, 12 % of July 2026 movements, 0.3 % of 2025) out of
``OTHER``. The identifier was chosen over the OSM apron cluster of ``taxi_distance``
(87-99 % of movements, 7-37 clusters) and over clustering the learned ADS-B stand
coordinates (LTFM 0 %, LFPG 86 %) because it covers every row with a stand at every
airport; coverage, group counts and the spatial spread of each group are in
internal note 2026-09-22_surface-flow.

Columns (t0 = own ``AOBT_3``, t1 = own takeoff; per airport and data island; the row itself
excluded; the surface interval and its validity exactly as in ``surface_queue``):

* ``sf_same_route_taxiing_at_push``  #other valid departures of the same (group, runway)
  with AOBT_3 ≤ t0 < MVT.
* ``sf_same_group_taxiing_at_push``  the same, same group, any runway.
* ``sf_share_same_route``            same-route count / airport-wide count taxiing at t0
  (``sq_taxiing_at_push``); null when that is 0.
* ``sf_arr_same_group_during_taxi``  arrivals landing in (t0, t1) whose stand is in the own
  stand group (they taxi in through the same apron).
* ``sf_same_route_taxiing_at_tm20`` / ``sf_same_group_taxiing_at_tm20``  the first two at
  x = t1 − 20 min, for every departure (no own AOBT_3 needed).

Own-anchored columns (``*_at_push``, share, arrivals) are null unless the row's own
interval is valid. Group columns are null when the row has no stand group; route columns
also when its runway is ``NA``. ``OTHER`` is a group like any other. Nothing reads the
target or the departure's ``BLOCK_TIME``.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from prc.data import schema as S
from prc.features.raw_extra import island_expr
from prc.features.registry import feature
from prc.features.surface_queue import (
    MAX_TAXI_S,
    NO_RUNWAY,
    US,
    _on_surface,
    _open_count,
)

#: A raw stand group below this share of the airport's movements is merged into OTHER.
MIN_GROUP_SHARE = 0.01
#: At most this many groups per airport, OTHER included.
MAX_GROUPS = 20
OTHER = "OTHER"
#: Takeoff-anchored look-back, minutes.
TM_MIN = 20

COUNT_COLS: tuple[str, ...] = (
    "sf_same_route_taxiing_at_push",
    "sf_same_group_taxiing_at_push",
    "sf_arr_same_group_during_taxi",
    f"sf_same_route_taxiing_at_tm{TM_MIN}",
    f"sf_same_group_taxiing_at_tm{TM_MIN}",
)
FLOAT_COLS: tuple[str, ...] = ("sf_share_same_route",)


def raw_stand_group() -> pl.Expr:
    """Raw stand group label from ``STAND_mvt`` (null when unparseable)."""
    s = pl.col(S.STAND).cast(pl.String).str.to_uppercase().str.strip_chars()
    alpha = s.str.extract(r"^([A-Z]+)", 1)
    long_num = s.str.extract(r"^(\d)\d\d", 1)
    short_num = s.str.extract(r"^(\d)\d?$", 1)
    return (
        pl.when(s == "UNKNOWN")
        .then(pl.lit(None, dtype=pl.String))
        .when(alpha.is_not_null())
        .then(alpha)
        .when(long_num.is_not_null())
        .then(pl.lit("#") + long_num)
        .when(short_num.is_not_null())
        .then(pl.lit("d") + short_num)
        .otherwise(pl.lit(None, dtype=pl.String))
    )


def stand_groups(df: pl.DataFrame) -> pl.DataFrame:
    """(airport, STAND_mvt) → ``sf_group``: raw groups, rare ones merged into OTHER. A raw
    group is kept when it holds at least ``MIN_GROUP_SHARE`` of the airport's movements
    (both phases) in **some** data island — so a pier that opens in 2026 (EDDF H / J,
    Terminal 3) is its own group — and ranks among the ``MAX_GROUPS − 1`` largest by that
    maximum share (ties by name)."""
    raw = (
        df.select(S.AIRPORT, S.STAND, S.SPLIT_MONTH)
        .with_columns(raw_stand_group().alias("_raw"), island_expr().alias("_island"))
        .filter(pl.col("_raw").is_not_null())
    )
    keys = [S.AIRPORT, "_island"]
    sizes = (
        raw.group_by(*keys, "_raw")
        .len()
        .with_columns((pl.col("len") / pl.col("len").sum().over(keys)).alias("_share"))
        .group_by(S.AIRPORT, "_raw")
        .agg(pl.col("_share").max())
        .sort([S.AIRPORT, "_share", "_raw"], descending=[False, True, False])
        .with_columns(pl.int_range(pl.len()).over(S.AIRPORT).alias("_rank"))
        .with_columns(
            pl.when((pl.col("_share") >= MIN_GROUP_SHARE) & (pl.col("_rank") < MAX_GROUPS - 1))
            .then(pl.col("_raw"))
            .otherwise(pl.lit(OTHER))
            .alias("sf_group")
        )
        .select(S.AIRPORT, "_raw", "sf_group")
    )
    return (
        raw.unique([S.AIRPORT, S.STAND])
        .join(sizes, on=[S.AIRPORT, "_raw"], how="left")
        .select(S.AIRPORT, S.STAND, "sf_group")
    )


def _codes(x: np.ndarray) -> np.ndarray:
    """Integer codes of an object array; -1 for None."""
    out = np.full(x.shape, -1, dtype=np.int64)
    ok = np.array([v is not None for v in x], dtype=bool)
    if ok.any():
        _, inv = np.unique(x[ok].astype(str), return_inverse=True)
        out[ok] = inv
    return out


def _group(dep: pl.DataFrame, arr: pl.DataFrame) -> dict[str, np.ndarray]:
    """All columns for the departures of one (airport, island)."""
    n = dep.height
    t = dep.get_column("_t").to_numpy()
    a = dep.get_column("_a").to_numpy()
    valid = dep.get_column("_valid").to_numpy()
    rwy = dep.get_column(S.RUNWAY).to_numpy()
    grp_d = dep.get_column("sf_group").to_numpy()
    grp_a = arr.get_column("sf_group").to_numpy() if arr.height else np.zeros(0, dtype=object)
    ta = arr.get_column("_t").to_numpy() if arr.height else np.zeros(0, dtype=np.int64)
    codes = _codes(np.concatenate([grp_d, grp_a]))
    gd, ga = codes[:n], codes[n:]
    no_rwy = np.array([r is None or r == NO_RUNWAY for r in rwy], dtype=bool)

    t0 = np.where(valid, a, 0)
    x = t - TM_MIN * 60 * US
    self_push = valid & (a < t)
    self_x = valid & (a <= x)

    route_push = np.zeros(n, dtype=np.int64)
    group_push = np.zeros(n, dtype=np.int64)
    route_x = np.zeros(n, dtype=np.int64)
    group_x = np.zeros(n, dtype=np.int64)
    arr_grp = np.zeros(n, dtype=np.int64)

    def fill(sel: np.ndarray, push_out: np.ndarray, x_out: np.ndarray) -> None:
        vs = valid[sel]
        a_v = np.sort(a[sel][vs])
        m_v = np.sort(t[sel][vs])
        push_out[sel] = _on_surface(a_v, m_v, t0[sel]) - self_push[sel]
        x_out[sel] = _on_surface(a_v, m_v, x[sel]) - self_x[sel]

    for g in np.unique(gd[gd >= 0]):
        sel = np.flatnonzero(gd == g)
        fill(sel, group_push, group_x)
        arr_grp[sel] = _open_count(np.sort(ta[ga == g]), t0[sel], t[sel])
        rsel = sel[~no_rwy[sel]]
        for r in np.unique(rwy[rsel]):
            fill(rsel[rwy[rsel] == r], route_push, route_x)

    # airport-wide taxiing at t0, surface_queue's definition, for the share
    a_all = np.sort(a[valid])
    m_all = np.sort(t[valid])
    total = _on_surface(a_all, m_all, t0) - self_push

    no_grp = gd < 0
    out: dict[str, np.ndarray] = {}
    for name, v in (
        ("sf_same_route_taxiing_at_push", route_push),
        ("sf_same_group_taxiing_at_push", group_push),
        ("sf_arr_same_group_during_taxi", arr_grp),
        (f"sf_same_route_taxiing_at_tm{TM_MIN}", route_x),
        (f"sf_same_group_taxiing_at_tm{TM_MIN}", group_x),
    ):
        f = v.astype(np.float64)
        f[no_grp] = np.nan
        if "_route_" in name:
            f[no_rwy] = np.nan
        if not name.endswith(f"_tm{TM_MIN}"):
            f[~valid] = np.nan
        out[name] = f
    rp = out["sf_same_route_taxiing_at_push"]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["sf_share_same_route"] = np.where(total > 0, rp / np.maximum(total, 1), np.nan)
    return out


@feature("surface_flow")
def surface_flow(df: pl.DataFrame) -> pl.DataFrame:
    """Departure rows only. Other departures taxiing at own pushback and 20 min before own
    takeoff on the own route (stand group × runway) and own stand group, the same-route share
    of the airport-wide count at pushback, and arrivals landing during own taxi whose stand
    is in the own stand group. Stand group = the stand identifier's letter prefix or
    hundreds digit, rare groups merged. Per (airport, data island). Counts Int16, share
    Float32. Definitions in the module docstring."""
    c = pl.col
    island = island_expr().alias("_island")
    df = df.select(S.MVT_ID, S.AIRPORT, S.PHASE, S.STAND, S.RUNWAY, S.MVT_TIME, S.AOBT_3,
                   S.SPLIT_MONTH)  # fmt: skip
    groups = stand_groups(df)
    proxy_s = (c(S.MVT_TIME) - c(S.AOBT_3)).dt.total_seconds()
    valid = proxy_s.is_between(0, MAX_TAXI_S).fill_null(False)
    deps = (
        df.filter(c(S.PHASE) == "DEP")
        .join(groups, on=[S.AIRPORT, S.STAND], how="left")
        .select(
            S.MVT_ID,
            S.AIRPORT,
            island,
            c(S.RUNWAY).fill_null(NO_RUNWAY),
            "sf_group",
            c(S.MVT_TIME).dt.epoch("us").alias("_t"),
            c(S.AOBT_3).dt.epoch("us").fill_null(0).alias("_a"),
            valid.alias("_valid"),
        )
        .sort([S.AIRPORT, "_island", "_t", S.MVT_ID])
    )
    arrs = (
        df.filter(c(S.PHASE) == "ARR")
        .join(groups, on=[S.AIRPORT, S.STAND], how="left")
        .select(S.AIRPORT, island, "sf_group", c(S.MVT_TIME).dt.epoch("us").alias("_t"))
        .drop_nulls("_t")
        .sort([S.AIRPORT, "_island", "_t"])
    )
    arr_groups = arrs.partition_by([S.AIRPORT, "_island"], as_dict=True, maintain_order=True)
    empty_arr = arrs.clear()
    parts = []
    for key, g in deps.partition_by(
        [S.AIRPORT, "_island"], as_dict=True, maintain_order=True
    ).items():
        cols = _group(g, arr_groups.get(key, empty_arr))
        parts.append(pl.DataFrame({S.MVT_ID: g.get_column(S.MVT_ID), **cols}))
    if not parts:
        return pl.DataFrame(
            {S.MVT_ID: []},
            schema={S.MVT_ID: pl.Float64, **{k: pl.Int16 for k in COUNT_COLS}}
            | {k: pl.Float32 for k in FLOAT_COLS},
        )
    out = pl.concat(parts, how="vertical")
    return out.select(
        S.MVT_ID,
        *[c(k).fill_nan(None).cast(pl.Int16) for k in COUNT_COLS],
        *[c(k).fill_nan(None).cast(pl.Float32) for k in FLOAT_COLS],
    )
