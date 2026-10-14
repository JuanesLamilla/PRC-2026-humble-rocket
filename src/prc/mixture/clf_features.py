"""Head-only signals: what the corruption classifier gets to see that the regressor does not.

The regressor's matrix is deliberately poor on the rows that decide the score: every surface
block is nulled on no-flight-plan departures (``mask_for_no_nm``, findings 2026-09-18) because
those columns let the tree fit catastrophic labels. The *classifier* has the opposite problem —
it is predicting whether the airport wrote the schedule into the off-block stamp, which is an
administrative event, not a taxi. It wants exactly the columns the regressor must not use,
plus a few signals nobody needs as a regressor feature.

Everything in this module is a pure function of the movement table (takeoff / schedule stamps,
stand, airport, NM match, flight number) and is therefore computable identically for the
ranking rows — except :func:`sched_eq_events` / :func:`sched_eq_prev24h` and
:func:`sched_default_stats`, which read ``BLOCK_TIME_UTC_mvt`` and are *fold-scoped by
construction*: the caller passes the rows a given head is allowed to learn from and gets back a
statistic that never includes the row it describes (the 24 h window is half-open,
``[t - 24 h, t)``; the encodings are measured on the passed rows only).

Two families live here, switched on independently from the YAML:

``mixture.corruption.extra_signals``
    :data:`SIGNAL_COLS` + :data:`SCHED_EQ_COL` — the departure's *state* (local hour, delay,
    the stand turnaround, the airport-day no-NM share).
``mixture.corruption.identity_signals`` (2026-09-18)
    :data:`IDENTITY_COLS` + :data:`SCHED_TE_COLS` — the departure's *identity*. On a
    no-flight-plan departure every ``_flt`` column is null and ``AIRCRAFT_TYPE_mvt`` is null on
    1,487 of 1,488 LIRF rows, so ``FLIGHT_mvt`` is the only identity column that survives —
    and the schedule-default substitution is a property of the airline / its handler, not of
    the flight's state (EDA §3: the ICAO prefix alone is AUC 0.814 on LIRF's no-NM departures,
    against 0.662 for the delay). The encodings are measured on **all** of a fold's training
    departures of that (airport, airline), which is where the rate is ~100x better powered
    than on the no-NM rows themselves (EDA §3.2, correlation 0.832 between the two).

Column prefixes: ``clf_`` for the signals below, ``clfx_`` for the unmasked copies of feature
blocks named in ``mixture.corruption.extra_blocks`` (joined by the runner).
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from prc.data import schema as S

#: IANA zone per airport, for the *local* hour. The schedule-default event is an airport
#: back-office artefact, so it follows local working hours, not UTC (findings 2026-09-18:
#: ``BLOCK == SCHED`` is ~2x more likely at night).
AIRPORT_TZ: dict[str, str] = {
    "EDDF": "Europe/Berlin",
    "EDDM": "Europe/Berlin",
    "EGLL": "Europe/London",
    "EHAM": "Europe/Amsterdam",
    "LEBL": "Europe/Madrid",
    "LEMD": "Europe/Madrid",
    "LFPG": "Europe/Paris",
    "LIRF": "Europe/Rome",
    "LSZH": "Europe/Zurich",
    "LTFM": "Europe/Istanbul",
}

SIGNAL_COLS: tuple[str, ...] = (
    "clf_local_hour",
    "clf_mvt_minus_sched",
    "clf_day_no_nm_share",
    "clf_stand_prev_inblock_gap_s",
    "clf_stand_prev_dep_gap_s",
    "clf_prev_occupant_no_nm",
    "clf_stand_next_dep_gap_min",
)

SCHED_EQ_COL = "clf_sched_eq_prev24h"
WINDOW_S = 86400

#: Pure-parse identity columns (``mixture.corruption.identity_signals``). ``clf_fi_airline`` is
#: the ICAO prefix of ``FLIGHT_mvt``, carried as a :class:`polars.Enum` so train and ranking
#: rows share one category ordering.
IDENTITY_COLS: tuple[str, ...] = (
    "clf_fi_airline",
    "clf_fi_null",
    "clf_fi_numeric_only",
    "clf_fi_alpha_len",
    "clf_fi_dup_prefix",
    "clf_reoccupied",
    "clf_ades_domestic",
)

TE_AIRLINE_COL = "clf_te_airline_sched"
TE_ADES_COL = "clf_te_ades_sched"
#: Fold-scoped encodings of the schedule-default label (:func:`sched_default_stats`).
SCHED_TE_COLS: tuple[str, ...] = (TE_AIRLINE_COL, TE_ADES_COL)
#: Rows of prior mass pulling a cell towards its airport's rate. 20 is the ``te_alpha`` the
#: regressor's target encodings use and the value the EDA measured its 0.814 / 0.791 with.
SCHED_TE_PRIOR = 20.0
#: Value of a categorical key that is missing or too short to parse.
NA_KEY = "NA"


def _local_hour() -> pl.Expr:
    """Hour of the takeoff in the airport's own time zone (UTC hour where it is unknown)."""
    expr: pl.Expr = pl.col(S.MVT_TIME).dt.hour()
    for airport in sorted(AIRPORT_TZ):
        expr = (
            pl.when(pl.col(S.AIRPORT) == airport)
            .then(pl.col(S.MVT_TIME).dt.convert_time_zone(AIRPORT_TZ[airport]).dt.hour())
            .otherwise(expr)
        )
    return expr.cast(pl.Int8)


def _stand_links(df: pl.DataFrame) -> pl.DataFrame:
    """Per departure: the previous arrival at the same (airport, stand) and whether that
    occupant had an NM flight plan, the gap to the previous departure from the stand, and the
    minutes to the next one. Departures only; arrivals contribute their in-block stream."""
    arr = (
        df.filter(
            (pl.col(S.PHASE) == "ARR")
            & pl.col(S.STAND).is_not_null()
            & pl.col(S.BLOCK_TIME).is_not_null()
        )
        .select(
            S.AIRPORT,
            S.STAND,
            pl.col(S.BLOCK_TIME).alias("_ib"),
            pl.col(S.FLIGHT_ID).is_null().cast(pl.Int8).alias("_occupant_no_nm"),
        )
        .unique(subset=[S.AIRPORT, S.STAND, "_ib"], keep="first")
        .sort([S.AIRPORT, S.STAND, "_ib"])
    )
    deps = (
        df.filter(pl.col(S.PHASE) == "DEP")
        .select(S.MVT_ID, S.AIRPORT, S.STAND, S.MVT_TIME)
        .sort([S.AIRPORT, S.STAND, S.MVT_TIME])
    )
    at_stand = pl.col(S.STAND).is_not_null()
    prev_dep = (
        pl.col(S.MVT_TIME) - pl.col(S.MVT_TIME).shift(1).over([S.AIRPORT, S.STAND])
    ).dt.total_seconds()
    next_dep = (
        pl.col(S.MVT_TIME).shift(-1).over([S.AIRPORT, S.STAND]) - pl.col(S.MVT_TIME)
    ).dt.total_seconds()
    deps = deps.with_columns(
        pl.when(at_stand).then(prev_dep).cast(pl.Float32).alias("clf_stand_prev_dep_gap_s"),
        pl.when(at_stand)
        .then(next_dep / 60.0)
        .cast(pl.Float32)
        .alias("clf_stand_next_dep_gap_min"),
    )
    out = deps.join_asof(
        arr,
        left_on=S.MVT_TIME,
        right_on="_ib",
        by=[S.AIRPORT, S.STAND],
        strategy="backward",
        allow_exact_matches=False,
        coalesce=False,
    )
    return out.select(
        S.MVT_ID,
        (pl.col(S.MVT_TIME) - pl.col("_ib"))
        .dt.total_seconds()
        .cast(pl.Float32)
        .alias("clf_stand_prev_inblock_gap_s"),
        pl.col("_occupant_no_nm").cast(pl.Int8).alias("clf_prev_occupant_no_nm"),
        pl.col("clf_stand_prev_dep_gap_s"),
        pl.col("clf_stand_next_dep_gap_min"),
    )


def signals(df: pl.DataFrame) -> pl.DataFrame:
    """One row per departure ``MVT_ID_mvt`` with :data:`SIGNAL_COLS`.

    Pure function of the movement table; reads no target and no off-block stamp of the row it
    describes, so train and ranking rows are described identically. The airport-day no-NM share
    is computed over *all* departures (train + test) — ``AOBT_3_flt`` is populated in the
    ranking set, so this is the same quantity on both sides.
    """
    deps = df.filter(pl.col(S.PHASE) == "DEP")
    day_share = (
        deps.select(
            S.AIRPORT,
            pl.col(S.MVT_TIME).dt.date().alias("_day"),
            pl.col(S.AOBT_3).is_null().cast(pl.Float32).alias("_no_nm"),
        )
        .group_by([S.AIRPORT, "_day"])
        .agg(pl.col("_no_nm").mean().cast(pl.Float32).alias("clf_day_no_nm_share"))
    )
    base = (
        deps.select(
            S.MVT_ID,
            S.AIRPORT,
            _local_hour().alias("clf_local_hour"),
            (pl.col(S.MVT_TIME) - pl.col(S.SCHED_TIME))
            .dt.total_seconds()
            .cast(pl.Float32)
            .alias("clf_mvt_minus_sched"),
            pl.col(S.MVT_TIME).dt.date().alias("_day"),
        )
        .join(day_share, on=[S.AIRPORT, "_day"], how="left")
        .drop(S.AIRPORT, "_day")
    )
    return base.join(_stand_links(df), on=S.MVT_ID, how="left").select(S.MVT_ID, *SIGNAL_COLS)


def sched_eq_events(rows: pl.DataFrame, tol_s: float) -> pl.DataFrame:
    """``(airport, takeoff time)`` of every row whose off-block stamp equals its schedule.

    Reads ``BLOCK_TIME_UTC_mvt``, i.e. the quantity the head is trying to predict, so the caller
    must pass **only the rows the head it is building is allowed to learn from** (a fold's
    training rows; all training rows for the test-time head).
    """
    gap = (pl.col(S.BLOCK_TIME) - pl.col(S.SCHED_TIME)).dt.total_seconds().abs()
    return (
        rows.filter(gap <= tol_s)
        .select(S.AIRPORT, pl.col(S.MVT_TIME).alias("_t"))
        .sort([S.AIRPORT, "_t"])
    )


def sched_eq_prev24h(
    rows: pl.DataFrame, events: pl.DataFrame, window_s: int = WINDOW_S
) -> pl.DataFrame:
    """How many of ``events`` fall in ``[t - window, t)`` at the row's airport.

    One row per input ``MVT_ID_mvt`` (input order is *not* preserved; join on the id). The
    window is half-open on the right, so a row never counts its own event even when it is one
    of the events — which is what makes the column usable as a training feature.
    """
    empty = pl.DataFrame(
        {S.MVT_ID: rows.get_column(S.MVT_ID), SCHED_EQ_COL: pl.zeros(rows.height, eager=True)}
    ).with_columns(pl.col(SCHED_EQ_COL).cast(pl.Float32))
    if events.is_empty() or rows.is_empty():
        return empty
    cum = (
        events.sort([S.AIRPORT, "_t"])
        .with_columns(pl.int_range(1, pl.len() + 1).over(S.AIRPORT).alias("_c"))
        .unique(subset=[S.AIRPORT, "_t"], keep="last")
        .sort([S.AIRPORT, "_t"])
    )
    q = rows.select(S.MVT_ID, S.AIRPORT, S.MVT_TIME).with_columns(
        (pl.col(S.MVT_TIME) - pl.duration(seconds=window_s)).alias("_lo")
    )
    hi = (
        q.sort([S.AIRPORT, S.MVT_TIME])
        .join_asof(
            cum,
            left_on=S.MVT_TIME,
            right_on="_t",
            by=S.AIRPORT,
            strategy="backward",
            allow_exact_matches=False,
            coalesce=False,
        )
        .select(S.MVT_ID, pl.col("_c").fill_null(0).alias("_hi"))
    )
    lo = (
        q.sort([S.AIRPORT, "_lo"])
        .join_asof(
            cum,
            left_on="_lo",
            right_on="_t",
            by=S.AIRPORT,
            strategy="backward",
            allow_exact_matches=False,
            coalesce=False,
        )
        .select(S.MVT_ID, pl.col("_c").fill_null(0).alias("_lo_c"))
    )
    return (
        hi.join(lo, on=S.MVT_ID, how="left")
        .select(S.MVT_ID, (pl.col("_hi") - pl.col("_lo_c")).cast(pl.Float32).alias(SCHED_EQ_COL))
        .sort(S.MVT_ID)
    )


# --------------------------------------------------------------------------- identity


def _airline_key() -> pl.Expr:
    """The ICAO prefix of ``FLIGHT_mvt``: its first three characters, :data:`NA_KEY` when the
    field is missing or empty. Pure parsing — the same expression on train and ranking rows."""
    key = pl.col(S.FLIGHT_NO).str.slice(0, 3)
    return pl.when(key.is_null() | (key.str.len_chars() == 0)).then(pl.lit(NA_KEY)).otherwise(key)


def _ades_key() -> pl.Expr:
    return pl.col(S.ADES).fill_null(NA_KEY)


def _reoccupied(df: pl.DataFrame) -> pl.DataFrame:
    """``clf_reoccupied``: an arrival went in-block at the departure's stand between its
    schedule and its takeoff — i.e. the stand was given away while the flight was nominally
    still on it (EDA §5: 0.68 vs 0.34, AUC 0.666). Reads only *arrival* in-block stamps, which
    are published for the ranking months, so it is computable on every row."""
    arr = (
        df.filter(
            (pl.col(S.PHASE) == "ARR")
            & pl.col(S.STAND).is_not_null()
            & pl.col(S.BLOCK_TIME).is_not_null()
        )
        .select(S.AIRPORT, S.STAND, pl.col(S.BLOCK_TIME).alias("_ibn"))
        .unique(subset=[S.AIRPORT, S.STAND, "_ibn"], keep="first")
        .sort([S.AIRPORT, S.STAND, "_ibn"])
    )
    deps = (
        df.filter(pl.col(S.PHASE) == "DEP")
        .select(S.MVT_ID, S.AIRPORT, S.STAND, S.SCHED_TIME, S.MVT_TIME)
        .sort([S.AIRPORT, S.STAND, S.SCHED_TIME])
    )
    return deps.join_asof(
        arr,
        left_on=S.SCHED_TIME,
        right_on="_ibn",
        by=[S.AIRPORT, S.STAND],
        strategy="forward",
        coalesce=False,
    ).select(
        S.MVT_ID,
        (pl.col("_ibn") < pl.col(S.MVT_TIME))
        .fill_null(False)
        .cast(pl.Int8)
        .alias("clf_reoccupied"),
    )


def identity(df: pl.DataFrame) -> pl.DataFrame:
    """One row per departure ``MVT_ID_mvt`` with :data:`IDENTITY_COLS`.

    Pure function of the movement table: the flight-number parse, the stand rotation on the
    arrival side and the destination's ICAO country prefix. It reads no target and no
    off-block stamp *of a departure*, so train and ranking rows are described identically —
    the fold-scoped part of this family is :func:`sched_default_stats`, not this.

    ``clf_fi_airline`` is a :class:`polars.Enum` whose categories are taken from *all*
    departures in ``df`` (train and ranking together), so the two sides convert to pandas
    categoricals with one shared ordering.
    """
    deps = df.filter(pl.col(S.PHASE) == "DEP")
    keys = deps.select(_airline_key().alias("_k")).get_column("_k")
    levels = sorted({*keys.drop_nulls().to_list(), NA_KEY})
    flight = pl.col(S.FLIGHT_NO)
    suffix = flight.str.replace_all(r"^[A-Z]+", "")
    base = deps.select(
        S.MVT_ID,
        _airline_key().cast(pl.Enum(levels)).alias("clf_fi_airline"),
        flight.is_null().cast(pl.Int8).alias("clf_fi_null"),
        suffix.str.contains(r"^[0-9]+$")
        .fill_null(False)
        .cast(pl.Int8)
        .alias("clf_fi_numeric_only"),
        flight.str.extract(r"^([A-Z]+)", 1)
        .str.len_chars()
        .cast(pl.Float32)
        .alias("clf_fi_alpha_len"),
        # the doubled-designator artefact: WZZ -> WZZZZ, ITY -> ITYTY, AEZ -> AEZZ (EDA §3.3)
        (
            (flight.str.slice(1, 2) == flight.str.slice(3, 2))
            | (flight.str.slice(2, 1) == flight.str.slice(3, 1))
        )
        .fill_null(False)
        .cast(pl.Int8)
        .alias("clf_fi_dup_prefix"),
        (pl.col(S.ADES).str.slice(0, 2) == pl.col(S.AIRPORT).str.slice(0, 2))
        .fill_null(False)
        .cast(pl.Int8)
        .alias("clf_ades_domestic"),
    )
    return base.join(_reoccupied(df), on=S.MVT_ID, how="left").select(S.MVT_ID, *IDENTITY_COLS)


# ------------------------------------------------------- fold-scoped label encodings


@dataclass(frozen=True)
class SchedDefaultStats:
    """Schedule-default rates measured on one set of rows, per airport and per key.

    Built only by :func:`sched_default_stats`, which is the fold-scoped entry point: whatever
    rows the caller passes are the only rows whose off-block stamp is ever read.
    """

    airport: pl.DataFrame  # airport -> _gm
    airline: pl.DataFrame  # (airport, airline key) -> _s_al, _n_al
    ades: pl.DataFrame  # (airport, ADES) -> _s_ad, _n_ad
    global_rate: float
    prior: float


def sched_default_stats(
    rows: pl.DataFrame, tol_s: float, prior: float = SCHED_TE_PRIOR
) -> SchedDefaultStats:
    """Count ``|BLOCK − SCHED| <= tol_s`` per (airport, airline) and (airport, destination).

    Reads ``BLOCK_TIME_UTC_mvt``, i.e. the quantity the head is trying to predict, so — exactly
    like :func:`sched_eq_events` — the caller must pass **only the rows the head it is building
    is allowed to learn from**: a fold's training departures, or all training departures for
    the test-time head. The statistics are measured on *all* of those departures, not only the
    no-flight-plan ones: per airline the two rates correlate at 0.83 and the with-plan
    population is ~100x larger (EDA §3.2 / conclusion 2).
    """
    lab = rows.select(
        S.AIRPORT,
        _airline_key().alias("_al"),
        _ades_key().alias("_ad"),
        ((pl.col(S.BLOCK_TIME) - pl.col(S.SCHED_TIME)).dt.total_seconds().abs() <= tol_s)
        .fill_null(False)
        .cast(pl.Float64)
        .alias("_y"),
    )
    return SchedDefaultStats(
        airport=lab.group_by(S.AIRPORT).agg(pl.col("_y").mean().alias("_gm")),
        airline=lab.group_by([S.AIRPORT, "_al"]).agg(
            pl.col("_y").sum().alias("_s_al"), pl.len().alias("_n_al")
        ),
        ades=lab.group_by([S.AIRPORT, "_ad"]).agg(
            pl.col("_y").sum().alias("_s_ad"), pl.len().alias("_n_ad")
        ),
        global_rate=float(lab.get_column("_y").mean() or 0.0),  # type: ignore[arg-type]
        prior=prior,
    )


def sched_default_encodings(
    rows: pl.DataFrame, stats: SchedDefaultStats, loo_tol_s: float | None = None
) -> pl.DataFrame:
    """:data:`SCHED_TE_COLS` for ``rows``, one row per input ``MVT_ID_mvt``.

    ``(Σy + prior·r_airport) / (n + prior)``: a cell the fold's training rows never saw falls
    back to that airport's own rate, and an airport they never saw to the pooled rate. Input
    order is not preserved; join on the id.

    ``loo_tol_s`` turns on **leave-one-out** for the rows that are inside ``stats``
    (2026-09-19, code review F4): ``(Σy − yᵢ + prior·r) / (n − 1 + prior)``. Without it, a row
    the head is *fitted* on has its own schedule-default label inside its own encoding —
    self-share up to 4.8 % in cells with n ≤ 5, ``corr(encoding, own label) ≈ 0.27`` — while
    the rows the head is *scored* on get clean out-of-sample encodings. That is train/serve
    skew, and it is confined to ``identity_signals: true``, which is exactly what exp0057 /
    exp0058 / exp0062 turned on and what was rejected for losing 8-31 s.

    Using it, ``rows`` must additionally carry ``BLOCK_TIME_UTC_mvt``, ``SCHED_TIME_UTC_mvt``
    and a boolean ``_in_fit`` marking the rows that contributed to ``stats``; rows with
    ``_in_fit = False`` are encoded exactly as they are without the flag.
    """
    cols = [S.MVT_ID, S.AIRPORT, _airline_key().alias("_al"), _ades_key().alias("_ad")]
    if loo_tol_s is not None:
        cols += [
            (
                pl.col("_in_fit")
                & (
                    (pl.col(S.BLOCK_TIME) - pl.col(S.SCHED_TIME)).dt.total_seconds().abs()
                    <= loo_tol_s
                ).fill_null(False)
            )
            .cast(pl.Float64)
            .alias("_own_y"),
            pl.col("_in_fit").cast(pl.Float64).alias("_own_n"),
        ]
    base = (
        rows.select(*cols)
        .join(stats.airport, on=S.AIRPORT, how="left")
        .join(stats.airline, on=[S.AIRPORT, "_al"], how="left")
        .join(stats.ades, on=[S.AIRPORT, "_ad"], how="left")
    )
    gm = pl.col("_gm").fill_null(stats.global_rate)
    w = stats.prior
    own_y = pl.col("_own_y") if loo_tol_s is not None else pl.lit(0.0)
    own_n = pl.col("_own_n") if loo_tol_s is not None else pl.lit(0.0)

    def smoothed(s: str, n: str) -> pl.Expr:
        num = pl.col(s).fill_null(0.0) - own_y + w * gm
        den = pl.col(n).fill_null(0) - own_n + w
        return (num / den).cast(pl.Float32)

    return base.select(
        S.MVT_ID,
        smoothed("_s_al", "_n_al").alias(TE_AIRLINE_COL),
        smoothed("_s_ad", "_n_ad").alias(TE_ADES_COL),
    )
