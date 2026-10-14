"""ADS-B surface observation of the departure (plan: ADS-B phase 3).

Open ADS-B (ADSB.lol, ODbL 1.0) sees many departures move off stand, taxi and lift off.
``scripts/adsb_build.py`` turns each day's release into one parquet of **derived** features
keyed by ``MVT_ID_mvt`` under ``data/external/adsblol_ground/``; this block is the join.

Every column is anchored on ``MVT_TIME_UTC_mvt``, which is populated on the ranking rows,
so all of them are computable at test time. Nothing here reads ``BLOCK_TIME_UTC_mvt`` or
the target — the ADS-B track is an *observation of the surface*, and the model is left to
learn the per-airport offset between the first observed movement and the airport's own
off-block stamp (that offset is stable: IQR ~230-280 s at EDDF/LEBL/LEMD).

**Only the drift-safe columns are exposed.** The 2025-vs-2026 check in
internal note 2026-09-19_adsb-pilot-phase2 §5 measured, per feature, the shift of its
median and mean between Jan/Jul 2025 and Jan/Jul 2026 days in units of its sd. The eight
kept here are all within 0.25 sd. Deliberately **not** exposed, and not to be added back
without re-measuring:

* ``n_stops`` — median shift 0.87 sd (0 in 2025, 1 in 2026): a cadence artefact.
* ``mvt_minus_first_seen_s`` — 0.35 sd.
* ``n_gnd`` — 0.45 sd, and not comparable between the heatmap and trace extraction routes.
* ``cov_airport_day`` — 1.36 sd. It *is* the coverage level, which is exactly what changes.
* every airport-state feature (aircraft moving on the ground, queue at the threshold) — the
  state-only control was **+4.8 s**, i.e. harmful; the movement table already encodes
  congestion better.

What does drift is how many rows carry the features at all (EDDM 31 % -> 99 %, EGLL 26 % ->
91 %, LIRF 64 % -> 33 % between 2025 and 2026). That is why ``adsb_matched`` and
``adsb_has_gnd`` are exposed as explicit flags rather than left implicit in the nulls.

Returns all-null columns when the archive is absent, so tests and fresh clones still run —
and the block declares ``external=`` so the cache can never serve an archive-less build in
place of a real one (the 2026-09-18 cache incident).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from prc import paths
from prc.data import schema as S
from prc.features.registry import external_digest, feature

#: Directory written by ``scripts/adsb_build.py``. Not date-stamped: it is an accumulating,
#: idempotent per-day archive rather than a single dated pull.
ARCHIVE = "adsblol_ground"

FLAGS: tuple[str, ...] = ("adsb_matched", "adsb_has_gnd")
FEATURES: tuple[str, ...] = (
    "adsb_mvt_minus_first_move_s",
    "adsb_mvt_minus_first_gnd_s",
    "adsb_gnd_path_m",
    "adsb_stop_total_s",
    "adsb_final_hold_s",
    "adsb_mean_gs_kt",
    "adsb_first_seen_dist_thr_m",
    "adsb_gnd_max_gap_s",
)
OUT: tuple[str, ...] = FLAGS + FEATURES


def archive_days() -> list[Path]:
    """Every per-day parquet ``scripts/adsb_build.py`` has written so far."""
    d = paths.EXTERNAL / ARCHIVE
    return sorted(d.glob("*.parquet")) if d.is_dir() else []


def _adsb_external() -> str:
    """Hash of every per-day parquet the block reads, plus this module's source.

    The archive grows day by day, so this digest changes as the build progresses and every
    partial state gets its own cache entry. That is the intended behaviour: a run must never
    be served a cache built from a smaller archive.
    """
    return external_digest(archive_days(), __name__)


def _archive() -> pl.DataFrame | None:
    files = archive_days()
    if not files:
        return None
    df = pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")
    # One row per departure; a day rebuilt at a newer builder version could overlap.
    return df.unique(subset=[S.MVT_ID], keep="last")


@feature("adsb_ground", external=_adsb_external)
def adsb_ground(df: pl.DataFrame) -> pl.DataFrame:
    """Open ADS-B surface observation of the departure: time from the first observed ground
    movement (and first ground point) to take-off, ground path length, time stopped, the
    final hold before line-up, mean taxi ground speed, how far from the runway threshold the
    aircraft was first seen, the worst gap in ground coverage, and two flags saying whether
    a track was matched at all and whether it had ground points."""
    base = df.select(S.MVT_ID)
    arch = _archive()
    if arch is None:
        return base.with_columns([pl.lit(None, dtype=pl.Float32).alias(c) for c in OUT])
    # The archive is written with the dtype of the real movement table (Float64), while the
    # synthetic fixture keys on Int64. Join on the *caller's* dtype so the block is correct
    # under both and the key it returns is always the one it was handed.
    key_dtype = base.schema[S.MVT_ID]
    arch = arch.select(S.MVT_ID, *OUT).with_columns(pl.col(S.MVT_ID).cast(key_dtype))
    out = base.join(arch, on=S.MVT_ID, how="left")
    return out.select(
        S.MVT_ID,
        # days that were never built stay null on the flags too, which is the honest state:
        # "we have no ADS-B archive for this day" is not the same as "no aircraft was seen".
        *[pl.col(c).cast(pl.Int8).alias(c) for c in FLAGS],
        *[pl.col(c).cast(pl.Float32).alias(c) for c in FEATURES],
    )
