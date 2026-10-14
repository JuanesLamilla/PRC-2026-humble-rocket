"""ADS-B surface observation, generation 2: where the aircraft was, not just when it moved.

`adsb_ground` (v1) describes *when* the departure moved: first movement to take-off, path
length, stops, final hold. This block adds *where* it was when we first saw it, measured
against the stand the movement table assigns the row:

* how far the first ground point and the first movement were from that stand — near zero
  means the track starts on stand and the first movement really is the pushback, hundreds
  of metres means the aircraft was picked up mid-taxi and its first-movement time is a
  lower bound, not the pushback;
* `adsb2_stand_thr_dist_m`, the stand-to-threshold straight line, and
  `adsb2_frac_remaining`, how much of it was still ahead at the first sighting — the
  coverage-corrected version of v1's `adsb_first_seen_dist_thr_m`;
* how long the aircraft was seen standing still before it moved, and over how many samples;
* the **pushback signature**: the first stop of at least one sample within 480 s of the
  first movement. A push is 2-4 kt for 1-4 minutes, then a stop, then the taxi; the
  duration, path, displacement and mean speed of that first segment separate a genuine
  pushback from an aircraft that simply started taxiing, and `adsb2_mvt_minus_taxi_start_s`
  is the taxi measured from the end of the push instead of from the off-block stamp.

The stand coordinate table is **learned from the tracks**, not from OSM: the median
position of the pre-movement stationary points of every departure off that stand and of the
last-movement point of every arrival to it, over the whole archive, for stands with at
least 5 observations (`scripts/adsb_v2.py archive`). `adsb2_stand_known` says whether the
row's stand got a coordinate at all.

Everything is anchored on `MVT_TIME_UTC_mvt` and on `STAND_mvt`, both populated on the
ranking rows, so every column is computable at test time; nothing reads the row's
`BLOCK_TIME_UTC_mvt` or the target.

**Measured**: on 119,003 labelled departures over 21
pilot days, as a residual on the v4 member OOF (which already contains v1), these columns
move the pooled RMSE 276.41 -> 275.70 (day-block CI [-1.36, -0.16]) and the ordinary lane
184.84 -> 183.78 (CI [-1.92, -0.30]); the gain is at the covered airports (EDDM -9, EHAM
-2.9, LSZH -2.3, LEBL -2.5, LEMD -1.4) and zero at LFPG / LIRF / LTFM. 2025-01 improves
with a CI clear of zero; **2025-07 does not** (-0.3, CI [-2.66, +1.17]) on the three July
days in the pilot sample.

Deliberately **not** exposed, and not to be added back without re-measuring:

* `a_extrap_full_taxi_s` (the observed duration scaled by the unobserved fraction) — 0.20 sd
  of 2025-vs-2026 shift and coverage-scale dependent by construction.
* every rotation / stamp-offset column (the inbound leg of the same airframe through
  `icao24`, the ADS-B-measured airport stamp offset per airport / day / 3 h / stand area /
  operator). They are a beautiful measurement — the offset between the inbound's last
  observed movement and the airport's in-block stamp has a median of -2 s at EHAM and -5 s
  at EDDM — but they did not pay: v1+B was flat to worse on every scope, and the aggregates
  drift 0.4-1.5 sd between 2025 and 2026.

Returns all-null columns when the archive is absent, and declares ``external=`` so the
feature cache can never serve an archive-less build in place of a real one.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from prc import paths
from prc.data import schema as S
from prc.features.registry import external_digest, feature

#: Directory written by ``scripts/adsb_v2.py archive`` from the points that
#: ``scripts/adsb_build.py`` (builder version 2) keeps under ``adsblol_points/``.
ARCHIVE = "adsblol_ground_v2"

FLAGS: tuple[str, ...] = ("adsb2_stand_known",)
FEATURES: tuple[str, ...] = (
    "adsb2_first_gnd_dist_stand_m",
    "adsb2_first_move_dist_stand_m",
    "adsb2_stand_thr_dist_m",
    "adsb2_frac_remaining",
    "adsb2_pre_move_stationary_s",
    "adsb2_n_pre_stationary",
    "adsb2_saw_stationary",
    "adsb2_pushback_dur_s",
    "adsb2_pushback_path_m",
    "adsb2_pushback_disp_m",
    "adsb2_pushback_mean_gs_kt",
    "adsb2_mvt_minus_taxi_start_s",
    "adsb2_mvt_minus_last_still_s",
)
OUT: tuple[str, ...] = FLAGS + FEATURES


def archive_days() -> list[Path]:
    """Every per-day parquet ``scripts/adsb_v2.py archive`` has written so far."""
    d = paths.EXTERNAL / ARCHIVE
    if not d.is_dir():
        return []
    # stand_table.parquet is an input to the per-day files, not one of them
    return sorted(f for f in d.glob("*.parquet") if f.name != "stand_table.parquet")


def _adsb2_external() -> str:
    """Hash of every per-day parquet the block reads, plus this module's source.

    The archive grows day by day while the rebuild runs, so this digest changes as it
    progresses and every partial state gets its own cache entry — a run must never be
    served a cache built from a smaller archive.
    """
    return external_digest(archive_days(), __name__)


def _archive() -> pl.DataFrame | None:
    files = archive_days()
    if not files:
        return None
    df = pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")
    # One row per departure; a day rebuilt at a newer archive version could overlap.
    return df.unique(subset=[S.MVT_ID], keep="last")


@feature("adsb_ground_v2", external=_adsb2_external)
def adsb_ground_v2(df: pl.DataFrame) -> pl.DataFrame:
    """Where the ADS-B track starts relative to the departure's own stand: distance of the
    first ground point and of the first movement from the stand, the stand-to-threshold
    distance and the fraction of it still ahead at the first sighting, how long the aircraft
    was seen standing before it moved, and the pushback signature (duration, path,
    displacement, mean speed, and the taxi measured from the end of the push)."""
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
        # a day that was never built stays null on the flag too: "we have no archive for
        # this day" is not the same as "this stand has no coordinate".
        *[pl.col(c).cast(pl.Int8).alias(c) for c in FLAGS],
        *[pl.col(c).cast(pl.Float32).alias(c) for c in FEATURES],
    )
