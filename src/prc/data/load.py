"""Load the challenge parquet files into one normalized movement table.

Raw layout (mirrored from the bucket into ``data/raw/``):
    training_2025-MM-01_2025-MM+1-01.parquet   x12
    ranking.parquet                             (Jan + Jul 2026; DEP targets blanked)
    submitting.parquet                          (MVT_ID_mvt, TAXITIME_SEC_mvt template)

``build_interim()`` writes ``data/interim/movements.parquet``: train + ranking rows
concatenated, timestamps parsed to UTC datetimes, ``airport`` / ``is_test`` / ``month``
derived columns added, sorted by (airport, MVT_TIME). Every downstream step reads this
one file, so schema fixes happen exactly once.
"""

from __future__ import annotations

import polars as pl

from prc import paths
from prc.data import schema as S


def list_training_files() -> list:
    files = sorted(paths.RAW.glob("training_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no training_*.parquet under {paths.RAW}; run `prc data pull`")
    return files


def _parse_times(df: pl.DataFrame) -> pl.DataFrame:
    exprs = []
    for c in S.TIME_COLS:
        if c not in df.columns:
            continue
        dt = df.schema[c]
        if dt == pl.Null:
            exprs.append(pl.col(c).cast(pl.Datetime("us", "UTC")))
        elif dt == pl.Utf8:
            exprs.append(pl.col(c).str.to_datetime(time_unit="us", time_zone="UTC", strict=False))
        elif isinstance(dt, pl.Datetime):
            exprs.append(
                pl.col(c).dt.replace_time_zone("UTC")
                if dt.time_zone is None
                else pl.col(c).dt.convert_time_zone("UTC")
            )
    return df.with_columns(exprs) if exprs else df


def _derive(df: pl.DataFrame, is_test: bool) -> pl.DataFrame:
    return df.with_columns(
        pl.when(pl.col(S.PHASE) == "DEP")
        .then(pl.col(S.ADEP))
        .otherwise(pl.col(S.ADES))
        .alias(S.AIRPORT),
        pl.lit(is_test).alias(S.IS_TEST),
        pl.col(S.MVT_TIME).dt.strftime("%Y-%m").alias(S.SPLIT_MONTH),
    )


def load_raw_training() -> pl.DataFrame:
    frames = [_parse_times(pl.read_parquet(f)) for f in list_training_files()]
    return pl.concat(frames, how="diagonal_relaxed")


def load_raw_ranking() -> pl.DataFrame:
    return _parse_times(pl.read_parquet(paths.RAW / "ranking.parquet"))


def load_submitting_template() -> pl.DataFrame:
    return pl.read_parquet(paths.RAW / "submitting.parquet")


def write_raw_manifest() -> dict:
    """Sizes + mtimes + sha256 of every raw file. The organizers reissued ranking.parquet on
    2026-09-04; diffing this manifest is how we notice the next reissue."""
    import hashlib
    import json
    from datetime import UTC, datetime

    files = {}
    for f in sorted(paths.RAW.glob("*.parquet")):
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        files[f.name] = {
            "bytes": f.stat().st_size,
            "sha256": h,
            "mtime": datetime.fromtimestamp(f.stat().st_mtime, UTC).isoformat(timespec="seconds"),
        }
    manifest = {"written_at": datetime.now(UTC).isoformat(timespec="seconds"), "files": files}
    (paths.RAW / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def build_interim(force: bool = False) -> pl.DataFrame:
    out = paths.INTERIM / "movements.parquet"
    if out.exists() and not force:
        return pl.read_parquet(out)
    paths.INTERIM.mkdir(parents=True, exist_ok=True)
    write_raw_manifest()
    train = _derive(load_raw_training(), is_test=False)
    test = _derive(load_raw_ranking(), is_test=True)
    df = pl.concat([train, test], how="diagonal_relaxed").sort([S.AIRPORT, S.MVT_TIME])
    df.write_parquet(out, compression="zstd")
    return df


def load_movements() -> pl.DataFrame:
    """The single normalized table (train + test). Build it first with ``prc data prepare``."""
    out = paths.INTERIM / "movements.parquet"
    if not out.exists():
        raise FileNotFoundError(f"{out} missing; run `prc data prepare`")
    return pl.read_parquet(out)


def departures(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(pl.col(S.PHASE) == "DEP")
