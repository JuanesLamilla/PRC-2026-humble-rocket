"""Feature registry with a content-addressed parquet cache.

A *feature block* is a function ``f(movements: pl.DataFrame) -> pl.DataFrame`` that
receives the FULL normalized movement table (train + test, DEP + ARR, sorted by
airport/time) and returns one row per ``MVT_ID_mvt`` with the new columns. Blocks
see test rows so that window / queue features are computed identically for both
(that's legitimate here: the challenge is post-ops, all takeoff times are known).

Blocks must NEVER read ``TAXITIME_SEC_mvt`` or ``BLOCK_TIME_UTC_mvt`` of the row they
describe. Target-encoding-style blocks that need the target must go through
``prc.features.target_enc`` which is fold-aware.

Cache key = block name + block source hash + input digest (+ an **external digest** when the
block declares one), so editing a block invalidates only its own cache.

The 2026-09-18 cache incident: the first three parts of that key do **not** see ``data/external/``.
On 2026-09-18 six archive-backed blocks built as all-null columns on the cluster, both
hashes matched, and the run would have trained on nulls. A block that reads files outside
the movement table must therefore declare them::

    @feature("winter_ops", external=lambda: external_digest(metar_dir()))

``external`` returns a short hash of the *bytes* of every file the block reads (or ``""``
when the archive is absent, which is itself a distinct key). It is appended to the cache
file name as ``__x<hash>``, so a block built against a different archive — or against no
archive at all — can never be served from the cache of another. Blocks without it keep
their historical file names byte-for-byte.
"""

from __future__ import annotations

import hashlib
import inspect
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from prc import paths
from prc.data import schema as S

FeatureFn = Callable[[pl.DataFrame], pl.DataFrame]


@dataclass(frozen=True)
class FeatureBlock:
    name: str
    fn: FeatureFn
    doc: str
    #: Optional: a hash of every file outside the movement table that the block reads.
    external: Callable[[], str] | None = None

    def source_hash(self) -> str:
        """Hash of the block's **module**, not just its decorated function.

        Hashing the function alone (the behaviour before 2026-09-19) left a stale parquet in
        place whenever a parsing helper such as ``_daily`` / ``_archive`` / ``_vis_metres``
        was fixed — the same silently-wrong-cache class as the 2026-09-18 cache incident, one level
        down (feature code review 2026-09-19, finding #4). The module is the smallest unit
        that certainly contains every helper a block calls.
        """
        mod = sys.modules.get(self.fn.__module__)
        src = inspect.getsource(mod) if mod is not None else inspect.getsource(self.fn)
        return hashlib.sha1(src.encode()).hexdigest()[:8]

    def external_hash(self) -> str:
        """``""`` for a block with no declared external inputs (historical file names)."""
        return self.external() if self.external is not None else ""


def external_digest(files: Iterable[Path], module: str | None = None) -> str:
    """Hash the *content* of the files a block reads, plus their names and sizes.

    An empty / missing set hashes to a distinct value of its own, so "built without the
    archive" and "built with it" are different cache entries rather than the same one.

    ``module`` (pass ``__name__``) additionally folds in the source of the block's whole
    module. :meth:`FeatureBlock.source_hash` only sees the decorated function, so without
    this a fix to a parsing helper would leave a stale parquet in place — the same class of
    bug as the 2026-09-18 cache incident, one level down.
    """
    h = hashlib.sha256()
    if module is not None:
        h.update(inspect.getsource(sys.modules[module]).encode())
    for f in sorted(Path(p) for p in files):
        if not f.is_file():
            continue
        h.update(f.name.encode())
        h.update(str(f.stat().st_size).encode())
        with f.open("rb") as fh:
            for chunk in iter(lambda fh=fh: fh.read(1 << 20), b""):  # type: ignore[misc]
                h.update(chunk)
    return h.hexdigest()[:8]


def archive_files(pattern: str, glob: str = "*") -> list[Path]:
    """Every file matching ``glob`` in the newest ``data/external/<pattern>`` directory."""
    dirs = sorted(paths.EXTERNAL.glob(pattern))
    return sorted(dirs[-1].glob(glob)) if dirs else []


REGISTRY: dict[str, FeatureBlock] = {}


def feature(
    name: str, external: Callable[[], str] | None = None
) -> Callable[[FeatureFn], FeatureFn]:
    """Register a block. ``external`` (see the module docstring) is mandatory in spirit for
    any block that reads ``data/external/``: without it the cache cannot tell an archive-less
    build from a real one."""

    def deco(fn: FeatureFn) -> FeatureFn:
        if name in REGISTRY:
            raise ValueError(f"feature block {name!r} registered twice")
        REGISTRY[name] = FeatureBlock(
            name=name, fn=fn, doc=(fn.__doc__ or "").strip(), external=external
        )
        return fn

    return deco


def list_features() -> list[str]:
    return sorted(REGISTRY)


def input_digest(df: pl.DataFrame) -> str:
    """Fingerprint of the movement table.

    Row count + schema + id range does not look at a single *value*: a reissue of the raw
    data with the same shape would serve every stale parquet (code review 2026-09-19,
    finding #10). So when ``df`` is the materialised movement table — same row count as
    ``data/interim/movements.parquet`` — the file's content hash goes into the digest as
    well. A caller that passes a *subset* (a sample, one airport) gets the shape fingerprint
    alone, exactly as before, because there is no file to hash it against.
    """
    from prc.cache import file_hash

    h = hashlib.sha1()
    h.update(str(df.height).encode())
    h.update(str(df.schema).encode())
    ids = df.get_column(S.MVT_ID)
    h.update(str(ids.min()).encode())
    h.update(str(ids.max()).encode())
    mv = paths.INTERIM / "movements.parquet"
    if mv.is_file():
        try:
            same_height = pl.scan_parquet(mv).select(pl.len()).collect().item() == df.height
        except (OSError, pl.exceptions.PolarsError):
            same_height = False
        if same_height:
            fh = file_hash(mv)
            if fh:
                h.update(fh.encode())
    return h.hexdigest()[:8]


def build(df: pl.DataFrame, names: list[str], use_cache: bool = True) -> pl.DataFrame:
    """Return a frame keyed by MVT_ID with all requested blocks joined."""
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise KeyError(f"unknown feature blocks: {unknown}; known: {list_features()}")
    digest = input_digest(df)
    out = df.select(S.MVT_ID)
    for n in names:
        block = REGISTRY[n]
        sh = block.source_hash()
        ext = block.external_hash()
        cache = paths.FEATURES / f"{n}__{sh}__{digest}{f'__x{ext}' if ext else ''}.parquet"
        if use_cache and cache.exists():
            part = pl.read_parquet(cache)
        else:
            part = block.fn(df)
            if S.MVT_ID not in part.columns:
                raise ValueError(f"block {n} must return {S.MVT_ID}")
            forbidden = {S.TARGET, S.BLOCK_TIME} & set(part.columns)
            if forbidden:
                raise ValueError(f"block {n} leaks target columns {forbidden}")
            if use_cache:
                paths.FEATURES.mkdir(parents=True, exist_ok=True)
                # Same code, same movement table, a different archive: the sibling is stale,
                # and leaving it would give `prc.cache.block_inputs` two matches and silently
                # disable the frame cache for every run on this machine.
                if ext:
                    for old in paths.FEATURES.glob(f"{n}__{sh}__{digest}__x*.parquet"):
                        if old != cache:
                            old.unlink()
                part.write_parquet(cache, compression="zstd")
        out = out.join(part, on=S.MVT_ID, how="left")
    return out
