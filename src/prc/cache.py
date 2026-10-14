"""Fold-level on-disk cache under ``data/cache/`` (gitignored with the rest of ``data/``).

Two things are cached, both of which a capacity sweep recomputes identically for every
variant:

``frames``  the assembled feature frame — blocks joined onto the departures, ``mask_for_no_nm``
            applied, folds joined, ``sample_frac`` taken — plus the per-fold target encodings
            (kind ``te``, one entry per fold and role).
``head``    the corruption head's outputs: out-of-fold ``p_k``, the nested normal-expert
            weights per fold, the test-time ``p_k``, and the head scalars that go in
            ``metrics.json``. Its key deliberately excludes the *normal regressor's*
            hyper-parameters, because the head does not see them — that is what makes seven
            capacity variants of R4 share one head — except under
            ``sched_head: z_regression``, where the head's target is built from a nested
            ``ŷ_normal`` fitted with exactly those parameters and they go back in.

THE RULE THIS MODULE EXISTS TO ENFORCE
--------------------------------------
Cache incident of 2026-09-18: the *feature-block* cache key encoded a spec hash and
an input-data hash, and **neither captured whether ``data/external/`` was present**. Six
blocks built as all-null columns, both hashes matched, ``features build`` reported success,
and the first real experiment would have trained on nulls and reported a plausible RMSE.

So here a key is a digest over the *content* of every input — the movement table's bytes,
every feature-block parquet's bytes, the source of the code that assembles them — not over
names, shapes or specs. :func:`file_hash` reading 400 MB of parquet costs well under a
second; a silently-wrong run costs a day. If any input cannot be hashed (a block has no
materialised parquet yet) the key builder returns ``None`` and the caller **recomputes**:
the cache fails open, never open-with-a-guess.

Perturbing any one of these inputs changes the key.

CONCURRENCY
-----------
Array tasks on the same shared filesystem race. Every write goes to a unique temp file in
the entry's own directory and is then ``os.replace``d (atomic within a filesystem), the
``READY`` marker is written last so a half-written multi-file entry is never read, and a
miss is handled under an exclusive ``flock`` on a per-key sidecar so that N tasks compute a
cold entry once rather than N times. The lock is only ever held by the task that is doing
the work; readers of a ready entry take no lock at all.
"""

from __future__ import annotations

import fcntl
import hashlib
import inspect
import json
import logging
import os
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import polars as pl

from prc import paths

log = logging.getLogger("prc.cache")

#: Bumped by hand when the *layout* of a cached payload changes in a way the key inputs
#: would not otherwise notice. Part of every key.
CACHE_VERSION = 1

_READY = "READY"
_CHUNK = 1 << 20


def cache_root() -> Path:
    """``data/cache/`` unless ``PRC_CACHE_DIR`` says otherwise."""
    env = os.environ.get("PRC_CACHE_DIR")
    return Path(env).expanduser() if env else paths.DATA / "cache"


def file_hash(path: str | Path) -> str | None:
    """blake2b of a file's bytes, or ``None`` if it is not there.

    Content, not metadata: mtime and size are exactly the kind of proxy that let the
    2026-09-18 null-column blocks through.
    """
    p = Path(path)
    try:
        h = hashlib.blake2b(digest_size=16)
        with open(p, "rb") as f:
            while chunk := f.read(_CHUNK):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def source_hash(*objs: ModuleType | Callable[..., Any]) -> str:
    """Hash of the source text of the modules / functions that produce a cached payload.

    A cache keyed only on *data* survives an edit to the code that transformed it, and then
    serves the old answer. Cheap insurance: the source of the assembly path is an input.
    """
    h = hashlib.blake2b(digest_size=16)
    for o in objs:
        try:
            h.update(inspect.getsource(o).encode())
        except (OSError, TypeError):  # built-in / C module: fall back to its identity
            h.update(repr(o).encode())
    return h.hexdigest()


def key_of(inputs: Mapping[str, Any]) -> str:
    """The cache key: sha256 over the canonical JSON of every input.

    ``default=str`` keeps it total (a stray Path or numpy scalar stringifies rather than
    raising), and ``sort_keys`` makes it independent of dict order.
    """
    blob = json.dumps(dict(inputs), sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:32]


# --------------------------------------------------------------------------------------
# entries
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Entry:
    """One cached payload: a directory named by ``<kind>/<key>``."""

    root: Path
    kind: str
    key: str
    inputs: dict

    @property
    def dir(self) -> Path:
        return self.root / self.kind / self.key

    @property
    def lock_path(self) -> Path:
        return self.root / self.kind / f"{self.key}.lock"

    def path(self, name: str) -> Path:
        return self.dir / name

    def ready(self) -> bool:
        return self.path(_READY).is_file()

    def mark_ready(self) -> None:
        """Written last, after every payload file is in place. Also records the inputs the
        key was taken over, so a human can diff two keys without re-deriving them."""
        atomic_write(
            self.path("inputs.json"),
            lambda p: p.write_text(json.dumps(self.inputs, sort_keys=True, indent=2, default=str)),
        )
        atomic_write(
            self.path(_READY),
            lambda p: p.write_text(json.dumps({"written_at": time.time(), "pid": os.getpid()})),
        )


def atomic_write(path: Path, writer: Callable[[Path], Any]) -> Path:
    """``writer`` fills a unique temp file beside ``path``; then one ``os.replace``.

    A reader therefore sees either the previous file or the complete new one, never a
    half-flushed parquet — which on a shared filesystem with concurrent array tasks is the
    difference between a cache and a corruption.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        writer(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return path


@contextmanager
def locked(entry: Entry, timeout_s: float = 3600.0) -> Iterator[bool]:
    """Exclusive flock on the entry's sidecar, for the duration of a cold computation.

    Yields True if the lock was taken, False on timeout (the caller then just computes
    without the cache — a slow cache is never a reason to fail a run). Readers of a ready
    entry never come here.
    """
    entry.lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_s
    got = False
    with open(entry.lock_path, "a+") as fh:
        try:
            while True:
                try:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    got = True
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        log.warning(
                            "cache lock %s busy for %.0fs; computing without it",
                            entry.lock_path,
                            timeout_s,
                        )
                        break
                    time.sleep(0.5)
            yield got
        finally:
            if got:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


# --------------------------------------------------------------------------------------
# payload helpers
# --------------------------------------------------------------------------------------


def write_frame(entry: Entry, name: str, df: pl.DataFrame) -> None:
    atomic_write(entry.path(name), lambda p: df.write_parquet(p, compression="zstd"))


def read_frame(entry: Entry, name: str) -> pl.DataFrame:
    return pl.read_parquet(entry.path(name))


def _savez(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    # np.savez appends ".npz" to a path that does not end in it, which would defeat the
    # temp-file-then-rename dance (the temp name ends in ".tmp"). Hand it a file object.
    with open(path, "wb") as fh:
        np.savez(fh, **dict(arrays))


def write_arrays(entry: Entry, name: str, arrays: Mapping[str, np.ndarray]) -> None:
    atomic_write(entry.path(name), lambda p: _savez(p, arrays))


def read_arrays(entry: Entry, name: str) -> dict[str, np.ndarray]:
    with np.load(entry.path(name), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def write_json(entry: Entry, name: str, obj: Any) -> None:
    atomic_write(entry.path(name), lambda p: p.write_text(json.dumps(obj, default=str)))


def read_json(entry: Entry, name: str) -> Any:
    return json.loads(entry.path(name).read_text())


# --------------------------------------------------------------------------------------
# key builders
# --------------------------------------------------------------------------------------


def block_inputs(names: Sequence[str]) -> list[dict] | None:
    """One record per feature block: its name, the hash of its **source**, and the hash of
    the **bytes** of the parquet it is materialised as.

    The parquet is found by globbing ``<name>__<source hash>__*.parquet`` rather than by
    re-deriving the movement table's input digest, so the key can be built without loading
    the movement table — which is the point, since a hit is meant to skip loading it.
    Anything other than exactly one match (none built yet, or two stale digests lying
    around) returns ``None`` and **disables the cache**: a key that cannot see one of its
    inputs is the 2026-09-18 null-column bug, and the safe answer is to recompute.
    """
    from prc.features.registry import REGISTRY

    out: list[dict] = []
    for n in names:
        block = REGISTRY[n]
        sh = block.source_hash()
        found = sorted(paths.FEATURES.glob(f"{n}__{sh}__*.parquet"))
        if len(found) != 1:
            log.info(
                "cache disabled: feature block %r has %d materialised parquet(s) matching "
                "%s__%s__*.parquet (need exactly 1)",
                n,
                len(found),
                n,
                sh,
            )
            return None
        h = file_hash(found[0])
        if h is None:
            log.info("cache disabled: cannot read %s", found[0])
            return None
        out.append({"name": n, "source": sh, "parquet": found[0].name, "blake2b": h})
    return out


def frames_inputs(cfg, assemble: Callable[..., Any]) -> dict | None:
    """Every input of the assembled train/test frames. ``None`` disables the cache.

    The ``code`` entry hashes the source of **everything the assembly calls**, not only
    ``_assemble``'s own body. ``inspect.getsource`` does not follow calls, and the 2026-09-19
    code review (F11) found the consequence: the entry also stores ``clf_extra_tr/te``, built
    by ``run.build_clf_extras`` -> :mod:`prc.mixture.clf_features`, and neither was in the
    key — so editing ``signals()`` or ``identity()`` served stale head tables from cache. That
    is worse than a plain miss, because the head's *own* key does hash ``clf_features``, so
    the head would recompute faithfully from the wrong input. Exactly the null-column class
    this module exists to prevent.
    """
    from prc.cv import folds as folds_mod
    from prc.cv.folds import SCHEMES
    from prc.data import load as load_mod
    from prc.eval import run as run_mod
    from prc.features import registry as registry_mod
    from prc.mixture import clf_features

    corr = cfg.corruption()
    extra_blocks = list(corr.extra_blocks) if corr else []
    blocks = block_inputs(list(cfg.features))
    mask_blocks = block_inputs(list(cfg.mask_for_no_nm))
    extra = block_inputs(extra_blocks)
    if blocks is None or mask_blocks is None or extra is None:
        return None
    mv_hash = file_hash(paths.INTERIM / "movements.parquet")
    if mv_hash is None:
        log.info("cache disabled: data/interim/movements.parquet not readable")
        return None
    scheme = cfg.cv.scheme
    return {
        "v": CACHE_VERSION,
        "kind": "frames",
        "movements_blake2b": mv_hash,
        "feature_blocks": blocks,
        "mask_for_no_nm": list(cfg.mask_for_no_nm),
        "mask_block_inputs": mask_blocks,
        "target_encoding": list(cfg.target_encoding),
        "te_alpha": cfg.te_alpha,
        "te_stats": list(cfg.te_stats),
        "cv_scheme": scheme,
        "cv_scheme_def": repr(SCHEMES.get(scheme)),
        "cv_group_by_airport": cfg.cv.group_by_airport,
        "folds_blake2b": file_hash(paths.FOLDS / f"{scheme}.parquet"),
        "sample_frac": cfg.sample_frac,
        "target_mode": cfg.target.mode,
        "target_split_groups": cfg.target.split_groups,
        "uses_regimes": cfg.uses_regimes(),
        "corruption_present": corr is not None,
        # the head-only tables are cached alongside the frames, so their spec is part of
        # this key too — they are what lets a hit skip loading the movement table at all
        "clf_extra_blocks": extra_blocks,
        "clf_extra_block_inputs": extra,
        "clf_extra_signals": bool(corr and corr.extra_signals),
        "clf_identity_signals": bool(corr and corr.identity_signals),
        "sample_frac_scope": cfg.sample_frac_scope,
        "code": source_hash(
            assemble,
            load_mod,
            run_mod.build_clf_extras,
            run_mod._align,
            clf_features,
            folds_mod,
            registry_mod,
        ),
    }


def te_inputs(frames_key: str, cfg, fold: int | str, role: str) -> dict:
    """Key of one fold-local target-encoding table. Anchored on the frames key, so the rows
    it was fitted on are part of it by construction."""
    from prc.features import target_enc

    return {
        "v": CACHE_VERSION,
        "kind": "te",
        "frames_key": frames_key,
        "groups": list(cfg.target_encoding),
        "alpha": cfg.te_alpha,
        "stats": list(cfg.te_stats),
        "fold": fold,
        "role": role,
        "code": source_hash(target_enc),
    }


def head_inputs(frames_key: str, cfg, head_seed: int) -> dict:
    """Key of the corruption head's outputs.

    Deliberately **excludes** ``model.params`` / ``num_boost_round`` /
    ``early_stopping_rounds`` on the classifier path: :func:`prc.mixture.head.classifier_config`
    reads only the engine (``kind``) and the ``device`` from the regressor's config, so the
    head of ``exp0065..0071`` — seven capacity variants of R4 — is literally the same
    computation and must hit one entry.

    Under ``sched_head: z_regression`` it puts them back, because ``_nested_yhat`` fits the
    *normal regressor* with those exact parameters to build the head's target.
    """
    from prc.mixture import clf_features, corruption, head, regimes

    corr = cfg.corruption()
    assert corr is not None
    z = corr.sched_head == "z_regression"
    return {
        "v": CACHE_VERSION,
        "kind": "head",
        "frames_key": frames_key,
        "corruption": corr.model_dump(exclude={"rollover_weight"}),  # regressor-only weight, not a head input
        "model_kind": cfg.model.kind,
        "model_device": cfg.model.device,
        "head_seed": head_seed,
        "target_transform": cfg.target.transform,
        "target_mode": cfg.target.mode,
        # `test_fit: full` adds one payload to the entry — the expert's weights over ALL
        # training rows (`HeadOutputs.weights_all`) — so an entry written by a `folds` run
        # must not be served to a `full` one. It is the payload, not the head, that differs:
        # every probability in the entry is identical either way.
        "weights_all": cfg.test_fit == "full" or z,
        # only the z head reads the normal regressor's capacity (via _nested_yhat)
        "regressor": (
            {
                "params": cfg.model.params,
                "num_boost_round": cfg.model.num_boost_round,
                "early_stopping_rounds": cfg.model.early_stopping_rounds,
            }
            if z
            else None
        ),
        "code": source_hash(head, corruption, clf_features, regimes),
    }


def entry(kind: str, inputs: Mapping[str, Any]) -> Entry:
    return Entry(root=cache_root(), kind=kind, key=key_of(inputs), inputs=dict(inputs))
