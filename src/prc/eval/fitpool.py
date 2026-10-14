"""One fold's regressor fits: the single implementation, run either in this process or
across one worker process per GPU.

The 2026-09-18 baseline of ``exp0047`` (3 seeds x 2 folds, one L40S, 8 cores) spent ~300 s
of its 361 s on **six sequential GPU fits**, on a node that has four GPUs. Nothing about
those fits is sequential in nature: a fit is a pure function of (fold matrix, seed, regime
group), they share no state, and the fold's training rows and weights are computed before
the seed loop begins. They were sequential because the loop was.

:func:`fit_one` is that pure function. :func:`run_fold_fits` runs a list of
``(seed, group)`` tasks over a ``fork`` process pool with one process per device, and
``pool.map`` hands the results back **in task order**, so the assembly into ``oof`` /
``test_pred`` happens in exactly the order the sequential loop would have used. The
sequential path calls the same :func:`fit_one`, so "parallel" and "sequential" are not two
implementations that have to be kept in agreement — they are one implementation and a
dispatch choice.

Why ``fork`` and module-level state rather than arguments: the context holds the fold's
design matrices, which at full size are gigabytes. Passing them as ``initargs`` would
pickle and copy them per worker; setting the module global in the parent *before* the pool
is created lets every child inherit them copy-on-write, which is free until something
writes. Nothing in :func:`fit_one` writes to the context.

A fit is a function of (fold, seed, regime group) and, since 2026-09-20, of an **airport
unit** as well (``cv.group_by_airport``): :func:`unit_masks` is the one place that turns a
task into the rows it owns, and with ``task.aps is None`` — the default — it returns exactly
the masks this module used before the option existed. :func:`fit_full` is the same pure
function without a fold: all training rows, a fixed number of rounds, no validation set and
no early stopping (``test_fit: full``).

THE CONSTRAINT THAT SHAPES ALL OF THIS (measured 2026-09-19)
------------------------------------------------------------
Two deadlocks, both reproduced in five lines on this machine:

1. **A process that has run one XGBoost fit hangs if it then forks a child that trains.**
   The child inherits an OpenMP runtime it cannot re-enter. ``nthread=1`` in the parent does
   not help; ``nthread=1`` in the child does not help. ``spawn`` and ``forkserver`` both
   work — and both pickle every task argument, which for a fold's design matrices is exactly
   the copy ``fork`` exists to avoid.
2. **A process that has done polars work hangs if it then forks a child that does polars
   work.** Same shape, rayon's thread pool instead of OpenMP.

Which is why the fits, and only the fits, are forked: :func:`fit_one` touches pandas, numpy
and the booster — never polars. The parent may have done all the polars it likes (loading,
joining, masking, target encodings) and a child that stays out of polars is fine; this is
the configuration that ran clean end to end at smoke size.

The remaining rule is (1): **the parent must not have trained.** The corruption head is the
only training the runner does before the fold loop, and it *does* use polars, so it cannot
be pushed into a forked child either. So with a corruption head, multi-GPU fit dispatch
wants ``PRC_CACHE_HEAD=1`` — the head then comes off disk and the parent stays clean.
:func:`run_fold_fits` checks :func:`prc.models.booster_built_here` before forking and falls
back to the sequential path with a loud, actionable warning rather than hanging: a wedged
four-GPU allocation is a far worse outcome than a slow one.

The one thing that is not identical between the two paths is the physical device: worker
``i`` fits on ``cuda:i``. XGBoost's GPU ``hist`` is deterministic at a fixed seed on a given
device; whether two cards of the same model agree bit-for-bit is a measurement, not an
assumption — see internal note 2026-09-19_runner-efficiency §3, and the
``bench-packing`` job that takes it.
"""

from __future__ import annotations

import gc
import logging
import multiprocessing as mp
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from prc.config import ExperimentConfig
from prc.eval.transform import inverse as _inverse
from prc.eval.transform import transform as _transform
from prc.models import booster_built_here, make_model


@dataclass
class FoldContext:
    """Everything one fold's fits read. Built once per fold, never mutated."""

    cfg: ExperimentConfig
    groups: list[list[str]]
    X_fit: pd.DataFrame
    X_va: pd.DataFrame | None  # None on the full-fit path, which has no validation set
    X_test: pd.DataFrame
    y_res: np.ndarray
    fit_rows: np.ndarray  # boolean over all training rows
    va: np.ndarray  # boolean over all training rows
    regime_tr: np.ndarray
    regime_te: np.ndarray
    w_fit: np.ndarray | None  # aligned to fit_rows
    n_threads: int | None = None
    #: Airport of every training / test row, for `cv.group_by_airport`. None when the option
    #: is off, and then no task carries an `aps` either.
    airport_tr: np.ndarray | None = None
    airport_te: np.ndarray | None = None
    #: `validation.early_stop: inner` (2026-09-23): boolean aligned to ``fit_rows`` (the rows
    #: of ``X_fit``) marking the held-out training airport-days the fold models early-stop
    #: on; they are kept out of the fit. None (the default) = early-stop on the fold's
    #: validation rows, the pre-2026-09-23 path.
    es_rows: np.ndarray | None = None


def inner_holdout_mask(block_key: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Rows whose block (an airport-day key) falls in a seeded random ``frac`` of the blocks.

    The draw is over the **sorted unique** keys, so it depends on the set of keys and the
    seed only — not on row order, the fold or the model seed. Every fold that trains on an
    airport-day therefore makes the same in/out decision for it."""
    keys, inv = np.unique(np.asarray(block_key), return_inverse=True)
    u = np.random.default_rng(seed).random(len(keys))
    return (u < frac)[inv]


@dataclass
class FitTask:
    seed: int
    gi: int
    device: str | None = None
    #: The airports this model owns (`cv.group_by_airport`), or None for the global model.
    #: A one-element tuple is "this airport's own model"; the longer tuple a hybrid config
    #: leaves over is the shared `rest` model.
    aps: tuple[str, ...] | None = None


@dataclass
class FitResult:
    """A fit's contribution, in the shapes the runner assembles from."""

    seed: int
    gi: int
    aps: tuple[str, ...] | None = None  # echoed from the task, so results key back to it
    g_va_pred: np.ndarray | None = None  # predictions on this group's validation rows
    g_te_pred: np.ndarray | None = None  # ... and on its test rows
    fallback: float | None = None  # set instead, when the group had no training rows
    best_iteration: int | None = None
    importance: pd.Series | None = None
    timings: dict[str, float] = field(default_factory=dict)
    elapsed_s: float = 0.0
    n_fit_rows: int = 0


def unit_masks(ctx: FoldContext, task: FitTask) -> tuple[np.ndarray, np.ndarray]:
    """The training and test rows one fit owns: its regime group, narrowed to its airports.

    ``task.aps is None`` is the pre-2026-09-20 path and returns exactly the two
    ``np.isin(regime, grp)`` masks the runner used before ``cv.group_by_airport`` existed.
    """
    grp = ctx.groups[task.gi]
    gm = np.isin(ctx.regime_tr, grp)
    gm_te = np.isin(ctx.regime_te, grp)
    if task.aps is None:
        return gm, gm_te
    if ctx.airport_tr is None or ctx.airport_te is None:
        raise ValueError("cv.group_by_airport needs the airport of every row in the context")
    return gm & np.isin(ctx.airport_tr, task.aps), gm_te & np.isin(ctx.airport_te, task.aps)


def fit_one(
    ctx: FoldContext,
    task: FitTask,
    matrices: Any | None = None,
    key: str | None = None,
) -> FitResult:
    """Fit one (seed, regime group) of this fold and predict its validation and test rows.

    ``matrices`` / ``key`` are the per-fold matrix cache (:mod:`prc.models.matrices`); both
    ``None`` is the pre-2026-09-19 path, which rebuilds every matrix inside every fit.
    """
    cfg = ctx.cfg
    transform = cfg.target.transform
    gm, gm_te = unit_masks(ctx, task)
    g_fit, g_va = gm[ctx.fit_rows], gm[ctx.va]
    t0 = time.time()
    res = FitResult(seed=task.seed, gi=task.gi, aps=task.aps, n_fit_rows=int(g_fit.sum()))

    if g_fit.sum() == 0:  # nothing to learn from: fall back to the fold's mean
        res.fallback = float(np.mean(ctx.y_res[ctx.fit_rows])) if ctx.fit_rows.sum() else 0.0
        res.elapsed_s = time.time() - t0
        return res

    assert ctx.X_va is not None, "fit_one is the fold path; the full fit calls fit_full"
    X_fit, X_va, X_test = ctx.X_fit, ctx.X_va, ctx.X_test
    if g_va.any():
        Xg_va = X_va if g_va.all() else X_va.reset_index(drop=True)[g_va]
        yg_va = _transform(ctx.y_res[ctx.va][g_va], transform)
    w_g = None if ctx.w_fit is None else ctx.w_fit[g_fit]
    g_es = None if ctx.es_rows is None else g_fit & ctx.es_rows
    if g_es is not None and g_es.any() and (g_fit & ~g_es).any():
        # `validation.early_stop: inner`: the held-out training airport-days are the
        # early-stopping set and stay out of the fit; the validation month is only predicted
        g_in = g_fit & ~g_es
        X_reset = X_fit.reset_index(drop=True)
        y_fit = ctx.y_res[ctx.fit_rows]
        Xg_tr, yg_tr = X_reset[g_in], _transform(y_fit[g_in], transform)
        Xg_es, yg_es = X_reset[g_es], _transform(y_fit[g_es], transform)
        w_g = None if ctx.w_fit is None else ctx.w_fit[g_in]
        res.n_fit_rows = int(g_in.sum())
        del X_reset
    else:
        # the default path. (Under `inner`, a group with no held-out day, or nothing else,
        # early-stops on its own training rows rather than on the scored month.)
        Xg_tr = X_fit if g_fit.all() else X_fit.reset_index(drop=True)[g_fit]
        yg_tr = _transform(ctx.y_res[ctx.fit_rows][g_fit], transform)
        if g_va.any() and g_es is None:
            Xg_es, yg_es = Xg_va, yg_va
        else:  # no validation rows in this fold (or `inner`): early-stop on its own rows
            Xg_es, yg_es = Xg_tr, yg_tr
    m = make_model(cfg.model, task.seed, matrices, task.device, ctx.n_threads)
    m.fit(Xg_tr, yg_tr, Xg_es, yg_es, w_g, key=key)
    if g_va.any():
        res.g_va_pred = _inverse(m.predict(Xg_va, key=key), transform)
    if gm_te.any():
        Xg_te = X_test if gm_te.all() else X_test[gm_te].reset_index(drop=True)
        res.g_te_pred = _inverse(
            m.predict(Xg_te, key=None if key is None else key + "/te"), transform
        )
        del Xg_te
    res.best_iteration = m.best_iteration
    res.importance = m.importance()
    res.timings = dict(getattr(m, "timings", {}) or {})
    del m, Xg_tr, yg_tr, Xg_es, yg_es
    if g_va.any():
        del Xg_va, yg_va
    gc.collect()
    res.elapsed_s = time.time() - t0
    return res


def fit_full(ctx: FoldContext, task: FitTask, rounds: int) -> FitResult:
    """One ``test_fit: full`` model: all training rows, ``rounds`` rounds, no validation set.

    The same pure function as :func:`fit_one` minus everything that is about a fold —
    there is no validation set, so no early stopping and no ``g_va_pred``. ``ctx.X_fit`` is
    the full design matrix (block features plus the target encodings fitted on all training
    rows, which is what the test rows already get), ``ctx.fit_rows`` the rows the expert is
    fitted on (all of them, or the clean ones under ``train_normal_on: clean``) and
    ``ctx.w_fit`` their weights. ``rounds`` comes from the fold models — see
    ``prc.eval.run._full_fit_rounds``.
    """
    cfg = ctx.cfg
    transform = cfg.target.transform
    gm, gm_te = unit_masks(ctx, task)
    g_fit = gm[ctx.fit_rows]
    t0 = time.time()
    res = FitResult(seed=task.seed, gi=task.gi, aps=task.aps, n_fit_rows=int(g_fit.sum()))
    if g_fit.sum() == 0 or not gm_te.any():
        res.fallback = float(np.mean(ctx.y_res[ctx.fit_rows])) if ctx.fit_rows.sum() else 0.0
        res.elapsed_s = time.time() - t0
        return res
    X_fit = ctx.X_fit
    Xg_tr = X_fit if g_fit.all() else X_fit.reset_index(drop=True)[g_fit]
    yg_tr = _transform(ctx.y_res[ctx.fit_rows][g_fit], transform)
    model_cfg = cfg.model.model_copy(update={"num_boost_round": int(rounds)})
    m = make_model(model_cfg, task.seed, None, task.device, ctx.n_threads)
    m.fit(Xg_tr, yg_tr, None, None, None if ctx.w_fit is None else ctx.w_fit[g_fit])
    Xg_te = ctx.X_test if gm_te.all() else ctx.X_test[gm_te].reset_index(drop=True)
    res.g_te_pred = _inverse(m.predict(Xg_te), transform)
    res.best_iteration = m.best_iteration
    res.importance = m.importance()
    res.timings = dict(getattr(m, "timings", {}) or {})
    del m, Xg_tr, yg_tr, Xg_te
    gc.collect()
    res.elapsed_s = time.time() - t0
    return res


# --------------------------------------------------------------------------------------
# parallel dispatch
# --------------------------------------------------------------------------------------

log = logging.getLogger("prc.fitpool")

_CTX: FoldContext | None = None


def _worker(task: FitTask) -> FitResult:
    assert _CTX is not None, "fold context not inherited; run_fold_fits must fork the pool"
    return fit_one(_CTX, task)


def can_fork() -> bool:
    """Is it safe for this process to fork a child that will train?

    False once anything here has built a booster — see constraint (1) in the module
    docstring. This is the guard that turns a deadlock into a slow run.
    """
    return not booster_built_here()


def run_fold_fits(ctx: FoldContext, tasks: list[FitTask], workers: int) -> list[FitResult]:
    """Run ``tasks`` over ``workers`` forked processes, results in task order.

    No matrix cache is passed to the children: each one builds the matrices it needs for the
    tasks it picks up, in its own address space. Reuse across seeds and reuse across
    processes are alternatives, not complements — which of the two wins on a 4-GPU node is
    what a packing benchmark measures.
    """
    global _CTX
    if workers <= 1 or len(tasks) <= 1:
        return [fit_one(ctx, t) for t in tasks]
    if not can_fork():
        log.warning(
            "fit_workers=%d requested, but this process has already trained a booster "
            "(the corruption head): forking now would deadlock, so the %d fits of this "
            "fold run SEQUENTIALLY. Set PRC_CACHE_HEAD=1 so the head comes off disk and "
            "the process stays a clean fork source. See prc.eval.fitpool.",
            workers,
            len(tasks),
        )
        return [fit_one(ctx, t) for t in tasks]
    _CTX = ctx
    try:
        mp_ctx = mp.get_context("fork")
        with mp_ctx.Pool(processes=min(workers, len(tasks))) as pool:
            return list(pool.map(_worker, tasks, chunksize=1))
    finally:
        _CTX = None


# --------------------------------------------------------------------------------------
# spawn dispatch for the torch members (2026-09-22)
# --------------------------------------------------------------------------------------
#
# ``run_fold_fits`` above forks, which is right for the boosters and wrong for torch: a
# process that has initialised CUDA cannot fork a child that uses CUDA, and on a large GPU a
# TabM cell is GPU-idle while the parent builds each fold's matrix and while the model's own
# quantile transform runs. So the torch members get a second dispatcher, over the SAME pure
# functions (``fit_one`` / ``fit_full``): a ``spawn`` ``ProcessPoolExecutor`` whose workers
# all share one GPU, fed one (fold, seed) fit at a time.
#
# * The fold's :class:`FoldContext` is spilled ONCE to a pickle (protocol 5, i.e. raw array
#   buffers) under a private temp dir, and each task carries only its path. A worker loads a
#   context once and keeps it for the next task of the same fold; ``X_test``, which is the
#   same object for every fold, is spilled once per run and kept by each worker. Nothing is
#   copied through a pipe, and the parent drops the fold's frames as soon as they are
#   spilled — so it can build the next fold while the workers train.
# * ``spawn`` children re-import ``prc`` and so start with the default ``prc.paths``; the
#   parent's layout (which tests redirect) is handed over explicitly, as the equivalence
#   harness does.
# * The fit a worker runs is the fit the sequential loop runs; the runner assembles the
#   results in fold, then task order. On CPU that makes the two paths bit-identical;
#   on CUDA each fit is only reproducible to float noise
#   anyway (cuBLAS / embedding-backward), and the pool adds nothing to that noise.

#: The torch members, the only kinds :class:`SpawnFitPool` is used for.
NN_KINDS = ("tabm", "setenc", "realmlp", "modernnca", "seqenc")

#: ``prc.paths`` attributes a spawned worker inherits from its parent.
_PATH_ATTRS = (
    "ROOT",
    "DATA",
    "OUT",
    "RAW",
    "INTERIM",
    "FEATURES",
    "EXTERNAL",
    "FOLDS",
    "RUNS",
    "SUBMISSIONS",
    "EXPERIMENTS",
)


@dataclass(frozen=True)
class NNRuntime:
    """The resolved torch-member switches (``runtime.nn_*`` / ``PRC_NN_*``)."""

    fold_workers: int = 1
    prep_threads: int = 0
    spill_dir: str | None = None

    def as_dict(self) -> dict:
        return {
            "nn_fold_workers": self.fold_workers,
            "nn_prep_threads": self.prep_threads,
            "nn_spill_dir": self.spill_dir,
        }


def resolve_nn(cfg: ExperimentConfig) -> NNRuntime:
    """Config -> environment, like :func:`prc.runtime.resolve`. The env var wins."""
    import os

    from prc.runtime import _env_int

    rc = cfg.runtime
    workers = _env_int("PRC_NN_FOLD_WORKERS", rc.nn_fold_workers) or 1
    prep = _env_int("PRC_NN_PREP_THREADS", rc.nn_prep_threads) or 0
    spill = os.environ.get("PRC_NN_SPILL_DIR") or rc.nn_spill_dir
    if workers < 1:
        raise ValueError(f"nn_fold_workers must be >= 1, got {workers}")
    if prep < 0:
        raise ValueError(f"nn_prep_threads must be >= 0, got {prep}")
    return NNRuntime(fold_workers=int(workers), prep_threads=int(prep), spill_dir=spill)


def set_prep_threads(n: int) -> None:
    """Hand ``runtime.nn_prep_threads`` to the torch members' shared preprocessing."""
    if n:
        from prc.models import tabm

        tabm.PREP_THREADS = int(n)


# worker-side state: the context of the fold this worker last fitted, and the test frame
_W: dict[str, Any] = {"ctx_path": None, "ctx": None, "test_path": None, "X_test": None}


def _spawn_init(
    path_snapshot: dict[str, str],
    log_file: str | None,
    prep_threads: int,
    torch_threads: int | None,
) -> None:
    from pathlib import Path

    from prc import paths

    for name, value in path_snapshot.items():
        setattr(paths, name, Path(value))
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [nnw %(process)d] %(message)s",
        handlers=handlers,
        force=True,
    )
    set_prep_threads(prep_threads)
    if torch_threads:
        import torch

        torch.set_num_threads(int(torch_threads))


def _load_pickle(path: str) -> Any:
    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)


def _spawn_fit(ctx_path: str, task: FitTask, rounds: int | None) -> FitResult:
    t0 = time.time()
    if _W["ctx_path"] != ctx_path:
        _W["ctx"] = _W["ctx_path"] = None
        gc.collect()
        light, test_path = _load_pickle(ctx_path)
        if _W["test_path"] != test_path:
            _W["X_test"] = None
            gc.collect()
            _W["X_test"] = _load_pickle(test_path)
            _W["test_path"] = test_path
        light.X_test = _W["X_test"]
        _W["ctx"], _W["ctx_path"] = light, ctx_path
        log.info("nn worker: loaded %s in %.1f s", ctx_path, time.time() - t0)
    ctx = _W["ctx"]
    res = fit_one(ctx, task) if rounds is None else fit_full(ctx, task, rounds)
    import sys

    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        # an idle worker must not sit on its predict-time cache while another one trains:
        # on a small card that pushes the pair into (WSL) shared-memory paging, 2-3x slower
        torch.cuda.empty_cache()
    res.timings = {**res.timings, "spawn_load": time.time() - t0 - res.elapsed_s}
    return res


def _dump_pickle(obj: Any, path: Any) -> None:
    import pickle

    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=5)


class SpawnFitPool:
    """``workers`` spawn processes sharing this node's GPU(s), fed (fold context, task) pairs.

    ``put(ctx)`` spills a context and returns its handle; ``submit(handle, task, rounds)``
    queues one fit (``rounds=None`` = :func:`fit_one`, else :func:`fit_full`) and returns a
    ``Future[FitResult]``; ``release(handle)`` deletes the spill once its fits are done.
    """

    def __init__(
        self,
        workers: int,
        spill_dir: str | None = None,
        log_file: str | None = None,
        prep_threads: int = 0,
        torch_threads: int | None = None,
    ) -> None:
        import shutil
        import tempfile
        import weakref
        from concurrent.futures import ProcessPoolExecutor
        from pathlib import Path

        from prc import paths

        self.workers = int(workers)
        self.dir = Path(tempfile.mkdtemp(prefix="prc-nnfold-", dir=spill_dir))
        snapshot = {n: str(getattr(paths, n)) for n in _PATH_ATTRS if hasattr(paths, n)}
        self._ex = ProcessPoolExecutor(
            max_workers=self.workers,
            mp_context=mp.get_context("spawn"),
            initializer=_spawn_init,
            initargs=(snapshot, log_file, int(prep_threads), torch_threads),
        )
        self._test_obj: pd.DataFrame | None = None
        self._test_path: str | None = None
        self._n = 0
        self.spill_s = 0.0
        self._finalizer = weakref.finalize(self, shutil.rmtree, str(self.dir), True)
        log.info(
            "nn fold pool: %d spawn workers, spill dir %s, prep threads %d, torch threads %s",
            self.workers,
            self.dir,
            prep_threads,
            torch_threads,
        )

    def put(self, ctx: FoldContext) -> str:
        import dataclasses

        t0 = time.time()
        if ctx.X_test is not self._test_obj:
            path = self.dir / f"x_test_{self._n}.pkl"
            _dump_pickle(ctx.X_test, path)
            self._test_obj, self._test_path = ctx.X_test, str(path)
        light = dataclasses.replace(ctx, X_test=None)
        path = self.dir / f"ctx_{self._n}.pkl"
        _dump_pickle((light, self._test_path), path)
        self._n += 1
        self.spill_s += time.time() - t0
        return str(path)

    def submit(self, handle: str, task: FitTask, rounds: int | None = None):
        return self._ex.submit(_spawn_fit, handle, task, rounds)

    def release(self, handle: str) -> None:
        import os

        try:
            os.unlink(handle)
        except FileNotFoundError:
            pass

    def close(self) -> None:
        self._ex.shutdown(wait=True, cancel_futures=True)
        self._test_obj = None
        self._finalizer()
