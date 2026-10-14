"""Resolve the performance switches of :class:`prc.config.RuntimeConfig` against the
environment, and answer the two questions the runner asks about the machine it is on.

Why an environment override at all: ``experiments/*.yaml`` are immutable once they have
produced a registry row, but how fast a cell runs is a property of the *node*,
not of the experiment. A packed suite on four L40S GPUs and the same YAML on a single-GPU
workstation want different ``fit_workers``. So every switch reads ``PRC_<UPPER_NAME>`` first and falls
back to the YAML, and the switches are excluded from the config digest
(:meth:`prc.config.ExperimentConfig.digest`) precisely so that this cannot change what a
run *is*.

Nothing in this module may change a prediction. The one thing that comes close is
``devices``: a fit dispatched to ``cuda:1`` instead of ``cuda:0`` runs on a different card.
XGBoost's GPU ``hist`` is deterministic at a fixed seed on a *given* device; whether two
physically identical L40S cards agree bit-for-bit is measured by the packing benchmark, not
assumed here.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

from prc.config import ExperimentConfig, RuntimeConfig

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None or not v.strip():
        return default
    s = v.strip().lower()
    if s in _TRUE:
        return True
    if s in _FALSE:
        return False
    raise ValueError(f"{name}={v!r} is not a boolean (use one of {sorted(_TRUE | _FALSE)})")


def _env_int(name: str, default: int | None) -> int | None:
    v = os.environ.get(name)
    if v is None or not v.strip():
        return default
    return int(v)


def visible_gpus() -> int:
    """How many GPUs this process may actually use.

    ``CUDA_VISIBLE_DEVICES`` wins when it is set, because that is what a packed suite uses
    to give each worker one card — asking ``nvidia-smi`` there would report the whole node
    and every worker would think it owned four GPUs.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        entries = [e for e in cvd.split(",") if e.strip() != ""]
        return len(entries)
    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return 0
    return sum(1 for line in out.stdout.splitlines() if line.startswith("GPU"))


@dataclass(frozen=True)
class Runtime:
    """The resolved switches, plus the device list a parallel fit pool dispatches over."""

    reuse_fold_matrix: bool
    cache_features: bool
    cache_head: bool
    fit_workers: int
    n_threads: int | None
    devices: tuple[str, ...]

    @property
    def parallel_fits(self) -> bool:
        return self.fit_workers > 1 and len(self.devices) > 0

    def as_dict(self) -> dict:
        return {
            "reuse_fold_matrix": self.reuse_fold_matrix,
            "cache_features": self.cache_features,
            "cache_head": self.cache_head,
            "fit_workers": self.fit_workers,
            "n_threads": self.n_threads,
            "devices": list(self.devices),
        }


def resolve(cfg: ExperimentConfig | None = None, device: str | None = None) -> Runtime:
    """Config -> environment -> the machine. ``device`` is ``cfg.model.device``.

    ``fit_workers`` is clamped to what exists: on a cuda config it does not exceed the
    number of visible GPUs unless ``PRC_ALLOW_GPU_OVERSUBSCRIBE=1`` says to over-pack on
    purpose; on a cpu config the "devices" are just N slots of ``cpu``.
    """
    rc: RuntimeConfig = cfg.runtime if cfg is not None else RuntimeConfig()
    device = device if device is not None else (cfg.model.device if cfg is not None else "cpu")

    reuse = _env_bool("PRC_REUSE_FOLD_MATRIX", rc.reuse_fold_matrix)
    cache_features = _env_bool("PRC_CACHE_FEATURES", rc.cache_features)
    cache_head = _env_bool("PRC_CACHE_HEAD", rc.cache_head)
    workers = _env_int("PRC_FIT_WORKERS", rc.fit_workers) or 1
    threads = _env_int("PRC_N_THREADS", rc.n_threads)
    if workers < 1:
        raise ValueError(f"fit_workers must be >= 1, got {workers}")
    if threads is not None and threads < 1:
        raise ValueError(f"n_threads must be >= 1, got {threads}")

    if device == "cuda":
        n_gpu = visible_gpus()
        if not n_gpu:
            workers = 1
            devices = ("cuda:0",)
        elif _env_bool("PRC_ALLOW_GPU_OVERSUBSCRIBE", False):
            # Deliberate over-packing: more fit processes than cards, dealt round-robin.
            # A ~100-feature `hist` problem does not saturate an L40S, so two processes on
            # one card can beat one — that is a measurement (a packing benchmark),
            # and it is also how a single-GPU workstation exercises the parallel path at all.
            devices = tuple(f"cuda:{i % n_gpu}" for i in range(workers))
        else:
            workers = max(1, min(workers, n_gpu))
            devices = tuple(f"cuda:{i}" for i in range(workers))
    else:
        devices = tuple("cpu" for _ in range(workers))

    return Runtime(
        reuse_fold_matrix=reuse,
        cache_features=cache_features,
        cache_head=cache_head,
        fit_workers=workers,
        n_threads=threads,
        devices=devices,
    )
