"""Run ONE experiment cell, with the overrides a job array needs.

``uv run prc train <yaml>`` takes only ``--out``: seeds, CV scheme and feature list
all come from the immutable YAML. A seed farm, an ablation or a ``month_loo`` array
needs to vary exactly those, one array task at a time, without editing a config that
has already produced a registry row. This wrapper does that outside ``src/prc``.

It changes no modelling behaviour: it loads the YAML with ``ExperimentConfig``,
applies overrides with ``model_copy``, renames the run so its registry row and output
dir are unique, and calls ``prc.eval.run.run_experiment`` — the same entry point
``prc train`` calls.

Examples (one array task each)::

    uv run python scripts/run_cell.py --config experiments/exp0227_r7c_realmlp_nomask.yaml \
        --seed 1 --scheme month_loo --out outputs/runs/loo-exp0227_r7c_realmlp_nomask
    uv run python scripts/run_cell.py --config experiments/exp0227_r7c_realmlp_nomask.yaml \
        --seed 1 --drop-block winter_ops --out outputs/runs/ablation-exp0227

``--dry-run`` resolves and validates everything (config, blocks, scheme, fold count,
output dir) and prints the plan without loading data or fitting anything.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

# Run from the repo root; make `src/` importable without an editable install step
# in case a job shell lost it.
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))


@contextmanager
def _registry_lock(path: Path):
    """Serialise ``experiments/registry.csv`` appends across concurrent array tasks.

    ``prc.eval.run.append_registry`` opens the CSV in append mode and writes one row.
    With a 12-task array all finishing near the same time on a shared filesystem, that
    is a real (if small) interleaving risk on a TRACKED file. We take an exclusive
    flock on a sidecar lock file around the append. ``metrics.json`` in each run dir
    stays the authoritative record either way.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _install_registry_lock() -> None:
    import prc.eval.run as run_mod
    from prc import paths

    inner = run_mod.append_registry
    lock_path = paths.EXPERIMENTS / ".registry.lock"

    def locked(metrics: dict, cfg) -> None:
        with _registry_lock(lock_path):
            inner(metrics, cfg)

    run_mod.append_registry = locked


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", required=True, help="base experiment YAML (never modified)")
    p.add_argument("--out", required=True, help="suite output root; the cell goes in <out>/<name>")
    p.add_argument("--seed", type=int, help="single seed; replaces the YAML's seeds list")
    p.add_argument("--seeds", help="comma-separated seeds; replaces the YAML's seeds list")
    p.add_argument("--scheme", help="CV scheme override (prc.cv.folds.SCHEMES)")
    p.add_argument("--sample-frac", type=float, help="subsample train rows (iteration only)")
    p.add_argument("--drop-block", help="feature block to remove (leave-one-out ablation)")
    p.add_argument("--only-blocks", help="comma-separated feature blocks to keep")
    p.add_argument("--name-suffix", help="extra suffix appended to the run name")
    p.add_argument("--dry-run", action="store_true", help="resolve and validate; do not fit")
    return p


def resolve(args: argparse.Namespace):
    from prc.config import ExperimentConfig
    from prc.cv.folds import SCHEMES, n_folds
    from prc.features import REGISTRY

    cfg = ExperimentConfig.load(args.config)
    updates: dict = {}
    parts: list[str] = []

    if args.seed is not None and args.seeds:
        raise SystemExit("pass --seed or --seeds, not both")
    if args.seed is not None:
        updates["seeds"] = [args.seed]
        parts.append(f"s{args.seed}")
    elif args.seeds:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        if len(set(seeds)) != len(seeds):
            raise SystemExit(f"duplicate seed in --seeds {args.seeds!r}")
        updates["seeds"] = seeds
        parts.append("s" + "-".join(str(s) for s in seeds))

    if args.drop_block and args.only_blocks:
        raise SystemExit("pass --drop-block or --only-blocks, not both")
    if args.drop_block:
        if args.drop_block not in cfg.features:
            raise SystemExit(
                f"--drop-block {args.drop_block!r} is not in {args.config}: {cfg.features}"
            )
        updates["features"] = [f for f in cfg.features if f != args.drop_block]
        parts.append(f"drop-{args.drop_block}")
    elif args.only_blocks:
        keep = [b.strip() for b in args.only_blocks.split(",") if b.strip()]
        unknown = [b for b in keep if b not in REGISTRY]
        if unknown:
            raise SystemExit(f"unknown feature block(s) {unknown}; see `uv run prc features list`")
        updates["features"] = keep
        parts.append("only-" + "+".join(keep))

    if args.scheme:
        if args.scheme not in SCHEMES:
            raise SystemExit(f"unknown cv scheme {args.scheme!r}; known: {sorted(SCHEMES)}")
        updates["cv"] = cfg.cv.model_copy(update={"scheme": args.scheme})
        parts.append(args.scheme)

    if args.sample_frac is not None:
        updates["sample_frac"] = args.sample_frac
        parts.append(f"frac{args.sample_frac:g}")

    if args.name_suffix:
        parts.append(args.name_suffix)

    if parts:
        updates["name"] = cfg.name + "__" + "__".join(parts)

    cfg = cfg.model_copy(update=updates)
    missing = [f for f in cfg.features if f not in REGISTRY]
    if missing:
        raise SystemExit(f"unknown feature block(s) {missing}; see `uv run prc features list`")
    return cfg, n_folds(cfg.cv.scheme)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg, folds = resolve(args)
    out_dir = Path(args.out) / cfg.name

    plan = {
        "name": cfg.name,
        "base_config": args.config,
        "out_dir": str(out_dir),
        "scheme": cfg.cv.scheme,
        "folds": folds,
        "seeds": cfg.seeds,
        "fits": folds * len(cfg.seeds),
        "n_feature_blocks": len(cfg.features),
        "features": cfg.features,
        "sample_frac": cfg.sample_frac,
        "model": cfg.model.kind,
        "device": cfg.model.device,
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        print("DRY RUN: nothing fitted, nothing written.", flush=True)
        return 0

    if out_dir.joinpath("metrics.json").is_file():
        print(f"SKIP (already complete): {out_dir}/metrics.json", flush=True)
        return 0

    _install_registry_lock()
    from prc.eval.run import run_experiment

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = run_experiment(cfg, out_dir)
    print(
        json.dumps(
            {
                "name": metrics["name"],
                "cv_rmse": metrics["cv_rmse"],
                "per_fold": metrics["per_fold"],
                "elapsed_s": metrics["elapsed_s"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
