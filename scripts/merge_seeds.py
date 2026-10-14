"""Merge per-seed cell runs into one run dir, and replay their registry rows.

A job array writes one run dir per cell and no ``experiments/registry.csv`` row (concurrent
appends from many nodes are avoided). This script turns those cell dirs into the shapes the
evaluation tools expect:

``merge-seeds <out root>``
    A seed farm array runs ONE seed per cell (``scripts/run_cell.py``), so one config+scheme
    arrives as N sibling directories
    ``<out root>/<exp>__s<seed>[__<other overrides>]/``. ``scripts/paired_delta.py``,
    ``scripts/lane_report.py`` and the blend checks want ONE run dir per experiment holding
    ``oof.parquet`` with ``pred_s<seed>`` columns plus ``pred``, and ``metrics.json`` with
    ``per_seed``. This command verifies that the cells really are the same config, the same
    CV scheme and the same rows (joined on ``MVT_ID_mvt`` — never on row order) and writes
    that merged run dir, with every metric recomputed by the repo's own metric functions
    (``prc.eval.metrics`` / ``prc.eval.lanes``).

``replay-registry <run dir> …``
    Appends the registry row the array did not write, through the runner's own
    :func:`prc.eval.run.append_registry` and the same ``flock`` ``scripts/run_cell.py`` takes.
    Idempotent: a name already in the registry is skipped, and no existing row is ever
    rewritten.

Both take ``--dry-run``.

    uv run python scripts/merge_seeds.py merge-seeds outputs/runs/loo-exp0227_r7c_realmlp_nomask
    uv run python scripts/merge_seeds.py replay-registry \\
        outputs/runs/exp0227_r7c_realmlp_nomask__s1-2-3__month_loo
    uv run python scripts/paired_delta.py <ref> <cand> --bootstrap 2000

**A merged run is not bit-identical to a native multi-seed run.**

*Without a corruption head*, the difference is only the clip: ``run_experiment`` averages the
*raw* per-seed predictions and applies ``target.clip_*`` once, while single-seed cells clip
inside each cell, so the merged ``pred`` is the mean of the clipped per-seed predictions.
Both are the seed mean of the same fits; they differ by ~1e-3 to 3e-2 s of CV RMSE on the
runs measured on 2026-09-19 (largest: ``exp0053_cat_r3_twin``, 0.033 s). ``per_seed`` is exact.

*With a corruption head, that claim was FALSE, and the fix is a config key.* The head used to
be seeded from ``cfg.seeds[0]``, so every single-seed cell of a farm fitted **its own head**
and a merge averaged three different mixtures — a 3-head ensemble against a native run's
1-head model. Measured on the three real ``exp0091`` cells, mean ``p_stamp_early`` was
0.00139 / 0.00332 / 0.00149 and head ``auc_oof[stamp_early]`` 0.9850 / 0.9700 / 0.9806. With
``head_delta_rmse`` around −69 s, R5's gate number (native) and its ``month_loo`` number
(merged) were **not the same model**. See the 2026-09-19 pipeline code review, F7.1.

Since 2026-09-19 a head run can pin ``mixture.corruption.head_seed`` in its YAML; every cell
then fits the identical head and the merge is the same model as the native run, up to the
clip above. A farm that did **not** pin it still merges — a campaign in flight must not be
bricked by a tooling change — but :func:`check_cells` prints a loud warning, the merged
``metrics.json`` **omits** cell 0's head diagnostics (``corruption``,
``corruption_calibration``) instead of attributing them to the merge, and
``merged.per_seed_heads`` records that this is a head ensemble. ``--strict-heads`` turns the
warning into a refusal. Runs merged before that date keep the ensembling benefit and cannot
be compared to a native multi-seed run of the same YAML.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from prc import paths
from prc.config import ExperimentConfig
from prc.data import schema as S
from prc.eval.lanes import lane_metrics
from prc.eval.metrics import bootstrap_rmse_std, rmse, rmse_breakdown
from prc.eval.run import append_registry
from prc.mixture.regimes import REGIME_COL

REPO = Path(__file__).resolve().parents[1]
RUN_CELL = REPO / "scripts" / "run_cell.py"
SEED_PREFIX = "pred_s"
#: identity columns ``prc.eval.run`` writes before the predictions
OOF_IDENTITY = (S.MVT_ID, S.AIRPORT, S.SPLIT_MONTH, "fold", S.TARGET)
TEST_IDENTITY = (S.MVT_ID,)
#: columns that must be *identical* across cells rather than averaged
EXACT_COLS = (S.AIRPORT, S.SPLIT_MONTH, "fold", S.TARGET, REGIME_COL)

_SEED_PART = re.compile(r"^s(\d+(?:-\d+)*)$")
_FRAC_PART = re.compile(r"^frac([0-9][0-9.e+-]*)$")


class IngestError(SystemExit):
    """A mismatch a human has to look at; printed as-is by ``main``."""


# --------------------------------------------------------------------------- run_cell reuse


def _run_cell_module():
    """``scripts/run_cell.py`` loaded as a module, so the override -> config -> name mapping
    used here is literally the one the cells ran."""
    spec = importlib.util.spec_from_file_location("run_cell_module", RUN_CELL)
    if spec is None or spec.loader is None:  # pragma: no cover - only if the file vanished
        raise IngestError(f"cannot import {RUN_CELL}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("run_cell_module", mod)
    spec.loader.exec_module(mod)
    return mod


def split_name(name: str) -> tuple[str, list[str]]:
    """``exp0047_x__s1__month_loo`` -> ``("exp0047_x", ["s1", "month_loo"])``."""
    base, *parts = name.split("__")
    return base, parts


def config_from_parts(base: str, parts: list[str], seeds: list[int] | None = None):
    """Rebuild a cell's :class:`ExperimentConfig` by replaying ``scripts/run_cell.py``.

    The run name is the only description of the overrides that comes home (``metrics.json``
    carries no CV scheme), so the parts are parsed back into ``run_cell``'s own flags and
    ``run_cell.resolve`` is asked to produce the config. ``seeds`` replaces the seed part —
    that is how a merged N-seed config is built from N single-seed cells.

    Returns ``(cfg, recipe)``; ``recipe`` is what :func:`config_from_recipe` needs to do it
    again from a merged ``metrics.json``.
    """
    from prc.cv.folds import SCHEMES

    yaml_path = paths.EXPERIMENTS / f"{base}.yaml"
    if not yaml_path.is_file():
        raise IngestError(
            f"no base config for run name {base!r}: expected {yaml_path}. "
            "Fetched runs are named <experiment>__<overrides>; pass --base-config if the "
            "YAML lives elsewhere."
        )
    ns = argparse.Namespace(
        config=str(yaml_path),
        seed=None,
        seeds=None,
        scheme=None,
        sample_frac=None,
        drop_block=None,
        only_blocks=None,
        name_suffix=None,
    )
    seed_part: str | None = None
    for i, part in enumerate(parts):
        if (m := _SEED_PART.match(part)) and seed_part is None:
            seed_part = part
            ns.seeds = ",".join(m.group(1).split("-"))
        elif part in SCHEMES:
            ns.scheme = part
        elif m := _FRAC_PART.match(part):
            ns.sample_frac = float(m.group(1))
        elif part.startswith("drop-"):
            ns.drop_block = part.removeprefix("drop-")
        elif part.startswith("only-"):
            ns.only_blocks = ",".join(part.removeprefix("only-").split("+"))
        elif i == len(parts) - 1:  # run_cell appends --name-suffix last
            ns.name_suffix = part
        else:
            raise IngestError(
                f"cannot parse run-name part {part!r} of {'__'.join([base, *parts])!r}; "
                "expected s<seed>[-<seed>…], a CV scheme, frac<x>, drop-<block>, "
                "only-<block>[+<block>…], or a trailing --name-suffix"
            )
    if seeds is not None:
        ns.seeds = ",".join(str(s) for s in seeds)
    cfg, _ = _run_cell_module().resolve(ns)
    recipe = {
        "base_config": str(
            yaml_path.relative_to(REPO) if yaml_path.is_relative_to(REPO) else yaml_path
        ),
        "parts": parts,
        "seeds": [int(s) for s in (ns.seeds or "").split(",") if s],
    }
    return cfg, recipe


def config_from_recipe(recipe: dict, name: str | None = None) -> ExperimentConfig:
    """The inverse of the ``recipe`` :func:`config_from_parts` returns (used by replay)."""
    base = Path(recipe["base_config"]).stem
    cfg, _ = config_from_parts(base, list(recipe.get("parts", [])), recipe.get("seeds") or None)
    if name and cfg.name != name:
        cfg = cfg.model_copy(update={"name": name})
    return cfg


def resolve_config(run_dir: Path, metrics: dict) -> tuple[ExperimentConfig, list[str]]:
    """The config of a fetched or merged run dir, plus any digest warnings.

    A merged run carries its recipe in ``metrics["merged"]``; a fetched run is reconstructed
    from its name. Either way the reconstruction is checked against the ``config_digest``
    the run itself recorded.
    """
    name = metrics.get("name") or run_dir.name
    if isinstance(metrics.get("merged"), dict) and metrics["merged"].get("recipe"):
        cfg = config_from_recipe(metrics["merged"]["recipe"], name)
    else:
        base, parts = split_name(name)
        cfg, _ = config_from_parts(base, parts)
        if cfg.name != name:
            raise IngestError(
                f"{run_dir}: rebuilt config is named {cfg.name!r} but the run is {name!r}; "
                "the run-name overrides could not be replayed"
            )
    warn: list[str] = []
    recorded = metrics.get("config_digest")
    if recorded and cfg.digest() != recorded:
        warn.append(
            f"config digest mismatch: {run_dir.name} recorded {recorded}, "
            f"{paths.EXPERIMENTS / (split_name(name)[0] + '.yaml')} now digests to "
            f"{cfg.digest()} — the YAML changed after the run, or the wrong YAML was found"
        )
    return cfg, warn


# --------------------------------------------------------------------------- cells


@dataclass
class Cell:
    path: Path
    metrics: dict
    name: str
    base: str
    parts: list[str]
    seeds: tuple[int, ...]
    cfg: ExperimentConfig
    warnings: list[str] = field(default_factory=list)

    @property
    def key(self) -> tuple[str, tuple[str, ...]]:
        """Config identity of the cell: everything in the name except the seeds."""
        return self.base, tuple(p for p in self.parts if not _SEED_PART.match(p))


def read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except OSError as e:
        raise IngestError(f"cannot read {p}: {e}") from e
    except json.JSONDecodeError as e:
        raise IngestError(f"{p} is not valid JSON: {e}") from e


def load_cell(run_dir: Path) -> Cell:
    metrics = read_json(run_dir / "metrics.json")
    name = metrics.get("name") or run_dir.name
    base, parts = split_name(name)
    cfg, warn = resolve_config(run_dir, metrics)
    return Cell(
        path=run_dir,
        metrics=metrics,
        name=name,
        base=base,
        parts=parts,
        seeds=tuple(sorted(cfg.seeds)),
        cfg=cfg,
        warnings=warn,
    )


def discover_cells(out_root: Path) -> list[Cell]:
    if not out_root.is_dir():
        raise IngestError(f"{out_root} is not a directory")
    dirs = sorted({p.parent for p in out_root.rglob("metrics.json")})
    dirs = [d for d in dirs if d != out_root or (d / "oof.parquet").is_file()]
    if not dirs:
        raise IngestError(f"no run dirs with metrics.json under {out_root}")
    return [load_cell(d) for d in dirs]


def one_group(cells: list[Cell]) -> list[Cell]:
    """The cells of ONE config+scheme, or a clear error naming the groups found."""
    groups: dict[tuple, list[Cell]] = {}
    for c in cells:
        groups.setdefault(c.key, []).append(c)
    if len(groups) > 1:
        lines = [f"{len(groups)} different configs under this out root; merge them one at a time:"]
        for (base, parts), members in sorted(groups.items()):
            lines.append(f"  {base}{''.join('__' + p for p in parts)}")
            lines += [f"    {m.path}" for m in sorted(members, key=lambda c: c.path)]
        raise IngestError("\n".join(lines))
    return sorted(next(iter(groups.values())), key=lambda c: c.seeds)


def per_seed_heads(cells: list[Cell]) -> bool:
    """Did these cells each fit their OWN corruption head?

    True when there is a head and no cell pinned ``mixture.corruption.head_seed``, so the
    head was seeded from that cell's single ``cfg.seeds[0]``. ``check_cells`` blanks ``seeds``
    before comparing configs — which is what let this through — so the question has to be
    asked separately. See the module docstring and the 2026-09-19 code review, F7.1.
    """
    corr = cells[0].cfg.corruption()
    return corr is not None and all((c.cfg.corruption() or corr).head_seed is None for c in cells)


def check_cells(cells: list[Cell], strict: bool = False) -> None:
    """Everything checkable before a parquet file is opened."""
    seen: dict[int, Cell] = {}
    for c in cells:
        if not c.seeds:
            raise IngestError(f"{c.path}: no seeds in the run name or config")
        for s in c.seeds:
            if s in seen:
                raise IngestError(
                    f"seed {s} appears in two cells: {seen[s].path} and {c.path}; "
                    "a merge needs one cell per seed"
                )
            seen[s] = c
    ref = cells[0]
    for c in cells[1:]:
        if c.cfg.cv.scheme != ref.cfg.cv.scheme:
            raise IngestError(
                f"CV scheme differs: {ref.path.name} is {ref.cfg.cv.scheme}, "
                f"{c.path.name} is {c.cfg.cv.scheme}"
            )
        for field_ in ("features", "n_features", "n_train", "n_test"):
            if c.metrics.get(field_) != ref.metrics.get(field_):
                raise IngestError(
                    f"{field_} differs between {ref.path.name} ({ref.metrics.get(field_)}) "
                    f"and {c.path.name} ({c.metrics.get(field_)})"
                )
        if c.cfg.model_copy(update={"name": "", "seeds": []}) != ref.cfg.model_copy(
            update={"name": "", "seeds": []}
        ):
            raise IngestError(
                f"{c.path.name} and {ref.path.name} rebuild to different configs "
                "(same name prefix, different YAML content?)"
            )
    # F7.1: `seeds` is blanked above so that per-seed cells compare equal — which is exactly
    # what let a per-cell corruption head through. The head is seeded from `cfg.seeds[0]`
    # unless the YAML pins `mixture.corruption.head_seed`, so without it these cells carry
    # DIFFERENT heads and the merge is a head ensemble, not the seed mean of one model.
    if per_seed_heads(cells):
        msg = (
            "these cells have a corruption head but no pinned "
            "mixture.corruption.head_seed, so each one fitted its OWN head (seed = its own "
            f"cfg.seeds[0]) and merging them averages {len(cells)} different mixtures rather "
            "than the seeds of one model. The merged run is a HEAD ENSEMBLE and is NOT "
            "comparable to a native multi-seed run of the same YAML (2026-09-19 code review, "
            "F7.1; head_delta_rmse is about -69 s on R5, so this is not a rounding "
            "difference). Pin `mixture.corruption.head_seed` in the YAML and re-run the farm "
            "to get the seed mean of one model."
        )
        if strict:
            raise IngestError(msg + " (--strict-heads)")
        print(f"WARNING: {msg}", flush=True)
        print(
            "WARNING: cell 0's head diagnostics (`corruption`, `corruption_calibration`) "
            "will be OMITTED from the merged metrics.json rather than attributed to the "
            "merge; `merged.per_seed_heads` records this.",
            flush=True,
        )


# --------------------------------------------------------------------------- frame merging


def _cols(df: pl.DataFrame, identity: tuple[str, ...]) -> tuple[list[str], list[str], list[str]]:
    """``(identity, per-seed, component)`` columns of a run frame."""
    ident = [c for c in identity if c in df.columns]
    seed_cols = sorted(c for c in df.columns if c.startswith(SEED_PREFIX))
    comp = [c for c in df.columns if c not in ident and c not in seed_cols and c != "pred"]
    return ident, seed_cols, comp


_PRESENT = "__present"


def _aligned(base_ids: pl.DataFrame, df: pl.DataFrame, cols: list[str], what: str) -> pl.DataFrame:
    """``df[cols]`` in the base's row order, joined on MVT_ID — never on row order."""
    if df.height != base_ids.height:
        raise IngestError(
            f"{what}: {df.height} rows, the first cell has {base_ids.height}; "
            "the cells did not score the same rows"
        )
    right = df.select(S.MVT_ID, *cols).with_columns(pl.lit(True).alias(_PRESENT))
    try:
        out = base_ids.join(right, on=S.MVT_ID, how="left", validate="1:1")
    except pl.exceptions.PolarsError as e:
        raise IngestError(f"{what}: cannot join on {S.MVT_ID} 1:1 ({e})") from e
    missing = int(out.select(pl.col(_PRESENT).is_null().sum()).item())
    if missing:
        raise IngestError(
            f"{what}: {missing} of {base_ids.height} {S.MVT_ID} values are not in this cell; "
            "the cells did not score the same rows"
        )
    return out.drop(_PRESENT)


def _same(a: pl.Series, b: pl.Series) -> int:
    """Number of rows on which two aligned columns differ (nulls compare equal)."""
    diff = pl.DataFrame({"a": a, "b": b}).select(pl.col("a").ne_missing(pl.col("b")).sum())
    return int(diff.item())


def merge_frames(
    cells: list[Cell], filename: str, identity: tuple[str, ...], seed_cols: bool
) -> tuple[pl.DataFrame, dict[int, np.ndarray]]:
    """Merge one artifact (``oof.parquet`` / ``test_pred.parquet``) across cells.

    ``pred`` becomes the seed mean; component columns (``y_hat_normal``, ``p_*``, anything
    else numeric) are averaged across seeds the same way the runner averages them; string
    columns (``regime``) and the identity columns must agree row for row. Returns the merged
    frame and, when the artifact carries them, the per-seed arrays.
    """
    base_path = cells[0].path / filename
    if not base_path.is_file():
        raise IngestError(f"{base_path} is missing; fetch it before merging")
    base = pl.read_parquet(base_path)
    ident, _, comp = _cols(base, identity)
    if S.MVT_ID not in ident:
        raise IngestError(f"{base_path}: no {S.MVT_ID} column")
    base_ids = base.select(S.MVT_ID)
    out = base.select(ident)
    n = base.height

    per_seed: dict[int, np.ndarray] = {}
    pred_sum: np.ndarray | None = None  # seed-weighted, for artifacts without per-seed columns
    comp_sum: dict[str, np.ndarray] = {}
    comp_str: dict[str, pl.Series] = {}
    n_seeds = 0
    for cell in cells:
        path = cell.path / filename
        if not path.is_file():
            raise IngestError(f"{path} is missing; fetch it before merging")
        df = pl.read_parquet(path)
        c_ident, c_seed, c_comp = _cols(df, identity)
        if sorted(c_comp) != sorted(comp) or sorted(c_ident) != sorted(ident):
            raise IngestError(
                f"{path} has columns {sorted(df.columns)}, {base_path} has "
                f"{sorted(base.columns)}; the cells are not the same kind of run"
            )
        if c_seed and sorted(int(c.removeprefix(SEED_PREFIX)) for c in c_seed) != list(cell.seeds):
            raise IngestError(
                f"{path}: per-seed columns {c_seed} do not match the cell's seeds "
                f"{list(cell.seeds)}"
            )
        want = [c for c in (*c_ident, *c_seed, *c_comp, "pred") if c != S.MVT_ID]
        a = _aligned(base_ids, df, want, str(path))
        for c in c_ident:
            if c != S.MVT_ID and c in EXACT_COLS and (bad := _same(out[c], a[c])):
                raise IngestError(f"{path}: {c} differs from {base_path} on {bad} of {n} rows")
        w = len(cell.seeds)
        if c_seed:
            for col in c_seed:
                per_seed[int(col.removeprefix(SEED_PREFIX))] = a[col].cast(pl.Float64).to_numpy()
        p = a["pred"].cast(pl.Float64).to_numpy() * w
        pred_sum = p if pred_sum is None else pred_sum + p
        n_seeds += w
        for c in c_comp:
            if a.schema[c].is_numeric():
                v = a[c].cast(pl.Float64).to_numpy() * w
                comp_sum[c] = v if c not in comp_sum else comp_sum[c] + v
            elif c in comp_str:
                if bad := _same(comp_str[c], a[c]):
                    raise IngestError(f"{path}: {c} differs from {base_path} on {bad} of {n} rows")
            else:
                comp_str[c] = a[c]
        del df, a
        gc.collect()

    seeds = sorted(per_seed)
    if seed_cols and len(seeds) != n_seeds:
        raise IngestError(
            f"{filename}: {n_seeds} seeds across the cells but {len(seeds)} per-seed columns"
        )
    assert pred_sum is not None
    pred = np.mean([per_seed[s] for s in seeds], axis=0) if seeds else pred_sum / n_seeds
    out = out.with_columns(
        pl.Series("pred", pred),
        *([pl.Series(f"{SEED_PREFIX}{s}", per_seed[s]) for s in seeds] if seed_cols else []),
    )
    # keep the runner's column order: identity, pred, per-seed, regime, y_hat_normal, p_*
    for c in comp:
        out = out.with_columns(
            comp_str[c] if c in comp_str else pl.Series(c, comp_sum[c] / n_seeds)
        )
    return out, per_seed


def merge_importance(cells: list[Cell], out_dir: Path) -> bool:
    """Gain importances averaged over the cells, in the runner's format."""
    import pandas as pd

    frames = []
    for cell in cells:
        p = cell.path / "importance.csv"
        if p.is_file():
            frames.append(pd.read_csv(p, index_col=0)["gain"])
    if not frames:
        return False
    pd.concat(frames, axis=1).fillna(0).mean(axis=1).sort_values(ascending=False).to_csv(
        out_dir / "importance.csv", header=["gain"]
    )
    return True


# --------------------------------------------------------------------------- metrics


def _lanes(oof: pl.DataFrame) -> dict:
    """``prc.eval.lanes.lane_metrics`` on the merged OOF, or ``{}`` with a warning when the
    movement table is not on this machine (the lane columns of the registry row then stay
    empty; everything else is unaffected)."""
    mv_path = paths.INTERIM / "movements.parquet"
    if not mv_path.is_file():
        print(f"WARNING: {mv_path} not found; lanes are left empty", file=sys.stderr)
        return {}
    mv = pl.read_parquet(mv_path, columns=[S.MVT_ID, S.MVT_TIME, S.AOBT_3])
    d = oof.select(S.MVT_ID, S.SPLIT_MONTH, S.TARGET, "pred").join(mv, on=S.MVT_ID, how="left")
    return lane_metrics(d)


def merged_metrics(
    cells: list[Cell],
    cfg: ExperimentConfig,
    oof: pl.DataFrame,
    per_seed_arr: dict[int, np.ndarray],
    n_test: int,
    recipe: dict,
) -> dict:
    """The runner's ``metrics.json``, recomputed on the merged frame with the repo's own
    metric functions, plus a ``merged`` provenance block."""
    y = oof[S.TARGET].cast(pl.Float64).to_numpy()
    pred = oof["pred"].to_numpy()
    ref = cells[0].metrics
    ref_cfg_corr = cells[0].cfg.corruption()
    per_seed = {str(s): float(rmse(y, per_seed_arr[s])) for s in sorted(per_seed_arr)}
    # a cell's own cv_rmse is the RMSE of exactly this column on exactly these rows: the
    # cheapest end-to-end check that the MVT_ID join reproduced each cell's predictions
    for cell in cells:
        for s in cell.seeds:
            got, want = per_seed[str(s)], cell.metrics.get("per_seed", {}).get(str(s))
            if want is not None and abs(got - want) > 1e-6:
                raise IngestError(
                    f"{cell.path.name}: seed {s} RMSE is {got:.6f} after the join but the cell "
                    f"recorded {want:.6f}; the merge did not reproduce the cell"
                )
    per_fold = rmse_breakdown(oof, S.TARGET, "pred", ["fold"])
    per_airport = rmse_breakdown(oof, S.TARGET, "pred", [S.AIRPORT])
    per_month = rmse_breakdown(oof, S.TARGET, "pred", [S.SPLIT_MONTH])
    gits = sorted({c.metrics.get("git", "") for c in cells} - {""})
    metrics: dict[str, Any] = {
        "name": cfg.name,
        "cv_rmse": float(rmse(y, pred)),
        "cv_rmse_boot_std": bootstrap_rmse_std(y, pred),
        "per_month": {r[S.SPLIT_MONTH]: float(r["rmse"]) for r in per_month.iter_rows(named=True)},
        "per_fold": {int(r["fold"]): float(r["rmse"]) for r in per_fold.iter_rows(named=True)},
        "per_airport": {r[S.AIRPORT]: float(r["rmse"]) for r in per_airport.iter_rows(named=True)},
        "per_seed": per_seed,
        "n_train": ref.get("n_train"),
        "n_oof": int(oof.height),
        "n_test": n_test,
        "n_features": ref.get("n_features"),
        "features": ref.get("features"),
        "best_iterations": {
            k: v for c in cells for k, v in (c.metrics.get("best_iterations") or {}).items()
        },
        "elapsed_s": round(sum(float(c.metrics.get("elapsed_s") or 0.0) for c in cells), 1),
        "git": gits[0] if len(gits) == 1 else ",".join(gits),
        "config_digest": cfg.digest(),
        "finished_at": max(c.metrics.get("finished_at", "") for c in cells)
        or datetime.now(UTC).isoformat(timespec="seconds"),
        "cv_scheme": cfg.cv.scheme,
        "seeds": list(cfg.seeds),
    }
    metrics["lanes"] = _lanes(oof)
    if REGIME_COL in oof.columns:
        per_regime = rmse_breakdown(oof, S.TARGET, "pred", [REGIME_COL])
        metrics["target_mode"] = ref.get("target_mode")
        metrics["regime_groups"] = ref.get("regime_groups")
        metrics["per_regime"] = {
            r[REGIME_COL]: {
                "n": int(r["n"]),
                "rmse": float(r["rmse"]),
                "sse_share": float(r["sse_share"]),
            }
            for r in per_regime.iter_rows(named=True)
        }
    # The head's configuration is seed-independent, and with `mixture.corruption.head_seed`
    # pinned so are its diagnostics — every cell fitted the identical head, so cell 0's
    # numbers describe the merge. Without it they describe cell 0 only; `check_cells` refuses
    # that case unless the caller opted in, and `merged.per_seed_heads` records which it was.
    # Anything computed from predictions is recomputed.
    cell0_only = {"corruption", "corruption_calibration"} if per_seed_heads(cells) else set()
    for key in (
        "corruption",
        "corruption_calibration",
        "corruption_train_normal_on",
        "corruption_normal_weight",
        "corruption_calibrate",
        "corruption_extra_blocks",
        "corruption_extra_signals",
        "corruption_identity_signals",
        "corruption_sched_head",
        "n_clf_features",
    ):
        # a per-seed-head merge has no single head, so cell 0's AUC / calibration tables
        # describe cell 0 and nothing else; the configuration echoes below are still true
        if key in ref and key not in cell0_only:
            metrics[key] = ref[key]
    if "y_hat_normal" in oof.columns:
        metrics["cv_rmse_no_head"] = float(rmse(y, oof["y_hat_normal"].to_numpy()))
        metrics["head_delta_rmse"] = metrics["cv_rmse"] - metrics["cv_rmse_no_head"]
    metrics["merged"] = {
        "tool": "scripts/merge_seeds.py merge-seeds",
        "merged_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "recipe": recipe,
        "note": (
            "pred is the mean of the per-seed predictions; a native multi-seed run averages "
            "before the mixture/clip, so cv_rmse can differ from one by ~1e-3-3e-2 s"
        ),
        # F7.1: True means every cell fitted its OWN corruption head and this run is a head
        # ensemble whose `corruption*` block describes cell 0 only. None = no head at all.
        "per_seed_heads": (None if ref_cfg_corr is None else bool(ref_cfg_corr.head_seed is None)),
        "cells": [
            {
                "dir": str(c.path.relative_to(REPO) if c.path.is_relative_to(REPO) else c.path),
                "name": c.name,
                "seeds": list(c.seeds),
                "cv_rmse": c.metrics.get("cv_rmse"),
                "config_digest": c.metrics.get("config_digest"),
                "git": c.metrics.get("git"),
                "elapsed_s": c.metrics.get("elapsed_s"),
                "finished_at": c.metrics.get("finished_at"),
                "slurm": _slurm_ids(c),
                "peak_rss_mb": (read_json(c.path / "status.json") or {}).get("peak_rss_mb")
                if (c.path / "status.json").is_file()
                else None,
            }
            for c in cells
        ],
    }
    return metrics


def _slurm_ids(cell: Cell) -> dict[str, Any]:
    """Whatever job identity the cell brought home (``metrics.json``/``status.json`` keys,
    else the ``SLURM_*`` lines the job scripts echo into ``log.txt``)."""
    out: dict[str, Any] = {}
    for src in (
        cell.metrics,
        read_json(cell.path / "status.json") if (cell.path / "status.json").is_file() else {},
    ):
        for k in ("slurm_job_id", "job_id", "slurm_array_task_id", "array_task_id", "host"):
            if src.get(k) is not None:
                out.setdefault(k, src[k])
    log = cell.path / "log.txt"
    if not out and log.is_file():
        try:
            text = log.read_text(encoding="utf-8", errors="replace")[:20000]
        except OSError:
            text = ""
        for k, pat in (
            ("slurm_job_id", r"SLURM_JOB_ID[=: ]+(\d+)"),
            ("slurm_array_task_id", r"SLURM_ARRAY_TASK_ID[=: ]+(\d+)"),
        ):
            if m := re.search(pat, text):
                out[k] = m.group(1)
    return out


# --------------------------------------------------------------------------- registry


def registry_names(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    with open(path, newline="") as f:
        return {(r.get("name") or "").strip() for r in csv.DictReader(f)}


def replay_one(run_dir: Path, dry_run: bool) -> str:
    """Append this run's registry row, or say why it was skipped. Idempotent."""
    metrics = read_json(run_dir / "metrics.json")
    if "cv_rmse" not in metrics:
        raise IngestError(f"{run_dir}/metrics.json has no cv_rmse; is this a finished run?")
    cfg, warn = resolve_config(run_dir, metrics)
    for w in warn:
        print(f"WARNING: {w}", file=sys.stderr)
    # a metrics.json written before one of these columns existed still gets a row
    metrics.setdefault("config_digest", cfg.digest())
    metrics.setdefault("git", "")
    metrics.setdefault("n_features", None)
    metrics.setdefault("elapsed_s", None)
    metrics.setdefault("finished_at", datetime.now(UTC).isoformat(timespec="seconds"))
    row = {
        "name": cfg.name,
        "cv_rmse": round(metrics["cv_rmse"], 4),
        "cv_scheme": cfg.cv.scheme,
        "model": cfg.model.kind,
        "seeds": len(cfg.seeds),
        "git": metrics.get("git"),
        "config_digest": metrics.get("config_digest"),
    }
    reg = paths.EXPERIMENTS / "registry.csv"
    if dry_run:  # a dry run reads; it takes no lock and leaves no lock file behind
        if cfg.name in registry_names(reg):
            return f"skip  {cfg.name}: already in {reg.name}"
        return f"WOULD append  {json.dumps(row)}"
    # the runner's own lock (scripts/run_cell.py), held across the check and the append so two
    # replays — or a replay and a local run — cannot both decide the row is missing
    run_cell = _run_cell_module()
    with run_cell._registry_lock(paths.EXPERIMENTS / ".registry.lock"):
        if cfg.name in registry_names(reg):
            return f"skip  {cfg.name}: already in {reg.name}"
        append_registry(metrics, cfg)
    return f"appended  {json.dumps(row)}"


# --------------------------------------------------------------------------- commands


def cmd_merge_seeds(args: argparse.Namespace) -> int:
    out_root = Path(args.out_root)
    cells = one_group(discover_cells(out_root))
    check_cells(cells, strict=getattr(args, "strict_heads", False))
    for c in cells:
        for w in c.warnings:
            print(f"WARNING: {w}", file=sys.stderr)
    seeds = sorted(s for c in cells for s in c.seeds)
    cfg, recipe = config_from_parts(cells[0].base, cells[0].parts, seeds)
    name = args.name or cfg.name
    if name != cfg.name:
        cfg = cfg.model_copy(update={"name": name})
    out_dir = paths.RUNS / name

    print(
        json.dumps(
            {
                "out_root": str(out_root),
                "cells": [str(c.path) for c in cells],
                "seeds": seeds,
                "scheme": cfg.cv.scheme,
                "base_config": recipe["base_config"],
                "merged_run": str(out_dir),
                "config_digest": cfg.digest(),
            },
            indent=2,
        ),
        flush=True,
    )
    if out_dir.exists() and not args.force and not args.dry_run:
        raise IngestError(f"{out_dir} already exists; pass --force to rewrite it")

    oof, per_seed_arr = merge_frames(cells, "oof.parquet", OOF_IDENTITY, seed_cols=True)
    test, _ = merge_frames(cells, "test_pred.parquet", TEST_IDENTITY, seed_cols=False)
    if args.dry_run:
        print(
            f"DRY RUN: {oof.height} OOF rows and {test.height} test rows would merge into "
            f"{out_dir}; nothing written."
        )
        return 0

    metrics = merged_metrics(cells, cfg, oof, per_seed_arr, test.height, recipe)
    out_dir.mkdir(parents=True, exist_ok=True)
    oof.write_parquet(out_dir / "oof.parquet")
    test.write_parquet(out_dir / "test_pred.parquet")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    merge_importance(cells, out_dir)
    (out_dir / "log.txt").write_text(
        "merged by scripts/merge_seeds.py from:\n"
        + "".join(f"  {c.path}\n" for c in cells)
        + f"cv_rmse={metrics['cv_rmse']:.4f} per_seed={metrics['per_seed']}\n"
    )
    print(
        json.dumps(
            {
                "name": metrics["name"],
                "cv_rmse": metrics["cv_rmse"],
                "per_seed": metrics["per_seed"],
                "per_month": metrics["per_month"],
                "per_fold": metrics["per_fold"],
                "n_oof": metrics["n_oof"],
            },
            indent=2,
        )
    )
    print(f"\nwrote {out_dir}")
    print(f"next:  uv run python scripts/merge_seeds.py replay-registry {out_dir}")
    return 0


def cmd_replay_registry(args: argparse.Namespace) -> int:
    for d in args.run_dirs:
        run_dir = Path(d)
        if not (run_dir / "metrics.json").is_file():
            raise IngestError(f"{run_dir}/metrics.json not found")
        print(replay_one(run_dir, args.dry_run), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("merge-seeds", help="merge per-seed cell dirs into one run dir")
    m.add_argument("out_root", help="suite out root holding the per-seed cell dirs")
    m.add_argument("--name", help="merged run name (default: <exp>__s<a>-<b>-<c>__<overrides>)")
    m.add_argument("--force", action="store_true", help="rewrite an existing merged run dir")
    m.add_argument(
        "--strict-heads",
        action="store_true",
        help="refuse (instead of warning) a corruption-head farm whose cells did not pin "
        "mixture.corruption.head_seed. Such a merge is a HEAD ENSEMBLE, not the seed mean "
        "of one model, and is not comparable to a native multi-seed run (code review F7.1).",
    )
    m.add_argument("--dry-run", action="store_true", help="verify and plan; write nothing")
    m.set_defaults(func=cmd_merge_seeds)

    r = sub.add_parser("replay-registry", help="append registry rows for fetched/merged runs")
    r.add_argument("run_dirs", nargs="+", help="run dirs under outputs/runs")
    r.add_argument("--dry-run", action="store_true", help="print the row; write nothing")
    r.set_defaults(func=cmd_replay_registry)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
