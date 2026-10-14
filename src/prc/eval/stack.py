"""H1 (2026-09-23): a member's out-of-fold prediction as a regressor input (``stack_member``).

A tree + sequence hybrid: the GRU sequence member's normal-expert prediction (``y_hat_normal``,
seconds, after the inverse ``sqrt`` transform and the clip) becomes one numeric column of the
CatBoost matrix.

The leak rule decides everything. For the consumer's fold ``k`` (training months ``T_k``,
excluded months ``X_k`` = the twelve minus ``T_k``):

* **training rows** of ``T_k`` get the member's out-of-fold prediction from an *inner run*:
  the member config, unchanged apart from its CV scheme, run on the scheme
  :func:`inner_scheme` (``stack_g<n>_x<X_k>``). That scheme splits ``T_k`` into ``n``
  interleaved month groups, validates each group on a fit over the other groups, and never
  lists a month of ``X_k`` as a training or validation month. So a training row's value comes
  from a fit that saw neither its own month group nor any month of ``X_k``, and the member's
  corruption head, target encodings and early stopping inside that run are fold-local too.
* **validation rows** get the member's ordinary run whose fold trains on exactly ``T_k``: the
  ``month_loo`` run when ``X_k`` is one month (``served__loo``), otherwise a run named per
  ``X_k`` (``served__x11-12`` = the ``tail_nov_dec`` run).
* **test rows** get the member's test prediction (``test_fit: full`` of the ``month_loo`` run),
  and under the consumer's own ``test_fit: full`` its training rows get ``served__loo``.

The producer side (:func:`run_inner`, :func:`build_table`) runs the member and writes one
table directory; the consumer side (:class:`StackColumns`) only reads it, in the runner.
Default ``stack_member: null`` executes none of this and is dropped from the digest.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from prc import paths
from prc.config import ExperimentConfig, StackMemberConfig
from prc.cv.folds import MONTHS_2025, SCHEMES, Fold, Scheme
from prc.data import schema as S

log = logging.getLogger(__name__)

VALUE_COL = "y_hat_normal"  # the member's normal-expert prediction, seconds
FALLBACK_COL = "pred"  # a member without a corruption head has no y_hat_normal
#: config keys an inner / served run may differ from the member YAML in
_RUN_KEYS = ("name", "description", "tags", "seeds", "cv", "test_fit", "runtime")


# --- schemes -----------------------------------------------------------------------------


def xkey(excluded: Iterable[str]) -> str:
    """``x01`` for January, ``x11-12`` for November + December."""
    ms = sorted(set(excluded))
    if not ms:
        raise ValueError("stack: an outer fold must exclude at least one month")
    return "x" + "-".join(m[-2:] for m in ms)


def excluded_months(train_months: Sequence[str]) -> list[str]:
    return [m for m in MONTHS_2025 if m not in set(train_months)]


def inner_groups(train_months: Sequence[str], n: int) -> list[list[str]]:
    """``n`` interleaved groups of the sorted training months (group ``j`` = every ``n``-th
    month from the ``j``-th), so each group spans the seasons."""
    ms = sorted(set(train_months))
    if not 2 <= n <= len(ms):
        raise ValueError(f"stack: inner_groups={n} needs 2..{len(ms)} training months")
    return [ms[j::n] for j in range(n)]


def inner_scheme(train_months: Sequence[str], n: int) -> str:
    """Register (idempotently) and return the inner scheme over ``train_months``: fold ``j``
    validates group ``j`` and trains on the other groups; the excluded months are neither."""
    name = f"stack_g{n}_{xkey(excluded_months(train_months))}"
    groups = inner_groups(train_months, n)
    folds = [Fold(val=g, train=[m for h in groups if h is not g for m in h]) for g in groups]
    sch = Scheme(name, folds, f"H1 inner folds over {sorted(train_months)} ({n} groups)")
    if name in SCHEMES and SCHEMES[name] != sch:
        raise ValueError(f"stack: scheme {name} already registered with other folds")
    SCHEMES[name] = sch
    return name


def outer_train_sets(schemes: Iterable[str]) -> list[list[str]]:
    """The distinct training-month sets of the consumer's folds (``month_loo`` and
    ``month_jan_jul`` share theirs), in a stable order."""
    seen: dict[str, list[str]] = {}
    for s in schemes:
        for f in SCHEMES[s].folds:
            t = sorted(f.train_months())
            seen.setdefault(xkey(excluded_months(t)), t)
    return [seen[k] for k in sorted(seen)]


# --- producer ------------------------------------------------------------------------------


def member_config(name: str) -> ExperimentConfig:
    return ExperimentConfig.load(paths.EXPERIMENTS / f"{name}.yaml")


def inner_run_name(member: str, seed: int, scheme: str) -> str:
    return f"{member}__s{seed}__{scheme}"


def inner_config(member: ExperimentConfig, train_months: Sequence[str], n: int, seed: int):
    """The member config on the inner scheme, one seed, test from the fold models (the inner
    runs' test predictions are never read, so a full fit would be wasted GPU)."""
    if member.season_weight is not None:
        raise ValueError("stack: a member with season_weight needs test_fit: full; unsupported")
    scheme = inner_scheme(train_months, n)
    return member.model_copy(
        update={
            "name": inner_run_name(member.name, seed, scheme),
            "cv": member.cv.model_copy(update={"scheme": scheme}),
            "seeds": [seed],
            "test_fit": "folds",
        }
    )


def run_inner(
    member: ExperimentConfig,
    train_sets: Sequence[Sequence[str]],
    seed: int,
    root: Path,
    n: int = 3,
    registry: bool = False,
) -> list[Path]:
    """Run the member's inner runs (one per training-month set) into ``root/<name>/``.
    Resumable: a run whose ``metrics.json`` exists is skipped. The registry row is skipped
    unless ``registry`` (an inner run is an intermediate, not an experiment)."""
    import prc.eval.run as run_mod

    out: list[Path] = []
    for t in train_sets:
        cfg = inner_config(member, t, n, seed)
        d = root / cfg.name
        out.append(d)
        if (d / "metrics.json").exists():
            log.info("stack inner %s: done already, skipped", cfg.name)
            continue
        _freeze_folds(cfg.cv.scheme)
        saved = run_mod.append_registry
        if not registry:
            run_mod.append_registry = lambda *_a, **_k: None
        try:
            run_mod.run_experiment(cfg, d)
        finally:
            run_mod.append_registry = saved
    return out


def _freeze_folds(scheme: str) -> None:
    """Write ``data/folds/<scheme>.parquet`` once, under a lock, temp file + rename: the
    array tasks of one inner scheme (one per seed) start together, and ``get_folds`` alone
    would let one read the file while another writes it."""
    import fcntl
    import os

    from prc.cv.folds import assign_folds
    from prc.data.load import load_movements

    path = paths.FOLDS / f"{scheme}.parquet"
    if path.exists():
        return
    paths.FOLDS.mkdir(parents=True, exist_ok=True)
    with open(paths.FOLDS / f".{scheme}.lock", "w") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            if not path.exists():
                tmp = path.with_suffix(f".{os.getpid()}.tmp")
                assign_folds(load_movements(), scheme).write_parquet(tmp)
                os.replace(tmp, path)
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _check_run(run_dir: Path, member: ExperimentConfig, scheme: str) -> None:
    """The run in ``run_dir`` must be ``member`` on ``scheme``: same config apart from the
    run keys (name, seeds, CV, test_fit, runtime, prose)."""
    if not (run_dir / "config.yaml").exists():  # synced runs keep metrics.json only
        m = json.loads((run_dir / "metrics.json").read_text())
        exp = member.model_copy(
            update={
                "name": m["name"],
                "seeds": [int(s) for s in m["per_seed"]],
                "cv": member.cv.model_copy(update={"scheme": scheme}),
            }
        )
        if exp.digest() != m["config_digest"]:
            raise ValueError(f"stack: {run_dir} digest is not {member.name} on {scheme}")
        return
    got = ExperimentConfig.load(run_dir / "config.yaml")
    if got.cv.scheme != scheme:
        raise ValueError(f"stack: {run_dir} ran scheme {got.cv.scheme}, expected {scheme}")
    a, b = got.model_dump(), member.model_dump()
    for k in _RUN_KEYS:
        a.pop(k, None)
        b.pop(k, None)
    if a != b:
        diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        raise ValueError(f"stack: {run_dir} is not {member.name} (differs in {diff})")


def _values(path: Path, name: str) -> pl.DataFrame:
    df = pl.read_parquet(path)
    col = VALUE_COL if VALUE_COL in df.columns else FALLBACK_COL
    return df.select(S.MVT_ID, pl.col(col).cast(pl.Float32).alias(name))


def _with_mean(df: pl.DataFrame, prefix: str, seeds: Sequence[int]) -> pl.DataFrame:
    cols = [f"{prefix}__s{s}" for s in seeds]
    return df.with_columns(pl.mean_horizontal(cols).cast(pl.Float32).alias(f"{prefix}__mean"))


def _join(frames: list[pl.DataFrame]) -> pl.DataFrame:
    out = frames[0]
    for f in frames[1:]:
        out = out.join(f, on=S.MVT_ID, how="full", coalesce=True)
    return out


def build_table(
    member: ExperimentConfig,
    seeds: Sequence[int],
    n: int,
    train_sets: Sequence[Sequence[str]],
    inner_root: Path,
    loo_run: str,
    served_runs: dict[str, str] | None = None,
    out_dir: Path | None = None,
) -> Path:
    """Collect the member's predictions into ``out_dir`` (default :func:`table_dir`).

    ``loo_run`` / ``served_runs`` values are run-directory templates with ``{seed}``;
    ``served_runs`` maps an excluded-month key (``x11-12``) to the member's run on the scheme
    whose one fold trains on exactly the complementary months (``tail_nov_dec``). Every run's
    ``config.yaml`` is checked to be the member on the expected scheme."""
    served_runs = served_runs or {}
    out_dir = out_dir or table_dir(member.name, n)
    sources: dict[str, str] = {}
    train_parts: list[pl.DataFrame] = []
    for t in train_sets:
        scheme = inner_scheme(t, n)
        prefix = f"inner__{xkey(excluded_months(t))}"
        per_seed = []
        for s in seeds:
            d = inner_root / inner_run_name(member.name, s, scheme)
            _check_run(d, member, scheme)
            per_seed.append(_values(d / "oof.parquet", f"{prefix}__s{s}"))
            sources[f"{prefix}__s{s}"] = str(d)
        train_parts.append(_with_mean(_join(per_seed), prefix, seeds))
    test_parts = []
    for key, tmpl, scheme in [("loo", loo_run, "month_loo")] + [
        (k, v, _served_scheme(k)) for k, v in served_runs.items()
    ]:
        prefix = f"served__{key}"
        per_seed = []
        for s in seeds:
            d = Path(tmpl.format(seed=s))
            _check_run(d, member, scheme)
            per_seed.append(_values(d / "oof.parquet", f"{prefix}__s{s}"))
            sources[f"{prefix}__s{s}"] = str(d)
            if key == "loo":
                test_parts.append(_values(d / "test_pred.parquet", f"test__s{s}"))
        train_parts.append(_with_mean(_join(per_seed), prefix, seeds))
    train = _join(train_parts).sort(S.MVT_ID)
    test = _with_mean(_join(test_parts), "test", seeds).sort(S.MVT_ID)
    out_dir.mkdir(parents=True, exist_ok=True)
    train.write_parquet(out_dir / "train.parquet")
    test.write_parquet(out_dir / "test.parquet")
    meta = {
        "member": member.name,
        "member_digest": member.digest(),
        "inner_groups": n,
        "seeds": list(seeds),
        "value": VALUE_COL,
        "inner_schemes": {
            xkey(excluded_months(t)): [f.val for f in SCHEMES[inner_scheme(t, n)].folds]
            for t in train_sets
        },
        "sources": sources,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    log.info("stack table %s: %d train rows x %d columns", out_dir, train.height, train.width)
    return out_dir


def _served_scheme(key: str) -> str:
    """The registered scheme with one fold that excludes exactly ``key``'s months."""
    for name, sch in SCHEMES.items():
        if (
            len(sch.folds) == 1
            and not name.startswith("stack_")
            and xkey(excluded_months(sch.folds[0].train_months())) == key
        ):
            return name
    raise ValueError(f"stack: no single-fold scheme excludes {key}")


def table_dir(member: str, n: int) -> Path:
    return paths.DATA / "stack" / f"{member}__g{n}"


# --- consumer (the runner) ---------------------------------------------------------------


class StackColumns:
    """The stack table aligned to the runner's training (and test) rows.

    ``names`` are the matrix columns: ``stk_mean`` (the member's seed average) and/or
    ``stk_matched`` (the member seed equal to this run's single seed)."""

    def __init__(self, cfg: ExperimentConfig, train: pl.DataFrame, test: pl.DataFrame | None):
        spec = cfg.stack_member
        assert spec is not None
        self.spec: StackMemberConfig = spec
        self.scheme = cfg.cv.scheme
        self.dir = Path(spec.table) if spec.table else table_dir(spec.config, spec.inner_groups)
        meta = json.loads((self.dir / "meta.json").read_text())
        if meta["member"] != spec.config or meta["inner_groups"] != spec.inner_groups:
            raise ValueError(f"stack: {self.dir} holds {meta['member']} g{meta['inner_groups']}")
        self.suffix: dict[str, str] = {}
        for c in spec.columns:
            if c == "mean":
                self.suffix["stk_mean"] = "mean"
            else:  # matched: one seed per run, and the member must have it
                if len(cfg.seeds) != 1 or cfg.seeds[0] not in meta["seeds"]:
                    raise ValueError(
                        f"stack: columns: [matched] needs one run seed among the member's "
                        f"{meta['seeds']}, got {cfg.seeds}"
                    )
                self.suffix["stk_matched"] = f"s{cfg.seeds[0]}"
        self.names = list(self.suffix)
        want = sorted({self._inner(k) for k in range(len(SCHEMES[self.scheme].folds))})
        want += sorted({self._served(k) for k in range(len(SCHEMES[self.scheme].folds))})
        want = list(dict.fromkeys([*want, "served__loo"]))
        cols = [f"{p}__{s}" for p in want for s in self.suffix.values()]
        tab = pl.read_parquet(self.dir / "train.parquet", columns=[S.MVT_ID, *cols])
        self.train = train.select(S.MVT_ID).join(
            tab, on=S.MVT_ID, how="left", maintain_order="left"
        )
        self.test_tab: pl.DataFrame | None = None
        if test is not None:
            tcols = [f"test__{s}" for s in self.suffix.values()]
            tt = pl.read_parquet(self.dir / "test.parquet", columns=[S.MVT_ID, *tcols])
            self.test_tab = test.select(S.MVT_ID).join(
                tt, on=S.MVT_ID, how="left", maintain_order="left"
            )

    def _train_months(self, k: int) -> list[str]:
        return SCHEMES[self.scheme].folds[k].train_months()

    def _inner(self, k: int) -> str:
        return f"inner__{xkey(excluded_months(self._train_months(k)))}"

    def _served(self, k: int) -> str:
        ex = excluded_months(self._train_months(k))
        return "served__loo" if len(ex) == 1 else f"served__{xkey(ex)}"

    def _block(self, frame: pl.DataFrame, prefix: str, rows: np.ndarray | None, what: str):
        cols = {n: f"{prefix}__{s}" for n, s in self.suffix.items()}
        sub = frame if rows is None else frame.filter(pl.Series(rows))
        out = sub.select([pl.col(c).alias(n) for n, c in cols.items()])
        bad = pl.any_horizontal([pl.col(c).is_null() | pl.col(c).is_nan() for c in out.columns])
        n_null = int(out.select(bad.sum()).item())
        if n_null:
            raise ValueError(f"stack: {n_null} {what} values missing in {prefix} ({self.dir})")
        return out.to_pandas()

    def fold(self, k: int, tr: np.ndarray, va: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fold ``k``: training rows from the inner run over fold ``k``'s training months,
        validation rows from the member's run trained on exactly those months."""
        return (
            self._block(self.train, self._inner(k), tr, f"fold {k} training"),
            self._block(self.train, self._served(k), va, f"fold {k} validation"),
        )

    def test(self) -> pd.DataFrame:
        assert self.test_tab is not None
        return self._block(self.test_tab, "test", None, "test")

    def full(self) -> pd.DataFrame:
        """``test_fit: full`` training rows: each row's value from the member fitted on the
        other eleven months (``month_loo``), the served semantics of the test rows."""
        return self._block(self.train, "served__loo", None, "full-fit training")
