"""``prc`` command line. Run ``uv run prc --help``."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import typer
from rich import print as rprint

app = typer.Typer(no_args_is_help=True, add_completion=False)
data_app = typer.Typer(no_args_is_help=True)
features_app = typer.Typer(no_args_is_help=True)
app.add_typer(data_app, name="data", help="Pull / prepare the challenge data.")
app.add_typer(features_app, name="features", help="Inspect / build feature blocks.")


@data_app.command("pull")
def data_pull(
    alias: str = typer.Option("prc", help="mc alias"), bucket: str | None = typer.Option(None)
) -> None:
    """Mirror the challenge bucket into data/raw with the MinIO client (mc)."""
    from prc import paths

    bucket = bucket or os.environ.get("PRC_DATA_BUCKET")
    if not bucket:
        raise typer.BadParameter("pass --bucket or set PRC_DATA_BUCKET (see .env.example)")
    paths.RAW.mkdir(parents=True, exist_ok=True)
    cmd = ["mc", "mirror", "--overwrite", f"{alias}/{bucket}", str(paths.RAW)]
    rprint("[bold]$", " ".join(cmd))
    subprocess.run(cmd, check=True)


@data_app.command("prepare")
def data_prepare(force: bool = False) -> None:
    """Build data/interim/movements.parquet from the raw monthly files + ranking."""
    from prc.data.load import build_interim

    df = build_interim(force=force)
    rprint(f"movements: {df.height:,} rows x {df.width} cols")
    rprint(df.group_by("is_test", "PHASE_mvt").len().sort("is_test", "PHASE_mvt"))


@features_app.command("list")
def features_list() -> None:
    from prc.features import REGISTRY

    for name, blk in sorted(REGISTRY.items()):
        rprint(
            f"[bold]{name}[/bold]  [dim]{blk.source_hash()}[/dim]  {blk.doc.splitlines()[0] if blk.doc else ''}"
        )


@features_app.command("build")
def features_build(
    names: list[str] = typer.Argument(None), all_: bool = typer.Option(False, "--all")
) -> None:
    """Compute (and cache) feature blocks."""
    from prc.data.load import load_movements
    from prc.features import build, list_features

    names = list_features() if all_ or not names else names
    out = build(load_movements(), names)
    rprint(f"built {len(out.columns) - 1} columns for {out.height:,} rows: {names}")


@app.command()
def train(
    config: Path, out: Path = typer.Option(None, help="override outputs/runs/<name>")
) -> None:
    """Run one experiment from a YAML config."""
    import traceback

    from prc import paths
    from prc.config import ExperimentConfig
    from prc.eval import events
    from prc.eval.run import run_experiment

    cfg = ExperimentConfig.load(config)
    run_dir = out or (paths.RUNS / cfg.name)
    try:
        m = run_experiment(cfg, out)
    except Exception:  # a crash must be visible to `prc status` / the monitor
        tb = traceback.format_exc()
        # 500 chars of traceback, keeping the tail: that is where the exception itself is
        events.emit(run_dir, "error", exp=cfg.name, message=tb if len(tb) <= 500 else tb[-500:])
        raise
    rprint(json.dumps({k: m[k] for k in ("name", "cv_rmse", "per_fold", "per_airport")}, indent=2))


@app.command()
def status(
    run: str = typer.Option(None, "--run", help="print this run's event timeline instead"),
    watch: float = typer.Option(0.0, "--watch", help="re-render every N seconds (Ctrl-C to stop)"),
    limit: int = typer.Option(0, "--limit", help="show only the N most recently active runs"),
) -> None:
    """What the runs under outputs/runs are doing (state, RMSE, fits done, ETA)."""
    import time

    from prc.eval.status import render, timeline

    while True:
        text = timeline(run) if run else render(limit=limit)
        print(("\033[2J\033[H" if watch else "") + text, flush=True)
        if watch <= 0:
            return
        try:
            time.sleep(watch)
        except KeyboardInterrupt:  # pragma: no cover - interactive
            return


@app.command()
def blend(
    runs: list[str],
    submit: bool = typer.Option(False),
    upload: bool = typer.Option(False),
    note: str = "",
) -> None:
    """NNLS-blend OOF predictions of several runs; optionally write (+upload) a submission."""
    from prc.blend.blend import blend as _blend
    from prc.submit.build import build_submission

    _weights, score, test = _blend(runs)
    if submit or upload:
        p = build_submission(test, source="blend:" + "+".join(runs), cv_rmse=score, note=note)
        rprint(f"wrote {p}")
        if upload:
            _upload(p)


@app.command()
def submit(run: str, note: str = "", upload: bool = typer.Option(False)) -> None:
    """Write a submission parquet from a single run's test predictions (+ optionally upload)."""
    import polars as pl

    from prc import paths
    from prc.submit.build import build_submission

    preds = pl.read_parquet(paths.RUNS / run / "test_pred.parquet")
    metrics = json.loads((paths.RUNS / run / "metrics.json").read_text())
    p = build_submission(preds, source=run, cv_rmse=metrics["cv_rmse"], note=note)
    rprint(f"wrote {p}")
    if upload:
        _upload(p)


def _upload(p: Path) -> None:
    from prc.submit.upload import upload as _up

    score = _up(p)
    rprint(f"leaderboard RMSE: {score}")


@app.command()
def upload(file: Path) -> None:
    """Upload an existing submission file and record its leaderboard score."""
    _upload(file)


@app.command()
def registry(n: int = 20) -> None:
    """Show the last n experiment registry rows sorted by CV RMSE."""
    import polars as pl

    from prc import paths

    path = paths.EXPERIMENTS / "registry.csv"
    if not path.exists():
        rprint("no runs yet")
        return
    df = pl.read_csv(path).sort("cv_rmse").head(n)
    rprint(df)


if __name__ == "__main__":
    app()
