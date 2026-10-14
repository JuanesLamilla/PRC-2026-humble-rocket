"""Filesystem layout. Every path is overridable by environment variable so the
same code runs on a laptop, a workstation and cluster project storage unchanged.

    PRC_ROOT      repo root (default: parent of src/)
    PRC_DATA_DIR  data root (default: $PRC_ROOT/data)
    PRC_OUT_DIR   outputs root (default: $PRC_ROOT/outputs)
"""

from __future__ import annotations

import os
from pathlib import Path


def _env_path(name: str, default: Path) -> Path:
    v = os.environ.get(name)
    return Path(v).expanduser().resolve() if v else default


ROOT: Path = _env_path("PRC_ROOT", Path(__file__).resolve().parents[2])


def load_dotenv(path: Path | None = None) -> dict[str, str]:
    """Read ``<ROOT>/.env`` (KEY=VALUE lines, ``#`` comments) into ``os.environ`` without
    overriding variables that are already set. Called once on import so ``uv run prc ...``
    sees ``PRC_TEAM_NAME`` / ``PRC_DATA_BUCKET`` / ``PRC_SUBMISSION_BUCKET`` without a
    manual ``set -a; . ./.env``. Returns what was loaded (for tests / diagnostics)."""
    path = path or ROOT / ".env"
    loaded: dict[str, str] = {}
    if not path.is_file():
        return loaded
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and value and key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded


load_dotenv()
# The organizers' parquet files carry an R `vctrs` extension type that polars does not know;
# loading it as its storage type is correct and silences a warning on every read.
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
DATA: Path = _env_path("PRC_DATA_DIR", ROOT / "data")
OUT: Path = _env_path("PRC_OUT_DIR", ROOT / "outputs")

RAW = DATA / "raw"  # untouched mirror of the challenge bucket
INTERIM = DATA / "interim"  # normalized single-file tables (train.parquet, ranking.parquet)
FEATURES = DATA / "features"  # hashed feature-block cache
EXTERNAL = DATA / "external"  # open external datasets (METAR, airport geometry, ...)
FOLDS = DATA / "folds"  # frozen fold assignments
RUNS = OUT / "runs"  # one dir per experiment run
SUBMISSIONS = ROOT / "submissions"
CONFIGS = ROOT / "configs"
EXPERIMENTS = ROOT / "experiments"


def ensure_dirs() -> None:
    for p in (RAW, INTERIM, FEATURES, EXTERNAL, FOLDS, RUNS, SUBMISSIONS):
        p.mkdir(parents=True, exist_ok=True)
