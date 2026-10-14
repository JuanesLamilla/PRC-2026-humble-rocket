"""Structured progress events for a running experiment.

Two files are written next to the run's other artifacts, and nothing else in the run is
touched: ``metrics.json``, ``log.txt``, the parquets and the registry keep their exact
shape.

``<run_dir>/events.jsonl``
    One JSON object per line, appended and fsynced as it happens, so a reader tailing the
    file sees a run's timeline live. Every line carries ``ts`` (ISO-8601 UTC, seconds),
    ``exp`` (the run name) and ``event``, plus that event's own fields. Events, in order:
    ``start``, ``features``, ``te_done``\\*, ``head_start``/``head_done``\\*, then
    ``fit_start``/``fit_done`` per (fold, seed, regime group), then ``done`` — or ``error``
    if the run raised (emitted by the CLI). Starred events only appear when the experiment
    uses target encodings / a corruption head.

``<run_dir>/status.json``
    The rolled-up state after every event, written to a temp file and renamed, so a reader
    never sees half a file. See :func:`rollup` for the keys.

Nothing here may ever kill a run: :func:`emit` swallows every exception.
"""

from __future__ import annotations

import json
import logging
import os
import socket
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EVENTS_NAME = "events.jsonl"
STATUS_NAME = "status.json"

#: events the run emits, in the order they first appear
EVENT_NAMES = (
    "start",
    "features",
    "te_done",
    "head_start",
    "head_done",
    "fit_start",
    "fit_done",
    # `test_fit: full` only (2026-09-20): the models fitted on all training rows after CV.
    # Deliberately NOT `fit_start`/`fit_done`, which count the CV fits that `fits_total` and
    # the ETA are defined by.
    "full_fit_start",
    "full_fit_done",
    "done",
    "error",
)

# run_dir -> the events emitted so far, so the rollup does not re-read the file each time
_SEEN: dict[str, list[dict]] = {}


def now_iso() -> str:
    """ISO-8601 UTC, second resolution — the ``ts`` of every event."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def host() -> str:
    try:
        return socket.gethostname()
    except OSError:  # pragma: no cover - hostname lookup does not fail in practice
        return "unknown"


def peak_rss_mb() -> float:
    """Peak resident set size of this process in MB (``resource.getrusage``)."""
    import resource

    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)


def _jsonable(v: Any) -> Any:
    """numpy scalars / arrays and paths to plain JSON types (a run is full of them)."""
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (str, bool, int, float)) or v is None:
        return v
    item = getattr(v, "item", None)  # numpy scalar
    if callable(item):
        try:
            return _jsonable(item())
        except (ValueError, TypeError):
            pass
    tolist = getattr(v, "tolist", None)  # numpy array
    if callable(tolist):
        try:
            return _jsonable(tolist())
        except (ValueError, TypeError):
            pass
    return str(v)


def emit(run_dir: str | Path, event: str, **fields: Any) -> None:
    """Append one event line to ``<run_dir>/events.jsonl`` and rewrite ``status.json``.

    ``exp`` defaults to the name carried by the run's ``start`` event, else the directory
    name; pass ``exp=`` to set it explicitly. Never raises: a logging failure must not be
    able to kill a run that has been training for an hour.
    """
    try:
        d = Path(run_dir)
        d.mkdir(parents=True, exist_ok=True)
        key = str(d.resolve())
        seen = _SEEN.setdefault(key, [])
        exp = fields.pop("exp", None)
        if exp is None:
            exp = next((e["exp"] for e in seen if e.get("event") == "start"), None) or d.name
        rec: dict[str, Any] = {"ts": now_iso(), "exp": str(exp), "event": event}
        rec.update(_jsonable(fields))
        with open(d / EVENTS_NAME, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
        seen.append(rec)
        _write_status(d, rollup(seen))
    except Exception:  # observability must never break the run
        # debug, not warning: a broken event file must not add noise to a run's log either
        logging.getLogger("prc.events").debug("event %r failed", event, exc_info=True)


def _write_status(run_dir: Path, status: dict) -> None:
    tmp = run_dir / (STATUS_NAME + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, run_dir / STATUS_NAME)


def _ts(e: dict) -> float:
    try:
        return datetime.fromisoformat(e["ts"]).timestamp()
    except (KeyError, TypeError, ValueError):
        return 0.0


def rollup(events: list[dict]) -> dict:
    """The state a reader wants at a glance, derived from the events seen so far.

    ``state`` is ``running`` until a ``done`` (or ``error``) event lands. ``eta_s`` is the
    mean time of the fits done so far times the fits still to come, plus the mean head time
    times the heads still to come; it is ``None`` until something has been timed.
    """
    by: dict[str, list[dict]] = {name: [] for name in EVENT_NAMES}
    for e in events:
        by.setdefault(str(e.get("event")), []).append(e)
    start = by["start"][0] if by["start"] else {}
    feats = by["features"][-1] if by["features"] else {}
    done = by["done"][-1] if by["done"] else {}
    err = by["error"][-1] if by["error"] else {}
    last = events[-1] if events else {}

    seeds = list(start.get("seeds") or [])
    groups = {str(e.get("group")) for e in by["fit_start"]} | {
        str(e.get("group")) for e in by["fit_done"]
    }
    n_groups = int(start.get("n_groups") or len(groups) or 1)
    n_folds = start.get("n_folds")
    # `cv.group_by_airport` splits every (fold, seed, regime group) into one fit per airport
    # unit; the count is known only once the frames are assembled, so it rides on `features`.
    n_units = int(feats.get("n_airport_units") or 1)
    fits_total = int(n_folds) * max(len(seeds), 1) * n_groups * n_units if n_folds else None
    fits_done = len(by["fit_done"])
    fit_times = [float(e["elapsed_s"]) for e in by["fit_done"] if e.get("elapsed_s") is not None]
    mean_fit_s = round(sum(fit_times) / len(fit_times), 1) if fit_times else None

    head_times = [float(e["elapsed_s"]) for e in by["head_done"] if e.get("elapsed_s") is not None]
    mean_head_s = round(sum(head_times) / len(head_times), 1) if head_times else None
    heads_done = len(by["head_done"])
    heads_total = (int(n_folds) + 1) if (by["head_start"] and n_folds) else None

    elapsed = float(done.get("elapsed_s") or 0.0)
    if not elapsed:
        totals = [float(e["total_elapsed_s"]) for e in by["fit_done"] if "total_elapsed_s" in e]
        elapsed = max(totals) if totals else (_ts(last) - _ts(start) if start else 0.0)

    eta: float | None = None
    if mean_fit_s is not None and fits_total:
        eta = mean_fit_s * max(fits_total - fits_done, 0)
        if mean_head_s is not None and heads_total:
            eta += mean_head_s * max(heads_total - heads_done, 0)
        eta = round(eta, 1)

    state = "error" if err else ("done" if done else ("running" if events else "unknown"))
    return {
        "exp": last.get("exp") or start.get("exp"),
        "state": state,
        "updated_at": now_iso(),
        "started_at": start.get("ts"),
        "last_event": last.get("event"),
        "last_event_ts": last.get("ts"),
        "pid": start.get("pid"),
        "host": start.get("host"),
        "git": start.get("git"),
        "config_digest": start.get("config_digest"),
        "scheme": start.get("scheme"),
        "model_kind": start.get("model_kind"),
        "device": start.get("device"),
        "n_folds": n_folds,
        "seeds": seeds,
        "n_groups": n_groups,
        "n_airport_units": n_units,
        "n_features": feats.get("n_features"),
        "n_train": feats.get("n_train"),
        "n_test": feats.get("n_test"),
        "fits_done": fits_done,
        "fits_total": fits_total,
        "fit_rmse": [e.get("rmse") for e in by["fit_done"]],
        "mean_fit_s": mean_fit_s,
        "heads_done": heads_done,
        "heads_total": heads_total,
        "mean_head_s": mean_head_s,
        "elapsed_s": round(elapsed, 1),
        "eta_s": eta,
        "cv_rmse": done.get("cv_rmse"),
        "per_month": done.get("per_month"),
        "cv_rmse_no_head": done.get("cv_rmse_no_head"),
        "peak_rss_mb": done.get("peak_rss_mb"),
        "error": err.get("message"),
    }


def read_events(run_dir: str | Path) -> list[dict]:
    """Every parsable line of ``<run_dir>/events.jsonl`` (a half-written tail is skipped)."""
    path = Path(run_dir) / EVENTS_NAME
    out: list[dict] = []
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def read_status(run_dir: str | Path) -> dict | None:
    """``<run_dir>/status.json``, or ``None`` when it is absent or unreadable."""
    path = Path(run_dir) / STATUS_NAME
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
