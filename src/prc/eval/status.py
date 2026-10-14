"""What the runs under ``outputs/runs`` are doing, as plain text (``uv run prc status``).

Reads only what a run leaves behind — ``status.json`` / ``events.jsonl`` (written by
:mod:`prc.eval.events`), ``metrics.json`` and ``experiments/registry.csv`` — so it is safe
to call at any time, including while a run is training, and costs nothing but a few small
file reads. Runs that finished before events existed still list: their state comes from
``metrics.json``.

The same numbers are on disk for any other reader (a terminal monitor can read
``status.json`` for the summary and ``events.jsonl`` for the timeline).
"""

from __future__ import annotations

import csv
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from prc import paths
from prc.eval.events import EVENTS_NAME, read_events, read_status


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def _read_json(p: Path) -> dict:
    try:
        out = json.loads(p.read_text(encoding="utf-8"))
        return out if isinstance(out, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def registry_digests(path: Path | None = None) -> dict[str, str]:
    """``run name -> config_digest`` from the registry (the last row for a name wins)."""
    path = path or paths.EXPERIMENTS / "registry.csv"
    out: dict[str, str] = {}
    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                name = (row.get("name") or "").strip()
                if name:
                    out[name] = (row.get("config_digest") or "").strip()
    except (OSError, csv.Error):
        return out
    return out


def _free_mb() -> str:
    try:
        out = subprocess.run(
            ["free", "-m"], capture_output=True, text=True, timeout=5, check=False
        ).stdout
        for line in out.splitlines():
            if line.lower().startswith("mem:"):
                return line.split()[-1] + " MB free"
    except (OSError, subprocess.SubprocessError):
        pass
    return "mem ?"


def _gpu() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""
    return "  |  gpu " + "; ".join(line.strip() for line in out.splitlines()) if out else ""


def machine_line() -> str:
    """``free -m`` available memory and, where nvidia-smi exists, GPU use — never fails."""
    now = datetime.now(UTC).isoformat(timespec="seconds")
    return f"{now}  |  {_free_mb()}{_gpu()}"


def _hms(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    s = round(float(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def _month_rmse(per_month: dict | None, suffix: str) -> float | None:
    for k, v in (per_month or {}).items():
        if str(k).endswith(suffix):
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
    return None


def _num(v: float | None, nd: int = 1) -> str:
    return "-" if v is None else f"{v:.{nd}f}"


def collect(runs_dir: Path | None = None) -> list[dict[str, Any]]:
    """One summary dict per run directory, newest activity first."""
    runs_dir = runs_dir or paths.RUNS
    digests = registry_digests()
    rows: list[dict[str, Any]] = []
    if not runs_dir.is_dir():
        return rows
    for d in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        st = read_status(d) or {}
        metrics = _read_json(d / "metrics.json")
        last = max(_mtime(d / EVENTS_NAME), _mtime(d / "metrics.json"), _mtime(d))
        state = st.get("state") or ("done" if metrics else "unknown")
        per_month = st.get("per_month") or metrics.get("per_month")
        rows.append(
            {
                "name": d.name,
                "state": state,
                "last": last,
                "cv_rmse": st.get("cv_rmse") or metrics.get("cv_rmse"),
                "jan": _month_rmse(per_month, "-01"),
                "jul": _month_rmse(per_month, "-07"),
                "fits_done": st.get("fits_done"),
                "fits_total": st.get("fits_total"),
                "eta_s": st.get("eta_s"),
                "elapsed_s": st.get("elapsed_s") or metrics.get("elapsed_s"),
                "peak_rss_mb": st.get("peak_rss_mb"),
                "digest": digests.get(d.name)
                or st.get("config_digest")
                or metrics.get("config_digest"),
                "last_event": st.get("last_event"),
                "error": st.get("error"),
            }
        )
    return sorted(rows, key=lambda r: r["last"], reverse=True)


_COLS = ("run", "state", "elapsed", "pooled", "jan", "jul", "fits", "eta", "rss_mb", "digest")


def render(runs_dir: Path | None = None, limit: int = 0) -> str:
    """The run table, newest activity first (``limit`` 0 = all runs)."""
    rows = collect(runs_dir)
    if limit:
        rows = rows[:limit]
    if not rows:
        return machine_line() + "\nno runs under " + str(runs_dir or paths.RUNS)
    w = max(len(r["name"]) for r in rows)
    head = f"{_COLS[0]:<{w}}  {'state':<8}{'elapsed':>9}{'pooled':>9}{'jan':>8}{'jul':>8}"
    head += f"{'fits':>9}{'eta':>8}{'rss_mb':>9}  digest"
    out = [machine_line(), head, "-" * len(head)]
    for r in rows:
        fits = f"{r['fits_done']}/{r['fits_total'] or '?'}" if r["fits_done"] is not None else "-"
        line = f"{r['name']:<{w}}  {r['state']!s:<8}{_hms(r['elapsed_s']):>9}"
        line += f"{_num(r['cv_rmse']):>9}{_num(r['jan']):>8}{_num(r['jul']):>8}"
        line += f"{fits:>9}{_hms(r['eta_s']) if r['state'] == 'running' else '-':>8}"
        line += f"{_num(r['peak_rss_mb'], 0):>9}  {r['digest'] or '-'}"
        out.append(line)
        if r["error"]:
            out.append(" " * (w + 2) + "error: " + str(r["error"]).splitlines()[0][:120])
    return "\n".join(out)


def _fmt_fields(e: dict) -> str:
    skip = {"ts", "exp", "event"}
    parts = []
    for k, v in e.items():
        if k in skip:
            continue
        if isinstance(v, float):
            v = round(v, 3)
        if isinstance(v, (dict, list)):
            v = json.dumps(v, separators=(",", ":"))
            if len(v) > 90:
                v = v[:87] + "..."
        parts.append(f"{k}={v}")
    return " ".join(parts)


def timeline(run: str, runs_dir: Path | None = None) -> str:
    """One line per event of a run, oldest first."""
    d = (runs_dir or paths.RUNS) / run
    events = read_events(d)
    st = read_status(d) or {}
    out = [machine_line(), f"{run}  [{st.get('state', 'unknown')}]"]
    if st.get("pid"):
        out[-1] += f"  pid {st['pid']} on {st.get('host')}"
    if not events:
        out.append(f"no {EVENTS_NAME} in {d} (a run started before progress events existed)")
        return "\n".join(out)
    for e in events:
        ts = str(e.get("ts", ""))[11:19] or "--:--:--"
        out.append(f"{ts}  {e.get('event', '?')!s:<10} {_fmt_fields(e)}")
    return "\n".join(out)
