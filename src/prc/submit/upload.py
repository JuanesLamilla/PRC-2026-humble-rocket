"""Upload a submission with the MinIO client and wait for the organizers' result file.

Mechanics (reported for 2026, unverified until our first upload): after
``mc cp <file> prc/<team-bucket>/`` a ``<file>_result.json`` appears in the same bucket
within ~15 s containing the score. We poll for it and write ``lb_rmse`` back into
``submissions/log.csv``.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from pathlib import Path

from prc import paths
from prc.submit.build import LOG_COLS


def upload(
    path: Path, *, alias: str = "prc", bucket: str | None = None, wait_s: int = 120
) -> float | None:
    bucket = bucket or os.environ.get("PRC_SUBMISSION_BUCKET")
    if not bucket:
        raise ValueError("set PRC_SUBMISSION_BUCKET (env) or pass bucket=")
    target = f"{alias}/{bucket}/{path.name}"
    result_key = f"{target}_result.json"
    # 2026-09-22: a file with the same name already existed in the team bucket; `mc cp` overwrote
    # it and the poll below read the stale result file of the earlier upload. Never overwrite a
    # name that exists in the team bucket, and only accept a result written after this upload.
    if _exists(target) or _exists(result_key):
        raise FileExistsError(
            f"{path.name} already exists in {alias}/{bucket}; pick the next free version"
        )
    t_up = time.time()
    subprocess.run(["mc", "cp", str(path), f"{alias}/{bucket}/"], check=True)
    deadline = time.time() + wait_s
    while time.time() < deadline:
        r = subprocess.run(["mc", "cat", result_key], capture_output=True, text=True, check=False)
        if r.returncode == 0 and r.stdout.strip() and _mtime(result_key) >= t_up - 5:
            info = json.loads(r.stdout)
            score = info.get("score") or info.get("rmse")
            print(json.dumps(info, indent=2))
            if score is not None:
                _record(path.name, float(score))
                return float(score)
            return None
        time.sleep(5)
    print(f"no result after {wait_s}s; check `mc ls {alias}/{bucket}/` later")
    return None


def _stat(key: str) -> dict | None:
    r = subprocess.run(["mc", "stat", "--json", key], capture_output=True, text=True, check=False)
    if r.returncode != 0 or not r.stdout.strip():
        return None
    try:
        return json.loads(r.stdout.splitlines()[0])
    except json.JSONDecodeError:
        return None


def _exists(key: str) -> bool:
    st = _stat(key)
    return bool(st) and st.get("status") != "error"


def _mtime(key: str) -> float:
    """Epoch seconds of the object's LastModified; 0 when unknown."""
    from datetime import datetime

    st = _stat(key) or {}
    ts = st.get("lastModified")
    if not ts:
        return 0.0
    try:
        return datetime.fromisoformat(ts).timestamp()
    except ValueError:
        return 0.0


def _record(filename: str, lb_rmse: float) -> None:
    logp = paths.SUBMISSIONS / "log.csv"
    if not logp.exists():
        return
    with open(logp, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if r["file"] == filename:
            r["lb_rmse"] = f"{lb_rmse:.4f}"
    with open(logp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLS)
        w.writeheader()
        w.writerows(rows)
