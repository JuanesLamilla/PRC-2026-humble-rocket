"""gate_runs.py ref cand out [--c-max SECONDS]: cand's prediction columns on rows with C = MVT - SCHED below
--c-max (default 3600, label-free), ref's elsewhere; oof + test_pred. Rule of 2026-09-27: a change is taken only
on the lanes where its validation gain is clear, the reference stays on the rest.
Prints the per-C-band OOF RMSE of ref, cand and the gated run."""

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

args = sys.argv[1:]
c_max = 3600.0
if "--c-max" in args:
    i = args.index("--c-max")
    c_max = float(args[i + 1])
    del args[i : i + 2]
ref, cand, out = args
R = Path("outputs/runs")
C = (
    pl.read_parquet("outputs/mixture_v2/exp0227_r7c_realmlp_nomask_lane__v2.parquet", columns=["MVT_ID_mvt", "c_sched"])
    .unique("MVT_ID_mvt")
)
(R / out).mkdir(exist_ok=True)
for f in ("oof", "test_pred"):
    a = pl.read_parquet(R / ref / f"{f}.parquet")
    b = a.select("MVT_ID_mvt").join(pl.read_parquet(R / cand / f"{f}.parquet"), on="MVT_ID_mvt", how="left", maintain_order="left")
    c = a.select("MVT_ID_mvt").join(C, on="MVT_ID_mvt", how="left", maintain_order="left")["c_sched"].to_numpy()
    assert not np.isnan(c).any(), f"{f}: rows without c_sched"
    take = c < c_max
    cols = [x for x in a.columns if x.startswith("pred") and x in b.columns]
    g = a.with_columns([pl.Series(x, np.where(take, b[x].to_numpy(), a[x].to_numpy())) for x in cols])
    g.write_parquet(R / out / f"{f}.parquet")
    print(f"{f}: cand on {int(take.sum()):,} of {len(take):,} rows (C < {c_max:.0f} s)")
    if f == "oof" and "TAXITIME_SEC_mvt" in a.columns:
        y = a["TAXITIME_SEC_mvt"].to_numpy().astype(float)
        for lo, hi in ((0, 1800), (1800, 3600), (3600, 7200), (7200, np.inf)):
            s = (c >= lo) & (c < hi)
            r = [np.sqrt(np.mean((d[s] - y[s]) ** 2)) for d in (a["pred"].to_numpy(), b["pred"].to_numpy(), g["pred"].to_numpy())]
            print(f"  C {lo:>5.0f}-{hi:<5.0f} rows {int(s.sum()):>9,}  ref {r[0]:8.2f}  cand {r[1]:8.2f}  gated {r[2]:8.2f}")
(R / out / "meta.json").write_text(json.dumps({"gated": {"ref": ref, "cand": cand, "rule": f"c_sched < {c_max:.0f} -> cand"}}, indent=1))
