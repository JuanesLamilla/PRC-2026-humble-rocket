"""route_no_ground.py base expert out [--flag FILE]: the expert run's prediction columns on rows without an ADS-B
ground track (adsb_has_gnd != 1, from the adsb_ground block), base's elsewhere; oof + test_pred.

--flag FILE reads (MVT_ID_mvt, adsb_has_gnd) from a parquet instead of the adsb_ground block. Once the ADS-B archive
holds the traces-rebuilt 2025-01-01..04-20 days, the base members (trained on the heatmap days) still need the heatmap
state: routing then reads the flag written from the heatmap archive (outputs/route/adsb_has_gnd_heatmap.parquet,
scripts/write_route_flag.py)."""

import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "src")
from prc.features.adsb_ground import adsb_ground

args = sys.argv[1:]
flag = None
if "--flag" in args:
    i = args.index("--flag")
    flag = pl.read_parquet(args[i + 1]).select("MVT_ID_mvt", "adsb_has_gnd")
    del args[i : i + 2]
base, exp, out = args


def seed_cols(columns: list[str]) -> list[str]:
    return sorted((c for c in columns if re.fullmatch(r"pred_s\d+", c)), key=lambda c: int(c[6:]))


args_flag = sys.argv[sys.argv.index("--flag") + 1] if flag is not None else "adsb_ground block"
R = Path("outputs/runs")
(R / out).mkdir(exist_ok=True)
for f in ("oof", "test_pred"):
    b = pl.read_parquet(R / base / f"{f}.parquet")
    e = pl.read_parquet(R / exp / f"{f}.parquet")
    g = flag if flag is not None else adsb_ground(b.select("MVT_ID_mvt")).select("MVT_ID_mvt", "adsb_has_gnd")
    nog = (
        b.select("MVT_ID_mvt")
        .join(g, on="MVT_ID_mvt", how="left", maintain_order="left")["adsb_has_gnd"]
        .fill_null(0)
        .to_numpy()
        != 1
    )
    e = b.select("MVT_ID_mvt").join(e, on="MVT_ID_mvt", how="left", maintain_order="left")
    cols = [
        c
        for c in b.columns
        if c.startswith("pred")
        or c in ("y_hat_normal", "p_sched_default", "p_rollover", "p_stamp_early")
    ]
    # seed slots: an expert merged from other seeds (e.g. s2-3-4 after a diverged seed) fills the base's
    # per-seed columns by position (seeds are exchangeable); pred itself is the seed mean either way
    src = {c: c for c in cols}
    bs, es = seed_cols(b.columns), seed_cols(e.columns)
    if bs != es:
        assert len(bs) == len(es), f"seed columns differ in number: {bs} vs {es}"
        src.update(dict(zip(bs, es, strict=True)))
        print(f, "seed slots", dict(zip(bs, es, strict=True)))
    b = b.with_columns(
        [pl.Series(c, np.where(nog, e[src[c]].to_numpy(), b[c].to_numpy())) for c in cols]
    )
    b.write_parquet(R / out / f"{f}.parquet")
    print(f, "routed rows", int(nog.sum()), "of", len(nog))
for extra in ("metrics.json",):
    if (R / base / extra).exists():
        shutil.copy(R / base / extra, R / out / extra)
(R / out / "meta.json").write_text(
    json.dumps({"routed": {"base": base, "expert": exp, "rule": "adsb_has_gnd != 1", "flag": args_flag}}, indent=1)
)
