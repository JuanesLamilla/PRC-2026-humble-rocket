"""write_route_flag.py run [--out FILE]: write the (MVT_ID_mvt, adsb_has_gnd) routing flag that
route_no_ground.py --flag reads, from the adsb_ground block over the run's oof + test_pred rows.

The flag of the final chain (outputs/route/adsb_has_gnd_heatmap.parquet) is written from the
**heatmap** ADS-B archive, before the 2025-01-01..04-20 traces rebuild; routing must use the heatmap
flag so that it matches the base members' training state (docs/05-reproduction.md). Run this while data/external/ holds the
heatmap archive. Rows missing from the archive keep a null flag (route_no_ground.py treats null as
"no ground track")."""

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, "src")
from prc.features.adsb_ground import adsb_ground

args = sys.argv[1:]
out = Path("outputs/route/adsb_has_gnd_heatmap.parquet")
if "--out" in args:
    i = args.index("--out")
    out = Path(args[i + 1])
    del args[i : i + 2]
(run,) = args
R = Path("outputs/runs") / run
ids = pl.concat(
    [pl.read_parquet(R / f"{f}.parquet", columns=["MVT_ID_mvt"]) for f in ("oof", "test_pred")]
).unique(maintain_order=True)
flag = adsb_ground(ids).select("MVT_ID_mvt", pl.col("adsb_has_gnd").cast(pl.Int8))
out.parent.mkdir(parents=True, exist_ok=True)
if out.exists():
    sys.exit(f"{out} exists; pass --out to write elsewhere")
flag.write_parquet(out)
print(out, flag.height, flag["adsb_has_gnd"].value_counts().sort("adsb_has_gnd").rows())
