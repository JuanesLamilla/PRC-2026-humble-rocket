"""avg_runs.py a b out: average the pred* columns of two run dirs (e.g. mixture applies of two seeds); report the shift of b vs a."""
import sys, json, shutil
from pathlib import Path
import polars as pl
a, b, out = sys.argv[1:]
R = Path("outputs/runs")
(R / out).mkdir(exist_ok=True)
for f in ("oof", "test_pred"):
    da = pl.read_parquet(R / a / f"{f}.parquet"); db = pl.read_parquet(R / b / f"{f}.parquet")
    assert da["MVT_ID_mvt"].equals(db["MVT_ID_mvt"])
    pc = [c for c in da.columns if c.startswith("pred")]
    for c in da.columns:
        if c not in pc and c != "MVT_ID_mvt" and da[c].dtype.is_numeric():
            assert (da[c] - db[c]).abs().max() < 1e-6, c
    d = (db["pred"] - da["pred"])
    print(f"{a[:28]:28s} {f:9s} shift mean {d.mean():+7.2f} rmsd {(d**2).mean()**.5:6.2f}", flush=True)
    da.with_columns([((pl.col(c) + db[c]) / 2).alias(c) for c in pc]).write_parquet(R / out / f"{f}.parquet")
m = json.loads((R / a / "meta.json").read_text()); m["averaged_from"] = [a, b]
(R / out / "meta.json").write_text(json.dumps(m, indent=1))
