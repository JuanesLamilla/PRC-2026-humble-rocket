"""write_candidate.py run out.parquet [ref.parquet]: a submission-shaped file from outputs/runs/<run>/test_pred.parquet
(clipped at 0, validated against the template); with ref, print the rmsd / mean move against it. Never uploads."""
import sys, polars as pl, numpy as np
from prc.data import schema as S
from prc.data.load import load_submitting_template
from prc.submit.build import validate
run, out, ref = sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None
tp = pl.read_parquet(f"outputs/runs/{run}/test_pred.parquet")
template = load_submitting_template()
tmpl = template.select(S.MVT_ID)
p = tp.select(S.MVT_ID, pl.col("pred").cast(pl.Float64).clip(lower_bound=0).alias(S.TARGET))
sub = tmpl.join(p, on=S.MVT_ID, how="left", maintain_order="left")
dt = template.schema[S.TARGET]
if dt.is_integer():
    sub = sub.with_columns(pl.col(S.TARGET).round(0).cast(dt))
validate(sub, tmpl)
sub.write_parquet(out)
print(out, sub.height, "neg clipped:", int((tp["pred"] < 0).sum()))
if ref:
    r = pl.read_parquet(ref).join(sub, on=S.MVT_ID, suffix="_new")
    d = (r[S.TARGET + "_new"].cast(pl.Float64) - r[S.TARGET].cast(pl.Float64)).to_numpy()
    print(f"vs {ref}: rmsd {np.sqrt((d**2).mean()):.2f} mean {d.mean():+.2f}")
