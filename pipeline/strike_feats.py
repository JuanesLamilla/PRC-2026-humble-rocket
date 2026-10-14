"""Per-departure strike / disruption features from assets/strike_calendar.csv (compiled from public sources 2026-10-06;
cancelled actions and long low-impact background strikes dropped). Writes work/strike_feats.parquet keyed by MVT_ID_mvt."""
import polars as pl
from common import load_movements, RESULTS
K = "MVT_ID_mvt"
TZ = {"EDDF": "Europe/Berlin", "EDDM": "Europe/Berlin", "EGLL": "Europe/London", "EHAM": "Europe/Amsterdam", "LEBL": "Europe/Madrid",
      "LEMD": "Europe/Madrid", "LFPG": "Europe/Paris", "LIRF": "Europe/Rome", "LSZH": "Europe/Zurich", "LTFM": "Europe/Istanbul"}
GROUP = {"ALL_EU": list(TZ), "ALL_IT": ["LIRF"], "ALL_FR": ["LFPG"], "ALL_DE": ["EDDF", "EDDM"], "ALL_ES": ["LEBL", "LEMD"],
         "ALL_UK": ["EGLL"], "ALL_NL": ["EHAM"], "ALL_CH": ["LSZH"], "ALL_TR": ["LTFM"]}
cal = pl.read_csv("assets/strike_calendar.csv", infer_schema_length=0)
notes = pl.col("notes").fill_null("")
cal = cal.filter(~notes.str.starts_with("CANCELLED") & ~notes.str.starts_with("LOW_IMPACT_BACKGROUND"))
rows = []
for r in cal.iter_rows(named=True):
    aps = []
    for a in (r["airports"] or "").split(";"):
        a = a.strip(); aps += GROUP.get(a, [a] if a in TZ else [])
    for a in set(aps):
        rows.append(dict(apt=a, date=r["date"], et=r["event_type"] or "other", carrier=(r["carrier"] or "").strip(),
                         s=r["local_start"] or "", e=r["local_end"] or ""))
E = pl.DataFrame(rows).with_columns(date=pl.col("date").str.to_date())
def mins(c): return pl.when(pl.col(c).str.contains(":")).then(pl.col(c).str.slice(0, 2).cast(pl.Int32) * 60 + pl.col(c).str.slice(3, 2).cast(pl.Int32))
E = E.with_columns(smin=mins("s").fill_null(0), emin=mins("e").fill_null(24 * 60))
print("events after filter:", cal.height, "airport-day rows:", E.height); print(E.group_by("et").len())
dep = load_movements().filter(pl.col("PHASE_mvt") == "DEP").select(K, apt="ADEP_mvt", ades="ADES_mvt", op=pl.col("FLIGHT_mvt").str.slice(0, 3), t="MVT_TIME_UTC_mvt")
parts = []
for a, tz in TZ.items():
    parts.append(dep.filter(pl.col("apt") == a).with_columns(loc=pl.col("t").dt.replace_time_zone("UTC").dt.convert_time_zone(tz).dt.replace_time_zone(None)))
dep = pl.concat(parts).with_columns(date=pl.col("loc").dt.date(), mn=pl.col("loc").dt.hour().cast(pl.Int32) * 60 + pl.col("loc").dt.minute().cast(pl.Int32))
J = dep.select(K, "apt", "date", "mn", "op").join(E, on=["apt", "date"], how="inner").with_columns(
    inwin=(pl.col("mn") >= pl.col("smin")) & (pl.col("mn") <= pl.col("emin")),
    staff=pl.col("et").is_in(["handling", "airport_staff", "security", "general_strike"]),
    crew_own=(pl.col("et") == "airline_crew") & (pl.col("carrier") == pl.col("op")))
F = J.group_by(K).agg(stk_any=pl.lit(1.0), stk_staff=pl.col("staff").max().cast(pl.Float32), stk_staff_inwin=(pl.col("staff") & pl.col("inwin")).max().cast(pl.Float32),
    stk_atc=(pl.col("et") == "atc").max().cast(pl.Float32), stk_crew_any=(pl.col("et") == "airline_crew").max().cast(pl.Float32),
    stk_crew_own=pl.col("crew_own").max().cast(pl.Float32), stk_disrupt=pl.col("et").is_in(["weather", "it_outage", "other"]).max().cast(pl.Float32),
    stk_inwin=pl.col("inwin").max().cast(pl.Float32), stk_n=pl.len().cast(pl.Float32))
prev = dep.select(K, "apt", date=pl.col("date") - pl.duration(days=1)).join(E.select("apt", "date").unique(), on=["apt", "date"], how="inner").select(K, stk_prevday=pl.lit(1.0))
fr = dep.filter(pl.col("ades").str.starts_with("LF")).select(K, "date").join(E.filter((pl.col("et") == "atc") & (pl.col("apt") == "LFPG")).select("date").unique(), on="date").select(K, stk_atc_dest_fr=pl.lit(1.0))
out = dep.select(K).join(F, on=K, how="left").join(prev, on=K, how="left").join(fr, on=K, how="left").fill_null(0.0)
out.write_parquet("work/strike_feats.parquet")
print(out.select([pl.col(c).mean() for c in out.columns if c != K]))
