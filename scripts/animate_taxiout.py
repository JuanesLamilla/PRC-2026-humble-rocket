"""Animated illustration of the label: departures at Zurich on 3 July 2025.

Draws each departure of a morning window as a horizontal bar from off-block to
take-off (the taxi-out time the challenge scores), coloured by recording class:
normal, schedule-default stamp (|BLOCK - SCHED| <= 10 s) and no-flight-plan.
Writes docs/assets/taxiout.gif. Illustration only; nothing feeds the model.

Needs matplotlib on top of the locked packages:
    uv run --with matplotlib python scripts/animate_taxiout.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter

RAW = "data/raw/training_2025-07-01_2025-08-01.parquet"
APT, DAY = "LSZH", "2025-07-03"
T0, T1 = pd.Timestamp(f"{DAY} 05:00", tz="UTC"), pd.Timestamp(f"{DAY} 08:00", tz="UTC")
OUT = "docs/assets/taxiout.gif"

COLORS = {"normal": "#4a5a9c", "schedule-default stamp": "#d9772e",
          "no flight plan": "#8a8f9c"}

df = pd.read_parquet(RAW, columns=[
    "ADEP_mvt", "PHASE_mvt", "BLOCK_TIME_UTC_mvt", "MVT_TIME_UTC_mvt",
    "SCHED_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "LOBT_flt"])
d = df[df.PHASE_mvt.eq("DEP") & df.ADEP_mvt.eq(APT) & df.TAXITIME_SEC_mvt.notna()].copy()
d["blk"] = pd.to_datetime(d.BLOCK_TIME_UTC_mvt)
d["mvt"] = pd.to_datetime(d.MVT_TIME_UTC_mvt)
d = d[(d.blk >= T0) & (d.blk < T1)].sort_values("blk").reset_index(drop=True)
sched = pd.to_datetime(d.SCHED_TIME_UTC_mvt)
d["cls"] = "normal"
d.loc[d.LOBT_flt.isna(), "cls"] = "no flight plan"
d.loc[(sched - d.blk).abs().dt.total_seconds() <= 10, "cls"] = "schedule-default stamp"

t_end = d.mvt.max() + pd.Timedelta(minutes=2)
frames = pd.date_range(T0, t_end, freq="2min")

fig, ax = plt.subplots(figsize=(9.4, 5.4), dpi=100)
fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.12)
ax.set_xlim(T0, t_end)
ax.set_ylim(-1, len(d))
ax.set_yticks([])
ax.set_ylabel("departures, in off-block order")
ax.set_xlabel(f"UTC, {DAY}")
ax.set_title(f"{APT} departures, {DAY} morning: off-block to take-off "
             "(the taxi-out time the challenge scores)", fontsize=11)
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
ax.legend(handles=[matplotlib.patches.Patch(color=col, label=c)
                   for c, col in COLORS.items()],
          loc="lower right", fontsize=8, framealpha=0.9)
now_line = ax.axvline(T0, color="#1f2430", lw=1)
bars = ax.barh(range(len(d)), [pd.Timedelta(0)] * len(d), left=d.blk,
               height=0.8, color=[COLORS[c] for c in d.cls])
info = ax.text(0.015, 0.97, "", transform=ax.transAxes, va="top", fontsize=9,
               bbox=dict(boxstyle="round", fc="white", alpha=0.85, ec="#4a5a9c"))


def update(now):
    now_line.set_xdata([now, now])
    started = d.blk <= now
    for bar, b, m, s in zip(bars, d.blk, d.mvt, started):
        bar.set_width(max((min(now, m) - b).total_seconds(), 0) / 86400)
        bar.set_alpha(1.0 if s and m > now else 0.6)
    taxiing = int((started & (d.mvt > now)).sum())
    done = d[d.mvt <= now]
    recent = done[done.mvt >= now - pd.Timedelta(minutes=30)]
    med = f"{recent.TAXITIME_SEC_mvt.median() / 60:.0f} min" if len(recent) else "n/a"
    info.set_text(f"{now:%H:%M} UTC   taxiing now: {taxiing:2d}   "
                  f"median taxi, last 30 min of take-offs: {med}")
    return [*bars, now_line, info]


anim = FuncAnimation(fig, update, frames=frames, blit=False)
anim.save(OUT, writer=PillowWriter(fps=9))
print(f"wrote {OUT}: {len(d)} departures, {len(frames)} frames; "
      f"classes: {d.cls.value_counts().to_dict()}")
