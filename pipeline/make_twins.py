"""Build work/twins.parquet (label-free): no-NM departures matched to an NM-joined ARRIVAL of the same
callsign and route at another challenge airport, arriving 10 min-6 h after take-off. ml = MVT - twin LOBT,
ma = MVT - twin AOBT_3. Prints 2025 diagnostics (these use labels; the output file does not).
usage: python pipeline/make_twins.py"""
import os, polars as pl, glob
from common import RAW, RANKING
D = RAW.rstrip("/") + "/"
a=pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(D+'training_2025-*.parquet'))]+[pl.read_parquet(D+RANKING)],how='diagonal_relaxed')
s=lambda x,y:(pl.col(x)-pl.col(y)).dt.total_seconds()
dep=a.filter(pl.col('PHASE_mvt')=='DEP'); arr=a.filter(pl.col('PHASE_mvt')=='ARR')
nofp=dep.filter(pl.col('LOBT_flt').is_null()).select('MVT_ID_mvt','ADEP_mvt','ADES_mvt','FLIGHT_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','TAXITIME_SEC_mvt')
tw=arr.filter(pl.col('LOBT_flt').is_not_null()).select(pl.col('ADEP_mvt').alias('A'),pl.col('ADES_mvt').alias('B'),'FLIGHT_mvt',pl.col('MVT_TIME_UTC_mvt').alias('ARR_MVT'),pl.col('LOBT_flt').alias('tLOBT'),pl.col('AOBT_3_flt').alias('tAOBT'),pl.col('EOBT_1_flt').alias('tEOBT'))
j=nofp.join(tw,left_on=['ADEP_mvt','ADES_mvt','FLIGHT_mvt'],right_on=['A','B','FLIGHT_mvt'])
j=j.filter(s('ARR_MVT','MVT_TIME_UTC_mvt').is_between(600,6*3600)).unique('MVT_ID_mvt',keep='first')
j=j.with_columns(ml=s('MVT_TIME_UTC_mvt','tLOBT'),ma=s('MVT_TIME_UTC_mvt','tAOBT'),C=s('MVT_TIME_UTC_mvt','SCHED_TIME_UTC_mvt'),y=pl.col('TAXITIME_SEC_mvt').cast(pl.Float64),
  yr=pl.col('MVT_TIME_UTC_mvt').dt.year())
print('twins by year', j.group_by('yr').len().sort('yr'))
print('test no-NM deps', dep.filter(pl.col('LOBT_flt').is_null() & (pl.col('MVT_TIME_UTC_mvt').dt.year()==2026)).height)
t=j.filter(pl.col('yr')==2025).with_columns(side=pl.when(pl.col('y')<pl.col('ml')-3606).then(pl.lit('below')).when(pl.col('y')>pl.col('ml')+3606).then(pl.lit('above')).otherwise(pl.lit('inside')))
print(t.group_by('side').agg(n=pl.len(),y_med=pl.col('y').median(),y_mean=pl.col('y').mean(),ml_med=pl.col('ml').median(),C_med=pl.col('C').median()))
print('2025 twins ml quantiles',[t['ml'].quantile(q) for q in (.05,.25,.5,.75,.95)])
print('ma - y (MVT-twinAOBT minus y) quantiles',[ (t['ma']-t['y']).quantile(q) for q in (.05,.25,.5,.75,.95)])
print('test twins ml quantiles', [j.filter(pl.col('yr')==2026)['ml'].quantile(q) for q in (.05,.25,.5,.75,.95)])
print(j.filter(pl.col('yr')==2026).group_by('ADEP_mvt').len().sort('ADEP_mvt'))
os.makedirs('work', exist_ok=True)
j.select('MVT_ID_mvt','ml','ma','yr').write_parquet('work/twins.parquet')
