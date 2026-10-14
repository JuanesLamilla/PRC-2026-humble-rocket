"""Reproduce every computed number, table value and figure of the manuscript.

Run from the repository root:
    python manuscript/scripts/make_figures_tables.py
Inputs : data/raw/training_2025-*.parquet, data/raw/final_ranking.parquet (inputs only, no labels),
         data/internal/fullstack_work/blend_oof.parquet (month-held blend predictions from final/blend.py)
Outputs: manuscript/figures/*.pdf|png and manuscript/scripts/out/*.csv|json (values quoted in main.tex;
         see manuscript/PROVENANCE.md). The method diagram is drawn separately:
    dot -Tpdf manuscript/scripts/method_overview.dot -o manuscript/figures/method_overview.pdf
"""
import os, glob, json, math
import numpy as np, polars as pl, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = "data/raw/"
OOF = "data/internal/fullstack_work/blend_oof.parquet"
OUTD = "manuscript/scripts/out/"; OUT = "manuscript/figures/"
os.makedirs(OUTD, exist_ok=True); os.makedirs(OUT, exist_ok=True)
def raw():
    fs=sorted(glob.glob(R+'training_2025-*.parquet'))
    cols=['MVT_ID_mvt','PHASE_mvt','ADEP_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','TAXITIME_SEC_mvt','LOBT_flt','EOBT_1_flt','AOBT_3_flt']
    return pl.concat([pl.read_parquet(f,columns=cols) for f in fs])


# ---- dataset statistics (Section 3, Table 1)
r=raw().filter(pl.col('PHASE_mvt')=='DEP')
s=lambda a,b:(pl.col(a)-pl.col(b)).dt.total_seconds()
r=r.with_columns(y=pl.col('TAXITIME_SEC_mvt'), bs=s('BLOCK_TIME_UTC_mvt','SCHED_TIME_UTC_mvt'), bl=s('BLOCK_TIME_UTC_mvt','LOBT_flt'),
  C=s('MVT_TIME_UTC_mvt','SCHED_TIME_UTC_mvt'), ml=s('MVT_TIME_UTC_mvt','LOBT_flt'), nm=pl.col('LOBT_flt').is_not_null())
r=r.with_columns(dflt=pl.col('bs').abs()<=10, ro=(pl.col('y')-86400).abs()<3600)
print('check y==MVT-BLOCK', (r['y']-s('MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt').alias('x').pipe(lambda e: r.select(e)['x'])).abs().max())
print('max |BL| NM', r.filter('nm')['bl'].abs().max())
print('default share', r['dflt'].mean(), 'rollovers', r['ro'].sum(), 'nonm', (~r['nm']).sum(), (~r['nm']).mean())
print('label stats', r['y'].describe())
print('neg labels', (r['y']<0).sum(), ' >3600', (r['y']>3600).sum())
ap=r.group_by('ADEP_mvt').agg(n=pl.len(), nm=pl.col('nm').mean(), dflt=pl.col('dflt').mean(), med=pl.col('y').median(), mean=pl.col('y').mean(), p90=pl.col('y').quantile(0.9), Cge1h=(pl.col('C')>=3600).mean()).sort('ADEP_mvt')
print(ap)
ap.write_csv(OUTD+'airport_stats.csv')
r.select('MVT_ID_mvt','ADEP_mvt','y','bs','bl','C','ml','nm','dflt','ro').write_parquet(OUTD+'deps.parquet')


# ---- month-held evaluation with day-block bootstrap (Tables 2 and 5, Figure 3)
o=pl.read_parquet(OOF)
d=pl.read_parquet(OUTD+'deps.parquet').select('MVT_ID_mvt','ml','bs','dflt','ro','bl')
o=o.join(d,on='MVT_ID_mvt',how='left')
print(o.null_count())
o=o.with_columns(nmb=pl.col('nm')==1)
# clip
o=o.with_columns(p=pl.when(pl.col('nmb')).then(pl.col('blend').clip(pl.col('ml')-3620,pl.col('ml')+3620)).otherwise(pl.col('blend')))
print('max change by clip', (o['p']-o['blend']).abs().max())
o=o.with_columns(e2=(pl.col('p')-pl.col('y'))**2)
rm=lambda df: float(np.sqrt(df['e2'].mean()))
x=o.filter(~pl.col('ro'))
print('ex_ro',rm(x),'all',rm(o), 'n ex_ro', x.height)
lane=pl.when(pl.col('nmb')&(pl.col('C')<3600)).then(pl.lit('nm_lt1h')).when(pl.col('nmb')).then(pl.lit('nm_ge1h')).when(pl.col('ADEP_mvt')=='LIRF').then(pl.lit('nonm_lirf')).otherwise(pl.lit('nonm_other'))
x=x.with_columns(lane=lane)
o=o.with_columns(lane=lane)
# day-block bootstrap: blocks airport x day, resample within airport-month
blk=x.group_by('ADEP_mvt','mon','day').agg(sse=pl.col('e2').sum(),n=pl.len())
rng=np.random.default_rng(20261012); B=2000
def boot(blkdf, groupkey=None):
    # returns array of bootstrap RMSE for the rows in blkdf
    tot_sse=np.zeros(B); tot_n=np.zeros(B)
    parts=blkdf.sort('ADEP_mvt','mon','day').partition_by(['ADEP_mvt','mon'],as_dict=True,maintain_order=True)
    for k in sorted(parts):
        g=parts[k]
        sse=g['sse'].to_numpy(); n=g['n'].to_numpy(); m=len(sse)
        idx=rng.integers(0,m,size=(B,m))
        tot_sse+=sse[idx].sum(1); tot_n+=n[idx].sum(1)
    return np.sqrt(tot_sse/tot_n)
res={}
b=boot(blk); res['overall_ex_ro']=(rm(x),*np.percentile(b,[2.5,97.5]))
for m in range(1,13):
    bm=boot(blk.filter(pl.col('mon')==m)); res[f'month_{m}']=(rm(x.filter(pl.col('mon')==m)),*np.percentile(bm,[2.5,97.5]))
for a in sorted(x['ADEP_mvt'].unique().to_list()):
    ba=boot(blk.filter(pl.col('ADEP_mvt')==a)); res[f'ap_{a}']=(rm(x.filter(pl.col('ADEP_mvt')==a)),*np.percentile(ba,[2.5,97.5]), x.filter(pl.col('ADEP_mvt')==a).height)
for l in ['nm_lt1h','nm_ge1h','nonm_other','nonm_lirf']:
    xl=x.filter(pl.col('lane')==l); bl=xl.group_by('ADEP_mvt','mon','day').agg(sse=pl.col('e2').sum(),n=pl.len())
    res[f'lane_{l}']=(rm(xl),*np.percentile(boot(bl),[2.5,97.5]), xl.height, float(xl['e2'].sum()/x['e2'].sum()))
# error share incl rollovers (all rows)
tot=o['e2'].sum()
grp=pl.when(pl.col('ro')).then(pl.lit('rollover')).when(~pl.col('nmb')).then(pl.lit('no_nm')).when(pl.col('C')>=3600).then(pl.lit('nm_Cge1h')).when(pl.col('y')>3600).then(pl.lit('nm_Clt1h_ygt1h')).otherwise(pl.lit('rest'))
es=o.with_columns(g=grp).group_by('g').agg(n=pl.len(),sse=pl.col('e2').sum(),rmse=pl.col('e2').mean().sqrt()).with_columns(share_rows=pl.col('n')/o.height,share_sse=pl.col('sse')/tot).sort('g')
print(es)
es.write_csv(OUTD+'error_share.csv')
json.dump({k:list(map(float,v)) for k,v in res.items()},open(OUTD+'oof_results.json','w'),indent=1)
for k,v in res.items(): print(k,[round(t,2) for t in v])
x.select('ADEP_mvt','mon','day','lane','y','p','C','nmb','dflt').write_parquet(OUTD+'oof_eval.parquet')


# ---- schedule coincidence, recording classes, single-record influence (Section 3.3, 6.1)
d=pl.read_parquet(OUTD+'deps.parquet')
w=lambda c:(pl.col('bs')-c).abs()<=10
sc=d.group_by('ADEP_mvt').agg(z=w(0).mean()*100,nb=((w(60).mean()+w(-60).mean())/2)*100,n=pl.len()).sort('ADEP_mvt')
sc.write_csv(OUTD+'schedule_coincidence.csv')
print('excess rows', float(((sc['z']-sc['nb'])/100*sc['n']).sum()))
near=(~pl.col('dflt'))&(pl.col('bs').abs()<=900)&(pl.col('C')>1800)
long=(pl.col('y')>3600)&(~pl.col('ro'))
cls=pl.when(pl.col('ro')).then(pl.lit('rollover')).when(pl.col('dflt')).then(pl.lit('default')).when(near).then(pl.lit('near')).when(long).then(pl.lit('long')).otherwise(pl.lit('normal'))
d.with_columns(k=cls).group_by('k').agg(n=pl.len(),y_med=pl.col('y').median()).write_csv(OUTD+'recording_classes.csv')
print('schedule infeasible share (matched)', d.filter('nm').select(((pl.col('bl')-pl.col('bs')).abs()>3616).mean()).item())
x=pl.read_parquet(OUTD+'oof_eval.parquet').with_columns(e2=(pl.col('p')-pl.col('y'))**2)
big=(pl.col('ADEP_mvt')=='LFPG')&(pl.col('y')>50000)   # the single 16.2-h unmatched LFPG record
print('16.2-h record', x.filter(big).select('y','day','lane'))
print('its share of SSE', x.filter(big)['e2'].sum()/x['e2'].sum())
xx=x.filter(~big); print('RMSE without it', math.sqrt(xx['e2'].mean()), 'Jan', math.sqrt(xx.filter(pl.col('mon')==1)['e2'].mean()), 'LFPG', math.sqrt(xx.filter(pl.col('ADEP_mvt')=='LFPG')['e2'].mean()), 'unmatched outside LIRF', math.sqrt(xx.filter(pl.col('lane')=='nonm_other')['e2'].mean()))
jd=x.filter(pl.col('mon')==1).group_by('ADEP_mvt','day').agg(sse=pl.col('e2').sum()).sort('sse',descending=True)
print('top airport-day share of January SSE', jd['sse'][0]/jd['sse'].sum(), jd.row(0))
print('LIRF unmatched labels > 20000 s', x.filter((pl.col('lane')=='nonm_lirf')&(pl.col('y')>20000)).height)
# ---- EOBT drift (Section 6.4, Figure 4): inputs only
s=lambda a,b:(pl.col(a)-pl.col(b)).dt.total_seconds()
def agg(df,tag):
    return df.filter((pl.col('PHASE_mvt')=='DEP')&pl.col('LOBT_flt').is_not_null()&pl.col('EOBT_1_flt').is_not_null()).with_columns(mon=pl.col('MVT_TIME_UTC_mvt').dt.month(), d=s('EOBT_1_flt','LOBT_flt')).group_by('ADEP_mvt','mon').agg(m=pl.col('d').mean(),n=pl.len()).with_columns(yr=pl.lit(tag))
f=pl.read_parquet(R+'final_ranking.parquet',columns=['ADEP_mvt','PHASE_mvt','MVT_TIME_UTC_mvt','LOBT_flt','EOBT_1_flt'])
pl.concat([agg(raw(),'2025'),agg(f,'2026')]).write_csv(OUTD+'eobt_drift.csv')


# ---- figures
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.spines.top':False,'axes.spines.right':False,
 'axes.edgecolor':'#52514e','axes.labelcolor':'#0b0b0b','xtick.color':'#52514e','ytick.color':'#52514e','axes.grid':True,'grid.color':'#e6e5e1','grid.linewidth':0.6,'axes.axisbelow':True})
B1,B2,B3,G='#2a78d6','#eb6834','#1baf7a','#8a8985'

d=pl.read_parquet(OUTD+'deps.parquet')
# Figure 1: label anatomy
fig,ax=plt.subplots(1,2,figsize=(7.2,2.8))
w=lambda c:(pl.col('bs')-c).abs()<=10
t=d.group_by('ADEP_mvt').agg(z=w(0).mean()*100,nb=((w(60).mean()+w(-60).mean())/2)*100).sort('ADEP_mvt')

x=np.arange(10); aps0=t['ADEP_mvt'].to_list()
ax[0].bar(x-0.2,t['z'],0.38,color=B1,label='within 10 s of schedule')
ax[0].bar(x+0.2,t['nb'],0.38,color=B3,label='within 10 s of schedule ± 60 s (mean)')
ax[0].set_xticks(x); ax[0].set_xticklabels(aps0,rotation=45,ha='right'); ax[0].set_ylabel('Share of departures (%)')
ax[0].legend(fontsize=6.8,frameon=False,loc='upper left')
ax[0].set_title('(a) Off-block stamp coinciding with the schedule',loc='left',fontsize=8.5)
bl=d.filter('nm')['bl'].to_numpy()
ax[1].hist(bl,bins=np.arange(-4000,4001,50),color=B1,edgecolor='white',linewidth=0.3)
ax[1].set_yscale('log'); ax[1].set_ylabel('Departures (log scale)'); ax[1].set_xlabel('Off-block stamp minus last flight-plan off-block time (s)')
for v in (-3606,3606): ax[1].axvline(v,color=B2,lw=1.2,ls='--')
ax[1].text(3606,ax[1].get_ylim()[1]*0.5,' ±3,606 s',color=B2,fontsize=7.5,va='top')
ax[1].set_xlim(-4000,4000)
ax[1].set_title('(b) Departures with a flight-plan record',loc='left',fontsize=8.5)
fig.tight_layout(); fig.savefig(OUT+'label_anatomy.pdf'); fig.savefig(OUT+'label_anatomy.png',dpi=220)
# Figure 3: validation by month and airport
res=json.load(open(OUTD+'oof_results.json'))
fig,ax=plt.subplots(1,2,figsize=(7.2,2.7),gridspec_kw={'width_ratios':[1.2,1]})
mons=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
v=np.array([res[f'month_{m}'] for m in range(1,13)])
ax[0].errorbar(range(12),v[:,0],yerr=[v[:,0]-v[:,1],v[:,2]-v[:,0]],fmt='o',color=B1,ms=4,capsize=2,lw=1)
ax[0].axhline(res['overall_ex_ro'][0],color=G,lw=1,ls='--'); ax[0].text(11.4,res['overall_ex_ro'][0]+4,'all months',ha='right',fontsize=7,color='#52514e')
ax[0].set_xticks(range(12)); ax[0].set_xticklabels(mons); ax[0].set_ylabel('RMSE (s)'); ax[0].set_ylim(0,330)
ax[0].set_title('(a) Held-out month, 2025',loc='left',fontsize=8.5)
aps=['EDDF','EDDM','EGLL','EHAM','LEBL','LEMD','LFPG','LIRF','LSZH','LTFM']
a=np.array([res[f'ap_{x}'][:3] for x in aps]); o=np.argsort(a[:,0])
ax[1].errorbar(a[o,0],range(10),xerr=[a[o,0]-a[o,1],a[o,2]-a[o,0]],fmt='o',color=B1,ms=4,capsize=2,lw=1)
ax[1].set_yticks(range(10)); ax[1].set_yticklabels([aps[i] for i in o]); ax[1].set_xlabel('RMSE (s)'); ax[1].set_xlim(0,450)
ax[1].set_title('(b) Airport',loc='left',fontsize=8.5)
fig.tight_layout(); fig.savefig(OUT+'validation_rmse.pdf'); fig.savefig(OUT+'validation_rmse.png',dpi=220)
# Figure 4: EOBT drift
a=pl.read_csv(OUTD+'eobt_drift.csv').pivot(on='yr',index=['ADEP_mvt','mon'],values='m').filter(pl.col('2026').is_not_null()).with_columns(diff=pl.col('2026')-pl.col('2025'))
fig,ax=plt.subplots(figsize=(7.2,2.5))
mk={1:'o',2:'s',6:'^',7:'D'}; nm={1:'January',2:'February',6:'June',7:'July'}
for i,ap in enumerate(aps):
    for m in (1,2,6,7):
        r=a.filter((pl.col('ADEP_mvt')==ap)&(pl.col('mon')==m))
        if r.height==0: continue
        val=r['diff'][0]; hi=(ap=='LFPG' and m in (6,7))
        ax.scatter(i+{1:-0.24,2:-0.08,6:0.08,7:0.24}[m],val,marker=mk[m],s=22,color=B2 if hi else B1,edgecolor='white',linewidth=0.5,zorder=3,label=nm[m] if i==0 else None)
ax.axhline(0,color='#52514e',lw=0.8)
ax.set_xticks(range(10)); ax.set_xticklabels(aps); ax.set_ylabel('Change in mean filed EOBT\nminus last off-block time (s)')
ax.annotate('LFPG, June and July',xy=(6.16,265),xytext=(7.2,250),fontsize=7.5,arrowprops=dict(arrowstyle='-',color=G,lw=0.8))
leg=ax.legend(title='2026 month vs same 2025 month',fontsize=7,title_fontsize=7,ncol=4,loc='upper left',frameon=False)
fig.tight_layout(); fig.savefig(OUT+'eobt_drift.pdf'); fig.savefig(OUT+'eobt_drift.png',dpi=220)
print('ok')
