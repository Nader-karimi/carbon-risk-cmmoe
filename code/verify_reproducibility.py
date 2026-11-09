#!/usr/bin/env python3
"""Independent verification of the published distributional and inference results.

Reads archived daily out-of-sample predictions; it does not refit models or
select parameters using the held-out test periods.
"""
from pathlib import Path
import math
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results'
DAILY=pd.read_csv(R/'comparative_predictions.csv')
TAB=pd.read_csv(R/'proper_score_comparisons.csv').set_index('Model')
INF=pd.read_csv(R/'ablation_inference.csv').set_index('Model')
DATA=pd.read_csv(R/'nested_predictions.csv')
assert set(['Year','Model','Date','IS90','IS95','Pinball01','Pinball05','QLIKE']).issubset(DAILY)
assert len(DATA)==1393 and len(DATA[['Year','Date']].drop_duplicates())==1393
assert DATA.Year.nunique()==6

# Model/date alignment, means for all published per-observation scores.
for model,g in DAILY.groupby('Model'):
    assert len(g)==1393 and len(g[['Year','Date']].drop_duplicates())==1393,model
    for daycol,summarycol in [('IS90','IntervalScore90'),('IS95','IntervalScore95'),
                              ('Pinball01','Pinball01'),('Pinball05','Pinball05'),
                              ('NLL','NLL'),('QLIKE','QLIKE')]:
        observed=float(g[daycol].mean());stored=float(TAB.loc[model,summarycol]);
        assert np.isclose(observed,stored,atol=1e-12,rtol=1e-10),(model,daycol,observed,stored)

# GARCH-family checks use the single integrated daily prediction file.
GARCH=pd.read_csv(R/'garch_predictions.csv')
assert len(GARCH)==3*1393
assert not GARCH[['Year','Model','Date']].duplicated().any()
assert {'IS90','IS95','Pinball01','Pinball05','CalThreshold90','Cover90','VaR01'}.issubset(GARCH)
pin=np.maximum(.01*(GARCH.Return-GARCH.VaR01),-.99*(GARCH.Return-GARCH.VaR01))
assert np.allclose(pin,GARCH.Pinball01,atol=1e-14,rtol=1e-12)
FITS=pd.read_csv(R/'garch_fit_details.csv')
merged=GARCH.merge(FITS[['Year','Model','scale_head']],on=['Year','Model'],validate='many_to_one')
assert len(merged)==len(GARCH) and not merged.scale_head.isna().any()
sd=np.sqrt(merged.VarForecast.to_numpy(float)*merged.scale_head.to_numpy(float))
threshold=merged.CalThreshold90.to_numpy(float)
y=np.abs(merged.Return.to_numpy(float))
is90=2*threshold*sd+20*np.maximum(y-threshold*sd,0)
assert np.allclose(is90,merged.IS90.to_numpy(float),atol=1e-11,rtol=1e-10)
assert np.array_equal(y<=threshold*sd,merged.Cover90.to_numpy(bool))
assert (GARCH.groupby(['Year','Model']).CalThreshold90.nunique()==1).all()
assert np.isfinite(GARCH.IS90).all() and (GARCH.IS90>=0).all()

# An additional raw-data check is run only when the user provides a licensed
# copy of the original OHLCV input at data/eua.csv; it is not redistributed here.
raw_path=ROOT/'data'/'eua.csv'
if raw_path.exists():
    from garch_benchmarks import filter_var
    from run_cmd_moe_nested import load_data
    from ast import literal_eval
    df=load_data(raw_path)
    ret=df.ret.to_numpy(float)
    dates=dict(zip(df.Date.dt.strftime('%Y-%m-%d'),np.arange(len(df))))
    for (year,model),g in GARCH.groupby(['Year','Model'],sort=False):
        mod={'GARCH-t':0,'GJR-GARCH-t':1,'EGARCH-t':2}[model]
        train=np.flatnonzero(df.Date<=pd.Timestamp(f'{year-2}-12-31'))
        val=np.flatnonzero((df.Date>=pd.Timestamp(f'{year-1}-01-01')) &
                           (df.Date<=pd.Timestamp(f'{year-1}-12-31')))
        x=100*(ret-ret[train].mean())
        record=FITS.loc[(FITS.Year==year)&(FITS.Model==model)].iloc[0]
        theta=np.array(literal_eval(record.theta),float)
        v0=max(np.var(x[train]),1e-4)
        forecasts=np.maximum(record.c_train*filter_var(x,mod,theta,v0)/10000,1e-12)
        cal=np.abs(ret[val])/np.sqrt(forecasts[val]*record.scale_head)
        thr90=np.quantile(cal,.90,method='higher')
        assert np.allclose(g.CalThreshold90,thr90,rtol=1e-10,atol=1e-12)
        ix=np.array([dates[d] for d in g.Date])
        assert np.allclose(forecasts[ix],g.VarForecast,rtol=1e-8,atol=1e-12)
else:
    print('SKIP: raw-data refiltering (licensed data/eua.csv not supplied)')

# Recompute full HAC and circular block bootstrap from *daily* paired QLIKE.
# Fixed pseudorandom seed and replicate ordering used in the published experiment.
models=['Ordinary exponential weighting','Equal weight','No Log-SV expert',
        'No Log-SV information','No EWMA experts','No nonlinear expert','No HAR-RV']
base=DAILY[DAILY.Model=='CMD-MoE'].sort_values(['Year','Date']).reset_index(drop=True)
seed=np.random.default_rng(20261028)
for model in models:
    other=DAILY[DAILY.Model==model].sort_values(['Year','Date']).reset_index(drop=True)
    assert np.array_equal(base.Date.to_numpy(),other.Date.to_numpy())
    d=other.QLIKE.to_numpy(float)-base.QLIKE.to_numpy(float)
    n=len(d);mean=d.mean();resid=d-mean
    rec=INF.loc[model]
    assert np.isclose(mean,rec.MeanLossGain,atol=2e-12)
    for lag in (3,7,14,21):
        v=np.dot(resid,resid)/n
        for k in range(1,lag+1):
            v+=2*(1-k/(lag+1))*np.dot(resid[k:],resid[:-k])/n
        stat=mean/np.sqrt(max(v,1e-18)/n)
        p=2*norm.sf(abs(stat))
        assert np.isclose(stat,rec[f'HAC_t_L{lag}'],atol=1e-9,rtol=1e-9)
        assert np.isclose(p,rec[f'HAC_p_L{lag}'],atol=1e-11,rtol=1e-8)
    for L in (10,20):
        sims=[]
        starts=math.ceil(n/L)
        for _ in range(1000):
            ids=(seed.integers(0,n,starts)[:,None]+np.arange(L)[None,:])%n
            sims.append(d[ids.ravel()[:n]].mean())
        ci=np.quantile(sims,[.025,.975])
        assert np.allclose(ci,[rec[f'Block{L}_CIlo'],rec[f'Block{L}_CIhi']],atol=1e-11,rtol=1e-9), (model,L,ci)

print('PASS: Pooled distributional scores match fold-level daily prediction records')
print('PASS: Integrated GARCH 90% interval scores and 1% quantile losses match archived forecasts')
print('PASS: HAC sensitivity (lags 3, 7, 14, 21) matches paired daily QLIKE')
print('PASS: 1,000-replication circular bootstrap (blocks 10, 20) exactly matches archived intervals')
print('PASS: Six dated out-of-sample folds and 1,393 matching observations')
