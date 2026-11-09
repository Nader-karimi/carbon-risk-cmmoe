from pathlib import Path
import json, math
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import het_arch

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results'; T=ROOT/'tables'; F=ROOT/'figures'; D=ROOT/'data'
T.mkdir(exist_ok=True); F.mkdir(exist_ok=True)
pred=pd.read_csv(R/'nested_predictions.csv',parse_dates=['Date'])
fold=pd.read_csv(R/'fold_metrics.csv')
abl=pd.read_csv(R/'ablation_summary.csv')
pool=json.loads((R/'pooled_metrics.json').read_text())
bench=pd.read_csv(R/'risk_benchmark_table.csv')

def fmt_p(p):
    return '<0.001' if p<.001 else f'{p:.3f}'

def write(name, s): (T/name).write_text(s)

# Load/clean data consistently enough for descriptive diagnostics.
df=pd.read_csv(D/'eua.csv',encoding='utf-8-sig')
df['Date']=pd.to_datetime(df['Date'],format='%m/%d/%Y',errors='coerce')
for c in ['Price','Open','High','Low']:
    df[c]=pd.to_numeric(df[c].astype(str).str.replace(',','',regex=False),errors='coerce')
def pv(v):
    if pd.isna(v): return np.nan
    s=str(v).strip().replace(',',''); m=1
    if not s or s=='-': return np.nan
    if s[-1:].upper()=='K': m=1e3; s=s[:-1]
    elif s[-1:].upper()=='M': m=1e6; s=s[:-1]
    elif s[-1:].upper()=='B': m=1e9; s=s[:-1]
    try:return float(s)*m
    except:return np.nan
df['Volume']=df['Vol.'].map(pv)
df=df.sort_values('Date')
df=df[(df.Date>=pd.Timestamp('2018-01-02'))&(df.Date<=pd.Timestamp('2026-06-11'))]
df=df.dropna(subset=['Date','Price','Open','High','Low','Volume']).copy()
df=df[(df[['Price','Open','High','Low']]>0).all(axis=1)]
df['logp']=np.log(df.Price); df['ret']=df.logp.diff(); df=df.dropna(subset=['ret']).copy()
ret=df.ret.to_numpy(); logp=df.logp.to_numpy()
adfp=adfuller(logp,autolag='AIC'); adfr=adfuller(ret,autolag='AIC')
try: kpssp=kpss(logp,regression='c',nlags='auto')
except: kpssp=(np.nan,np.nan,None,None)
try: kpssr=kpss(ret,regression='c',nlags='auto')
except: kpssr=(np.nan,np.nan,None,None)
arch=het_arch(ret,nlags=10)
rows=[
 ('Observations',f'{len(df):,}','NA'),
 ('Price mean',f'{df.Price.mean():.3f}','EUR/t'),
 ('Price std.',f'{df.Price.std(ddof=1):.3f}','EUR/t'),
 ('Return mean',f'{ret.mean():.6f}','log return'),
 ('Return std.',f'{ret.std(ddof=1):.6f}','log return'),
 ('Return skewness',f'{stats.skew(ret,bias=False):.3f}','NA'),
 ('Return excess kurtosis',f'{stats.kurtosis(ret,fisher=True,bias=False):.3f}','NA'),
 ('ADF, log price',f'{adfp[0]:.3f}',f'p={adfp[1]:.3f}'),
 ('KPSS, log price',f'{kpssp[0]:.3f}',f'p={kpssp[1]:.3f}'),
 ('ADF, return',f'{adfr[0]:.3f}',f'p={adfr[1]:.3f}'),
 ('KPSS, return',f'{kpssr[0]:.3f}',f'p={kpssr[1]:.3f}'),
 ('ARCH-LM (10 lags)',f'{arch[0]:.3f}',f'p={arch[1]:.3g}'),
]
s='\\begin{tabular}{lrr}\n\\toprule\nStatistic & Value & Detail \\\\\n\\midrule\n'+'\n'.join(f'{a} & {b} & {c} \\\\' for a,b,c in rows)+'\n\\bottomrule\n\\end{tabular}\n'
write('data_diagnostics.tex',s)

# Fold design table
fr=[]
for _,r in fold.iterrows():
    y=int(r.Year); val=y-1; train_end=y-2
    test=f'{y}' if y<2026 else '2026 (to 11 Jun)'
    fr.append((f'2018-{train_end}',str(val),test,str(int(r.N))))
s='\\begin{tabular}{lccc}\n\\toprule\nTraining window & Validation & Test & $N_{test}$ \\\\\n\\midrule\n'+'\n'.join(f'{a} & {b} & {c} & {d} \\\\' for a,b,c,d in fr)+'\n\\bottomrule\n\\end{tabular}\n'
write('fold_design.tex',s)

# point forecast table
rows=[]
for _,r in fold.iterrows():
    gain=100*(r.RWPriceRMSE-r.PriceRMSE)/r.RWPriceRMSE
    rows.append((int(r.Year),int(r.N),r.PriceRMSE,r.RWPriceRMSE,gain,r.MeanReturnRMSE))
rows.append(('Pooled',int(fold.N.sum()),pool['PriceRMSE_CMD'],float(np.sqrt(np.mean((pred.ActualPrice-pred.ActualPrice/np.exp(pred.Return))**2))),np.nan,pool['MeanReturnRMSE_CMD']))
s='\\begin{tabular}{lrrrrr}\n\\toprule\nPeriod & $N$ & CMD price RMSE & No-change RMSE & Gain (\\%) & Return RMSE \\\\\n\\midrule\n'
for a,b,c,d,e,f in rows:
    eg='NA' if not np.isfinite(e) else f'{e:.2f}'
    s+=f'{a} & {b} & {c:.4f} & {d:.4f} & {eg} & {f:.5f} \\\\\n'
s+='\\bottomrule\n\\end{tabular}\n';write('point_forecast.tex',s)

# pooled risk table
order=['CMD-MoE','Equal-weight pool','EWMA94','HAR','EWMA97','LogSV','ML']
labels={'CMD-MoE':'CMD-MoE','Equal-weight pool':'Equal weight','EWMA94':'EWMA(0.94)','HAR':'HAR-RV','EWMA97':'EWMA(0.97)','LogSV':'Log-SV','ML':'Multiscale ML'}
ql={'CMD-MoE':pool['QLIKE_CMD'],'Equal-weight pool':pool['QLIKE_Equal']}
vr={'CMD-MoE':pool['VarianceRMSE_CMD'],'Equal-weight pool':pool['VarianceRMSE_Equal']}
for k in ['EWMA94','HAR','EWMA97','LogSV','ML']:
    ql[k]=pool['QLIKE_'+k];vr[k]=pool['VarianceRMSE_'+k]
s='\\begin{tabular}{lrrrrr}\n\\toprule\nModel & QLIKE & Variance RMSE & Gain vs. CMD (\\%) & HAC $t$ & $p$ \\\\\n\\midrule\n'
for k in order:
    if k=='CMD-MoE': gain=t=p='NA'
    else:
        gain=f'{100*(ql[k]-ql["CMD-MoE"])/ql[k]:.2f}'
        key='Equal' if k=='Equal-weight pool' else k
        hac=pool['HAC_vs_'+key];t=f'{hac[1]:.3f}';p=fmt_p(hac[2])
    s+=f'{labels[k]} & {ql[k]:.4f} & {vr[k]:.7f} & {gain} & {t} & {p} \\\\\n'
s+='\\bottomrule\n\\end{tabular}\n';write('pooled_risk.tex',s)

# annual QLIKE table from prediction exact losses
models=[('CMDVariance','CMD-MoE'),('LogSVVariance','Log-SV'),('HARVariance','HAR-RV'),('EWMA94Variance','EWMA(.94)'),('EWMA97Variance','EWMA(.97)'),('MLVariance','ML'),('EqualVariance','Equal')]
def qlike(y,v):
    rr=y/np.clip(v,1e-12,None); return rr-np.log(rr)-1
annual=[]
for y,g in pred.groupby('Year'):
    row=[int(y)]
    for c,_ in models: row.append(qlike(g.RVTarget.to_numpy(),g[c].to_numpy()).mean())
    annual.append(row)
s='\\begin{tabular}{lrrrrrrr}\n\\toprule\nYear & CMD-MoE & Log-SV & HAR-RV & EWMA(.94) & EWMA(.97) & ML & Equal \\\\\n\\midrule\n'
for row in annual:s+=str(row[0])+' & '+' & '.join(f'{x:.4f}' for x in row[1:])+' \\\\\n'
s+='\\bottomrule\n\\end{tabular}\n';write('annual_qlike.tex',s)

# distribution calibration
s='\\begin{tabular}{lrrrrr}\n\\toprule\nPeriod & NLL & Cov. 90\\% & Width 90\\% & Cov. 95\\% & Width 95\\% \\\\\n\\midrule\n'
for _,r in fold.iterrows():s+=f'{int(r.Year)} & {r.NLL:.3f} & {100*r.Coverage90:.2f} & {r.Width90:.4f} & {100*r.Coverage95:.2f} & {r.Width95:.4f} \\\\\n'
s += ('Pooled & {:.3f} & {:.2f} & {:.4f} & {:.2f} & {:.4f} \\\n'.format(pool['NLL'],100*pool['Coverage90'],pool['Width90'],100*pool['Coverage95'],pool['Width95']))
s+='\\bottomrule\n\\end{tabular}\n';write('distribution.tex',s)

# var/es backtest table
s='\\begin{tabular}{lrrrrrr}\n\\toprule\nTail level & Hit rate & Kupiec $p$ & Indep. $p$ & ES-ID $t$ & ES-ID $p$ & Decision \\\\\n\\midrule\n'
for a,label in [(.05,'95\\%'),(.01,'99\\%')]:
    hit=pool[f'hit_{a}'];ku=pool[f'kupiec_{a}'][1];ch=pool[f'christoffersen_{a}'][1];es=pool[f'es_identification_{a}']
    dec='Not rejected at 5\\%'
    s+=f'{label} & {100*hit:.2f}\\% & {ku:.3f} & {ch:.3f} & {es[1]:.3f} & {es[2]:.3f} & {dec} \\\\\n'
s+='\\bottomrule\n\\end{tabular}\n';write('var_backtest.tex',s)

# ablation
labels_abl={
    'Full_fixed_share':'Full fixed-share',
    'Exponential_weighting':'Exponential weighting ($\\gamma=0$)',
    'Equal_weight':'Equal weight',
    'No_LogSV_expert':'No Log-SV expert',
    'No_LogSV_information':'No Log-SV information',
    'No_EWMA_experts':'No EWMA experts',
    'No_ML':'No ML expert',
    'No_HAR':'No HAR-RV expert',
}
order_abl=list(labels_abl)
s='\\begin{tabular}{lrrr}\n\\toprule\nSpecification & Pooled QLIKE & Change vs full (\\%) & Mean annual QLIKE \\\\\n\\midrule\n'
full=float(abl.loc[abl.Variant=='Full_fixed_share','Pooled_QLIKE'].iloc[0])
for key in order_abl:
    r=abl.loc[abl.Variant==key].iloc[0]
    ch=100*(r.Pooled_QLIKE/full-1)
    s+=f'{labels_abl[key]} & {r.Pooled_QLIKE:.4f} & {ch:+.2f} & {r.Mean_Annual_QLIKE:.4f} \\\\\n'
s+='\\bottomrule\n\\end{tabular}\n';write('ablation.tex',s)

# expert weights table
wc=['VarW_LogSV','VarW_HAR','VarW_EWMA94','VarW_EWMA97','VarW_ML']
s='\\begin{tabular}{lrrrrr}\n\\toprule\nYear & Log-SV & HAR-RV & EWMA(.94) & EWMA(.97) & ML \\\\\n\\midrule\n'
for y,g in pred.groupby('Year'):
    vals=g[wc].mean().to_numpy();s+=f'{int(y)} & '+' & '.join(f'{x:.3f}' for x in vals)+' \\\\\n'
vals=pred[wc].mean().to_numpy();s+='Pooled & '+' & '.join(f'{x:.3f}' for x in vals)+' \\\\\n\\bottomrule\n\\end{tabular}\n';write('weights.tex',s)

# hyperparameter selected by fold
s='\\begin{tabular}{lrrrrrr}\n\\toprule\nYear & Mean $\\eta$ & Mean $\\gamma$ & Var $\\eta$ & Var $\\gamma$ & $\\nu$ & Variance scale \\\\\n\\midrule\n'
for _,r in fold.iterrows():s+=f'{int(r.Year)} & {r.mean_eta:g} & {r.mean_gamma:g} & {r.var_eta:g} & {r.var_gamma:g} & {r.nu:g} & {r.scale_mult:g} \\\\\n'
s+='\\bottomrule\n\\end{tabular}\n';write('hyperparams.tex',s)

# Figures
bench2=pd.DataFrame({'Model':[labels[k] for k in order],'QLIKE':[ql[k] for k in order]}).sort_values('QLIKE')
fig,ax=plt.subplots(figsize=(7.2,4.4));ax.barh(bench2.Model,bench2.QLIKE);ax.set_xlabel('Pooled QLIKE (lower is better)');ax.set_title('Pooled conditional-variance forecast performance');fig.tight_layout();fig.savefig(F/'pooled_qlike.png',dpi=220);plt.close(fig)

fig,ax=plt.subplots(figsize=(7.2,4.5))
for c,l in models[:5]:
    vals=[]
    for y,g in pred.groupby('Year'): vals.append(qlike(g.RVTarget.to_numpy(),g[c].to_numpy()).mean())
    ax.plot(sorted(pred.Year.unique()),vals,marker='o',label=l)
ax.set_xlabel('Test year');ax.set_ylabel('QLIKE');ax.set_title('Annual out-of-sample QLIKE');ax.legend(ncol=2,fontsize=8);fig.tight_layout();fig.savefig(F/'annual_qlike.png',dpi=220);plt.close(fig)

wg=pred.groupby('Year')[wc].mean();fig,ax=plt.subplots(figsize=(7.2,4.5))
for c,l in zip(wc,['Log-SV','HAR-RV','EWMA(.94)','EWMA(.97)','ML']):ax.plot(wg.index,wg[c],marker='o',label=l)
ax.set_xlabel('Test year');ax.set_ylabel('Average fixed-share weight');ax.set_ylim(0,1);ax.set_title('Causal variance-expert weights by test year');ax.legend(ncol=2,fontsize=8);fig.tight_layout();fig.savefig(F/'expert_weights.png',dpi=220);plt.close(fig)

fig,ax=plt.subplots(figsize=(6.6,4.2));levels=['90%','95%'];nom=[.90,.95];obs=[pool['Coverage90'],pool['Coverage95']];x=np.arange(2);w=.34;ax.bar(x-w/2,nom,w,label='Nominal');ax.bar(x+w/2,obs,w,label='Observed');ax.set_xticks(x,levels);ax.set_ylim(.84,1);ax.set_ylabel('Coverage');ax.set_title('Conformal interval calibration');ax.legend();fig.tight_layout();fig.savefig(F/'interval_coverage.png',dpi=220);plt.close(fig)

# cumulative loss advantage
fig,ax=plt.subplots(figsize=(7.4,4.4)); y=pred.RVTarget.to_numpy(); lc=qlike(y,pred.CMDVariance.to_numpy())
for c,l in [('LogSVVariance','Log-SV'),('HARVariance','HAR-RV'),('EWMA94Variance','EWMA(.94)'),('EqualVariance','Equal weight')]:
    d=qlike(y,pred[c].to_numpy())-lc;ax.plot(pred.Date,np.cumsum(d),label=l)
ax.axhline(0,linewidth=.8);ax.set_ylabel('Cumulative QLIKE advantage of CMD-MoE');ax.set_xlabel('Date');ax.set_title('Accumulated out-of-sample loss differential');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(F/'cumulative_loss_gain.png',dpi=220);plt.close(fig)

# risk path for last 300 observations
p=pred.tail(300);fig,ax=plt.subplots(figsize=(7.4,4.4));ax.plot(p.Date,p.Return,label='Return',linewidth=.8);ax.plot(p.Date,p.Q95,label='95% VaR',linewidth=1);ax.plot(p.Date,p.ES95,label='95% ES',linewidth=1);ax.set_ylabel('Log return');ax.set_xlabel('Date');ax.set_title('Tail-risk forecasts in the final evaluation period');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(F/'tail_risk_path.png',dpi=220);plt.close(fig)

# variance path last 180 observations
p=pred.tail(180);fig,ax=plt.subplots(figsize=(7.4,4.4));ax.plot(p.Date,p.RVTarget,label='OHLC variance proxy',linewidth=.8);ax.plot(p.Date,p.CMDVariance,label='CMD-MoE',linewidth=1);ax.plot(p.Date,p.LogSVVariance,label='Log-SV',linewidth=.8);ax.set_ylabel('Variance');ax.set_xlabel('Date');ax.set_title('Conditional variance forecasts in the final evaluation window');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(F/'variance_path.png',dpi=220);plt.close(fig)

print('generated tables:',len(list(T.glob('*.tex'))),'figures:',len(list(F.glob('*.png'))))
