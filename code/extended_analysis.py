#!/usr/bin/env python3
"""Additional EUA risk-forecast evaluation from submitted source/data: no test-set parameter selection.
Experiment scores use same original test years, outcomes, and OHLCV information set.
Primary parameter selection for comparators uses the complete preceding validation year;
an independent, chronologically split validation is reported as a robustness diagnostic.
"""
from pathlib import Path
import math, json, itertools, sys, time
import numpy as np, pandas as pd
from scipy import stats
from scipy.optimize import minimize
from statsmodels.stats.diagnostic import acorr_ljungbox
from numba import njit
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from run_cmd_moe_nested import (load_data,qlike, fs_sequence,tune_fixed_share,student_nll,select_distribution,
    dist_quantile,dist_es_lower,kupiec,christoffersen,hac_test)
from run_ablation import tune_exponential_weighting
RES=ROOT/'results'; RES.mkdir(parents=True,exist_ok=True)
YEARS=list(range(2021,2027)); COLS=['LogSV','HAR','EWMA94','EWMA97','ML','ML_NoSV']

# Load exact arrays cached from the author's causal training specification (recomputed separately).
folds={yr:dict(np.load(RES/'extended_cache'/f'fold_{yr}.npz',allow_pickle=True)) for yr in YEARS}
base=pd.read_csv(RES/'nested_predictions.csv'); fold_original=pd.read_csv(RES/'fold_metrics.csv');
# Ensure independently rerun folds reproduce the original released model exactly.
for yr,d in folds.items():
 z=base.loc[base.Year==yr]
 assert np.allclose(z.RVTarget.to_numpy(),d['rv_test'])
 assert np.allclose(z[[c+'Variance' for c in COLS[:5]]].to_numpy(),d['P_test'][:,:5],rtol=1e-7,atol=1e-8)

VARIANTS={'CMD-MoE':list(range(5)),'Ordinary exponential weighting':list(range(5)),
          'Equal weight':list(range(5)),'Log-SV':[0],'HAR-RV':[1],
          'EWMA(0.94)':[2],'EWMA(0.97)':[3],'Multiscale ExtraTrees':[4],
          'No Log-SV expert':[1,2,3,4], 'No Log-SV information':[1,2,3,5],
          'No EWMA experts':[0,1,4],'No nonlinear expert':[0,1,2,3],
          'No HAR-RV':[0,2,3,4]}


def predict_variant(name,d,tune_part=1.,cap=3.,eta_override=None,gamma_override=None):
 Pva=d['P_val'][:,VARIANTS[name]];Pte=d['P_test'][:,VARIANTS[name]];yv=d['rv_val'];yt=d['rv_test']
 vi=np.arange(max(1,int(len(yv)*tune_part)))
 if name=='Equal weight':
  return Pva.mean(1),Pte.mean(1),float('nan'),float('nan')
 if len(VARIANTS[name])==1:
  return Pva[:,0],Pte[:,0],float('nan'),float('nan')
 if name=='Ordinary exponential weighting':
  best=tune_exponential_weighting(vi,yv,Pva);eta,gamma=best[1],0.0
  w=best[2]
 else:
  best=tune_fixed_share(vi,yv,Pva,'qlike');eta,gamma=best[1],best[2];w=best[3]
 if eta_override is not None: eta=eta_override
 if gamma_override is not None:gamma=gamma_override
 # If overwriting parameters or using split tuning, reset and replay chronological validation
 vpred,w=fs_sequence(range(len(yv)),yv,Pva,eta,gamma,loss='qlike',cap=cap)
 tpred,w=fs_sequence(range(len(yt)),yt,Pte,eta,gamma,winit=w,loss='qlike',cap=cap)
 return vpred,tpred,eta,gamma

# Main model-specific probabilistic comparisons: same zero-mean location for all risk models.
ALL=list(VARIANTS)
PRED={}; VVAL={}; SEL=[]
for year,d in folds.items():
 for name in ALL:
  vp,tp,eta,gamma=predict_variant(name,d)
  if name=='CMD-MoE':
   actual=base.loc[base.Year==year,'CMDVariance'].to_numpy()
   assert np.allclose(tp,actual,atol=1e-7,rtol=1e-7), (year,float(np.max(np.abs(tp-actual))))
  PRED[year,name]=tp;VVAL[year,name]=vp;SEL.append({'Year':year,'Model':name,'eta':eta,'gamma':gamma})

# A transparent, proper score for joint left-tail VaR and ES; applicable when ES<0.
def fz0(y,q,e,alpha):
 e=np.minimum(np.asarray(e,float),-1e-9)
 return -( (np.asarray(y)<=np.asarray(q))*(np.asarray(q)-np.asarray(y)) / (alpha*e)) +np.asarray(q)/e+np.log(-e)-1

def interval_score(y,lo,hi,alpha):
 return (hi-lo)+(2/alpha)*(lo-y)*(y<lo)+(2/alpha)*(y-hi)*(y>hi)

def pinball(y,q,alpha):
 u=y-q;return np.maximum(alpha*u,(alpha-1)*u)

# Fair density/interval/tail comparison: every model has its own validation-selected t df/scale.
# Benchmark mean deliberately fixed to zero for all to isolate distributional variance performance.
DETAIL=[];PER_YEAR=[]; PARAM=[]
for year,d in folds.items():
 yv=d['r_val'];yt=d['r_test'];dates=d['dates_test']
 for name in ALL:
  vv=VVAL[year,name];vt=PRED[year,name]
  best=select_distribution(yv,vv);nu,sm=best[1],best[2]
  zval=np.abs(yv)/np.sqrt(np.maximum(vv*sm,1e-12));c90=np.quantile(zval,.90,method='higher');c95=np.quantile(zval,.95,method='higher')
  sd=np.sqrt(np.maximum(vt*sm,1e-12));lo90=-c90*sd;hi90=c90*sd;lo95=-c95*sd;hi95=c95*sd
  q05=dist_quantile(0,vt,nu,sm,.05);q01=dist_quantile(0,vt,nu,sm,.01);q95=dist_quantile(0,vt,nu,sm,.95);q99=dist_quantile(0,vt,nu,sm,.99)
  e05=dist_es_lower(0,vt,nu,sm,.05)
  e95=-dist_es_lower(0,vt,nu,sm,.05)
  dct={'Year':year,'Model':name,'Date':dates,'Return':yt,'RV':d['rv_test'],'VarForecast':vt,
       'NLL':student_nll(yt,vt,nu,sm),'QLIKE':qlike(d['rv_test'],vt),
       'IS90':interval_score(yt,lo90,hi90,.1),'IS95':interval_score(yt,lo95,hi95,.05),
       'Cover90':(yt>=lo90)&(yt<=hi90),'Cover95':(yt>=lo95)&(yt<=hi95),
       'Pinball05':pinball(yt,q05,.05),'Pinball01':pinball(yt,q01,.01),
       'FZ05':fz0(yt,q05,e05,.05),'VaR05':q05,'VaR01':q01,'VaR95up':q95,'VaR99up':q99,'ES05':e05,'ES95up':e95,
       'Left95Hit':yt<q05,'Left99Hit':yt<q01,'Right95Hit':yt>q95,'Right99Hit':yt>q99,'Width95':hi95-lo95,'Width90':hi90-lo90}
  DETAIL.append(pd.DataFrame(dct));PARAM.append({'Year':year,'Model':name,'nu':nu,'scale':sm,'c90':c90,'c95':c95})
full=pd.concat(DETAIL,ignore_index=True);full.to_csv(RES/'comparative_predictions.csv',index=False)
pd.DataFrame(PARAM).to_csv(RES/'comparative_calibration.csv',index=False)
pd.DataFrame(SEL).to_csv(RES/'routing_selections.csv',index=False)
summary=full.groupby('Model',sort=False).agg(N=('NLL','size'),QLIKE=('QLIKE','mean'),NLL=('NLL','mean'),
              IntervalScore90=('IS90','mean'),IntervalScore95=('IS95','mean'),Pinball05=('Pinball05','mean'),
              Pinball01=('Pinball01','mean'),FZ05=('FZ05','mean'), Coverage90=('Cover90','mean'),Coverage95=('Cover95','mean'),
              VaR95_hit=('Left95Hit','mean'),VaR99_hit=('Left99Hit','mean'),Upper95_hit=('Right95Hit','mean'), Upper99_hit=('Right99Hit','mean')).reset_index()
summary.to_csv(RES/'proper_score_comparisons.csv',index=False)

# Annual exception counts and binomial Wilson intervals, plus upper and lower independence tests.
def wilson(x,n,conf=.95):
 z=stats.norm.ppf(1-(1-conf)/2);q=x/n;den=1+z*z/n;ctr=(q+z*z/(2*n))/den;rad=z*np.sqrt(q*(1-q)/n+z*z/(4*n*n))/den;return (float(ctr-rad),float(ctr+rad))
exc=[]
for yr in YEARS:
 x=full[(full.Year==yr)&(full.Model=='CMD-MoE')]
 for col,alpha,tail in [('Left95Hit',.05,'left95'),('Left99Hit',.01,'left99'),('Right95Hit',.05,'right95'),('Right99Hit',.01,'right99')]:
  n=len(x);h=x[col].astype(int).to_numpy();L,U=wilson(int(h.sum()),n)
  exc.append({'Year':yr,'Tail':tail,'N':n,'Exceptions':int(h.sum()),'Rate':h.mean(),'Wilson95Lo':L,'Wilson95Hi':U,'KupiecP':kupiec(h,alpha)[1],'IndepP':christoffersen(h)[1]})
exc=pd.DataFrame(exc);exc.to_csv(RES/'annual_exceptions.csv',index=False)

# Test split-calibration 60% routing/density selection, final 40% conformal (train precedes test).
split=[]
for yr,d in folds.items():
 va=d['r_val'];te=d['r_test'];n=len(va);n1=int(.60*n)
 for name in ['CMD-MoE','Ordinary exponential weighting','Equal weight']:
  # For the two dynamic routers use first 60% for tuning, then update through calibration without refitting.
  v,t,eta,gamma=predict_variant(name,d,tune_part=.60)
  nu,sm=select_distribution(va[:n1],v[:n1])[1:]
  z=np.abs(va[n1:])/np.sqrt(np.maximum(v[n1:]*sm,1e-12));
  def split_q(a):
   # finite-sample conformal quantile with (n_cal+1) factor; capped to largest calibration score
   k=min(len(z),math.ceil((len(z)+1)*(1-a)))
   return float(np.sort(z)[k-1])
  c90=split_q(.1);c95=split_q(.05);sd=np.sqrt(t*sm)
  split.append({'Year':yr,'Model':name,'N':len(te),'eta':eta,'gamma':gamma,'nu':nu,'sm':sm,
                'QLIKE':float(np.mean(qlike(d['rv_test'],t))),
                'NLL':float(np.mean(student_nll(te,t,nu,sm))),
                'Cover90':float(np.mean(np.abs(te)<=c90*sd)),
                'Cover95':float(np.mean(np.abs(te)<=c95*sd)),
                'IS95':float(np.mean(interval_score(te,-c95*sd,c95*sd,.05)))})
split=pd.DataFrame(split);split.to_csv(RES/'chronological_calibration.csv',index=False)

# Hyperparameters η,γ and update cap: validation -> held-out test; original is optimized full year.
sens=[]
for eta,gamma,cap in itertools.product([.02,.1,.5,1,2],[0,.02,.05,.1,.2,.4],[1.,3.,10.]):
 ytest=[];yvar=[];yy=[];vs=[];rets=[];nus=[];sms=[]
 for yr,d in folds.items():
  vv,tt,_,_=predict_variant('CMD-MoE',d,eta_override=eta,gamma_override=gamma,cap=cap)
  bb=select_distribution(d['r_val'],vv);nu,sm=bb[1:]; q=dist_quantile(0,tt,nu,sm,.05)
  ytest.extend(d['rv_test']);yvar.extend(tt);rets.extend(d['r_test']);
  yy.extend((d['r_test']<q).astype(int))
 sens.append({'eta':eta,'gamma':gamma,'cap':cap,'QLIKE':float(np.mean(qlike(np.asarray(ytest),np.asarray(yvar)))),
              'Hit95':float(np.mean(yy))})
sens=pd.DataFrame(sens);sens.to_csv(RES/'hyperparameter_sensitivity.csv',index=False)

# Dependence-aware HAC and stationary circular block bootstrap for equal/exp/ablated daily losses.
boot=[];rng=np.random.default_rng(20261028)
cmd=full.loc[full.Model=='CMD-MoE'].sort_values(['Year','Date']).reset_index(drop=True)
for name in ['Ordinary exponential weighting','Equal weight','No Log-SV expert','No Log-SV information','No EWMA experts','No nonlinear expert','No HAR-RV']:
 cmp=full.loc[full.Model==name].sort_values(['Year','Date']).reset_index(drop=True)
 assert np.all(cmp.Date.to_numpy()==cmd.Date.to_numpy())
 dif=(cmp.QLIKE.to_numpy()-cmd.QLIKE.to_numpy());x=dif.mean()
 item={'Model':name,'MeanLossGain':x}
 for lag in [3,7,14,21]:
  ht=hac_test(dif,lag=lag);item[f'HAC_t_L{lag}']=ht[1];item[f'HAC_p_L{lag}']=ht[2]
 for L in [10,20]:
  sims=[];n=len(dif);starts=int(math.ceil(n/L))
  for j in range(1000):
   idx=(rng.integers(0,n,starts)[:,None]+np.arange(L)[None,:])%n
   sims.append(float(np.mean(dif[idx.ravel()[:n]])))
  lo,hi=np.quantile(sims,[.025,.975]);item[f'Block{L}_CIlo']=lo;item[f'Block{L}_CIhi']=hi
 boot.append(item)
boot=pd.DataFrame(boot);boot.to_csv(RES/'ablation_inference.csv',index=False)

# Alternative OHLC target sensitivity: close/open gap + Parkinson's range estimator.
df=load_data(ROOT/'data'/'eua.csv');alt=(np.log(df.Open/df.Price.shift(1))**2+df['rv_p']).clip(lower=1e-10)
altmap=dict(zip(df.Date.dt.strftime('%Y-%m-%d'),alt));alts=[]
for name in ['CMD-MoE','Ordinary exponential weighting','Equal weight','Log-SV','HAR-RV','EWMA(0.94)','EWMA(0.97)','Multiscale ExtraTrees']:
 x=full.loc[full.Model==name].sort_values(['Year','Date'])
 y=np.array([altmap[z] for z in x.Date],float)
 alts.append({'Model':name,'QLIKE_OriginalGK':float(x.QLIKE.mean()),
    'QLIKE_OvernightParkinson_FixedForecast':float(np.mean(qlike(y,x.VarForecast.values))),
    'RMSE_OvernightParkinson_FixedForecast':float(np.sqrt(np.mean((y-x.VarForecast.values)**2)))})
pd.DataFrame(alts).to_csv(RES/'alternative_proxy.csv',index=False)

# Concrete reserve decision for 10,000 EUA long compliance demand; score = reserve + 20 x shortfall.
# We use actual one-day price changes and upper-tail 95%-quantiles, all model-specific calibrated.
prices=base[['Date','ActualPrice','Return']].copy();prices['Date']=pd.to_datetime(prices.Date).dt.strftime('%Y-%m-%d')
prices['PrevPrice']=prices.ActualPrice/np.exp(prices.Return)
price_map=prices.set_index('Date')
reserve=[]
for name in ['CMD-MoE','Ordinary exponential weighting','Equal weight','Log-SV','HAR-RV','Multiscale ExtraTrees']:
 x=full.loc[full.Model==name].sort_values(['Year','Date']).copy()
 s0=price_map.loc[x.Date,'PrevPrice'].to_numpy();s1=price_map.loc[x.Date,'ActualPrice'].to_numpy()
 actual=1e4*np.maximum(s1-s0,0)
 buf=1e4*s0*np.maximum(np.exp(x.VaR95up.to_numpy())-1,0)
 shortfall=np.maximum(actual-buf,0)
 reserve.append({'Model':name,'MeanReserveEUR':float(buf.mean()),'MeanShortfallEUR':float(shortfall.mean()),
       'BreachRate':float(np.mean(actual>buf)),'DecisionLossEUR':float(np.mean(buf+20*shortfall))})
pd.DataFrame(reserve).to_csv(RES/'compliance_reserve_exercise.csv',index=False)

# Pooled complementary conditional diagnostics: serial correlation in indicator beyond lag-1.
D={}
for tail,alpha in [('Left95Hit',.05),('Left99Hit',.01),('Right95Hit',.05),('Right99Hit',.01)]:
 h=cmd[tail].astype(int).to_numpy();D[tail]={'count':int(h.sum()),'rate':float(h.mean()),'wilson':wilson(int(h.sum()),len(h)),
  'kupiec':kupiec(h,alpha),'independence':christoffersen(h),
  'LjungBox5_p':float(acorr_ljungbox(h,lags=[5],return_df=True).lb_pvalue.iloc[0]),
  'LjungBox10_p':float(acorr_ljungbox(h,lags=[10],return_df=True).lb_pvalue.iloc[0])}
(RES/'tail_diagnostics.json').write_text(json.dumps(D,indent=2))
print('\n--- COMPARATORS ---');print(summary[['Model','QLIKE','NLL','IntervalScore95','Pinball05','FZ05','Coverage95','VaR95_hit','Upper95_hit']].round(5).to_string(index=False))
print('\n--- SPLIT ---');print(split.groupby('Model').apply(lambda x:pd.Series({'QLIKE':np.average(x.QLIKE,weights=x.N),'Coverage95':np.average(x.Cover95,weights=x.N)})).to_string())
print('\n--- INFERENCE ---');print(boot[['Model','MeanLossGain','HAC_p_L7','Block20_CIlo','Block20_CIhi']].round(5).to_string(index=False))
print('\n--- RESERVE ---');print(pd.DataFrame(reserve).round(2).to_string(index=False))
