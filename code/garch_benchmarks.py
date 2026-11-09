#!/usr/bin/env python3
"""GARCH/EGARCH/GJR-GARCH, Student-t innovations; fitted solely on training returns.
The distributional output is independently calibrated on the preceding validation year.
This implementation is self-contained (scipy + numba), no future data used for optimization.
"""
import math,sys,time
from pathlib import Path
import numpy as np,pandas as pd
from scipy.optimize import minimize
from scipy import stats
from numba import njit
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'))
from run_cmd_moe_nested import load_data,build_samples,fit_logsv,filter_logsv,qlike,select_distribution,student_nll,dist_quantile,dist_es_lower
RES=ROOT/'results'
@njit
def filter_var(x,mod,theta,v0):
 n=len(x);h=np.empty(n);h[0]=v0
 if mod==0:
  w=np.exp(theta[0]);e=np.exp(theta[1]);f=np.exp(theta[2]);alpha=.995*e/(1+e+f);beta=.995*f/(1+e+f)
 elif mod==1:
  w=np.exp(theta[0]);e=np.exp(theta[1]);f=np.exp(theta[2]);g=np.exp(theta[3]);alpha=.995*e/(1+e+f+g);beta=.995*f/(1+e+f+g);gamma=1.99*g/(1+e+f+g)
 else:
  w=theta[0];beta=theta[1];alpha=theta[2];gamma=theta[3]
 for t in range(1,n):
  if mod==0: h[t]=w+alpha*x[t-1]*x[t-1]+beta*h[t-1]
  elif mod==1:h[t]=w+(alpha+gamma*(x[t-1]<0))*x[t-1]*x[t-1]+beta*h[t-1]
  else:
   z=x[t-1]/math.sqrt(max(h[t-1],1e-15))
   ln_h=w+(1-beta)*math.log(v0)+beta*math.log(max(h[t-1],1e-15))+alpha*(abs(z)-.79788456)+gamma*z
   h[t]=math.exp(min(20,max(-12,ln_h)))
  h[t]=max(h[t],1e-10)
 return h
@njit
def loglik(x,h,nu):
 c=math.lgamma((nu+1)/2)-math.lgamma(nu/2)-.5*math.log((nu-2)*math.pi)
 s=0.
 for i in range(len(x)):
  s+= .5*math.log(h[i])-c+ .5*(nu+1)*math.log(1+x[i]*x[i]/(h[i]*(nu-2)))
 return s/len(x)

def fit(mod,x):
 v0=max(float(np.var(x)),1e-4)
 if mod==0:
  st=[math.log(v0*.025),math.log(.07/(.995-.97)),math.log(.90/(.995-.97)),7.]
  bounds=[(-9,4),(-6,6),(-6,8),(3.01,30.)]
 elif mod==1:
  st=[math.log(v0*.025),math.log(.045/(.995-.945)),math.log(.86/(.995-.945)),math.log(.04/(.995-.945)),7.]
  bounds=[(-9,4),(-6,6),(-6,8),(-6,6),(3.01,30.)]
 else:
  st=[0.,.93,.18,-.12,7.];bounds=[(-3,3),(.45,.999),(-.5,1.5),(-1,1),(3.01,30.)]
 def obj(th):
  h=filter_var(x,mod,np.array(th),v0)
  z=loglik(x,h,float(th[-1]))
  return z if np.isfinite(z) else 1e8
 best=None
 for guess in (st,st[:-1]+[12.]):
  opt=minimize(obj,guess,method='L-BFGS-B',bounds=bounds,options={'maxiter':180,'ftol':2e-9})
  if best is None or opt.fun<best.fun:best=opt
 return best,v0

if __name__=='__main__':
 df=load_data(ROOT/'data'/'eua.csv'); r=df.ret.to_numpy(float);records=[];annual=[];fits=[];start_all=time.perf_counter()
 # Warm up Numba compilation before timing estimation.
 filter_var(np.random.default_rng(0).normal(size=60),0,np.array([0.,0.,1.,7.]),1.)
 loglik(np.array([.1,.2,.3]),np.array([1.,1.,1.]),7.)
 for yr in range(2021,2027):
  year_start=time.perf_counter();valyr=yr-1;te_end=pd.Timestamp('2026-06-11') if yr==2026 else pd.Timestamp(f'{yr}-12-31')
  tr_end=pd.Timestamp(f'{valyr-1}-12-31');va1=pd.Timestamp(f'{valyr}-12-31');trix=np.where(df.Date<=tr_end)[0]
  yraw=np.asarray(r[trix],float);mu=float(np.mean(yraw));x=100*(r-mu)
  # Align forecasts for target date t with the actual OHLC variance proxy at date t.
  va=np.where((df.Date>tr_end)&(df.Date<=va1))[0];te=np.where((df.Date>va1)&(df.Date<=te_end))[0]
  rv=df.rv_target.to_numpy(float)
  for mod,name in enumerate(['GARCH-t','GJR-GARCH-t','EGARCH-t']):
   t0=time.perf_counter();opt,v0=fit(mod,x[trix]);hh=filter_var(x,mod,np.asarray(opt.x,float),v0)/10000
   train_index=trix[trix>=61];train_index=train_index[np.isfinite(rv[train_index])]
   # train-only multiplicative OHLC-proxy scale calibration, as in author's pool.
   c=float(np.mean(rv[train_index]/np.maximum(hh[train_index],1e-12)));c=float(np.clip(c,.5,2.0))
   pred=np.maximum(c*hh,1e-12)
   best=select_distribution(r[va],pred[va]);nu,sm=best[1],best[2]
   s95=dist_quantile(0,pred[te],nu,sm,.05);s01=dist_quantile(0,pred[te],nu,sm,.01)
   su95=dist_quantile(0,pred[te],nu,sm,.95);es=dist_es_lower(0,pred[te],nu,sm,.05)
   zz=np.abs(r[va])/np.sqrt(np.maximum(pred[va]*sm,1e-12));c90=np.quantile(zz,.90,method='higher');c95=np.quantile(zz,.95,method='higher');sd=np.sqrt(pred[te]*sm)
   for j,idx in enumerate(te):
    records.append({'Year':yr,'Date':df.Date.iloc[idx].strftime('%Y-%m-%d'),'Model':name,'Return':r[idx],
       'RV':rv[idx],'VarForecast':pred[idx],'QLIKE':qlike(rv[idx],pred[idx]),
       'NLL':student_nll(r[idx],pred[idx],nu,sm),'VaR05':s95[j], 'VaR01':s01[j],'VaR95up':su95[j], 'ES05':es[j],
       'Cover90':abs(r[idx])<=c90*sd[j], 'CalThreshold90':c90, 'Cover95':abs(r[idx])<=c95*sd[j],
       'IS90':2*c90*sd[j]+20*max(abs(r[idx])-c90*sd[j],0),
       'IS95':2*c95*sd[j]+40*max(abs(r[idx])-c95*sd[j],0),
       'Pinball01':max(.01*(r[idx]-s01[j]),-.99*(r[idx]-s01[j])),
       'Pinball05':max(.05*(r[idx]-s95[j]),-.95*(r[idx]-s95[j])),
       'FZ05': -( (r[idx]<=s95[j])*(s95[j]-r[idx])/(.05*min(es[j],-1e-9)))+s95[j]/min(es[j],-1e-9)+math.log(-min(es[j],-1e-9))-1,
       'Left95Hit':r[idx]<s95[j],'Upper95Hit':r[idx]>su95[j]})
   fits.append({'Year':yr,'Model':name,'Objective':opt.fun,'Converged':bool(opt.success),'FitWallSeconds':time.perf_counter()-t0,'t_nu_innovation':float(opt.x[-1]),
      't_nu_distribution':nu,'c_train':c,'scale_head':sm,'theta':str(opt.x.tolist())})
   print('fit',yr,name,'QLIKE',np.mean(qlike(rv[te],pred[te])),'FitWallSecs',round(time.perf_counter()-t0,3),flush=True)
 print('total seconds',round(time.perf_counter()-start_all,1))
 pd.DataFrame(records).to_csv(RES/'garch_predictions.csv',index=False)
 pd.DataFrame(fits).to_csv(RES/'garch_fit_details.csv',index=False)
 print(pd.DataFrame(records).groupby('Model')[['QLIKE','NLL','Cover95','IS95','Pinball05','FZ05','Left95Hit','Upper95Hit']].mean().to_string())
