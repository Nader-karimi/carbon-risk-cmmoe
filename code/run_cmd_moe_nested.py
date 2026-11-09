#!/usr/bin/env python3
"""Causal Multiscale Distributional Mixture-of-Experts (CMD-MoE).

Strict annual expanding-window evaluation on the supplied EUA OHLCV data.
For each test year Y: train <=Y-2, validate Y-1, test Y. 2026 ends 2026-06-11.
All model choices (fixed-share eta/gamma; Student-t df/scale) use only the validation year.
The script writes exact predictions, fold metrics, pooled tests, and ablations.
"""
from __future__ import annotations
import argparse, json, math, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy import stats
from sklearn.ensemble import ExtraTreesRegressor
warnings.filterwarnings('ignore')
LOG_CHI2_MEAN=-1.2703628454614782
LOG_CHI2_VAR=math.pi**2/2


def parse_volume(v):
    if pd.isna(v): return np.nan
    s=str(v).strip().replace(',','')
    if s in {'','-','nan','NaN'}: return np.nan
    m=1.0
    if s[-1:].upper()=='K': m=1e3; s=s[:-1]
    elif s[-1:].upper()=='M': m=1e6; s=s[:-1]
    elif s[-1:].upper()=='B': m=1e9; s=s[:-1]
    try: return float(s)*m
    except Exception: return np.nan


def load_data(path):
    df=pd.read_csv(path,encoding='utf-8-sig')
    df['Date']=pd.to_datetime(df['Date'],format='%m/%d/%Y',errors='coerce')
    for c in ['Price','Open','High','Low']:
        df[c]=pd.to_numeric(df[c].astype(str).str.replace(',','',regex=False),errors='coerce')
    df['Volume']=df['Vol.'].map(parse_volume)
    df=df.sort_values('Date').reset_index(drop=True)
    df=df[(df.Date>=pd.Timestamp('2018-01-02'))&(df.Date<=pd.Timestamp('2026-06-11'))]
    df=df.dropna(subset=['Date','Price','Open','High','Low','Volume']).copy()
    df=df[(df[['Price','Open','High','Low']]>0).all(axis=1)].copy()
    df['ret']=np.log(df.Price/df.Price.shift(1)); df['abs_ret']=df.ret.abs(); df['sq_ret']=df.ret**2
    df['range_hl']=np.log(df.High/df.Low).clip(lower=0); df['ret_co']=np.log(df.Price/df.Open)
    df['rv_p']=df.range_hl**2/(4*np.log(2))
    df['rv_gk']=(0.5*df.range_hl**2-(2*np.log(2)-1)*df.ret_co**2).clip(lower=0)
    df['log_volume']=np.log1p(df.Volume.clip(lower=0)); df['dlog_volume']=df.log_volume.diff()
    df=df.dropna(subset=['ret','dlog_volume']).reset_index(drop=True)
    # Overnight-adjusted Garman-Klass proxy: aligns OHLC risk target with close-to-close return variance.
    df['rv_target']=(np.log(df.Open/df.Price.shift(1))**2+df.rv_gk).clip(lower=1e-10)
    return df


def _sv_unpack(th): return float(th[0]),float(.999*np.tanh(th[1])),float(np.exp(th[2])+1e-6)
def _sv_nll(th,yobs):
    om,ph,sg=_sv_unpack(th);q=sg*sg;P=q/max(1-ph*ph,1e-5);m=om;ll=0.0
    for y in yobs:
        F=P+LOG_CHI2_VAR; inn=y-m
        if F<=0 or not np.isfinite(F): return 1e12
        ll += .5*(math.log(2*math.pi)+math.log(F)+inn*inn/F)
        K=P/F; m=m+K*inn; P=max((1-K)*P,1e-12)
        m=om+ph*(m-om); P=ph*ph*P+q
    return ll

def fit_logsv(r):
    mu=float(np.mean(r)); y=np.log((r-mu)**2+1e-12)-LOG_CHI2_MEAN; om=float(np.mean(y))
    starts=[np.array([om,np.arctanh(.95/.999),math.log(.2)]),np.array([om,np.arctanh(.98/.999),math.log(.1)]),np.array([om,np.arctanh(.9/.999),math.log(.3)])]
    best=None
    for st in starts:
        z=minimize(_sv_nll,st,args=(y,),method='L-BFGS-B',bounds=[(-20,2),(-5,5),(-8,1)],options={'maxiter':1000})
        if best is None or z.fun<best.fun: best=z
    return (mu,*_sv_unpack(best.x))

def filter_logsv(r,fit):
    mu,om,ph,sg=fit; y=np.log((r-mu)**2+1e-12)-LOG_CHI2_MEAN; q=sg*sg
    m=om;P=q/max(1-ph*ph,1e-5);v1=[];vh=[];pp=[];nu=[]
    for x,yy in zip(r,y):
        prior=math.exp(np.clip(m+.5*P,-30,10));nu.append((x-mu)/math.sqrt(prior+1e-12))
        F=P+LOG_CHI2_VAR;K=P/F;m=m+K*(yy-m);P=max((1-K)*P,1e-12);vh.append(m);pp.append(P)
        m1=om+ph*(m-om);P1=ph*ph*P+q;v1.append(math.exp(np.clip(m1+.5*P1,-30,10)));m,P=m1,P1
    return np.array(v1),np.array(vh),np.array(pp),np.array(nu)

def qlike(y,v):
    rr=np.asarray(y)/np.clip(np.asarray(v),1e-12,None); return rr-np.log(rr)-1

def ewma(r,lam,trainmask):
    v=np.nanvar(r[trainmask]);out=[]
    for i in range(len(r)):
        if i>0:v=lam*v+(1-lam)*r[i-1]**2
        out.append(v)
    return np.array(out)

def build_samples(df,v1,vh,pp,nu,lookback=60):
    rv=df.rv_target.to_numpy(float);ret=df.ret.to_numpy(float);rng=df.range_hl.to_numpy(float);dv=df.dlog_volume.to_numpy(float);co=df.ret_co.to_numpy(float)
    Xv=[];Xm=[];yv=[];ym=[];dates=[];orig=[]
    for t in range(lookback,len(df)-1):
        if not np.isfinite(rv[t+1]):continue
        rvrow=[];mrow=[]
        for h in [1,3,5,10,22,60]:
            sl=slice(t-h+1,t+1)
            rvrow += [np.nanmean(ret[sl]),np.nanstd(ret[sl]),np.nanmean(np.abs(ret[sl])),np.nanmean(ret[sl]**2),np.nanmean(rv[sl]),np.nanstd(rv[sl]),np.nanmean(rng[sl]),np.nanmean(dv[sl])]
            mrow += [np.nanmean(ret[sl]),np.nanstd(ret[sl]),np.nanmean(np.abs(ret[sl])),np.nanmean(ret[sl]**2),np.nanmean(rng[sl]),np.nanmean(rv[sl]),np.nanmean(dv[sl]),np.nanmean(co[sl])]
        rvrow += [vh[t],v1[t],pp[t],nu[t]**2,abs(np.log(max(v1[t],1e-12))-np.log(max(rv[t],1e-12)))]
        Xv.append(rvrow);Xm.append(mrow);yv.append(rv[t+1]);ym.append(ret[t+1]);dates.append(df.Date.iloc[t+1]);orig.append(t)
    return np.array(Xv),np.array(Xm),np.clip(np.array(yv),1e-10,None),np.array(ym),pd.DatetimeIndex(dates),np.array(orig)

def fs_sequence(idx,y,P,eta,gamma,winit=None,loss='qlike',scale=1.0,cap=3.0,return_weights=False):
    K=P.shape[1];w=np.ones(K)/K if winit is None else np.array(winit,float);pred=[];weights=[]
    for i in idx:
        pred.append(float(P[i]@w));weights.append(w.copy())
        if loss=='qlike': ell=qlike(np.repeat(y[i],K),P[i])
        else: ell=((y[i]-P[i])**2)*scale
        ell=np.minimum(ell,cap);z=w*np.exp(-eta*ell);z/=z.sum();w=(1-gamma)*z+gamma/K
    if return_weights:return np.array(pred),w,np.array(weights)
    return np.array(pred),w

def tune_fixed_share(val_idx,y,P,loss,scale=1.0):
    best=(1e99,None,None,None)
    for eta in [.02,.05,.1,.2,.5,1.0,2.0]:
        for gamma in [.02,.05,.1,.2,.4]:
            pv,w=fs_sequence(val_idx,y,P,eta,gamma,loss=loss,scale=scale)
            sc=float(np.mean(qlike(y[val_idx],pv))) if loss=='qlike' else float(np.mean((y[val_idx]-pv)**2))
            if sc<best[0]:best=(sc,eta,gamma,w)
    return best

def student_nll(resid,var,nu,sm=1.0):
    var=np.clip(var*sm,1e-12,None)
    if nu>200:return .5*(np.log(2*np.pi*var)+resid**2/var)
    scale=np.sqrt(var*(nu-2)/nu);z=resid/scale;return -stats.t.logpdf(z,df=nu)+np.log(scale)
def select_distribution(resid,var):
    best=(1e99,None,None)
    for nu in [4,5,6,8,10,15,30,1000]:
        for sm in [.7,.85,1.0,1.15,1.3,1.5]:
            sc=float(np.mean(student_nll(resid,var,nu,sm)))
            if sc<best[0]:best=(sc,nu,sm)
    return best
def dist_quantile(mean,var,nu,sm,alpha):
    vv=np.clip(var*sm,1e-12,None)
    if nu>200:return mean+np.sqrt(vv)*stats.norm.ppf(alpha)
    s=np.sqrt(vv*(nu-2)/nu);return mean+s*stats.t.ppf(alpha,nu)
def dist_es_lower(mean,var,nu,sm,alpha):
    vv=np.clip(var*sm,1e-12,None)
    if nu>200:
        z=stats.norm.ppf(alpha);return mean-np.sqrt(vv)*stats.norm.pdf(z)/alpha
    s=np.sqrt(vv*(nu-2)/nu);q=stats.t.ppf(alpha,nu);f=stats.t.pdf(q,nu);return mean-s*((nu+q*q)/(nu-1))*f/alpha

def kupiec(hits,alpha):
    h=np.asarray(hits,int);n=len(h);x=h.sum();ph=x/n
    if x==0 or x==n:return (np.nan,np.nan)
    ll0=(n-x)*np.log(1-alpha)+x*np.log(alpha);ll1=(n-x)*np.log(1-ph)+x*np.log(ph);lr=-2*(ll0-ll1);return float(lr),float(1-stats.chi2.cdf(lr,1))
def christoffersen(hits):
    h=np.asarray(hits,int);n00=n01=n10=n11=0
    for a,b in zip(h[:-1],h[1:]):
        if a==0 and b==0:n00+=1
        elif a==0 and b==1:n01+=1
        elif a==1 and b==0:n10+=1
        else:n11+=1
    p01=n01/max(n00+n01,1);p11=n11/max(n10+n11,1);p=(n01+n11)/max(n00+n01+n10+n11,1)
    def tm(n,x):return 0 if n==0 else n*np.log(max(x,1e-12))
    l0=tm(n01+n11,p)+tm(n00+n10,1-p);l1=tm(n01,p01)+tm(n00,1-p01)+tm(n11,p11)+tm(n10,1-p11);lr=-2*(l0-l1);return float(lr),float(1-stats.chi2.cdf(lr,1))
def hac_test(d,lag=7):
    d=np.asarray(d,float);n=len(d);m=d.mean();x=d-m;vv=np.dot(x,x)/n
    for k in range(1,min(lag,n-1)+1):vv+=2*(1-k/(lag+1))*np.dot(x[k:],x[:-k])/n
    se=np.sqrt(max(vv,1e-18)/n);t=m/se;p=2*(1-stats.norm.cdf(abs(t)));return float(m),float(t),float(p)

def run_fold(df,year):
    r=df.ret.to_numpy(float);vy=year-1;tr_end=pd.Timestamp(f'{vy-1}-12-31');va0=pd.Timestamp(f'{vy}-01-01');va1=pd.Timestamp(f'{vy}-12-31');te0=pd.Timestamp(f'{year}-01-01');te1=pd.Timestamp(f'{year}-12-31') if year<2026 else pd.Timestamp('2026-06-11');trainmask=df.Date<=tr_end
    if trainmask.sum()<500:return None
    fit=fit_logsv(r[trainmask]);v1,vh,pp,nu0=filter_logsv(r,fit);Xv,Xm,yv,ym,dates,orig=build_samples(df,v1,vh,pp,nu0)
    tr=np.where(dates<=tr_end)[0];va=np.where((dates>=va0)&(dates<=va1))[0];te=np.where((dates>=te0)&(dates<=te1))[0]
    if len(tr)<400 or len(va)<100 or len(te)<50:return None
    # ----- mean expert pool: zero/no-change, AR(5), multiscale ExtraTrees -----
    Pm=np.zeros((len(ym),3));p=5;A=[];b=[]
    for i in tr:
        t=orig[i];A.append([1]+[r[t-j] for j in range(p)]);b.append(ym[i])
    coef=np.linalg.lstsq(np.asarray(A),np.asarray(b),rcond=None)[0]
    Pm[:,1]=np.array([coef[0]+sum(coef[j+1]*r[t-j] for j in range(p)) for t in orig])
    mlm=ExtraTreesRegressor(n_estimators=400,min_samples_leaf=40,max_features=.7,random_state=0,n_jobs=-1).fit(Xm[tr],ym[tr]);Pm[:,2]=mlm.predict(Xm)
    mscale=1/max(np.var(ym[tr]),1e-8);mb=tune_fixed_share(va,ym,Pm,'mse',mscale);mval,mw=fs_sequence(va,ym,Pm,mb[1],mb[2],loss='mse',scale=mscale);mtest,mw,mW=fs_sequence(te,ym,Pm,mb[1],mb[2],winit=mb[3],loss='mse',scale=mscale,return_weights=True)
    # ----- variance expert pool -----
    rv=df.rv_target.to_numpy(float);ex={'LogSV':np.clip(v1[orig],1e-12,None)};F=[]
    for t in orig:F.append([1,np.log(rv[t]+1e-12),np.nanmean(np.log(rv[t-4:t+1]+1e-12)),np.nanmean(np.log(rv[t-21:t+1]+1e-12)),np.nanmean(np.log(rv[t-59:t+1]+1e-12))])
    F=np.asarray(F);hc=np.linalg.lstsq(F[tr],np.log(yv[tr]),rcond=None)[0];ex['HAR']=np.exp(np.clip(F@hc,-30,5));ex['EWMA94']=ewma(r,.94,trainmask)[orig];ex['EWMA97']=ewma(r,.97,trainmask)[orig]
    mlv=ExtraTreesRegressor(n_estimators=400,min_samples_leaf=40,max_features=.7,random_state=0,n_jobs=-1).fit(Xv[tr],np.log(yv[tr]));ex['ML']=np.exp(np.clip(mlv.predict(Xv),-30,5))
    for k,pred in list(ex.items()):
        c=float(np.mean(yv[tr]/np.clip(pred[tr],1e-12,None)));ex[k]=np.clip(np.clip(c,.5,2)*pred,1e-12,None)
    keys=list(ex);Pv=np.column_stack([ex[k] for k in keys]);vb=tune_fixed_share(va,yv,Pv,'qlike');vval,vw=fs_sequence(va,yv,Pv,vb[1],vb[2],loss='qlike');vtest,vw,vW=fs_sequence(te,yv,Pv,vb[1],vb[2],winit=vb[3],loss='qlike',return_weights=True)
    # distribution calibration uses validation residuals only
    db=select_distribution(ym[va]-mval,vval);dnu,dsm=db[1],db[2];zval=np.abs(ym[va]-mval)/np.sqrt(np.clip(vval*dsm,1e-12,None));c90=float(np.quantile(zval,.90,method='higher'));c95=float(np.quantile(zval,.95,method='higher'))
    resid=ym[te]-mtest;nll=float(np.mean(student_nll(resid,vtest,dnu,dsm)));sd=np.sqrt(vtest*dsm);lo90=mtest-c90*sd;hi90=mtest+c90*sd;lo95=mtest-c95*sd;hi95=mtest+c95*sd
    q95=dist_quantile(mtest,vtest,dnu,dsm,.05);q99=dist_quantile(mtest,vtest,dnu,dsm,.01);es95=dist_es_lower(mtest,vtest,dnu,dsm,.05);es99=dist_es_lower(mtest,vtest,dnu,dsm,.01)
    # price mean forecast uses mean-return branch only; distributional variance does not improve point median mechanically.
    s0=df.Price.to_numpy(float)[orig[te]];actual_price=df.Price.to_numpy(float)[orig[te]+1];price_pred=s0*np.exp(mtest)
    metrics={'Year':year,'N':len(te),'MeanReturnRMSE':float(np.sqrt(np.mean((ym[te]-mtest)**2))),'PriceRMSE':float(np.sqrt(np.mean((actual_price-price_pred)**2))),'RWPriceRMSE':float(np.sqrt(np.mean((actual_price-s0)**2))),'QLIKE':float(np.mean(qlike(yv[te],vtest))),'NLL':nll,'nu':dnu,'scale_mult':dsm,'Coverage90':float(np.mean((ym[te]>=lo90)&(ym[te]<=hi90))),'Width90':float(np.mean(hi90-lo90)),'Coverage95':float(np.mean((ym[te]>=lo95)&(ym[te]<=hi95))),'Width95':float(np.mean(hi95-lo95)),'VaR95_hit':float(np.mean(ym[te]<q95)),'VaR99_hit':float(np.mean(ym[te]<q99)),'mean_eta':mb[1],'mean_gamma':mb[2],'var_eta':vb[1],'var_gamma':vb[2]}
    out=pd.DataFrame({'Date':dates[te],'Year':year,'Return':ym[te],'MeanForecast':mtest,'ActualPrice':actual_price,'PriceForecast':price_pred,'RVTarget':yv[te],'CMDVariance':vtest,'Q95':q95,'Q99':q99,'ES95':es95,'ES99':es99,'MeanW_NoChange':mW[:,0],'MeanW_AR5':mW[:,1],'MeanW_ML':mW[:,2]})
    for j,k in enumerate(keys):out[k+'Variance']=ex[k][te];out['VarW_'+k]=vW[:,j]
    # add equal-weight control
    out['EqualVariance']=Pv[te].mean(1)
    return metrics,out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default=str(Path(__file__).resolve().parents[1]/'data'/'eua.csv'));ap.add_argument('--out',default=str(Path(__file__).resolve().parents[1]/'results'));args=ap.parse_args();outdir=Path(args.out);outdir.mkdir(parents=True,exist_ok=True)
    df=load_data(args.data);foldrows=[];frames=[]
    for yr in [2021,2022,2023,2024,2025,2026]:
        z=run_fold(df,yr)
        if z is not None:foldrows.append(z[0]);frames.append(z[1]);print(yr,z[0])
    folds=pd.DataFrame(foldrows);pred=pd.concat(frames,ignore_index=True);folds.to_csv(outdir/'fold_metrics.csv',index=False);pred.to_csv(outdir/'nested_predictions.csv',index=False)
    y=pred.RVTarget.to_numpy(float);full=pred.CMDVariance.to_numpy(float);Lf=qlike(y,full);pool={'N':len(pred),'QLIKE_CMD':float(Lf.mean()),'VarianceRMSE_CMD':float(np.sqrt(np.mean((y-full)**2))),'MeanReturnRMSE_CMD':float(np.sqrt(np.mean((pred.Return-pred.MeanForecast)**2))),'PriceRMSE_CMD':float(np.sqrt(np.mean((pred.ActualPrice-pred.PriceForecast)**2)))}
    for k in ['LogSV','HAR','EWMA94','EWMA97','ML']:
        p=pred[k+'Variance'].to_numpy(float);Lb=qlike(y,p);pool['QLIKE_'+k]=float(Lb.mean());pool['VarianceRMSE_'+k]=float(np.sqrt(np.mean((y-p)**2)));pool['gain_pct_vs_'+k]=float(100*(Lb.mean()-Lf.mean())/Lb.mean());pool['HAC_vs_'+k]=hac_test(Lb-Lf)
    eq=pred.EqualVariance.to_numpy(float);Le=qlike(y,eq);pool['QLIKE_Equal']=float(Le.mean());pool['VarianceRMSE_Equal']=float(np.sqrt(np.mean((y-eq)**2)));pool['gain_pct_vs_Equal']=float(100*(Le.mean()-Lf.mean())/Le.mean());pool['HAC_vs_Equal']=hac_test(Le-Lf)
    for a,col,escol in [(.05,'Q95','ES95'),(.01,'Q99','ES99')]:
        yy=pred.Return.to_numpy(float);qv=pred[col].to_numpy(float);ev=pred[escol].to_numpy(float);hits=yy<qv;pool[f'hit_{a}']=float(hits.mean());pool[f'kupiec_{a}']=kupiec(hits,a);pool[f'christoffersen_{a}']=christoffersen(hits)
        # Fissler-Ziegel identification function V2(q,e,y)=e-q+I(y<=q)(q-y)/alpha.
        # Under correctly specified conditional VaR and ES, E[V2|F_t]=0. We report a HAC test of the pooled mean.
        v2=ev-qv+(hits.astype(float)/a)*(qv-yy);pool[f'es_identification_{a}']=hac_test(v2)
    # pooled interval and NLL summaries weighted by fold N
    n=folds.N.sum()
    for c in ['NLL','Coverage90','Width90','Coverage95','Width95']:
        pool[c]=float((folds[c]*folds.N).sum()/n)
    (outdir/'pooled_metrics.json').write_text(json.dumps(pool,indent=2))
    # Transparent constituent/ablation table from exact saved predictions.
    rows=[{'Model':'CMD-MoE','QLIKE':pool['QLIKE_CMD']},{'Model':'Equal-weight pool','QLIKE':pool['QLIKE_Equal']}]
    for k in ['LogSV','HAR','EWMA94','EWMA97','ML']:rows.append({'Model':k,'QLIKE':pool['QLIKE_'+k]})
    pd.DataFrame(rows).sort_values('QLIKE').to_csv(outdir/'risk_benchmark_table.csv',index=False)
    print(json.dumps(pool,indent=2))
if __name__=='__main__':main()
