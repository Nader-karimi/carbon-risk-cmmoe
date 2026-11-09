#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from run_cmd_moe_nested import (
    load_data, fit_logsv, filter_logsv, build_samples, ewma, qlike,
    fs_sequence, tune_fixed_share
)

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / 'data' / 'eua.csv'
ETA_GRID = [.02, .05, .1, .2, .5, 1.0, 2.0]


def tune_exponential_weighting(val_idx, y, P):
    best = (np.inf, None, None)
    for eta in ETA_GRID:
        pv, w = fs_sequence(val_idx, y, P, eta, 0.0, loss='qlike')
        score = float(np.mean(qlike(y[val_idx], pv)))
        if score < best[0]:
            best = (score, eta, w)
    return best


def calibrated_experts(df, trainmask, tr, yv, orig, v1, Xv):
    r = df.ret.to_numpy(float)
    rv = df.rv_target.to_numpy(float)
    ex = {'LogSV': np.clip(v1[orig], 1e-12, None)}

    F = []
    for t in orig:
        F.append([
            1,
            np.log(rv[t] + 1e-12),
            np.nanmean(np.log(rv[t-4:t+1] + 1e-12)),
            np.nanmean(np.log(rv[t-21:t+1] + 1e-12)),
            np.nanmean(np.log(rv[t-59:t+1] + 1e-12)),
        ])
    F = np.asarray(F)
    coef = np.linalg.lstsq(F[tr], np.log(yv[tr]), rcond=None)[0]
    ex['HAR'] = np.exp(np.clip(F @ coef, -30, 5))
    ex['EWMA94'] = ewma(r, .94, trainmask)[orig]
    ex['EWMA97'] = ewma(r, .97, trainmask)[orig]

    ml = ExtraTreesRegressor(
        n_estimators=400, min_samples_leaf=40, max_features=.7,
        random_state=0, n_jobs=1
    ).fit(Xv[tr], np.log(yv[tr]))
    ex['ML'] = np.exp(np.clip(ml.predict(Xv), -30, 5))

    Xv_no_sv = Xv[:, :48]
    ml_no_sv = ExtraTreesRegressor(
        n_estimators=400, min_samples_leaf=40, max_features=.7,
        random_state=0, n_jobs=1
    ).fit(Xv_no_sv[tr], np.log(yv[tr]))
    ex['ML_NoSV'] = np.exp(np.clip(ml_no_sv.predict(Xv_no_sv), -30, 5))

    for k, pred in list(ex.items()):
        c = float(np.mean(yv[tr] / np.clip(pred[tr], 1e-12, None)))
        ex[k] = np.clip(np.clip(c, .5, 2.0) * pred, 1e-12, None)
    return ex


def evaluate_adaptive(name, keys, ex, va, te, yv, mode='fixed_share'):
    P = np.column_stack([ex[k] for k in keys])
    if mode == 'fixed_share':
        best = tune_fixed_share(va, yv, P, 'qlike')
        pt, _ = fs_sequence(te, yv, P, best[1], best[2], winit=best[3], loss='qlike')
        eta, gamma = best[1], best[2]
    elif mode == 'exponential':
        best = tune_exponential_weighting(va, yv, P)
        pt, _ = fs_sequence(te, yv, P, best[1], 0.0, winit=best[2], loss='qlike')
        eta, gamma = best[1], 0.0
    else:
        raise ValueError(mode)
    return {
        'Variant': name,
        'QLIKE': float(np.mean(qlike(yv[te], pt))),
        'eta': eta,
        'gamma': gamma,
    }


def run_fold(df, year):
    r = df.ret.to_numpy(float)
    validation_year = year - 1
    tr_end = pd.Timestamp(f'{validation_year-1}-12-31')
    va0 = pd.Timestamp(f'{validation_year}-01-01')
    va1 = pd.Timestamp(f'{validation_year}-12-31')
    te0 = pd.Timestamp(f'{year}-01-01')
    te1 = pd.Timestamp(f'{year}-12-31') if year < 2026 else pd.Timestamp('2026-06-11')
    trainmask = df.Date <= tr_end
    if trainmask.sum() < 500:
        return None

    fit = fit_logsv(r[trainmask])
    v1, vh, pp, nu0 = filter_logsv(r, fit)
    Xv, _, yv, _, dates, orig = build_samples(df, v1, vh, pp, nu0)
    tr = np.where(dates <= tr_end)[0]
    va = np.where((dates >= va0) & (dates <= va1))[0]
    te = np.where((dates >= te0) & (dates <= te1))[0]
    if len(tr) < 400 or len(va) < 100 or len(te) < 50:
        return None

    ex = calibrated_experts(df, trainmask, tr, yv, orig, v1, Xv)
    full_keys = ['LogSV', 'HAR', 'EWMA94', 'EWMA97', 'ML']
    variants = []
    variants.append(evaluate_adaptive('Full_fixed_share', full_keys, ex, va, te, yv, 'fixed_share'))
    variants.append(evaluate_adaptive('Exponential_weighting', full_keys, ex, va, te, yv, 'exponential'))
    variants.append(evaluate_adaptive('No_LogSV_expert', ['HAR', 'EWMA94', 'EWMA97', 'ML'], ex, va, te, yv))
    variants.append(evaluate_adaptive('No_LogSV_information', ['HAR', 'EWMA94', 'EWMA97', 'ML_NoSV'], ex, va, te, yv))
    variants.append(evaluate_adaptive('No_EWMA_experts', ['LogSV', 'HAR', 'ML'], ex, va, te, yv))
    variants.append(evaluate_adaptive('No_ML', ['LogSV', 'HAR', 'EWMA94', 'EWMA97'], ex, va, te, yv))
    variants.append(evaluate_adaptive('No_HAR', ['LogSV', 'EWMA94', 'EWMA97', 'ML'], ex, va, te, yv))

    Pfull = np.column_stack([ex[k] for k in full_keys])
    variants.append({
        'Variant': 'Equal_weight',
        'QLIKE': float(np.mean(qlike(yv[te], Pfull[te].mean(axis=1)))),
        'eta': np.nan,
        'gamma': np.nan,
    })

    # Validation/test arrays cached for independent, reproducible sensitivity and scoring analyses.
    cache_dir = ROOT / 'results' / 'extended_cache'
    cache_dir.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(cache_dir / f'fold_{year}.npz',
        dates_val=dates[va].strftime('%Y-%m-%d').to_numpy(), dates_test=dates[te].strftime('%Y-%m-%d').to_numpy(),
        rv_val=yv[va],rv_test=yv[te],r_val=df.ret.to_numpy(float)[orig[va]+1],r_test=df.ret.to_numpy(float)[orig[te]+1],
        P_val=np.column_stack([ex[k][va] for k in ['LogSV','HAR','EWMA94','EWMA97','ML','ML_NoSV']]),
        P_test=np.column_stack([ex[k][te] for k in ['LogSV','HAR','EWMA94','EWMA97','ML','ML_NoSV']]))
    for row in variants:
        row['Year'] = year
        row['N'] = len(te)
    return variants


def _run_year(year):
    df = load_data(CSV)
    return run_fold(df, year)


def main():
    from concurrent.futures import ProcessPoolExecutor
    years = [2021, 2022, 2023, 2024, 2025, 2026]
    rows = []
    with ProcessPoolExecutor(max_workers=min(6, len(years))) as ex_pool:
        for out in ex_pool.map(_run_year, years):
            if out:
                rows.extend(out)

    res = pd.DataFrame(rows)
    res = res[['Year', 'Variant', 'N', 'QLIKE', 'eta', 'gamma']]
    res.to_csv(ROOT / 'results' / 'ablation_folds.csv', index=False)

    summary = (
        res.groupby('Variant', sort=False)
        .apply(lambda g: pd.Series({
            'Pooled_QLIKE': np.average(g.QLIKE, weights=g.N),
            'Mean_Annual_QLIKE': g.QLIKE.mean(),
            'Median_QLIKE': g.QLIKE.median(),
        }))
        .reset_index()
    )
    order = [
        'Full_fixed_share', 'Exponential_weighting', 'Equal_weight',
        'No_LogSV_expert', 'No_LogSV_information', 'No_EWMA_experts',
        'No_ML', 'No_HAR'
    ]
    summary['order'] = summary.Variant.map({v: i for i, v in enumerate(order)})
    summary = summary.sort_values('order').drop(columns='order')
    summary.to_csv(ROOT / 'results' / 'ablation_summary.csv', index=False)

    print(res.pivot(index='Year', columns='Variant', values='QLIKE').round(6))
    print('\nPOOLED')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
