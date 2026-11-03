"""Metrics and decision policies that do not depend on the held-out test set."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

HORIZONS=['5','15','30','60','90','120','close']
THRESHOLDS=[5,10,15,20,30,50]
SEED=731
TARGET_PRECISION=.80
MIN_CALLS=25
MIN_DAYS=10

def signed_log(x):
    x=np.asarray(x,dtype=float)
    return np.sign(x)*np.log1p(np.abs(x))

def make_splits(dates):
    dates=np.array(sorted(set(dates)))
    a,b,c=[int(len(dates)*p) for p in (.60,.75,.85)]
    # Full-session embargo at each boundary; no stock-day is split across periods.
    return {'train':dates[:a], 'validation':dates[a+1:b],
            'threshold':dates[b+1:c], 'test':dates[c+1:], 'embargo':dates[[a,b,c]]}

def complete_mask(df):
    return df.entry_available.fillna(False).astype(bool)&df.missing_minutes.eq(0)

def truth(df, threshold):
    return df[f'runner_{threshold}pct'].fillna(False).to_numpy(dtype=bool)

def wilson(success,n):
    if not n:return (np.nan,np.nan)
    z=1.95996398454;p=success/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return float(center-half),float(center+half)

def metrics(df, score, cutoff, threshold):
    score=np.asarray(score);selected=score>=cutoff
    observed=truth(df,threshold);known=df[f'runner_{threshold}pct'].notna().to_numpy()
    complete=complete_mask(df).to_numpy();n=int(selected.sum());tp=int((selected&observed).sum())
    fp=int((selected&known&~observed).sum());unk=int((selected&~known).sum())
    ci=wilson(tp,n)
    return {'calls':n,'hit':tp,'confirmed_failure':fp,'unknown':unk,
            'signal_days':int(df.loc[selected,'date'].nunique()),
            'precision_conservative':tp/n if n else np.nan,
            'precision_upper_bound':(tp+unk)/n if n else np.nan,
            'precision_known':tp/(tp+fp) if tp+fp else np.nan,
            'precision_wilson_low':ci[0],'precision_wilson_high':ci[1],
            'recall_complete':int((selected&observed&complete).sum())/int((observed&complete).sum()) if (observed&complete).any() else np.nan,
            'coverage':n/len(df) if len(df) else np.nan,
            'complete_rows':int(complete.sum()),'complete_positive':int((complete&observed).sum()),
            'baseline_hit_fraction':float(observed.mean()) if len(df) else np.nan,
            'average_precision_complete':float(average_precision_score(observed[complete],score[complete])) if complete.any() and observed[complete].any() else np.nan}

def select_cutoff(df, score, threshold, min_calls=MIN_CALLS, min_days=MIN_DAYS):
    # Only threshold-validation rows may enter this function in the experiment.
    score=np.asarray(score)
    order=np.argsort(-score,kind='stable');s=score[order];y=truth(df,threshold)[order]
    running=np.cumsum(y);ends=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1]
    choices=[]
    for i in ends:
        n=int(i+1)
        if n<min_calls:continue
        ix=order[:n]
        if df.iloc[ix].date.nunique()<min_days:continue
        hits=int(running[i]);low,high=wilson(hits,n)
        choices.append({'cutoff':float(s[i]),'calls':n,'precision':hits/n,'wilson_low':low})
    if not choices:
        return {'cutoff':1.000001,'qualified':False,'reason':'insufficient threshold-validation signal support','calibration':None}
    qualified=[x for x in choices if x['precision']>=TARGET_PRECISION and x['wilson_low']>=.65]
    if qualified:
        chosen=max(qualified,key=lambda x:(x['calls'],x['wilson_low']))
        return {'cutoff':chosen['cutoff'],'qualified':True,'reason':'80% threshold-validation precision target met with support','calibration':chosen}
    chosen=max(choices,key=lambda x:(x['wilson_low'],x['precision'],x['calls']))
    return {'cutoff':chosen['cutoff'],'qualified':False,'reason':'research-only fallback; 80% precision target not met','calibration':chosen}

def cluster_interval(df, selected, threshold, seed=SEED, repeats=2000):
    x=pd.DataFrame({'date':df.date.to_numpy(),'call':np.asarray(selected,dtype=int),'hit':np.asarray(selected,dtype=int)*truth(df,threshold).astype(int)})
    # Resample all held-out dates, including dates with no calls.
    grouped=x.groupby('date')[['call','hit']].sum().to_numpy()
    if not grouped[:,0].sum():return (np.nan,np.nan)
    rng=np.random.default_rng(seed);sample=rng.integers(0,len(grouped),size=(repeats,len(grouped)))
    sums=grouped[sample].sum(axis=1);valid=sums[:,0]>0
    vals=sums[valid,1]/sums[valid,0]
    return tuple(map(float,np.quantile(vals,[.025,.975])))
