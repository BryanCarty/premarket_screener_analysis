import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from modeling.core import make_splits

HORIZONS=['5','15','30','60','90','120','close']
TIMES=np.array([0,5,15,30,60,90,120,375],dtype=float)
DYNAMIC=['elapsed_session_fraction','position','price_return_since_945','observed_max_gain_since_945','observed_max_decline_since_945','last_interval_return','position_return','position_drawdown_at_landmarks','holding_fraction','remaining_fraction']
EXCLUDE=['date','ticker','identity','feature_cutoff_utc','opening_available','opening_minutes_observed','opening_missing_minutes']

def load(source):
    source=Path(source);f=pd.read_parquet(source/'features.parquet');l=pd.read_parquet(source/'labels.parquet')
    splits=make_splits(f.date)
    f=f[f.opening_missing_minutes.eq(0)].sort_values(['date','ticker','identity']).reset_index(drop=True)
    keys=['date','ticker','identity'];features=[c for c in f if c not in EXCLUDE]
    assert not f.duplicated(keys).any() and not l.duplicated(keys+['horizon']).any()
    arrays=[]
    for h in HORIZONS:
        g=l[l.horizon.eq(h)]
        arrays.append(f[keys].merge(g,on=keys,how='left',validate='one_to_one'))
    entry=arrays[0].entry_price_proxy.to_numpy(float)
    price=np.column_stack([entry]+[entry*(1+a.end_return_pct.to_numpy(float)/100) for a in arrays])
    high=np.column_stack([np.zeros(len(f))]+[a.max_gain_pct.to_numpy(float) for a in arrays])
    low=np.column_stack([np.zeros(len(f))]+[a.max_decline_pct.to_numpy(float) for a in arrays])
    times=np.tile(TIMES,(len(f),1));times[:,-1]=arrays[-1].interval_end_minute.to_numpy(float)-585
    # Past extrema can be unavailable even with an endpoint price; fall back to known endpoints only.
    for t in range(1,8):
        endpoint=(price[:,t]/price[:,0]-1)*100
        high[:,t]=np.where(np.isfinite(high[:,t]),high[:,t],endpoint)
        low[:,t]=np.where(np.isfinite(low[:,t]),low[:,t],endpoint)
    raw=f[features].replace([np.inf,-np.inf],np.nan).to_numpy(float)
    complete=np.isfinite(price).all(axis=1)&(price>0).all(axis=1)
    trainmask=f.date.isin(splits['train']).to_numpy()&complete
    imputer=SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True).fit(raw[trainmask])
    static=imputer.transform(raw).astype(np.float32)
    return {'frame':f[keys],'features':features,'static':static,'price':price,'high':high,'low':low,'times':times,'complete':complete,'splits':splits,'imputer':imputer,
            'source_sha256':{name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in ['features.parquet','labels.parquet']}}

def observation(data,ix,t,entry):
    ix=np.asarray(ix,dtype=int);n=len(ix);p=data['price'][ix];tm=data['times'][ix]
    out=np.zeros((n,len(DYNAMIC)),dtype=np.float32)
    out[:,0]=tm[:,t]/tm[:,-1];out[:,1]=float(entry>=0)
    out[:,9]=(tm[:,-1]-tm[:,t])/tm[:,-1]
    if t:
        out[:,2]=(p[:,t]/p[:,0]-1)*100
        out[:,3]=data['high'][ix,t];out[:,4]=data['low'][ix,t]
        out[:,5]=(p[:,t]/p[:,t-1]-1)*100
    if entry>=0:
        out[:,6]=(p[:,t]/p[:,entry]-1)*100
        peak=np.max(p[:,entry:t+1],axis=1)
        out[:,7]=(p[:,t]/peak-1)*100
        out[:,8]=(tm[:,t]-tm[:,entry])/tm[:,-1]
    assert np.isfinite(out).all(), 'A future/missing price entered the current observation'
    return np.column_stack([data['static'][ix],out])
