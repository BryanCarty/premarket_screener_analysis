"""Validate produced join keys, eligibility, horizon coverage, and cutoff metadata."""
import sys
from pathlib import Path
import pandas as pd
from .engine import HORIZONS, THRESHOLDS

def validate(root):
    root=Path(root);f=pd.read_parquet(root/'features.parquet');l=pd.read_parquet(root/'labels.parquet')
    keys=['date','ticker','identity']
    assert not f.duplicated(keys).any(), 'Duplicate features'
    if f.empty:
        assert l.empty
        print('Valid empty dataset; no evidence of prevalence yet.');return
    assert (f.pm_rvol>=5-1e-10).all() and (f.pm_gap_pct>=10-1e-10).all()
    assert not l.duplicated(keys+['horizon']).any()
    assert len(l)==7*len(f)
    joined=l.merge(f[keys],on=keys,how='outer',indicator=True)
    assert joined['_merge'].eq('both').all()
    assert set(l.horizon)==set(map(str,HORIZONS))
    assert l.groupby(keys).size().eq(7).all()
    local=pd.to_datetime(f.feature_cutoff_utc,utc=True).dt.tz_convert('America/New_York')
    assert (local.dt.hour.eq(9)&local.dt.minute.eq(45)).all()
    assert (local.dt.strftime('%Y-%m-%d')==f.date).all()
    for t in THRESHOLDS:
        s=l[f'runner_{t}pct']
        assert s.dropna().isin([True,False]).all()
        assert not ((s==False)&l.missing_minutes.gt(0)).any()
        assert not ((s==True)&(l.max_gain_pct<t-1e-8)).any()
    print(f'Validated {len(f)} feature rows and {len(l)} horizon rows.')

if __name__=='__main__':validate(sys.argv[1])
