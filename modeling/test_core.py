import numpy as np
import pandas as pd
from modeling.core import make_splits,metrics,select_cutoff,complete_mask

def test_chronological_splits_and_embargo():
    dates=pd.date_range('2024-01-01',periods=100).strftime('%Y-%m-%d')
    s=make_splits(dates)
    assert len(set(np.concatenate(list(s.values()))))==100
    assert max(s['train'])<min(s['validation'])<max(s['validation'])<min(s['threshold'])<max(s['threshold'])<min(s['test'])
    assert not set(s['embargo'])&set(np.r_[s['train'],s['validation'],s['threshold'],s['test']])

def test_unknown_calls_lower_precision_and_incomplete_positives_excluded_training():
    d=pd.DataFrame({'date':['2025-01-01']*3,'runner_10pct':[True,False,None],'entry_available':[True]*3,'missing_minutes':[1,0,2]})
    m=metrics(d,[.9,.8,.7],.5,10)
    assert m['precision_conservative']==1/3
    assert m['precision_known']==.5
    assert m['precision_upper_bound']==2/3
    assert complete_mask(d).tolist()==[False,True,False]

def test_threshold_qualification_needs_support_and_ties_are_included():
    d=pd.DataFrame({'date':[f'2025-01-{i%20+1:02}' for i in range(100)],'runner_10pct':[True]*40+[False]*60})
    score=np.r_[np.full(40,.9),np.full(60,.1)]
    p=select_cutoff(d,score,10)
    assert p['qualified'] and p['calibration']['calls']==40
    assert p['cutoff']==.9
    small=select_cutoff(d.iloc[:20],score[:20],10)
    assert not small['qualified']
