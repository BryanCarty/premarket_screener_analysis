import pandas as pd
import numpy as np
from runner.engine import labels, features
from runner.collect import normalize

def bars(lo=240,hi=960):
    m=np.arange(lo,hi)
    return pd.DataFrame({'minute':m,'open':100.,'close':100.,'high':100.,'low':100.,'volume':100.,'transactions':10.})

def test_horizon_boundaries_and_entry():
    b=bars();b.loc[b.minute==584,'high']=1000
    b.loc[b.minute==590,'high']=120
    r=labels(b,960)
    assert r[0]['runner_5pct'] is False
    assert r[1]['runner_20pct'] is True
    assert r[1]['minutes_to_20pct']==5
    assert r[0]['entry_price_proxy']==100

def test_missing_bars_are_unknown_unless_hit():
    b=bars();b=b[b.minute!=586]
    assert labels(b,960)[0]['runner_5pct'] is None
    b.loc[b.minute==587,'high']=110
    assert labels(b,960)[0]['runner_5pct'] is True
    assert labels(b[b.minute!=585],960)[0]['runner_5pct'] is None

def test_missing_endpoint_and_early_close():
    b=bars(240,780)
    r=labels(b,780)[-1]
    assert r['expected_minutes']==195
    assert r['end_return_pct']==0
    assert np.isnan(labels(b[b.minute!=779],780)[-1]['end_return_pct'])

def test_features_do_not_need_future():
    b=bars();hist=pd.DataFrame({'reg_volume':[100]*10,'pm_volume':[10]*10})
    cutoff=pd.Timestamp('2025-01-06 14:45',tz='UTC')
    f=features(b[b.minute<585],90,100,hist,cutoff)
    b.loc[b.minute>=585,['open','close','high','low']]=10000
    assert features(b[b.minute<585],90,100,hist,cutoff)==f

def test_dst_conversion():
    for utc in ('2025-01-06 14:30','2025-07-07 13:30'):
        x={'ticker':'X','window_start':pd.Timestamp(utc,tz='UTC').value,'open':1,'close':1,'high':1,'low':1,'volume':1,'transactions':1}
        assert normalize(pd.DataFrame([x])).minute.iloc[0]==570
