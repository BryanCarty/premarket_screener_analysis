import pandas as pd
from modeling.exit_study.analyze import simulate

def bars():
    return pd.DataFrame({'minute':range(585,960),'open':100.,'high':100.,'low':100.,'close':100.})

def test_stop_before_later_target_and_same_bar_ambiguity():
    b=bars();b.loc[b.minute.eq(586),'low']=94;b.loc[b.minute.eq(587),'high']=110
    assert simulate(b,5,5,'close',960)['gross_pct']<0
    b=bars();b.loc[b.minute.eq(586),['high','low']]=[110,94]
    r=simulate(b,5,5,'close',960)
    assert r['reason']=='stop' and r['ambiguous_bar']

def test_stop_gap_and_time_exit():
    b=bars();b.loc[b.minute.eq(586),['open','low','high','close']]=[90,89,91,90]
    assert simulate(b,5,5,'close',960)['reason']=='stop_gap'
    assert abs(simulate(b,5,5,'close',960)['gross_pct']+10)<1e-9
    b.loc[b.minute.eq(645),'open']=105
    assert abs(simulate(b,None,None,60,960)['gross_pct']-5)<1e-9

def test_gaps_and_entry():
    b=bars();b=b[~b.minute.eq(586)];b.loc[b.minute.eq(587),'high']=110
    assert simulate(b,5,5,'close',960)['missing_before_exit']
    assert not simulate(b,None,None,'close',960)['missing_before_exit']
    assert simulate(b[b.minute.ne(585)],5,5,'close',960)['reason']=='missing_entry'

def test_exact_decimal_target_and_static_target_with_gap():
    b=bars();b[['open','high','low','close']]=.4
    b=b[b.minute.ne(586)];b.loc[b.minute.eq(587),'high']=.44
    r=simulate(b,10,None,'close',960)
    assert abs(r['gross_pct']-10)<1e-8 and not r['missing_before_exit']
