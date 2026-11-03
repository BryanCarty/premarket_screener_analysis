import numpy as np
import pandas as pd
from modeling.barrier_study.build import barrier_label,intraday_features


def bars():
    return pd.DataFrame(dict(minute=range(240,645),open=100.,high=100.5,low=99.5,close=100.,volume=1000.))


def test_up_first_not_changed_by_later_collapse():
    b=bars();b.loc[b.minute.eq(586),'high']=104;b.loc[b.minute.eq(587),'low']=90
    r=barrier_label(b,.03)
    assert r['outcome']=='up' and r['exit_minute']==586


def test_down_first_and_stop_gap_fill():
    b=bars();b.loc[b.minute.eq(586),['open','low','high','close']]=[95,94,96,95]
    r=barrier_label(b,.03)
    assert r['outcome']=='down' and r['reason']=='stop_gap' and np.isclose(r['gross_return_pct'],-5)


def test_same_bar_tie_is_unresolved_unless_open_establishes_order():
    b=bars();b.loc[b.minute.eq(586),['high','low']]=[110,90]
    assert barrier_label(b,.03)['reason']=='ambiguous_same_minute'
    b.loc[b.minute.eq(586),'open']=104
    assert barrier_label(b,.03)['outcome']=='up'


def test_missing_before_exit_is_unresolved_but_after_does_not_matter():
    b=bars();b.loc[b.minute.eq(587),'high']=104
    assert barrier_label(b[b.minute.ne(586)],.03)['reason']=='missing_before_exit'
    b.loc[b.minute.eq(586),'high']=104
    assert barrier_label(b[b.minute.ne(587)],.03)['outcome']=='up'


def test_timeout_requires_full_grid_and_uses_final_close():
    b=bars();b.loc[b.minute.eq(644),'close']=100.2
    r=barrier_label(b,.03)
    assert r['outcome']=='neither' and np.isclose(r['gross_return_pct'],.2)
    assert barrier_label(b[b.minute.ne(644)],.03)['reason']=='missing_before_timeout'


def test_post_cutoff_prices_cannot_change_features():
    b=bars();before=intraday_features(b)
    b.loc[b.minute.ge(585),['open','high','low','close','volume']]=[200,250,150,220,100000]
    assert intraday_features(b)==before


def test_pre_entry_and_after_horizon_barriers_are_ignored():
    b=bars();b.loc[b.minute.eq(584),['high','low']]=[150,50]
    extra=pd.DataFrame(dict(minute=[645],open=[100.],high=[150.],low=[50.],close=[100.],volume=[1000.]))
    assert barrier_label(pd.concat([b,extra],ignore_index=True),.03)['outcome']=='neither'
