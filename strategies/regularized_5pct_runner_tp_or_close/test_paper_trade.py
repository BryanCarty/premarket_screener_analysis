import pandas as pd
import pytest
from strategies.regularized_5pct_runner_tp_or_close.paper_trade import (
    stamp, safe_features, future_entry, evaluate_order, summarize, atomic,
)
from runner.collect import normalize, COLUMNS

DAY='2026-10-30'

def bars(minutes,high=101,close=99):
    return normalize(pd.DataFrame([dict(ticker='TEST',volume=100,open=100,close=close,
        high=high,low=min(close,99),window_start=stamp(DAY,m).value,transactions=10)
        for m in minutes],columns=COLUMNS))


def order():
    return dict(decision_at=stamp(DAY,600).isoformat(),entry_at=stamp(DAY,601).isoformat(),notional=1000)


def test_feature_cutoff_and_completion_are_enforced():
    b=bars([583,584,585,586])
    assert safe_features(b,stamp(DAY,585),stamp(DAY,584)+pd.Timedelta(seconds=30)).minute.tolist()==[583]
    assert safe_features(b,stamp(DAY,585),stamp(DAY,600)).minute.tolist()==[583,584]


def test_entry_strictly_after_decision():
    t=stamp(DAY,585)+pd.Timedelta(seconds=45)
    assert future_entry(t)==stamp(DAY,586)
    o=order();o['entry_at']=o['decision_at']
    with pytest.raises(ValueError):evaluate_order(o,bars([600]),stamp(DAY,960),stamp(DAY,961))


def test_pre_entry_target_is_not_used():
    b=bars([600],high=110)
    r=evaluate_order(order(),b,stamp(DAY,960),stamp(DAY,961))
    assert r['status']=='unresolved' and r['reason']=='missing_entry'


def test_unfinished_target_bar_is_not_used():
    b=bars([601],high=110)
    assert evaluate_order(order(),b,stamp(DAY,960),stamp(DAY,601)+pd.Timedelta(seconds=59))['status']=='pending'
    r=evaluate_order(order(),b,stamp(DAY,960),stamp(DAY,602))
    assert r['gross_return_pct']==5 and r['exit_price']==105


def test_observed_fixed_target_resolves_despite_interior_gap():
    b=bars([601,603]);b.loc[b.minute.eq(603),'high']=106
    assert evaluate_order(order(),b,stamp(DAY,960),stamp(DAY,604))['reason']=='target'


def test_missing_path_cannot_establish_no_target():
    r=evaluate_order(order(),bars([601,959]),stamp(DAY,960),stamp(DAY,961))
    assert r['status']=='unresolved'


def test_complete_path_exits_at_close_and_respects_early_close():
    r=evaluate_order(order(),bars(range(601,780)),stamp(DAY,780),stamp(DAY,781))
    assert r['reason']=='close' and r['gross_return_pct']==pytest.approx(-1)


def test_summary_separates_unknown_and_persists(tmp_path):
    o=order();o.update(status='closed',reason='target',gross_return_pct=5)
    state=dict(date=DAY,round_trip_cost_pct=.5,feed_delay_seconds=900,orders={'TEST':o,'OTHER':{'status':'unresolved'}},finished=True)
    r=summarize(state,tmp_path)
    assert r['closed']==1 and r['unresolved_or_pending']==1
    assert r['mean_net_pct']==4.5 and r['total_resolved_net_pnl_dollars']==45
    assert (tmp_path/'SUMMARY.md').exists()
    atomic(tmp_path/'state.json',state)
