import numpy as np
import pandas as pd
from modeling.open_occupancy.build import aggregate_window,opening_targets,summarize,assemble,CUTS
from runner.collect import save


def bars(day,active=False):
    midnight=pd.Timestamp(day,tz='America/New_York').value
    b=pd.DataFrame(dict(ticker='TEST',minute=list(range(540,569))+list(range(570,960))))
    b['window_start']=midnight+b.minute*60_000_000_000
    for col in ('open','close'):b[col]=115. if active else 100.
    b['high']=b.close+1;b['low']=b.close-1;b['volume']=10000. if active else 1000.;b['transactions']=10.;b['invalid']=False
    benchmarks=[]
    for ticker in ('SPY','QQQ'):
        z=b.copy();z['ticker']=ticker;benchmarks.append(z)
    return pd.concat([b]+benchmarks,ignore_index=True)


def master():
    return pd.DataFrame(dict(ticker=['TEST','SPY','QQQ'],identity=['id','benchmark_SPY','benchmark_QQQ'],reg_close=[100.,np.nan,np.nan],reg_volume=[400000.,np.nan,np.nan]))


def test_target_is_close_above_open_with_fixed_denominator_and_no_stale_fill():
    b=bars('2025-01-02');b.loc[b.minute.eq(570),'open']=100
    b.loc[b.minute.between(570,581),'close']=101
    y=opening_targets(b)
    assert y.loc['TEST','above_pct']==80
    b=b[~b.minute.eq(584)]
    assert np.isnan(opening_targets(b).loc['TEST','above_pct'])


def test_post_cutoff_changes_do_not_change_window_features():
    b=bars('2025-01-02');a=aggregate_window(b,240,554)
    b.loc[b.minute.ge(554),['open','high','low','close','volume']]=[200,250,150,220,1000000]
    pd.testing.assert_frame_equal(a,aggregate_window(b,240,554))


def test_cohort_does_not_depend_on_current_opening_coverage_or_volume(tmp_path):
    days=pd.bdate_range('2025-01-02',periods=11).strftime('%Y-%m-%d').tolist()
    for day in days:
        save(summarize(bars(day,active=day==days[-1]),master(),day,960),tmp_path/f'{day}.parquet')
    original,_=assemble(days,tmp_path,pd.DataFrame(),{days[-1]})
    assert len(original)==len(CUTS)
    b=bars(days[-1],active=True);b=b[~b.minute.eq(571)]
    b.loc[b.minute.ge(570),'volume']=1
    save(summarize(b,master(),days[-1],960),tmp_path/f'{days[-1]}.parquet')
    changed,_=assemble(days,tmp_path,pd.DataFrame(),{days[-1]})
    assert len(changed)==len(CUTS) and changed.target_above_pct.isna().all()
    cols=[c for c in original if not c.startswith('target_')]
    pd.testing.assert_frame_equal(original[cols],changed[cols])


def test_prior_missing_regular_candle_excludes_opportunity(tmp_path):
    days=pd.bdate_range('2025-01-02',periods=11).strftime('%Y-%m-%d').tolist()
    for day in days:
        b=bars(day,active=day==days[-1])
        if day==days[-2]:b=b[~b.minute.eq(800)]
        save(summarize(b,master(),day,960),tmp_path/f'{day}.parquet')
    result,_=assemble(days,tmp_path,pd.DataFrame(),{days[-1]})
    assert result.empty


def test_early_close_prior_grid_uses_short_session():
    b=bars('2025-07-03');b=b[b.minute.lt(780)]
    s=summarize(b,master(),'2025-07-03',780).set_index('ticker')
    assert s.loc['TEST','reg_expected']==210 and s.loc['TEST','reg_complete']
    s=summarize(b[b.minute.ne(779)],master(),'2025-07-03',780).set_index('ticker')
    assert not s.loc['TEST','reg_complete']


def test_fractional_estimator_has_stable_serialization(tmp_path):
    import joblib
    from modeling.open_occupancy.estimators import FractionalLogit
    x=pd.DataFrame({'a':np.ones(30),'b':np.zeros(30)});y=np.tile([0.,50.,100.],10)
    model=FractionalLogit().fit(x,y)
    assert model.__class__.__module__=='modeling.open_occupancy.estimators'
    file=tmp_path/'model.joblib';joblib.dump(model,file)
    assert np.allclose(joblib.load(file).predict(x),50,atol=.01)
