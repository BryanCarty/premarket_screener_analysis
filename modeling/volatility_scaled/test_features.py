import numpy as np
import pandas as pd
import pytest
from modeling.volatility_scaled.build import history_dates,prior_features,adjust_volume_history
import exchange_calendars as xcals


def fixture():
    h=pd.DataFrame({'date':pd.bdate_range('2025-01-01',periods=20).strftime('%Y-%m-%d'),'entry_close_log_return':np.linspace(-.03,.03,20)})
    r={'date':'2025-02-03'}
    for name in ('pm','pm_last30','open15'):
        h[name+'_volume']=np.arange(1,21)*100.
        h[name+'_rv_pct']=np.linspace(1,3,20)
        h[name+'_range_pct']=np.linspace(2,4,20)
        h[name+'_bars']=15
        r[name+'_volume']=2100.;r[name+'_realized_vol_pct']=4.;r[name+'_range_pct']=6.
    return r,h


def test_prior_features_use_same_windows_and_reject_future():
    r,h=fixture();f=prior_features(r,h)
    assert f['hist20_pm_volume_ratio']==pytest.approx(2)
    assert f['hist20_open15_rv_pct_ratio']==pytest.approx(2)
    assert f['sigma20_log']==pytest.approx(h.entry_close_log_return.std(ddof=1))
    h.loc[0,'date']=r['date']
    with pytest.raises(ValueError):prior_features(r,h)


def test_sparse_and_constant_history_is_not_fabricated():
    r,h=fixture();f=prior_features(r,h.iloc[:10])
    assert np.isnan(f['sigma20_log']) and np.isnan(f['hist20_pm_volume_z'])
    h['pm_volume']=1.
    assert np.isnan(prior_features(r,h)['hist20_pm_volume_z'])


def test_split_alignment_excludes_future_actions_and_keeps_returns():
    r,h=fixture();events=[dict(execution_date='2025-02-01',split_to=2,split_from=1),dict(execution_date='2025-02-10',split_to=100,split_from=1)]
    adjusted=adjust_volume_history(h,events,'2025-02-03')
    assert np.allclose(adjusted.pm_volume,h.pm_volume*2)
    assert np.allclose(adjusted.entry_close_log_return,h.entry_close_log_return)
    assert np.allclose(h.pm_volume,np.arange(1,21)*100)


def test_history_excludes_today_and_handles_exchange_holidays():
    dates=history_dates('2025-07-07',xcals.get_calendar('XNYS'))
    assert len(dates)==20 and dates[-1]=='2025-07-03' and max(dates)<'2025-07-07'
