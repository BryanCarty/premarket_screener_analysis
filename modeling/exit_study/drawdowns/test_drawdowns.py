import itertools
import numpy as np
import pandas as pd
from modeling.exit_study.drawdowns.analyze import ohlc_bounds,route_drawdown,paired_distribution_test

def test_mdd_bounds_match_exhaustive_intrabar_ordering():
    b=pd.DataFrame({'minute':[585,586],'open':[100,105],'high':[110,115],'low':[90,95],'close':[105,100]})
    expected=[]
    for choices in itertools.product([0,1],repeat=2):
        path=[100]
        for r,choice in zip(b.itertuples(),choices):path+=([r.open,r.high,r.low,r.close] if choice==0 else [r.open,r.low,r.high,r.close])
        expected.append(route_drawdown(path,100)[0])
    result=ohlc_bounds(b,100,100,587,False)
    assert np.isclose(result['peak_to_trough_lower_pct'],min(expected))
    assert np.isclose(result['peak_to_trough_upper_pct'],max(expected))

def test_exit_bar_does_not_include_after_exit_high_and_low_as_certain():
    b=pd.DataFrame({'minute':[585],'open':[100],'high':[150],'low':[80],'close':[90]})
    result=ohlc_bounds(b,100,105,585,True)
    assert np.isclose(result['mfe_pct'],5)
    assert result['mae_lower_pct']==0 and np.isclose(result['mae_upper_pct'],20)
    assert result['peak_to_trough_lower_pct']==0 and np.isclose(result['peak_to_trough_upper_pct'],20)

def test_exact_paired_distribution_permutations_preserve_clusters():
    assert paired_distribution_test([5]*4,[1]*4,np.arange(4))['p_value']==.125
    assert paired_distribution_test([5]*4,[1]*4,np.zeros(4))['p_value']==1
