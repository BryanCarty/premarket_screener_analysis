import json
from types import SimpleNamespace
import pandas as pd
from runner import collect

def test_pipeline_eligibility_splits_and_resume(tmp_path,monkeypatch):
    target='2025-01-06'
    class FakeAPI:
        def __init__(self,root,pace):
            self.cache=root/'api_cache';self.cache.mkdir(exist_ok=True)
        def pages(self,path,**kw):
            return [{'ticker':'X','execution_date':target,'split_from':1,'split_to':2}]
        def get(self,path,**kw):
            day=path.rsplit('/',1)[1]
            return {'results':[{'T':'X','c':100 if day<target else 65,'v':1000}]}
    monkeypatch.setattr(collect,'API',FakeAPI)
    monkeypatch.setattr(collect,'universe',lambda api,day:pd.DataFrame([{'ticker':'X','type':'CS','share_class_figi':'ID'}]))
    def raw(day,tickers,api):
        # 2-for-1 split means 100 prior close -> 50; PM120 shares -> baseline20 shares.
        pm_price=60 if day==target else 100
        times=[240,569]+list(range(570,960))
        rows=[]
        for minute in times:
            price=pm_price if minute<570 else (60 if day==target else 100)
            ts=pd.Timestamp(day,tz='America/New_York')+pd.Timedelta(minutes=minute)
            rows.append(dict(ticker='X',window_start=ts.value,open=price,close=price,high=price,low=price,volume=(60 if day==target else 5) if minute<570 else 1,transactions=1))
        return collect.normalize(pd.DataFrame(rows))
    monkeypatch.setattr(collect,'rest_day',raw)
    args=SimpleNamespace(start=target,end=target,source='rest-sample',tickers='X',output=str(tmp_path),pace=0)
    collect.run(args)
    f=pd.read_parquet(tmp_path/'features.parquet');l=pd.read_parquet(tmp_path/'labels.parquet')
    assert len(f)==1 and len(l)==7
    assert f.previous_close.iloc[0]==50
    assert f.pm_rvol.iloc[0]==6
    assert abs(f.pm_gap_pct.iloc[0]-20)<1e-8
    assert (l.runner_5pct==False).all()
    original=f.copy()
    collect.run(args)
    pd.testing.assert_frame_equal(original,pd.read_parquet(tmp_path/'features.parquet'))
    assert json.loads((tmp_path/'summary.json').read_text())['qualifying_rows']==1
