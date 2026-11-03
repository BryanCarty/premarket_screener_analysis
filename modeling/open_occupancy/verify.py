"""Verify pre-open eligibility, prior-only baselines, phase timing and frozen forecasts."""
import hashlib,json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import exchange_calendars as xcals
from threadpoolctl import threadpool_limits
from .build import CUTS
from .evaluate import predict,KEYS


def main():
    data=Path('modeling/open_occupancy/data');out=Path('modeling/open_occupancy/results')
    d=pd.read_parquet(data/'dataset.parquet');p=pd.read_parquet(out/'predictions.parquet');m=json.loads((out/'manifest.json').read_text())
    assert hashlib.sha256((data/'dataset.parquet').read_bytes()).hexdigest()==m['dataset_sha256']
    assert json.loads(Path('modeling/open_occupancy/plan.json').read_text())==m['plan']
    frozen=json.loads((out/'frozen_selection.json').read_text());cal=xcals.get_calendar('XNYS');events=json.loads((data/'split_events.json').read_text())
    known=d.target_above_pct.notna()
    assert d.loc[known,'target_above_pct'].between(0,100).all()
    assert np.allclose(d.loc[known,'target_above_pct']*15/100,np.round(d.loc[known,'target_above_pct']*15/100))
    assert d.loc[known,'target_observed'].eq(15).all()
    assert d.loc[known,'target_entire_above_pct'].le(d.loc[known,'target_above_pct']+1e-8).all()
    assert not d.duplicated(KEYS).any()
    cache={}
    def day_frame(day):
        if day not in cache:cache[day]=pd.read_parquet(data/'daily'/f'{day}.parquet').set_index(['ticker','identity'])
        return cache[day]
    sample=d.iloc[::max(1,len(d)//50)]
    for row in sample.to_dict('records'):
        day=row['date'];phase=row['phase'];key=(row['ticker'],row['identity']);cut=CUTS[phase]
        expected=(pd.Timestamp(day,tz='America/New_York')+pd.Timedelta(minutes=cut)).tz_convert('UTC')
        assert pd.Timestamp(row['feature_cutoff_utc'])==expected
        previous=str(cal.previous_session(pd.Timestamp(day)).date());prior=day_frame(previous).loc[key]
        assert bool(prior.reg_complete) and prior.reg_rows==prior.reg_unique==prior.reg_expected
        assert prior.reg_invalid==0
        current=day_frame(day).loc[key];volumes=[];past_open=[];prior_close=None
        for offset in range(-10,0):
            hd=str(cal.session_offset(pd.Timestamp(day),offset).date());assert hd<day
            h=day_frame(hd).loc[key]
            factor=np.prod([float(e['split_to'])/float(e['split_from']) for e in events if e['ticker']==row['ticker'] and hd<e['execution_date']<=day])
            volumes.append(h[phase+'_pm_volume']*factor);past_open.append(h.y_above_pct)
            if offset==-1:prior_close=h.official_close/factor
        assert np.isclose(row['pm_rvol'],current[phase+'_pm_volume']/np.mean(volumes))
        assert np.isclose(row['pm_gap_pct'],100*(current[phase+'_pm_last']/prior_close-1))
        assert np.isclose(row['prior10_opening_above_mean'],np.nanmean(past_open),equal_nan=True)
        assert row['pm_gap_pct']>=10-1e-8 and row['pm_rvol']>=5-1e-8
        assert np.isclose(row['target_above_pct'],current.y_above_pct,equal_nan=True)
    metrics=pd.read_csv(out/'regression_metrics.csv');precision=pd.read_csv(out/'precision_metrics.csv')
    for candidate,g in p[p.split.eq('test')].groupby('candidate'):
        artifact=joblib.load(out/'models'/f'{candidate}.joblib')
        assert g.phase.eq(artifact['phase']).all()
        assert g.date.isin(m['split_dates']['test']).all()
        assert not any(c.startswith('target_') or c in ('date','ticker','identity','feature_cutoff_utc') for c in artifact['features'])
        assert not any(c.startswith('open15_') for c in artifact['features'])
        x=g[KEYS].merge(d,on=KEYS,validate='one_to_one')
        with threadpool_limits(limits=2):pred=predict(artifact['estimator'],x[artifact['features']])
        assert np.allclose(pred,g.prediction_pct)
        assert np.array_equal(pred/100>=artifact['policy']['cutoff'],g.called)
        exact=g.target_above_pct.notna().to_numpy();y=g.loc[exact,'target_above_pct'].to_numpy()
        r=metrics[metrics.candidate.eq(candidate)&metrics.split.eq('test')].iloc[0]
        assert np.isclose(np.mean(np.abs(y-pred[exact])),r.mae_percentage_points)
        assert np.isclose(np.sqrt(np.mean((y-pred[exact])**2)),r.rmse_percentage_points)
        policy=precision[precision.candidate.eq(candidate)&precision.split.eq('test')].iloc[0]
        n=int(g.called.sum());hits=int((g.called&g.target_above_pct.ge(80-1e-10)).sum())
        assert n==policy.calls and hits==policy.hits
        if n:assert np.isclose(hits/n,policy.precision)
    assert (out/'frozen_selection.json').stat().st_mtime<=(out/'predictions.parquet').stat().st_mtime
    assert not p.duplicated(KEYS+['candidate','split']).any()
    print(f'Verified {len(sample)} prior-grid/cutoff/history samples, exact target counts, source/plan hashes, all saved test scores/errors/precision and chronology.')

if __name__=='__main__':main()
