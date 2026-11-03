"""Check headline timing, delayed targets, OOF chronology and frozen precision metrics."""
import hashlib,json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import exchange_calendars as xcals
from threadpoolctl import threadpool_limits
from .news import news_features
from .evaluate import mask,score,top_calls,KEYS
from modeling.barrier_study.build import barrier_label


def equal(value,actual):
    if value is None:return pd.isna(actual)
    if isinstance(value,str):return value==actual
    return np.isclose(value,actual,equal_nan=True)


def main():
    data=Path('modeling/catalyst_precision/data');out=Path('modeling/catalyst_precision/results')
    d=pd.read_parquet(data/'dataset.parquet');p=pd.read_parquet(out/'predictions.parquet');oof=pd.read_parquet(out/'development_predictions.parquet')
    manifest=json.loads((out/'manifest.json').read_text());frozen=json.loads((out/'frozen_selection.json').read_text());cal=xcals.get_calendar('XNYS')
    assert hashlib.sha256((data/'dataset.parquet').read_bytes()).hexdigest()==manifest['dataset_sha256']
    source=manifest['data_manifest'];assert hashlib.sha256(Path(source['source']).read_bytes()).hexdigest()==source['source_sha256']
    assert hashlib.sha256(Path('modeling/catalyst_precision/plan.json').read_bytes()).hexdigest()==source['plan_sha256']
    for row in d.iloc[::10].to_dict('records'):
        token=hashlib.sha256((row['date']+'|'+row['ticker']).encode()).hexdigest()[:24]
        archive=json.loads((data/'news_parts'/f'{token}.json').read_text());cutoff=pd.Timestamp(row['date']+' 09:45',tz='America/New_York').tz_convert('UTC')
        prev=cal.previous_session(pd.Timestamp(row['date']));prior_close=cal.session_close(prev)
        features,audit=news_features(archive['articles'] if archive['identity_match'] else [],row['ticker'],archive['company_name'],cutoff,prior_close)
        for field,value in features.items():assert equal(value,row[field]),field
        assert all(pd.Timestamp(a['published_utc'])<cutoff for a in audit)
        b=pd.read_parquet(data/'paths'/f'{token}.parquet')
        for prefix,start,end in [('target',605,665),('late',615,675)]:
            target=barrier_label(b,row['sigma20_log'],start=start,end=end)
            for field,value in target.items():assert equal(value,row[prefix+'_'+field]),field
    selected={s['id']:s for s in frozen['candidates']}
    for (candidate,fold),g in oof.groupby(['candidate','fold']):
        selection=selected[candidate]
        artifact=joblib.load(out/'development_models'/f'{selection["segment"]}__{selection["feature_set"]}__{selection["model"]}__fold{fold}.joblib')
        assert max(artifact['train_dates'])<artifact['embargo_date']<min(artifact['validation_dates'])
        assert g.date.isin(artifact['validation_dates']).all()
        x=g[KEYS].merge(d,on=KEYS,validate='one_to_one')
        with threadpool_limits(limits=2):s=score(artifact['estimator'],x[artifact['features']])
        assert np.allclose(s,g.score)
        assert np.array_equal(top_calls(s,.2),g.top20)
    metrics=pd.read_csv(out/'metrics.csv');trades=pd.read_csv(out/'strategy_metrics.csv')
    for candidate,g in p[p.split.eq('test')].groupby('candidate'):
        artifact=joblib.load(out/'models'/f'{candidate}.joblib');x=g[KEYS].merge(d,on=KEYS,validate='one_to_one')
        assert mask(x,artifact['segment']).all()
        assert g.date.isin(manifest['split_dates']['test']).all()
        assert not any(c.startswith(('target_','late_','label_')) for c in artifact['features'])
        with threadpool_limits(limits=2):s=score(artifact['estimator'],x[artifact['features']])
        assert np.allclose(s,g.score)
        called=s>=artifact['policy']['cutoff'];assert np.array_equal(called,g.called)
        m=metrics[metrics.candidate.eq(candidate)&metrics.split.eq('test')].iloc[0]
        assert int(called.sum())==m.calls
        hits=int((g.called&g.target_outcome.eq('up')).sum());unknown=int((g.called&g.target_outcome.isna()).sum())
        assert hits==m.hits and unknown==m.unknown_calls
        if m.calls:assert np.isclose(hits/m.calls,m.precision)
        for entry,prefix in [('10:05','target'),('10:15_sensitivity','late')]:
            t=trades[trades.candidate.eq(candidate)&trades.split.eq('test')&trades.entry.eq(entry)].iloc[0]
            known=g.called & g[prefix+'_outcome'].notna();returns=g.loc[known,prefix+'_gross_return_pct']
            assert len(returns)==t.resolved and int(called.sum())==t.signals
            if len(returns):assert np.isclose(returns.mean(),t.mean_gross_pct)
    for selection in frozen['candidates']:
        if selection['approved']:
            m=selection['validation_metrics'];r=selection['validation_returns']
            assert m['precision']>=.8 and m['calls']>=25 and m['call_dates']>=10 and m['wilson_low']>=.65
            assert r['mean_net_50bps_pct']>0 and r['worst_pct']>=-10
    assert not p.duplicated(KEYS+['candidate','split']).any()
    assert (out/'frozen_selection.json').stat().st_mtime<=(out/'predictions.parquet').stat().st_mtime
    print('Verified 82 headline and delayed-target samples, all chosen OOF probabilities/chronology/top20 calls, source/plan hashes, frozen test scores/precision/returns, and validation gates.')

if __name__=='__main__':main()
