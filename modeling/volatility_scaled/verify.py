"""Reproduce prior-only baselines, frozen scores and test cohort accounting."""
import hashlib,json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import exchange_calendars as xcals
from sklearn.metrics import mean_absolute_error
from threadpoolctl import threadpool_limits
from .build import prior_features,adjust_volume_history,history_dates,KEYS
from .evaluate import convert


def main():
    data=Path('modeling/volatility_scaled/data');out=Path('modeling/volatility_scaled/results')
    f=pd.read_parquet(data/'features.parquet');h=pd.read_parquet(data/'history_summaries.parquet');p=pd.read_parquet(out/'predictions.parquet')
    events=json.loads((data/'split_events.json').read_text());manifest=json.loads((out/'manifest.json').read_text())
    for file,digest in manifest['source_sha256'].items():assert hashlib.sha256((data/file).read_bytes()).hexdigest()==digest
    source_manifest=json.loads((data/'manifest.json').read_text())
    for file,digest in source_manifest['source_sha256'].items():assert hashlib.sha256((Path(source_manifest['source'])/file).read_bytes()).hexdigest()==digest
    cal=xcals.get_calendar('XNYS')
    for row in f.iloc[::10].to_dict('records'):
        dates=history_dates(row['date'],cal)
        hist=h[h.ticker.eq(row['ticker'])&h.identity.eq(row['identity'])&h.date.isin(dates)]
        assert hist.date.lt(row['date']).all()
        hist=adjust_volume_history(hist,[e for e in events if e['ticker']==row['ticker']],row['date'])
        expected=prior_features(row,hist)
        for feature,value in expected.items():assert np.isclose(value,row[feature],equal_nan=True),feature
    rows=pd.read_csv(out/'metrics.csv')
    for name,g in p[p.split.eq('test')].groupby('name'):
        artifact=joblib.load(out/'models'/f'{name}.joblib')
        x=g[KEYS].merge(f,on=KEYS,validate='one_to_one')
        with threadpool_limits(limits=2):raw=artifact['estimator'].predict(x[artifact['features']])
        scores=convert(raw,artifact['direction'],artifact['units'],x.sigma20_log.to_numpy())
        assert np.allclose(scores,g.prediction_sigma)
        valid=g.actual_sigma.notna()
        saved=rows[rows.split.eq('test')&rows.name.eq(name)].iloc[0]
        assert np.isclose(mean_absolute_error(g.loc[valid,'actual_sigma'],scores[valid]),saved.mae_sigma)
        assert g.date.isin(manifest['split_dates']['test']).all()
    rates=pd.read_csv(out/'base_rates.csv')
    assert np.allclose(rates[['both_1sigma_pct','neither_1sigma_pct','up_only_1sigma_pct','down_only_1sigma_pct']].sum(axis=1),100)
    assert not p.duplicated(KEYS+['split','name']).any()
    assert (out/'frozen_selection.json').stat().st_mtime<=(out/'predictions.parquet').stat().st_mtime
    print('Verified prior-only split-adjusted baselines on 93 rows, source hashes, all saved test regression scores/MAEs, split dates and base-rate accounting.')

if __name__=='__main__':main()
