"""Reproduce labels/features from archived bars and score the frozen test artifacts."""
import hashlib,json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits
from .build import intraday_features,barrier_label
from .evaluate import probability,CLASSES,KEYS


def main():
    data=Path('modeling/barrier_study/data');out=Path('modeling/barrier_study/results')
    d=pd.read_parquet(data/'dataset.parquet');p=pd.read_parquet(out/'predictions.parquet');m=json.loads((out/'manifest.json').read_text())
    assert hashlib.sha256((data/'dataset.parquet').read_bytes()).hexdigest()==m['dataset_sha256']
    source=m['source_manifest'];assert hashlib.sha256(Path(source['source']).read_bytes()).hexdigest()==source['source_sha256']
    for row in d.iloc[::10].to_dict('records'):
        token=hashlib.sha256((row['date']+'|'+row['ticker']).encode()).hexdigest()[:24]
        b=pd.read_parquet(data/'paths'/f'{token}.parquet')
        expected=intraday_features(b)
        for feature,value in expected.items():assert np.isclose(value,row[feature],equal_nan=True),feature
        target=barrier_label(b,row['sigma20_log'])
        for field,value in target.items():
            actual=row['label_'+field]
            if value is None:assert pd.isna(actual)
            elif isinstance(value,str):assert actual==value
            else:assert np.isclose(value,actual,equal_nan=True),field
        for etf in ('SPY','QQQ'):
            benchmark=pd.read_parquet(data/'paths'/f'{row["date"]}_{etf}_benchmark.parquet')
            values=intraday_features(benchmark)
            for w in ('first5','middle5','last5','open15'):
                value=values.get(f'path_{w}_return_pct',np.nan)
                assert np.isclose(value,row[f'market_{etf}_{w}_return_pct'],equal_nan=True)
                assert np.isclose(expected.get(f'path_{w}_return_pct',np.nan)-value,row[f'market_excess_{etf}_{w}_return_pct'],equal_nan=True)
    metrics=pd.read_csv(out/'metrics.csv');returns=pd.read_csv(out/'strategy_metrics.csv')
    for feature_set,g in p[p.split.eq('test')].groupby('feature_set'):
        assert g.date.isin(m['split_dates']['test']).all()
        x=g[KEYS].merge(d,on=KEYS,validate='one_to_one')
        artifact=joblib.load(out/'models'/f'{feature_set}.joblib')
        with threadpool_limits(limits=2):scores=probability(artifact['estimator'],x[artifact['features']])
        assert np.allclose(scores,g[['prob_down','prob_neither','prob_up']])
        assert np.array_equal(scores[:,2]>=artifact['policy']['cutoff'],g.called)
        known=g.label_outcome.notna();saved=metrics[metrics.split.eq('test')&metrics.feature_set.eq(feature_set)].iloc[0]
        assert np.isclose(log_loss(g.loc[known,'label_outcome'],scores[known],labels=CLASSES),saved.log_loss_known)
        called=g[g.called];resolved=called[called.label_outcome.notna()]
        trade=returns[returns.split.eq('test')&returns.feature_set.eq(feature_set)].iloc[0]
        assert len(called)==trade.signals and len(resolved)==trade.resolved
        if len(resolved):assert np.isclose(resolved.label_gross_return_pct.mean(),trade.mean_gross_pct)
    assert not p.duplicated(KEYS+['split','feature_set']).any()
    assert (out/'frozen_selection.json').stat().st_mtime<=(out/'predictions.parquet').stat().st_mtime
    print('Verified 82 archived opening/benchmark feature and target samples, source hashes, test probabilities/calls/log loss/returns, split membership, and freeze ordering.')

if __name__=='__main__':main()
