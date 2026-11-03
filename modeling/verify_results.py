"""Check model-study artifacts against frozen policies and held-out prediction rows."""
import hashlib
import joblib
from threadpoolctl import threadpool_limits
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .core import truth

def verify(out=Path('modeling/results'),source=Path('data/master')):
    out=Path(out);source=Path(source)
    manifest=json.loads((out/'manifest.json').read_text());frozen=json.loads((out/'frozen_selection.json').read_text())
    for name,digest in manifest['source_sha256'].items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest,'Source dataset changed'
    forbidden=('runner_','entry_price','max_gain','max_decline','end_return','minutes_to','horizon','identity','ticker','date')
    assert not any(any(c.startswith(x) for x in forbidden) for c in manifest['features'])
    assert len(manifest['features'])==len(set(manifest['features']))
    pred=pd.read_parquet(out/'test_predictions.parquet');r=pd.read_csv(out/'test_results.csv')
    assert pred.date.isin(manifest['split_dates']['test']).all()
    assert not pred.duplicated(['definition','date','ticker','identity']).any()
    assert set(r.definition)==set(d['definition'] for d in frozen['definitions'])
    for _,row in r.iterrows():
        q=pred[pred.definition.eq(row.definition)];called=q[q.predicted_runner]
        assert len(q)==row.test_rows and len(called)==row.calls
        hits=int(called.label.fillna(False).sum());unk=int(called.label.isna().sum())
        assert hits==row.hit and unk==row.unknown
        if len(called):assert np.isclose(hits/len(called),row.precision_conservative)
        assert (q.predicted_runner==(q.score>=row.cutoff)).all()
        definition=next(d for d in frozen['definitions'] if d['definition']==row.definition)
        assert definition['policy']['cutoff']==row.cutoff or np.isclose(definition['policy']['cutoff'],row.cutoff)
        if definition['policy']['qualified']:
            m=definition['threshold_metrics']
            assert m['precision_conservative']>=.8 and m['calls']>=25 and m['signal_days']>=10 and m['precision_wilson_low']>=.65
    assert (out/'frozen_selection.json').stat().st_mtime<=(out/'test_results.csv').stat().st_mtime
    # Reload saved artifacts and reproduce a held-out score sample.
    features=pd.read_parquet(source/'features.parquet')
    for name in frozen['preselected_priority_definitions']:
        q=pred[pred.definition.eq(name)].head(8)
        x=q[['date','ticker','identity']].merge(features,on=['date','ticker','identity'],validate='one_to_one')
        artifact=joblib.load(out/'models'/f'{name}.joblib')
        with threadpool_limits(limits=2):
            score=artifact['estimator'].predict_proba(x[artifact['features']].replace([np.inf,-np.inf],np.nan))[:,1]
        assert np.allclose(score,q.score), 'Saved model score mismatch'
    print(f'Verified {len(r)} frozen definitions, {len(pred):,} held-out predictions, metric totals, activity-policy gates, and unchanged source hashes.')

if __name__=='__main__':verify()
