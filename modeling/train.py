"""Chronological model comparison, threshold tuning, then a frozen holdout test."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from .core import HORIZONS,THRESHOLDS,SEED,make_splits,complete_mask,truth,metrics,select_cutoff,cluster_interval,signed_log

KEYS=['date','ticker','identity']
EXCLUDE=KEYS+['feature_cutoff_utc','opening_available','opening_minutes_observed','opening_missing_minutes']

def models():
    return {
      'logistic':make_pipeline(FunctionTransformer(signed_log),SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),LogisticRegression(C=.5,max_iter=2000,random_state=SEED)),
      'random_forest':make_pipeline(SimpleImputer(strategy='median',add_indicator=True),RandomForestClassifier(n_estimators=220,min_samples_leaf=12,max_depth=12,max_features=.8,n_jobs=2,random_state=SEED)),
      'extra_trees':make_pipeline(SimpleImputer(strategy='median',add_indicator=True),ExtraTreesClassifier(n_estimators=220,min_samples_leaf=8,max_depth=14,max_features=.8,n_jobs=2,random_state=SEED)),
      'hist_gradient_boosting':HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=15,min_samples_leaf=30,learning_rate=.05,l2_regularization=5,early_stopping=False,random_state=SEED)}

def dump(obj,path):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)));tmp.replace(path)

def dataset(source):
    f=pd.read_parquet(source/'features.parquet');l=pd.read_parquet(source/'labels.parquet')
    assert not f.duplicated(KEYS).any() and not l.duplicated(KEYS+['horizon']).any()
    assert not any(c.startswith('runner_') for c in f.columns)
    # Fixed pre-cutoff activity screen; do not filter by future outcome or entry price.
    f=f[f.opening_missing_minutes.eq(0)].copy()
    cols=[c for c in f if c not in EXCLUDE]
    assert all(pd.api.types.is_numeric_dtype(f[c]) for c in cols)
    f[cols]=f[cols].replace([np.inf,-np.inf],np.nan)
    return f,l,cols

def run(args):
    source=Path(args.source);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    # Refuse silently reusing an already evaluated holdout for a new experiment.
    if (out/'test_results.csv').exists():raise RuntimeError('Completed results exist; use a new output directory and document any further holdout reuse.')
    f,l,cols=dataset(source)
    dates=pd.read_parquet(source/'features.parquet',columns=['date']).date
    splits=make_splits(dates)
    manifest={'seed':SEED,'python':platform.python_version(),'sklearn':sklearn.__version__,
              'features':cols,'source_sha256':{name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in ('features.parquet','labels.parquet')},
              'split_dates':{k:list(v) for k,v in splits.items()},
              'split_summary':{k:{'first':str(v[0]),'last':str(v[-1]),'sessions':len(v),'cohort_rows':int(f.date.isin(v).sum())} for k,v in splits.items()},
              'cohort_rule':'All 15 opening minute bars observed before 09:45; no future quality filter for scoring',
              'training_rule':'Only complete outcome windows for both positives and negatives; unknowns are not negatives in training',
              'selection_rule':'Highest Wilson lower bound on conservative precision among highest-scored 10% of validation cohort; then average precision on complete windows',
              'threshold_rule':'Separate later threshold set: >=80% conservative precision, >=25 signals over >=10 dates, Wilson lower >=65%; maximize call count. If unmet, show best-supported fallback as research-only.',
              'test_rule':'Models and thresholds fixed before any test scoring; no refit after calibration; same ticker may recur over time; one session embargo at every boundary',
              'minimum_training_positives':30,'minimum_validation_positives':5,'minimum_threshold_positives':5}
    dump(manifest,out/'manifest.json')
    out.joinpath('models').mkdir(exist_ok=True);out.joinpath('checkpoints').mkdir(exist_ok=True)
    validation_rows=[];val_predictions=[];definitions=[];skipped=[]
    started=time.monotonic()
    for h in HORIZONS:
        d=f.merge(l[l.horizon.eq(h)],on=KEYS,validate='one_to_one').sort_values(KEYS).reset_index(drop=True)
        for threshold in THRESHOLDS:
            name=f'h{h}_gain{threshold}';checkpoint=out/'checkpoints'/f'{name}.json'
            if checkpoint.exists():
                cached=json.loads(checkpoint.read_text());definitions.append(cached['definition']);validation_rows.extend(cached['comparisons'])
                print(name,'resumed',flush=True);continue
            train=d[d.date.isin(splits['train'])&complete_mask(d)].copy()
            valid=d[d.date.isin(splits['validation'])].copy();cal=d[d.date.isin(splits['threshold'])].copy()
            y=truth(train,threshold).astype(int)
            counts={'train_positive':int(y.sum()),'train_negative':int((1-y).sum()),'validation_observed_positive':int(truth(valid,threshold).sum()),'threshold_observed_positive':int(truth(cal,threshold).sum())}
            if min(counts['train_positive'],counts['train_negative'])<30 or counts['validation_observed_positive']<5 or counts['threshold_observed_positive']<5:
                skipped.append({'definition':name,'horizon':h,'gain_threshold':threshold,**counts,'reason':'too few positive/negative examples for a supported precision estimate'})
                print(name,'SKIP',counts,flush=True);continue
            comparisons=[]
            for family,model in models().items():
                t0=time.monotonic();model.fit(train[cols],y)
                score=model.predict_proba(valid[cols])[:,1]
                k=max(25,int(np.ceil(len(valid)*.1)));cut=float(np.sort(score)[-min(k,len(score))])
                m=metrics(valid,score,cut,threshold)
                row={'definition':name,'horizon':h,'gain_threshold':threshold,'model':family,'seconds':time.monotonic()-t0,**m}
                comparisons.append(row);validation_rows.append(row)
                frame=valid[KEYS].copy();frame['definition']=name;frame['model']=family;frame['score']=score;frame['observed_hit']=truth(valid,threshold);frame['complete']=complete_mask(valid).to_numpy();val_predictions.append(frame)
            winner=max(comparisons,key=lambda r:(r['precision_wilson_low'],r['average_precision_complete']))
            family=winner['model'];model=models()[family]
            refit=d[d.date.isin(np.r_[splits['train'],splits['validation']])&complete_mask(d)]
            model.fit(refit[cols],truth(refit,threshold).astype(int))
            cal_score=model.predict_proba(cal[cols])[:,1];policy=select_cutoff(cal,cal_score,threshold)
            cm=metrics(cal,cal_score,policy['cutoff'],threshold)
            artifact={'estimator':model,'features':cols,'definition':name,'horizon':h,'gain_threshold':threshold,'policy':policy,'manifest':'../manifest.json','cohort_rule':manifest['cohort_rule']}
            joblib.dump(artifact,out/'models'/f'{name}.joblib',compress=3)
            definition={'definition':name,'horizon':h,'gain_threshold':threshold,'model':family,'train_rows':len(train),'refit_rows':len(refit),**counts,'policy':policy,'threshold_metrics':cm}
            definitions.append(definition);dump({'definition':definition,'comparisons':comparisons},checkpoint)
            pd.DataFrame(validation_rows).to_csv(out/'validation_model_comparison.csv',index=False)
            print(name,family,'threshold precision',round(cm['precision_conservative'],3),'calls',cm['calls'],'qualified',policy['qualified'],f'elapsed {(time.monotonic()-started)/60:.1f}m',flush=True)
    pd.DataFrame(validation_rows).to_csv(out/'validation_model_comparison.csv',index=False)
    if val_predictions:pd.concat(val_predictions,ignore_index=True).to_parquet(out/'validation_predictions.parquet',compression='zstd',index=False)
    dump(skipped,out/'skipped_definitions.json')
    # Global recommendation uses validation/threshold data only. Frozen before holdout scoring.
    priorities=sorted(definitions,key=lambda r:(r['policy']['qualified'],r['threshold_metrics']['precision_wilson_low'],r['threshold_metrics']['calls']),reverse=True)
    prereg={'definitions':definitions,'preselected_priority_definitions':[r['definition'] for r in priorities[:5]],'skipped':skipped,'test_accessed':False}
    dump(prereg,out/'frozen_selection.json')
    print('MODEL/THRESHOLD SELECTION FROZEN. Beginning held-out evaluation.',flush=True)
    results=[];predictions=[]
    for definition in definitions:
        h=definition['horizon'];threshold=definition['gain_threshold'];name=definition['definition']
        d=f.merge(l[l.horizon.eq(h)],on=KEYS,validate='one_to_one');test=d[d.date.isin(splits['test'])].copy().reset_index(drop=True)
        artifact=joblib.load(out/'models'/f'{name}.joblib');score=artifact['estimator'].predict_proba(test[cols])[:,1]
        cutoff=definition['policy']['cutoff'];selected=score>=cutoff
        m=metrics(test,score,cutoff,threshold);ci=cluster_interval(test,selected,threshold)
        row={'definition':name,'horizon':h,'gain_threshold':threshold,'model':definition['model'],'threshold_qualified':definition['policy']['qualified'],
             'cutoff':cutoff,'test_rows':len(test),'bootstrap_day_ci_low':ci[0],'bootstrap_day_ci_high':ci[1],
             'threshold_precision':definition['threshold_metrics']['precision_conservative'],'threshold_calls':definition['threshold_metrics']['calls'],**m}
        results.append(row)
        frame=test[KEYS+['missing_minutes','entry_available',f'runner_{threshold}pct']].copy().rename(columns={f'runner_{threshold}pct':'label'})
        frame['definition']=name;frame['score']=score;frame['predicted_runner']=selected;frame['qualified_policy']=definition['policy']['qualified'];frame['complete_outcome']=complete_mask(test).to_numpy()
        # No costs or trade ordering assumptions: no profit estimate from high prices.
        predictions.append(frame)
    pd.DataFrame(results).to_csv(out/'test_results.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_parquet(out/'test_predictions.parquet',compression='zstd',index=False)
    print('Held-out evaluation complete:',len(results),'definitions',flush=True)
    from .report import build_report
    build_report(out)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',default='data/master');p.add_argument('--output',default='modeling/results')
    with threadpool_limits(limits=2):run(p.parse_args())
