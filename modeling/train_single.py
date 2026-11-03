"""One training-only gradient boosting fit, one validation cutoff, three comparable splits."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from .train import dataset, models, KEYS
from .core import complete_mask, truth, metrics, select_cutoff, cluster_interval


def run(source, output, regularized=False, split_manifest=None):
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError('Choose a new output directory; existing results are preserved')
    f, l, cols = dataset(source)
    d = f.merge(l[l.horizon.eq('close')],on=KEYS,validate='one_to_one').sort_values(KEYS).reset_index(drop=True)
    dates = np.array(sorted(d.date.unique()))
    a,b = int(len(dates)*.60),int(len(dates)*.85)
    splits = {'train':dates[:a],'validation':dates[a+1:b],'test':dates[b+1:]}
    embargo_dates = list(dates[[a,b]])
    if split_manifest:
        original = json.loads(Path(split_manifest).read_text())
        splits = {k:np.array(original['split_dates'][k]) for k in ('train','validation','test')}
        embargo_dates = original['embargo_dates']
        assert set(dates) <= set(np.concatenate(list(splits.values()))) | set(embargo_dates)
    assert max(splits['train']) < min(splits['validation']) < max(splits['validation']) < min(splits['test'])
    parts = {k:d[d.date.isin(v)].copy() for k,v in splits.items()}
    train = parts['train'].loc[complete_mask(parts['train'])]
    comparisons = []
    if regularized:
        # Small fixed search, selected entirely before test scoring. All fits use training only.
        validation = parts['validation'].loc[complete_mask(parts['validation'])]
        candidates = []
        for leaves, leaf_size, penalty in [(2,150,50), (3,100,30), (4,80,20)]:
            for iterations in [40,80,120]:
                candidate = HistGradientBoostingClassifier(max_iter=iterations,max_leaf_nodes=leaves,
                    max_depth=2,min_samples_leaf=leaf_size,learning_rate=.03,
                    l2_regularization=penalty,early_stopping=False,random_state=731)
                candidate.fit(train[cols],truth(train,5).astype(int))
                ts = candidate.predict_proba(train[cols])[:,1]
                vs = candidate.predict_proba(validation[cols])[:,1]
                ap = average_precision_score(truth(validation,5),vs)
                comparisons.append(dict(leaves=leaves,min_samples_leaf=leaf_size,l2=penalty,
                    iterations=iterations,validation_ap=ap,
                    train_auc=roc_auc_score(truth(train,5),ts),
                    validation_auc=roc_auc_score(truth(validation,5),vs)))
                candidates.append(candidate)
        chosen = max(range(len(comparisons)),key=lambda i:comparisons[i]['validation_ap'])
        model = candidates[chosen]
    else:
        model = models()['hist_gradient_boosting']
        model.fit(train[cols],truth(train,5).astype(int))
    validation_scores = model.predict_proba(parts['validation'][cols])[:,1]
    policy = select_cutoff(parts['validation'],validation_scores,5)
    output.mkdir(parents=True)
    if comparisons:
        pd.DataFrame(comparisons).to_csv(output/'regularization_comparison.csv',index=False)
    manifest = {'source':str(source.resolve()),'source_sha256':{name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in ('features.parquet','labels.parquet')},'target':'+5% from 09:45 minute open, reached before session close','model':'hist_gradient_boosting','parameters':model.get_params(),'features':cols,'split_dates':{k:list(v) for k,v in splits.items()},'embargo_dates':embargo_dates,'training_rows_used':len(train),'policy':policy,'refit':False,'disclosure':'Calendar test dates were examined in previous experiments. This is not a fresh independent holdout. Training diagnostics are in sample; validation metrics are used for cutoff selection.','training_rule':'Complete outcome windows for both classes; score every row in each split; unknowns remain unknown.'}
    manifest['split_manifest_source'] = str(split_manifest) if split_manifest else None
    if (source/'manifest.json').exists():
        manifest['source_cohort_manifest'] = json.loads((source/'manifest.json').read_text())
    manifest['model_selection'] = 'Nine fixed regularized candidates; highest complete-window validation average precision. Same validation period also selects cutoff; validation is optimistic after tuning.' if regularized else 'Fixed original model settings'
    # Freeze the single fit and validation-selected policy before test scoring.
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    cohort_rule = 'Complete opening grid and >=$1m opening turnover'
    if manifest.get('source_cohort_manifest',{}).get('rule'):
        cohort_rule += '; ' + manifest['source_cohort_manifest']['rule']
    joblib.dump({'estimator':model,'features':cols,'definition':'hclose_gain5','horizon':'close','gain_threshold':5,'policy':policy,'cohort_rule':cohort_rule},output/'model.joblib',compress=3)
    rows=[];predictions=[]
    for split,p in parts.items():
        scores = model.predict_proba(p[cols])[:,1]
        called = scores >= policy['cutoff']
        m = metrics(p,scores,policy['cutoff'],5)
        complete = complete_mask(p).to_numpy();y=truth(p,5)[complete];pred=called[complete]
        row={'split':split,'first_date':p.date.min(),'last_date':p.date.max(),'sessions':p.date.nunique(),'stock_days':len(p),'cutoff':policy['cutoff'],**m,'unknown_total':int(p.runner_5pct.isna().sum()),'accuracy_complete':accuracy_score(y,pred),'precision_complete':precision_score(y,pred,zero_division=0),'recall_complete':recall_score(y,pred,zero_division=0),'f1_complete':f1_score(y,pred,zero_division=0),'roc_auc_complete':roc_auc_score(y,scores[complete]),'average_precision_complete':average_precision_score(y,scores[complete]),'baseline_complete':float(y.mean())}
        row['precision_day_ci_low'],row['precision_day_ci_high']=cluster_interval(p,called,5)
        rows.append(row)
        q=p[KEYS+['runner_5pct','missing_minutes','entry_available']].copy();q['split']=split;q['score']=scores;q['predicted_runner']=called;q['complete_outcome']=complete;predictions.append(q)
    result=pd.DataFrame(rows);result.to_csv(output/'metrics.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_parquet(output/'predictions.parquet',index=False,compression='zstd')
    reloaded=joblib.load(output/'model.joblib')
    assert np.allclose(reloaded['estimator'].predict_proba(parts['test'][cols])[:,1],predictions[-1].score)
    assert sum(len(p) for p in parts.values())+int(d.date.isin(embargo_dates).sum())==len(d)
    for name,digest in manifest['source_sha256'].items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest
    report=['# Single-model +5% before-close experiment','',manifest['disclosure'],'','One chronological 60% training / 25% validation / 15% test split by session, with one embargo session at each boundary. Same histogram gradient boosting settings as the earlier experiment. Fit once on complete training windows; no refit. One validation-selected cutoff is used unchanged on every split. The validation policy targets 80% conservative precision with support; otherwise it reports a research fallback.','',f"Training fit used {len(train):,} complete stock-days. Cutoff: {policy['cutoff']:.6f}. Validation target met: {policy['qualified']}.",'','| Split | Dates | Rows | Signals | Hits | Failures | Unknown signals | Conservative precision | Complete recall | Complete AP |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:report.append(f"| {r['split']} | {r['first_date']} to {r['last_date']} | {r['stock_days']} | {r['calls']} | {r['hit']} | {r['confirmed_failure']} | {r['unknown']} | {r['precision_conservative']:.2%} | {r['recall_complete']:.2%} | {r['average_precision_complete']:.2%} |")
    report+=['','Conservative precision includes every called observation; unknowns count against it. Complete metrics use only complete future windows and describe a different, more observable subset. Training is scored on all training-period rows, including incomplete windows excluded from fitting. This avoids presenting training precision on an easier complete-only denominator while evaluation includes unknowns. Training remains in sample for fitted rows. Validation is not independent after cutoff tuning. Scores are not calibrated probabilities.','', 'Minute-high hits do not measure executable profit. No spread, slippage, stop-loss, or feed latency is modeled. Opening selection requires data through 09:45. Market-cap, exchange, and exact premarket completeness remain unverified.','', 'Verification passed: chronological boundaries, row accounting, source hashes, and reloaded-model test scores. Full metrics are in metrics.csv; predictions and the single fitted model are alongside.']
    if regularized:
        report = [line.replace('Same histogram gradient boosting settings as the earlier experiment.',
            'Stronger regularization: at most four leaves and depth two, larger leaf populations, L2 penalties, lower learning rate, and validation-selected iteration count from nine fixed candidates. Highest complete-window validation AP selects the model; the same validation period selects the cutoff, so validation results are optimistic after tuning.') for line in report]
        report += ['', 'Selected model parameters: `' + json.dumps(model.get_params()) + '`. Candidate comparisons: regularization_comparison.csv.']
    report += ['', 'Source cohort: ' + cohort_rule + '.']
    if split_manifest:
        report += ['', 'Calendar split dates were reused from ' + str(split_manifest) + '; percentages describe the original calendar allocation, not recomputed filtered-date proportions.']
    (output/'REPORT.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(policy,indent=2));print(result.to_string(index=False));print(output/'REPORT.md')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',default='data/master_open15_1m');p.add_argument('--output',default='modeling/results_single_5pct_close')
    p.add_argument('--regularized',action='store_true')
    p.add_argument('--split-manifest',help='Reuse frozen calendar split dates when filtering a source cohort')
    args=p.parse_args()
    with threadpool_limits(limits=2):run(args.source,args.output,args.regularized,args.split_manifest)
