"""Frozen-validation ablation study for an up/down/neither first-touch target."""
import hashlib,json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer,StandardScaler
from sklearn.metrics import log_loss,accuracy_score,balanced_accuracy_score,f1_score,average_precision_score,roc_auc_score,confusion_matrix
from threadpoolctl import threadpool_limits
from modeling.core import signed_log,select_cutoff,wilson

OUT=Path('modeling/barrier_study/results')
DATA=Path('modeling/barrier_study/data')
KEYS=['date','ticker','identity']
CLASSES=['down','neither','up']


def families():
    result={'prior':DummyClassifier(strategy='prior'),
        'logistic':make_pipeline(FunctionTransformer(signed_log),SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),StandardScaler(),LogisticRegression(C=.1,max_iter=2000,random_state=731))}
    for i,(leaves,size,l2) in enumerate(((2,150,50),(3,100,30),(4,80,20))):
        result[f'hgb_{i}']=HistGradientBoostingClassifier(max_leaf_nodes=leaves,min_samples_leaf=size,l2_regularization=l2,max_iter=80,max_depth=2,learning_rate=.03,early_stopping=False,random_state=731)
    return result


def probability(model,x):
    raw=model.predict_proba(x);out=np.zeros((len(x),len(CLASSES)))
    for j,klass in enumerate(model.classes_):out[:,CLASSES.index(klass)]=raw[:,j]
    return out


def evaluate(g,p,cutoff):
    known=g.label_outcome.notna().to_numpy();y=g.loc[known,'label_outcome'].to_numpy();q=p[known]
    pred=np.array(CLASSES)[q.argmax(axis=1)];up=y=='up';score=p[:,2]
    called=score>=cutoff;ck=called&known;hits=int((ck&(g.label_outcome.eq('up').fillna(False).to_numpy(dtype=bool))).sum());unknown=int((called&~known).sum())
    calls=int(called.sum());lo,hi=wilson(hits,calls)
    return dict(rows=len(g),known_rows=int(known.sum()),unknown_rows=int((~known).sum()),log_loss_known=log_loss(y,q,labels=CLASSES),
        accuracy_known=accuracy_score(y,pred),balanced_accuracy_known=balanced_accuracy_score(y,pred),macro_f1_known=f1_score(y,pred,labels=CLASSES,average='macro',zero_division=0),
        up_base_rate_known=float(up.mean()),up_ap_known=average_precision_score(up,q[:,2]) if up.any() else np.nan,
        up_auc_known=roc_auc_score(up,q[:,2]) if up.any() and not up.all() else np.nan,
        up_brier_known=float(np.mean((q[:,2]-up)**2)),calls=calls,up_hits=hits,known_non_up=int(ck.sum())-hits,unknown_calls=unknown,
        precision_conservative=hits/calls if calls else np.nan,precision_upper=(hits+unknown)/calls if calls else np.nan,
        precision_wilson_low=lo,precision_wilson_high=hi,up_recall_known=hits/int(up.sum()) if up.any() else np.nan)


def pnl(g,score,cutoff,calendar_dates):
    called=g[score>=cutoff].copy();known=called.label_outcome.notna();r=called.loc[known,'label_gross_return_pct'].to_numpy();net=r-.5
    wins=r[r>1e-10];losses=r[r< -1e-10]
    day=called.loc[known].groupby('date').label_gross_return_pct.agg(['sum','size']).reindex(calendar_dates,fill_value=0).to_numpy()
    rng=np.random.default_rng(731);total=day[rng.integers(0,len(day),size=(3000,len(day)))].sum(axis=1)
    valid=total[:,1]>0;ci=np.quantile(total[valid,0]/total[valid,1],[.025,.975]) if valid.any() else [np.nan,np.nan]
    return dict(signals=len(called),resolved=len(r),unresolved=int((~known).sum()),wins=len(wins),losses=len(losses),breakeven=int((np.abs(r)<=1e-10).sum()),
        mean_gross_pct=float(r.mean()) if len(r) else np.nan,median_gross_pct=float(np.median(r)) if len(r) else np.nan,
        mean_net_50bps_pct=float(net.mean()) if len(r) else np.nan,mean_net_100bps_pct=float((r-1).mean()) if len(r) else np.nan,
        average_win_pct=float(wins.mean()) if len(wins) else np.nan,average_loss_pct=float(losses.mean()) if len(losses) else np.nan,
        best_pct=float(r.max()) if len(r) else np.nan,worst_pct=float(r.min()) if len(r) else np.nan,
        mean_gross_day_ci_low=ci[0],mean_gross_day_ci_high=ci[1])


def main():
    if (OUT/'metrics.csv').exists():raise ValueError('Completed results exist; use a fresh experiment path')
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'models').mkdir(exist_ok=True)
    d=pd.read_parquet(DATA/'dataset.parquet')
    original=json.loads(Path('modeling/volatility_scaled/results/manifest.json').read_text())
    base=original['features']['base'];relative=original['features']['enriched']
    path=[c for c in d if c.startswith('path_')];market=[c for c in d if c.startswith('market_')]
    sets={'base':base,'opening_path':base+path,'path_market':base+path+market,'full':relative+path+market}
    for features in sets.values():
        assert len(features)==len(set(features))
        assert not any(c.startswith('label_') for c in features)
        assert all(pd.api.types.is_numeric_dtype(d[c]) for c in features)
    groups={s:d[d.date.isin(dates)].copy() for s,dates in original['split_dates'].items()}
    known={s:g[g.label_outcome.notna()].copy() for s,g in groups.items()}
    train=known['train'];valid=known['validation'];comparisons=[];frozen=[]
    manifest=dict(source_manifest=json.loads((DATA/'manifest.json').read_text()),dataset_sha256=hashlib.sha256((DATA/'dataset.parquet').read_bytes()).hexdigest(),
        split_dates=original['split_dates'],embargo_dates=original['embargo_dates'],features=sets,
        selection='Each feature set chooses minimum known-label validation multiclass log loss among prior baseline, regularized logistic, and three fixed shallow HGB settings. No balancing or refit. Separate policy cutoff selection uses the same validation period and original 80% conservative precision/support rule; unqualified fallbacks are exploratory.',
        warning='Previously viewed calendar test dates are reused. This is a follow-up experiment, not independent confirmation. Validation is also tuned, and is not an independent evaluation.',classes=CLASSES)
    for feature_set,features in sets.items():
        local=[]
        for family,model in families().items():
            model.fit(train[features],train.label_outcome)
            p=probability(model,valid[features]);loss=log_loss(valid.label_outcome,p,labels=CLASSES)
            row=dict(feature_set=feature_set,model=family,validation_log_loss=loss,
                validation_up_ap=average_precision_score(valid.label_outcome.eq('up'),p[:,2]))
            comparisons.append(row);local.append((row,model))
        row,model=min(local,key=lambda x:x[0]['validation_log_loss'])
        val=groups['validation'];val_p=probability(model,val[features]);tuning=val[KEYS].copy()
        tuning['runner_5pct']=pd.Series([None if pd.isna(o) else o=='up' for o in val.label_outcome],index=val.index,dtype=object)
        policy=select_cutoff(tuning,val_p[:,2],5)
        artifact=dict(estimator=model,features=features,classes=CLASSES,policy=policy,selection=row)
        joblib.dump(artifact,OUT/'models'/f'{feature_set}.joblib',compress=3)
        frozen.append(dict(feature_set=feature_set,model=row['model'],validation_log_loss=row['validation_log_loss'],policy=policy))
    # Freeze the global primary choice by validation only; do not choose from test results.
    primary=min(frozen,key=lambda x:x['validation_log_loss'])['feature_set']
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (OUT/'frozen_selection.json').write_text(json.dumps(dict(primary_feature_set=primary,models=frozen),indent=2)+'\n')
    pd.DataFrame(comparisons).to_csv(OUT/'validation_comparison.csv',index=False)
    print('Frozen primary:',primary,'; beginning reused-test evaluation.',flush=True)
    rows=[];pnl_rows=[];predictions=[];counts=[];confusions=[]
    # Always retain a separate prior-only reference, even if not selected in any feature set.
    reference=DummyClassifier(strategy='prior');reference.fit(train[base],train.label_outcome)
    joblib.dump(dict(estimator=reference,features=base,classes=CLASSES),OUT/'models'/'prior_reference.joblib')
    for split,g in groups.items():
        counts.append(dict(split=split,eligible=len(g),known=int(g.label_outcome.notna().sum()),unknown=int(g.label_outcome.isna().sum()),**{c:int(g.label_outcome.eq(c).sum()) for c in CLASSES},
            ambiguous=int(g.label_reason.eq('ambiguous_same_minute').sum()),missing=int(g.label_reason.str.startswith('missing',na=False).sum())))
        for selected in frozen:
            name=selected['feature_set'];artifact=joblib.load(OUT/'models'/f'{name}.joblib')
            p=probability(artifact['estimator'],g[artifact['features']]);cut=artifact['policy']['cutoff']
            rows.append(dict(split=split,feature_set=name,model=selected['model'],cutoff=cut,qualified=artifact['policy']['qualified'],**evaluate(g,p,cut)))
            pnl_rows.append(dict(split=split,feature_set=name,**pnl(g,p[:,2],cut,manifest['split_dates'][split])))
            frame=g[KEYS+[c for c in g if c.startswith('label_')]].copy();frame['split']=split;frame['feature_set']=name
            frame['prob_down']=p[:,0];frame['prob_neither']=p[:,1];frame['prob_up']=p[:,2];frame['called']=p[:,2]>=cut;predictions.append(frame)
            known_mask=g.label_outcome.notna();cm=confusion_matrix(g.loc[known_mask,'label_outcome'],np.array(CLASSES)[p[known_mask].argmax(axis=1)],labels=CLASSES)
            for i,true in enumerate(CLASSES):
                for j,predicted in enumerate(CLASSES):confusions.append(dict(split=split,feature_set=name,actual=true,predicted=predicted,count=int(cm[i,j])))
        ref=probability(reference,g[base]);rows.append(dict(split=split,feature_set='prior_reference',model='prior',cutoff=1.000001,qualified=False,**evaluate(g,ref,1.000001)))
    result=pd.DataFrame(rows);returns=pd.DataFrame(pnl_rows);rates=pd.DataFrame(counts)
    result.to_csv(OUT/'metrics.csv',index=False);returns.to_csv(OUT/'strategy_metrics.csv',index=False);rates.to_csv(OUT/'base_rates.csv',index=False)
    pd.DataFrame(confusions).to_csv(OUT/'confusion_matrices.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_parquet(OUT/'predictions.parquet',index=False,compression='zstd')
    assert hashlib.sha256((DATA/'dataset.parquet').read_bytes()).hexdigest()==manifest['dataset_sha256']
    assert (OUT/'frozen_selection.json').stat().st_mtime<=(OUT/'predictions.parquet').stat().st_mtime
    report(manifest,frozen,primary,result,returns,rates)
    print('BASE RATES\n',rates.to_string(index=False),flush=True)
    print('TEST METRICS\n',result[result.split.eq('test')].to_string(index=False),flush=True)
    print('TEST RETURNS\n',returns[returns.split.eq('test')].to_string(index=False),flush=True)


def report(manifest,frozen,primary,result,returns,rates):
    text=['# First-touch +/-1 sigma barrier study','',manifest['warning'],'',
        '## Experiment','',
        'Entry at historical 09:45 minute open. First upper/lower log-price barrier entry*exp(+/-sigma20) in [09:45,10:45) gives up/down. Neither means a complete 60-minute path touched neither barrier, exiting at 10:44 close. A missing minute before the deciding event or an unresolved both-barriers minute leaves the outcome unknown. A bar opening beyond one barrier establishes ordering at the open. Stops gapping lower fill at that worse open; take-profits are capped at target.', '',
        'Sigma20 is the existing prior-20-session 09:45-to-close log-return standard deviation, not a new 60-minute sigma. It is known before entry; one unit is not a probability guarantee. This fixes scale across experiments. The source retains complete opening candles, >=$1m opening turnover and complete prior regular-session candles. Every feature-eligible row is scored. Unknown labels never become training negatives.', '',
        '## Inputs and selection','',
        'Base = original features plus shared prior volatility scale/return history. Opening-path additions: three five-minute price/turnover windows, acceleration, pullback/bounce, close-based drawdown, upper-wick fractions, running-VWAP crossings, and premarket-high break/hold/failure measures. Market additions: contemporaneous SPY/QQQ first/middle/last-five and opening-15 returns plus stock excess returns. Full also adds the prior-window relative activity features from the earlier study. Features stop strictly before 09:45. Historical quote probe returned 403, so no spread or order-flow features are represented.', '',
        manifest['selection'],'',f'Primary feature set, frozen using validation only: **{primary}**.','',
        '## Outcome counts','', '```',rates.to_string(index=False),'```','',
        '## Test performance','', '```',result[result.split.eq('test')].to_string(index=False),'```','',
        'Log loss and Brier: lower is better. AP must be compared with the same cohort\'s up-first base rate. Macro F1 and balanced accuracy expose class imbalance; ordinary accuracy alone can reward majority prediction. Probabilities are uncalibrated estimates. Precision includes unknown selected outcomes in its denominator. Policy settings failing the 80% validation gate remain research fallbacks.', '',
        '## Test paper exit simulation','', '```',returns[returns.split.eq('test')].to_string(index=False),'```','',
        'The preselected cutoff flags long entries. Exit at up barrier, down barrier, or 60-minute timeout. Returns equally weight resolved trades, not portfolio capital. Unknown outcomes remain disclosed and excluded from return averages; the apparent mean can be biased. Cost scenarios subtract 0.5 or 1 percentage point per position. Bracket fills, spreads, latency, halts and market impact are not reconstructed. Historical 09:45 entry differs from the live delayed-feed paper runner.', '',
        '## Limits','',
        '- Same-minute ties remain unknown rather than assuming favorable ordering. Complete labels require only complete information through the deciding event; missing bars after an observed exit do not affect that exit.',
        '- Requeried minute bars and market features can contain vendor revisions. The study is event-time research, not an as-received replay.',
        '- Candidate models/settings and operating cutoffs use the same validation period. Test periods have already been inspected in earlier research. Neither is independent confirmation after repeated experimentation.',
        '- No feature set, target threshold or model is selected on test outcomes. The fixed time horizon and barriers are not optimized here.',
        '- No oversampling or class weighting changes the observed class frequencies. A constant training-prior reference is always reported.',
        '- Existing paper strategy files and model remain unchanged.','',
        'Artifacts: validation_comparison.csv, frozen_selection.json, metrics.csv, strategy_metrics.csv, base_rates.csv, confusion_matrices.csv, predictions.parquet and models/.']
    (OUT/'REPORT.md').write_text('\n'.join(text)+'\n')

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
