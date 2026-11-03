"""Validation-selected excursion regressors, feature ablation and +5% classifier control."""
import json
from pathlib import Path
import hashlib
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor,HistGradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer,StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,average_precision_score,roc_auc_score
from threadpoolctl import threadpool_limits
from modeling.core import signed_log,complete_mask,metrics,select_cutoff,truth

DATA=Path('modeling/volatility_scaled/data')
OUT=Path('modeling/volatility_scaled/results')
KEYS=['date','ticker','identity']
CONFIGS=[dict(max_leaf_nodes=2,min_samples_leaf=150,l2_regularization=50),dict(max_leaf_nodes=3,min_samples_leaf=100,l2_regularization=30),dict(max_leaf_nodes=4,min_samples_leaf=80,l2_regularization=20)]
COMMON=dict(max_iter=80,max_depth=2,learning_rate=.03,early_stopping=False,random_state=731)


def regressors():
    result={'median':DummyRegressor(strategy='median'),
        'ridge':make_pipeline(FunctionTransformer(signed_log),SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),Ridge(alpha=10))}
    result.update({f'hgb_{i}':HistGradientBoostingRegressor(loss='absolute_error',**COMMON,**params) for i,params in enumerate(CONFIGS)})
    return result


def constrained(pred,direction):
    return np.maximum(pred,0) if direction=='up' else np.minimum(np.maximum(pred,-99.999999),0)


def convert(pred,direction,units,sigma):
    # Percent models must respect the physical -100% floor; sigma predictions are log units.
    pred=constrained(pred,direction) if units=='percent' else (np.maximum(pred,0) if direction=='up' else np.minimum(pred,0))
    return np.log1p(pred/100)/sigma if units=='percent' else pred


def regression_metrics(y,pred,sigma,direction):
    sign=1 if direction=='up' else -1
    observed=sign*y;score=sign*pred
    result=dict(mae_sigma=mean_absolute_error(y,pred),rmse_sigma=np.sqrt(mean_squared_error(y,pred)),r2_sigma=r2_score(y,pred),
        spearman=float(spearmanr(y,pred).statistic) if np.std(pred)>1e-12 else np.nan,
        mae_percentage_points=mean_absolute_error(100*np.expm1(y*sigma),100*np.expm1(pred*sigma)))
    for barrier in (1,2):
        target=observed>=barrier
        result[f'base_rate_{barrier}sigma']=float(target.mean())
        result[f'ap_{barrier}sigma']=average_precision_score(target,score) if target.any() else np.nan
        result[f'auc_{barrier}sigma']=roc_auc_score(target,score) if target.any() and not target.all() else np.nan
    return result


def main():
    if (OUT/'metrics.csv').exists():raise ValueError('Completed results already exist; preserve the viewed evaluation')
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'models').mkdir(exist_ok=True)
    f=pd.read_parquet(DATA/'features.parquet');t=pd.read_parquet(DATA/'targets.parquet')
    # sigma is already in features, so avoid suffixed duplicate scale columns.
    d=f.merge(t.drop(columns=['sigma20_log']),on=KEYS,validate='one_to_one')
    previous=json.loads(Path('modeling/results_prior_complete_5pct_close/manifest.json').read_text())
    splits=previous['split_dates']
    base=list(json.loads(Path('modeling/results_regularized_5pct_close/manifest.json').read_text())['features'])
    # The common scale is available to both ablations; activity history is the difference.
    base+=['sigma20_log','historical_return_mean_log','historical_return_count']
    enriched=base+[c for c in f if c.startswith('hist20_') or c=='history_sessions_matched']
    assert not set(enriched)&set(['up_sigma','down_sigma','max_gain_pct','max_decline_pct','runner_5pct','missing_minutes','entry_available'])
    cols={'base':base,'enriched':enriched}
    d[enriched]=d[enriched].replace([np.inf,-np.inf],np.nan)
    cohort=d[d.sigma20_log.notna()].copy()
    groups={s:cohort[cohort.date.isin(dates)].copy() for s,dates in splits.items()}
    # Future quality affects regression target availability, never score eligibility.
    observed={s:g[g.up_sigma.notna()&g.down_sigma.notna()].copy() for s,g in groups.items()}
    manifest=dict(source_manifest=json.loads((DATA/'manifest.json').read_text()),
        source_sha256={name:hashlib.sha256((DATA/name).read_bytes()).hexdigest() for name in ('features.parquet','targets.parquet')},
        split_dates=splits,embargo_dates=previous['embargo_dates'],features=cols,
        selection='Choose model family/settings by complete-window validation MAE in sigma units separately for each feature set, target and output unit. Median baseline included. No refit. Classification control selects shallow settings by validation AP, cutoff by original supported precision policy.',
        note='Same validation period used for settings and cutoffs. Previously inspected calendar test dates are reused: exploratory, not independent confirmation. All models fit training only.',
        eligible_rows={s:len(g) for s,g in groups.items()},exact_target_rows={s:len(g) for s,g in observed.items()})
    comparisons=[];frozen=[]
    for direction in ('up','down'):
        for units in ('sigma','percent'):
            target=direction+'_sigma' if units=='sigma' else ('max_gain_pct' if direction=='up' else 'max_decline_pct')
            for feature_set,features in cols.items():
                local=[]
                for family,model in regressors().items():
                    train,valid=observed['train'],observed['validation']
                    model.fit(train[features],train[target])
                    pred=convert(model.predict(valid[features]),direction,units,valid.sigma20_log.to_numpy())
                    row=dict(direction=direction,units=units,feature_set=feature_set,model=family,**regression_metrics(valid[direction+'_sigma'].to_numpy(),pred,valid.sigma20_log.to_numpy(),direction))
                    comparisons.append(row);local.append((row,model))
                row,model=min(local,key=lambda x:x[0]['mae_sigma'])
                name=f'{direction}_{units}_{feature_set}'
                artifact=dict(estimator=model,features=features,direction=direction,units=units,selection=row)
                joblib.dump(artifact,OUT/'models'/f'{name}.joblib',compress=3)
                frozen.append(dict(name=name,task='regression',**row))
    # Quantile models are fixed independently of test results, trained on the same exact targets.
    for direction in ('up','down'):
        for q in (.1,.5,.9):
            model=HistGradientBoostingRegressor(loss='quantile',quantile=q,**COMMON,**CONFIGS[1])
            model.fit(observed['train'][enriched],observed['train'][direction+'_sigma'])
            joblib.dump(dict(estimator=model,features=enriched,direction=direction,quantile=q),OUT/'models'/f'{direction}_q{q}.joblib',compress=3)
    class_frozen=[];class_comparisons=[]
    for feature_set,features in cols.items():
        train=groups['train'].loc[complete_mask(groups['train'])]
        val=groups['validation'];complete=complete_mask(val).to_numpy();local=[]
        for i,params in enumerate(CONFIGS):
            model=HistGradientBoostingClassifier(**COMMON,**params)
            model.fit(train[features],truth(train,5).astype(int))
            score=model.predict_proba(val[features])[:,1]
            ap=average_precision_score(truth(val,5)[complete],score[complete])
            row=dict(feature_set=feature_set,config=i,validation_ap=ap);class_comparisons.append(row);local.append((row,model,score))
        row,model,score=max(local,key=lambda x:x[0]['validation_ap'])
        policy=select_cutoff(val,score,5)
        joblib.dump(dict(estimator=model,features=features,policy=policy,selection=row),OUT/'models'/f'classifier_{feature_set}.joblib',compress=3)
        class_frozen.append(dict(feature_set=feature_set,selection=row,policy=policy))
    pd.DataFrame(comparisons).to_csv(OUT/'validation_comparison.csv',index=False)
    pd.DataFrame(class_comparisons).to_csv(OUT/'classifier_validation_comparison.csv',index=False)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (OUT/'frozen_selection.json').write_text(json.dumps(dict(regressions=frozen,classifiers=class_frozen),indent=2)+'\n')
    print('Selection frozen before test scoring.',flush=True)
    metrics_rows=[];predictions=[];interval_rows=[];base_rates=[];class_results=[]
    for split,g in groups.items():
        known=g.up_sigma.notna()&g.down_sigma.notna();obs=g[known]
        base_rates.append(dict(split=split,eligible_rows=len(g),exact_rows=len(obs),unknown_excursions=int((~known).sum()),
            sigma_median_pct=100*obs.sigma20_log.median(),up_1sigma_pct=100*obs.up_sigma.ge(1).mean(),down_1sigma_pct=100*obs.down_sigma.le(-1).mean(),
            both_1sigma_pct=100*(obs.up_sigma.ge(1)&obs.down_sigma.le(-1)).mean(),neither_1sigma_pct=100*(obs.up_sigma.lt(1)&obs.down_sigma.gt(-1)).mean(),
            up_only_1sigma_pct=100*(obs.up_sigma.ge(1)&obs.down_sigma.gt(-1)).mean(),down_only_1sigma_pct=100*(obs.up_sigma.lt(1)&obs.down_sigma.le(-1)).mean(),
            close_up_pct=100*obs.end_return_pct.gt(0).mean(),close_down_pct=100*obs.end_return_pct.lt(0).mean(),
            up_5pct_pct=100*obs.max_gain_pct.ge(5-1e-8).mean(),down_5pct_pct=100*obs.max_decline_pct.le(-5+1e-8).mean()))
        for selected in frozen:
            artifact=joblib.load(OUT/'models'/f'{selected["name"]}.joblib')
            score=convert(artifact['estimator'].predict(g[artifact['features']]),artifact['direction'],artifact['units'],g.sigma20_log.to_numpy())
            actual=g[artifact['direction']+'_sigma'].to_numpy()
            result=regression_metrics(actual[known],score[known],g.loc[known,'sigma20_log'].to_numpy(),artifact['direction'])
            metrics_rows.append(dict(split=split,name=selected['name'],direction=artifact['direction'],units=artifact['units'],feature_set=selected['feature_set'],model=selected['model'],scored=len(g),evaluated=int(known.sum()),**result))
            frame=g[KEYS].copy();frame['split']=split;frame['name']=selected['name'];frame['prediction_sigma']=score;frame['prediction_pct']=100*np.expm1(score*g.sigma20_log);frame['actual_sigma']=actual;frame['sigma20_log']=g.sigma20_log;predictions.append(frame)
        for direction in ('up','down'):
            raw=np.column_stack([joblib.load(OUT/'models'/f'{direction}_q{q}.joblib')['estimator'].predict(g[enriched]) for q in (.1,.5,.9)])
            crossing=int((raw[:,0]>raw[:,2]).sum())
            raw=np.maximum(raw,0) if direction=='up' else np.minimum(raw,0)
            # Rearrangement guarantees ordered bands; disclose pre-rearrangement crossing count.
            ordered=np.sort(raw,axis=1);y=g[direction+'_sigma'].to_numpy()
            interval_rows.append(dict(split=split,direction=direction,evaluated=int(known.sum()),nominal_coverage=.8,actual_coverage=float(((y[known]>=ordered[known,0])&(y[known]<=ordered[known,2])).mean()),mean_width_sigma=float((ordered[known,2]-ordered[known,0]).mean()),raw_crossing_rows=crossing))
            frame=g[KEYS].copy();frame['split']=split;frame['direction']=direction
            for j,q in enumerate((.1,.5,.9)):frame[f'q{q}']=ordered[:,j]
            save_path=OUT/f'interval_predictions_{split}_{direction}.parquet';frame.to_parquet(save_path,index=False)
        for selection in class_frozen:
            artifact=joblib.load(OUT/'models'/f'classifier_{selection["feature_set"]}.joblib')
            score=artifact['estimator'].predict_proba(g[artifact['features']])[:,1]
            result=metrics(g,score,artifact['policy']['cutoff'],5)
            class_results.append(dict(split=split,feature_set=selection['feature_set'],qualified=artifact['policy']['qualified'],**result))
    result=pd.DataFrame(metrics_rows);result.to_csv(OUT/'metrics.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_parquet(OUT/'predictions.parquet',index=False,compression='zstd')
    pd.DataFrame(interval_rows).to_csv(OUT/'interval_metrics.csv',index=False)
    pd.DataFrame(base_rates).to_csv(OUT/'base_rates.csv',index=False)
    pd.DataFrame(class_results).to_csv(OUT/'classifier_metrics.csv',index=False)
    # Independent median baseline remains visible even if validation chooses a flexible model.
    baseline=[]
    for direction in ('up','down'):
        center=observed['train'][direction+'_sigma'].median()
        for split,g in observed.items():
            baseline.append(dict(split=split,direction=direction,median_prediction=center,**regression_metrics(g[direction+'_sigma'].to_numpy(),np.full(len(g),center),g.sigma20_log.to_numpy(),direction)))
    pd.DataFrame(baseline).to_csv(OUT/'median_baselines.csv',index=False)
    associations=[]
    for split in ('train','validation'):
        g=observed[split]
        for feature in [c for c in enriched if c.endswith(('_z','_ratio'))]:
            for direction in ('up','down'):
                valid=g[[feature,direction+'_sigma']].dropna()
                rho=float(spearmanr(valid[feature],valid[direction+'_sigma']).statistic) if len(valid)>2 and valid[feature].nunique()>1 else np.nan
                associations.append(dict(split=split,feature=feature,direction=direction,rows=len(valid),spearman=rho))
    pd.DataFrame(associations).to_csv(OUT/'feature_associations.csv',index=False)
    for file,digest in manifest['source_sha256'].items():assert hashlib.sha256((DATA/file).read_bytes()).hexdigest()==digest
    assert not set(enriched)&set(['up_sigma','down_sigma','max_gain_pct','max_decline_pct','runner_5pct'])
    assert (OUT/'frozen_selection.json').stat().st_mtime <= (OUT/'predictions.parquet').stat().st_mtime
    report(manifest,result,pd.DataFrame(base_rates),pd.DataFrame(interval_rows),pd.DataFrame(class_results),pd.DataFrame(baseline))
    print('BASE RATES\n',pd.DataFrame(base_rates).to_string(index=False),flush=True)
    print('TEST REGRESSION\n',result[result.split.eq('test')].to_string(index=False),flush=True)
    print('CLASSIFIER\n',pd.DataFrame(class_results).to_string(index=False),flush=True)


def report(manifest,result,rates,intervals,class_results,baseline):
    text=['# Volatility-scaled intraday excursion study','',manifest['note'],'',
        '## Definitions','',
        'At 09:45, predict the largest subsequent rise and largest subsequent fall through regular-session close. Scale = sample standard deviation of the same stock\'s 09:45-open to close log returns over the previous 20 exchange sessions, requiring at least 15 identity-matched endpoint returns. Scale is determined entirely before today. A +1 sigma upside means a high of entry * exp(sigma); -1 sigma downside means a low of entry * exp(-sigma). It is not a normal-distribution confidence statement. Up/down excursions can both occur; neither gives their ordering.', '',
        'Activity features compare PM (04:00–09:30), final PM 30 minutes and opening 15 minutes with their corresponding prior-20-session windows: volume/RV/range mean ratios and z-scores. Volume z-scores use log1p. Sparse and zero-variance baselines remain NaN; volume=0 assumes no qualifying bars only where the security exists and other day bars are observed. Missing historical candles can understate observed activity; historical bar-count features are retained. Original features are unchanged. Both ablations get the same historical scale and same rows.', '',
        'Exact max/min targets require complete future windows. Every feature-eligible row is scored, including rows with unknown excursions; exact metrics use only observed complete paths and therefore carry observability selection bias. No post-entry outcome quality selects the pre-entry scoring cohort. No balancing or oversampling is used.', '',
        '## Base rates','', '```',rates.to_string(index=False),'```','',
        '## Validation-selected models: test metrics','',
        '| Direction | Output training units | Inputs | Model | Exact rows | MAE (sigma) | R² | Spearman | Error (percentage points) |',
        '|---|---|---|---|---:|---:|---:|---:|---:|']
    for r in result[result.split.eq('test')].itertuples():
        text.append(f'| {r.direction} | {r.units} | {r.feature_set} | {r.model} | {r.evaluated} | {r.mae_sigma:.3f} | {r.r2_sigma:.3f} | {r.spearman:.3f} | {r.mae_percentage_points:.2f} |')
    text+=['','Lower MAE is better; negative R² means worse squared error than predicting the evaluation-set mean. A median predictor can outperform an unreliable flexible model for absolute error. Standardizing the target does not guarantee predictability or improve economic returns. Models are selected by validation MAE in common sigma units; raw-percent and sigma training are evaluated on the same exact cases. Medians, ridge and three shallow boosting configurations are compared. Model and hyperparameter selection may favor a constant model when data support is weak.','',
        '## Constant median reference','', '```',baseline.to_string(index=False),'```','',
        '## +5% classifier feature control','', '```',class_results.to_string(index=False),'```','',
        'The +5% control compares original/risk-baseline features with the enriched activity features on identical rows and original chronological splits. Complete training windows avoid retaining incomplete positives while dropping incomplete negatives. Unknown selected outcomes count against conservative precision. Validation is reused for setting and cutoff selection, not an independent score.','',
        '## Quantile uncertainty','', '```',intervals.to_string(index=False),'```','',
        'Fixed enriched 10th/50th/90th quantile regressors form an intended 80% interval. Quantiles are rearranged into increasing order; pre-rearrangement crossing counts are disclosed. Actual coverage is measured, not guaranteed. Method: [scikit-learn histogram boosting/quantile regression](https://scikit-learn.org/stable/auto_examples/ensemble/plot_hgbt_regression.html).','',
        '## Interpretation and limits','',
        '- A stock can hit both +1 sigma and -1 sigma: these are two independent excursion outcomes, not mutually exclusive winner/loser labels. Fixed 5% upside/downside rates are also shown on the same complete cases.',
        '- The dataset remains the selected premarket-gap, high-relative-volume, complete-opening, >=$1m-turnover, prior-day-complete universe. It is not the market-wide population.',
        '- Only prior sessions form the 20-day features. Reference identities and splits follow historical vendor data, which may contain corrections. Percentage returns and log-return volatility are split invariant within each session; historical share volumes are split-aligned separately before building their baselines.',
        '- Unusual activity need not predict positive direction. Extrema are not executable strategy returns, and independently predicted extrema do not determine target/stop ordering, probabilities of fill, or an optimal trade.',
        '- Test dates were already examined in earlier studies. No test result chooses a model here, but this is exploratory reuse and requires future confirmation. No live strategy was replaced.','',
        'Files: manifest.json, frozen_selection.json, validation_comparison.csv, metrics.csv, predictions.parquet, base_rates.csv, median_baselines.csv, classifier_metrics.csv, interval_metrics.csv, feature_associations.csv and models/.']
    (OUT/'REPORT.md').write_text('\n'.join(text)+'\n')

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
