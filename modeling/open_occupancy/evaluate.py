"""Opening occupancy regression with a separately frozen precision-first operating view."""
import hashlib,json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer,StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,average_precision_score
from threadpoolctl import threadpool_limits
from modeling.core import signed_log,select_cutoff,wilson
from .estimators import FractionalLogit

DATA=Path('modeling/open_occupancy/data');OUT=Path('modeling/open_occupancy/results')
KEYS=['date','ticker','identity','phase']


def models():
    return {'mean':DummyRegressor(strategy='mean'),'median':DummyRegressor(strategy='median'),
        'ridge':make_pipeline(FunctionTransformer(signed_log),SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),StandardScaler(),Ridge(alpha=10)),
        'fractional_logit':FractionalLogit(),
        'hgb_absolute':HistGradientBoostingRegressor(loss='absolute_error',max_iter=80,max_leaf_nodes=3,max_depth=2,min_samples_leaf=40,l2_regularization=30,learning_rate=.03,early_stopping=False,random_state=731),
        'hgb_squared':HistGradientBoostingRegressor(loss='squared_error',max_iter=80,max_leaf_nodes=3,max_depth=2,min_samples_leaf=40,l2_regularization=30,learning_rate=.03,early_stopping=False,random_state=731)}


def predict(model,x):return np.clip(model.predict(x),0,100)


def rank_calls(p,fraction):
    k=max(1,int(np.ceil(len(p)*fraction)))
    return np.asarray(p)>=np.sort(p)[-k]


def regression_metrics(g,p):
    known=g.target_above_pct.notna().to_numpy();y=g.loc[known,'target_above_pct'].to_numpy();pred=np.asarray(p)[known]
    return dict(rows=len(g),known=int(known.sum()),unknown=int((~known).sum()),mae_percentage_points=mean_absolute_error(y,pred),
        rmse_percentage_points=np.sqrt(mean_squared_error(y,pred)),r2=r2_score(y,pred),
        spearman=float(spearmanr(y,pred).statistic) if np.std(pred)>1e-12 else np.nan,
        mean_actual_pct=float(y.mean()),mean_prediction_pct=float(pred.mean()),
        mean_bias_percentage_points=float((pred-y).mean()),within_one_candle_fraction=float((np.abs(pred-y)<=100/15+1e-10).mean()))


def precision_metrics(g,p,called):
    called=np.asarray(called,dtype=bool);known=g.target_above_pct.notna().to_numpy();high=g.target_above_pct.ge(80-1e-10).fillna(False).to_numpy(dtype=bool)
    n=int(called.sum());hit=int((called&high).sum());unknown=int((called&~known).sum());lo,hi=wilson(hit,n)
    y=high[known];pred=np.asarray(p)[known]
    return dict(rows=len(g),calls=n,signal_dates=int(g.loc[called,'date'].nunique()),hits=hit,confirmed_non_high=int((called&known&~high).sum()),unknown_calls=unknown,
        precision=hit/n if n else np.nan,precision_upper=(hit+unknown)/n if n else np.nan,wilson_low=lo,wilson_high=hi,
        coverage=n/len(g),base_high_known=float(y.mean()),ap_high_known=average_precision_score(y,pred) if y.any() else np.nan)


def day_interval(g,called):
    called=np.asarray(called,dtype=bool)
    hit=called & g.target_above_pct.ge(80-1e-10).fillna(False).to_numpy(dtype=bool)
    a=pd.DataFrame({'date':g.date.to_numpy(),'calls':called.astype(int),'hits':hit.astype(int)}).groupby('date')[['calls','hits']].sum().to_numpy()
    if not a[:,0].sum():return {'day_ci_low':np.nan,'day_ci_high':np.nan}
    rng=np.random.default_rng(731);totals=a[rng.integers(0,len(a),size=(2000,len(a)))].sum(axis=1);valid=totals[:,0]>0
    lo,hi=np.quantile(totals[valid,1]/totals[valid,0],[.025,.975])
    return {'day_ci_low':float(lo),'day_ci_high':float(hi)}


def main():
    if (OUT/'regression_metrics.csv').exists():raise ValueError('Preserve completed test results')
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'models').mkdir(exist_ok=True)
    d=pd.read_parquet(DATA/'dataset.parquet');manifest=json.loads((DATA/'manifest.json').read_text());splits=manifest['split_dates']
    pm=[c for c in d if c.startswith(('pm_','last5_','last15_','last30_'))]
    context=pm+[c for c in d if c.startswith(('prior_','prior10_','market_'))]
    sets={'premarket':pm,'context':context}
    for cols in sets.values():
        assert len(cols)==len(set(cols))
        assert not any(c.startswith('target_') or c in ('ticker','identity','date','feature_cutoff_utc') for c in cols)
        assert all(pd.api.types.is_numeric_dtype(d[c]) for c in cols)
    comparison=[];selections=[];feature_artifacts=[]
    for phase,data in d.groupby('phase'):
        groups={s:data[data.date.isin(dates)].copy() for s,dates in splits.items()}
        train=groups['train'];train=train[train.target_above_pct.notna()]
        val=groups['validation']
        for feature_set,cols in sets.items():
            choices=[]
            for name,model in models().items():
                model.fit(train[cols],train.target_above_pct)
                p=predict(model,val[cols]);reg=regression_metrics(val,p);prec=precision_metrics(val,p,rank_calls(p,.2))
                row=dict(phase=phase,feature_set=feature_set,model=name,**reg,top20_precision=prec['precision'],top20_wilson_low=prec['wilson_low'])
                comparison.append(row);choices.append((row,model))
            for objective in ('regression','precision'):
                row,model=min(choices,key=lambda x:x[0]['rmse_percentage_points']) if objective=='regression' else max(choices,key=lambda x:(x[0]['top20_wilson_low'],-x[0]['rmse_percentage_points']))
                ident=f'{phase}__{feature_set}__{objective}'
                p=predict(model,val[cols]);tuning=val[['date','ticker','identity']].copy()
                tuning['runner_5pct']=pd.Series([None if pd.isna(v) else v>=80-1e-10 for v in val.target_above_pct],index=val.index,dtype=object)
                # Core policy uses 0..1 scores. These are scaled occupancy predictions, not calibrated class probabilities.
                policy=select_cutoff(tuning,p/100,5)
                artifact=dict(estimator=model,features=cols,phase=phase,objective=objective,policy=policy,
                    target='Percent of 09:30–09:44 minute closes strictly above 09:30 open',selection=row)
                joblib.dump(artifact,OUT/'models'/f'{ident}.joblib',compress=3)
                selections.append(dict(id=ident,phase=phase,feature_set=feature_set,objective=objective,model=row['model'],selection=row,policy=policy))
                feature_artifacts.append(ident)
    # Freeze the primary choice within each timing assumption using validation only.
    primaries={}
    for phase in d.phase.unique():
        candidates=[s for s in selections if s['phase']==phase and s['objective']=='regression']
        regression=min(candidates,key=lambda x:x['selection']['rmse_percentage_points'])['id']
        candidates=[s for s in selections if s['phase']==phase and s['objective']=='precision']
        precision=max(candidates,key=lambda x:(x['selection']['top20_wilson_low'],-x['selection']['rmse_percentage_points']))['id']
        primaries[phase]=dict(regression=regression,precision=precision)
    run_manifest=dict(source_manifest=manifest,plan=json.loads(Path('modeling/open_occupancy/plan.json').read_text()),dataset_sha256=hashlib.sha256((DATA/'dataset.parquet').read_bytes()).hexdigest(),features=sets,split_dates=splits,
        caveat='Chronological calendar splits reused, but test dates were examined in previous experiments. Validation chooses both models and operating cutoffs. Exploratory reuse, not independent confirmation.',
        objectives='Primary percentage forecast uses validation RMSE (a squared-error/conditional-mean objective), with MAE also reported. Precision companion uses Wilson lower bound for >=80% occupancy among validation top20%; operating cutoff uses original supported 80% precision policy. Occupancy predictions are not probabilities of high occupancy.')
    (OUT/'manifest.json').write_text(json.dumps(run_manifest,indent=2)+'\n')
    (OUT/'frozen_selection.json').write_text(json.dumps(dict(primaries=primaries,candidates=selections),indent=2)+'\n')
    pd.DataFrame(comparison).to_csv(OUT/'validation_comparison.csv',index=False)
    reg_rows=[];policies=[];rank_rows=[];predictions=[];calibration=[];baselines=[];rates=[]
    for phase,data in d.groupby('phase'):
        train=data[data.date.isin(splits['train'])&data.target_above_pct.notna()]
        centers={'mean':train.target_above_pct.mean(),'median':train.target_above_pct.median()}
        for split,dates in splits.items():
            group=data[data.date.isin(dates)]
            rates.append(dict(phase=phase,split=split,rows=len(group),known=int(group.target_above_pct.notna().sum()),high=int(group.target_above_pct.ge(80-1e-10).sum()),mean_pct=group.target_above_pct.mean(),median_pct=group.target_above_pct.median()))
            for name,center in centers.items():baselines.append(dict(phase=phase,split=split,model=name,**regression_metrics(group,np.full(len(group),center))))
            for name,col in [('previous_day','prior_opening_above_pct'),('previous10_mean','prior10_opening_above_mean')]:
                baselines.append(dict(phase=phase,split=split,model=name,**regression_metrics(group,group[col].fillna(centers['mean']).clip(0,100).to_numpy())))
    for selection in selections:
        artifact=joblib.load(OUT/'models'/f'{selection["id"]}.joblib');data=d[d.phase.eq(selection['phase'])]
        for split,dates in splits.items():
            g=data[data.date.isin(dates)].copy();p=predict(artifact['estimator'],g[artifact['features']]);called=p/100>=artifact['policy']['cutoff']
            reg_rows.append(dict(candidate=selection['id'],phase=selection['phase'],split=split,model=selection['model'],**regression_metrics(g,p)))
            policies.append(dict(candidate=selection['id'],phase=selection['phase'],split=split,qualified=artifact['policy']['qualified'],occupancy_prediction_cutoff_pct=100*artifact['policy']['cutoff'],**precision_metrics(g,p,called),**day_interval(g,called)))
            for fraction in (.05,.1,.2):rank_rows.append(dict(candidate=selection['id'],split=split,nominal_fraction=fraction,**precision_metrics(g,p,rank_calls(p,fraction))))
            frame=g[KEYS+['target_above_pct','target_entire_above_pct','target_entry_price','target_observed','target_end_return_pct']].copy()
            frame['candidate']=selection['id'];frame['split']=split;frame['prediction_pct']=p;frame['called']=called;predictions.append(frame)
            q=pd.DataFrame({'prediction':p,'actual':g.target_above_pct.to_numpy()})
            q['bin']=pd.cut(q.prediction,[-1,20,40,60,80,101])
            for band,z in q.groupby('bin',observed=False):calibration.append(dict(candidate=selection['id'],split=split,bin=str(band),rows=len(z),known=int(z.actual.notna().sum()),mean_prediction=z.loc[z.actual.notna(),'prediction'].mean(),mean_actual=z.actual.mean()))
    pd.DataFrame(reg_rows).to_csv(OUT/'regression_metrics.csv',index=False);pd.DataFrame(policies).to_csv(OUT/'precision_metrics.csv',index=False)
    pd.DataFrame(rank_rows).to_csv(OUT/'ranking_precision.csv',index=False);pd.DataFrame(baselines).to_csv(OUT/'baselines.csv',index=False)
    pd.DataFrame(rates).to_csv(OUT/'cohort_summary.csv',index=False);pd.DataFrame(calibration).to_csv(OUT/'calibration.csv',index=False)
    all_predictions=pd.concat(predictions,ignore_index=True)
    all_predictions.to_parquet(OUT/'predictions.parquet',index=False,compression='zstd')
    common=d.groupby(['date','ticker','identity']).phase.nunique().eq(2)
    common=common[common].reset_index()[['date','ticker','identity']]
    matched=[]
    for phase,choices in primaries.items():
        for objective,candidate in choices.items():
            frame=all_predictions[all_predictions.candidate.eq(candidate)].merge(common,on=['date','ticker','identity'],validate='many_to_one')
            for split,g in frame.groupby('split'):
                matched.append(dict(phase=phase,objective=objective,split=split,candidate=candidate,**regression_metrics(g,g.prediction_pct.to_numpy()),**{'policy_'+k:v for k,v in precision_metrics(g,g.prediction_pct.to_numpy(),g.called).items()}))
    pd.DataFrame(matched).to_csv(OUT/'matched_cohort_metrics.csv',index=False)
    assert hashlib.sha256((DATA/'dataset.parquet').read_bytes()).hexdigest()==run_manifest['dataset_sha256']
    assert (OUT/'frozen_selection.json').stat().st_mtime<=(OUT/'predictions.parquet').stat().st_mtime
    print('PRIMARY MODELS',json.dumps(primaries),flush=True)
    print('COHORTS\n',pd.DataFrame(rates).to_string(index=False),flush=True)
    print('TEST REGRESSION\n',pd.DataFrame(reg_rows).query("split=='test'").to_string(index=False),flush=True)
    print('TEST PRECISION\n',pd.DataFrame(policies).query("split=='test'").to_string(index=False),flush=True)
    report(run_manifest,primaries,selections,pd.DataFrame(reg_rows),pd.DataFrame(policies),pd.DataFrame(baselines))


def report(manifest,primaries,selections,reg,precision,baseline):
    text=['# Predicting opening 15-minute close occupancy above opening price','',manifest['caveat'],'',
        '## Target and timing','',manifest['source_manifest']['target'],'',
        'The target is 15 closing snapshots, not the percentage of clock time spent above entry or the gain magnitude. An auxiliary all-of-candle-low-above-entry target is saved but not optimized. Missing opening minutes leave the exact target unknown; no observed-only denominator or stale entry is substituted. Every pre-open-eligible stock is scored regardless of later coverage.','',
        'Two timing assumptions are separate, not a test-selected winner: delayed_0914 uses only bars ending by 09:14, leaving approximately one minute after a 15-minute delay for a hypothetical decision; realtime_0929 uses bars ending by 09:29 and requires timely real-time data. Actual receipt, publication latency, auction order deadlines and broker fills are not reconstructed. The 09:30 minute open is a research reference price, not a guaranteed executable fill.','',
        '## Leakage controls','',
        'The opportunity set was rebuilt market-wide from premarket data and historical identities, rather than reusing the earlier >=$1m opening-turnover filter. Eligibility: same-cutoff PM gap >=10%, PM relative volume >=5 versus previous ten matching sessions, ten matched historical identities and a complete immediately prior regular-session grid (early closes honored). No current opening volume/price/coverage participates in selection. All current opening data lives under target_ columns and is excluded from model inputs.','',
        'Features include PM price/volume/transactions/RV/range/VWAP/location, final 30/15/5-minute momentum and volume concentration; prior-day and prior-ten-day returns, liquidity, PM baselines and historical opening occupancy; contemporaneous pre-cutoff SPY/QQQ returns and stock excess returns. Split corrections only use execution dates up to the target date. Prior opening outcomes are already known before the current open; current opening outcomes are never features.','',
        '## Models and selection','',manifest['objectives'],'',
        'Candidates: mean/median baselines, regularized ridge, fractional logistic regression for expected occupancy, and two shallow gradient-boosting regressors (absolute/squared loss). Fractional-logit weights sum to one per stock-day; no claim that the 15 candles are independent trials. Models fit exact training targets only, and score all validation/test opportunities. There is no refit after validation selection.','',
        '## Frozen primary models','', '```',json.dumps(primaries,indent=2),'```','',
        '## Test regression','', '```',reg[reg.split.eq('test')].to_string(index=False),'```','',
        '## Constant references','', '```',baseline[baseline.split.eq('test')].to_string(index=False),'```','',
        '## Test high-occupancy precision','', '```',precision[precision.split.eq('test')].to_string(index=False),'```','',
        'High occupancy means >=12 of 15 closes above entry (>=80%). Precision is the fraction of called stock-days meeting that condition, counting unknown calls against precision. A predicted occupancy of 80% is not an 80% probability of meeting the condition. Supported cutoff policy needs >=80% validation precision, >=25 calls over >=10 dates and Wilson lower >=65%; otherwise a research fallback is disclosed. Top5/10/20 diagnostics retain ties and are not deployable period-wide rank cutoffs.','',
        'This is an occupancy forecast, not a backtested profitable strategy. Spending most sampled minutes above entry does not guarantee a profitable exit or favorable risk. Historical vendor corrections and previously viewed calendar test dates limit inference. The existing paper strategy/model are unchanged.','',
        'Artifacts: validation_comparison.csv, frozen_selection.json, regression_metrics.csv, precision_metrics.csv, ranking_precision.csv, baselines.csv, cohort_summary.csv, calibration.csv, predictions.parquet and models/.']
    (OUT/'REPORT.md').write_text('\n'.join(text)+'\n')

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
