"""Precision-first, chronological catalyst/setup study with frozen delayed-entry labels."""
import hashlib,json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer,StandardScaler
from sklearn.metrics import average_precision_score,roc_auc_score
from threadpoolctl import threadpool_limits
from modeling.core import signed_log,select_cutoff,wilson

DATA=Path('modeling/catalyst_precision/data');OUT=Path('modeling/catalyst_precision/results')
KEYS=['date','ticker','identity']
SEGMENTS=['all','headline_positive','financing','no_classified_fresh_event','continuation','failed_breakout','positive_continuation']


def mask(d,segment):
    if segment=='all':return pd.Series(True,index=d.index)
    if segment=='headline_positive':return d.news_positive_candidate.eq(1)
    if segment=='financing':return d.news_has_financing.eq(1)
    if segment=='no_classified_fresh_event':return d.news_none_classified.eq(1)&d.news_api_ok.eq(1)&d.news_identity_match.eq(1)
    if segment=='continuation':return d.setup_continuation.eq(1)
    if segment=='failed_breakout':return d.setup_failed_breakout.eq(1)
    if segment=='positive_continuation':return d.news_positive_candidate.eq(1)&d.setup_continuation.eq(1)
    raise ValueError(segment)


def family(name):
    if name=='logistic':return make_pipeline(FunctionTransformer(signed_log),SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),StandardScaler(),LogisticRegression(C=.1,max_iter=2000,random_state=731))
    return HistGradientBoostingClassifier(max_leaf_nodes=3,max_depth=2,min_samples_leaf=40,l2_regularization=30,max_iter=80,learning_rate=.03,early_stopping=False,random_state=731)


def score(model,x):
    return model.predict_proba(x)[:,list(model.classes_).index(1)]


def stats(d,p,called):
    p=np.asarray(p);called=np.asarray(called,dtype=bool)
    known=d.target_outcome.notna().to_numpy();up=d.target_outcome.eq('up').fillna(False).to_numpy(dtype=bool)
    n=int(called.sum());hit=int((called&up).sum());unknown=int((called&~known).sum());low,high=wilson(hit,n)
    return dict(rows=len(d),known=int(known.sum()),calls=n,call_dates=int(d.loc[called,'date'].nunique()),hits=hit,confirmed_non_up=int((called&known&~up).sum()),unknown_calls=unknown,
        precision=hit/n if n else np.nan,precision_upper=(hit+unknown)/n if n else np.nan,wilson_low=low,wilson_high=high,
        coverage=n/len(d) if len(d) else np.nan,base_rate_conservative=float(up.mean()) if len(d) else np.nan,
        ap_known=average_precision_score(up[known],p[known]) if known.any() and up[known].any() else np.nan,
        auc_known=roc_auc_score(up[known],p[known]) if known.any() and up[known].any() and not up[known].all() else np.nan)


def returns(d,called,prefix='target'):
    subset=d[np.asarray(called,dtype=bool)];known=subset[prefix+'_outcome'].notna();r=subset.loc[known,prefix+'_gross_return_pct'].to_numpy()
    wins=r[r>1e-10];losses=r[r< -1e-10]
    return dict(signals=len(subset),resolved=len(r),unresolved=int((~known).sum()),wins=len(wins),losses=len(losses),breakeven=int((np.abs(r)<=1e-10).sum()),
        mean_gross_pct=float(r.mean()) if len(r) else np.nan,mean_net_50bps_pct=float(r.mean()-.5) if len(r) else np.nan,
        mean_net_100bps_pct=float(r.mean()-1) if len(r) else np.nan,average_win_pct=float(wins.mean()) if len(wins) else np.nan,
        average_loss_pct=float(losses.mean()) if len(losses) else np.nan,best_pct=float(r.max()) if len(r) else np.nan,worst_pct=float(r.min()) if len(r) else np.nan)


def day_precision(d,called):
    called=np.asarray(called,dtype=bool)
    hits=called & d.target_outcome.eq('up').fillna(False).to_numpy(dtype=bool)
    grouped=pd.DataFrame({'date':d.date.to_numpy(),'calls':called.astype(int),'hits':hits.astype(int)}).groupby('date')[['calls','hits']].sum().to_numpy()
    if not len(grouped) or grouped[:,0].sum()==0:return {'day_ci_low':np.nan,'day_ci_high':np.nan}
    rng=np.random.default_rng(731);sample=grouped[rng.integers(0,len(grouped),size=(2000,len(grouped)))].sum(axis=1)
    valid=sample[:,0]>0;v=sample[valid,1]/sample[valid,0]
    lo,hi=np.quantile(v,[.025,.975])
    return {'day_ci_low':float(lo),'day_ci_high':float(hi)}


def top_calls(p,fraction):
    k=max(1,int(np.ceil(len(p)*fraction)))
    return np.asarray(p)>=np.sort(p)[-k]


def fit_support(d):
    known=d[d.target_outcome.notna()];up=int(known.target_outcome.eq('up').sum())
    return len(known)>=40 and up>=10 and len(known)-up>=10


def tuning_frame(d):
    out=d[KEYS].copy()
    out['runner_5pct']=pd.Series([None if pd.isna(o) else o=='up' for o in d.target_outcome],index=d.index,dtype=object)
    return out


def main():
    if (OUT/'metrics.csv').exists():raise ValueError('Preserve completed evaluated results')
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'models').mkdir(exist_ok=True);(OUT/'development_models').mkdir(exist_ok=True)
    d=pd.read_parquet(DATA/'dataset.parquet');original=json.loads(Path('modeling/barrier_study/results/manifest.json').read_text());splits=original['split_dates'];calendar=splits['train']
    price=original['features']['base']+['path_last5_return_pct','path_close_vs_pm_high_pct','path_pullback_from_high_pct','path_last5_above_running_vwap_fraction','setup_continuation','setup_failed_breakout']
    sets={'price':price,'price_news':price+[c for c in d if c.startswith('news_')]}
    for cols in sets.values():
        assert len(cols)==len(set(cols))
        assert not any(c.startswith(('label_','target_','late_')) for c in cols)
    plan=json.loads(Path('modeling/catalyst_precision/plan.json').read_text())
    manifest=dict(plan=plan,data_manifest=json.loads((DATA/'manifest.json').read_text()),dataset_sha256=hashlib.sha256((DATA/'dataset.parquet').read_bytes()).hexdigest(),split_dates=splits,embargo_dates=original['embargo_dates'],features=sets,
        note='Development model ranking uses OOF data entirely inside original training dates. Later validation chooses cutoffs and candidates. Test dates have already been viewed. Publication-time news proxies do not certify original arrival/version timing.')
    development=[];selections=[];skipped=[];oof_frames=[]
    for segment in SEGMENTS:
        subset=d[mask(d,segment)].copy()
        for feature_set,cols in sets.items():
            choices=[]
            for name in ('logistic','hgb'):
                frames=[]
                for fold,(a,b) in enumerate(((150,200),(200,250),(250,300)),1):
                    train=subset[subset.date.isin(calendar[:a])];val=subset[subset.date.isin(calendar[a+1:b])]
                    if not fit_support(train) or val.empty:continue
                    train=train[train.target_outcome.notna()]
                    model=family(name);model.fit(train[cols],train.target_outcome.eq('up').astype(int))
                    joblib.dump(dict(estimator=model,features=cols,train_dates=calendar[:a],embargo_date=calendar[a],validation_dates=calendar[a+1:b]),OUT/'development_models'/f'{segment}__{feature_set}__{name}__fold{fold}.joblib',compress=3)
                    p=score(model,val[cols]);frame=val.copy();frame['score']=p;frame['top20']=top_calls(p,.2);frame['fold']=fold;frames.append(frame)
                if len(frames)!=3:
                    development.append(dict(segment=segment,feature_set=feature_set,model=name,supported=False,reason='Three development folds lack training-class or validation support'))
                    continue
                oof=pd.concat(frames,ignore_index=True);m=stats(oof,oof.score,oof.top20)
                supported=m['calls']>=20 and m['call_dates']>=10
                development.append(dict(segment=segment,feature_set=feature_set,model=name,supported=supported,**m))
                if supported:choices.append((m,name,oof))
            if not choices:
                skipped.append(dict(segment=segment,feature_set=feature_set,reason='Insufficient OOF support; no high-precision policy claimed'));continue
            m,name,oof=max(choices,key=lambda x:(x[0]['wilson_low'],x[0]['ap_known']))
            train=subset[subset.date.isin(splits['train'])&subset.target_outcome.notna()]
            val=subset[subset.date.isin(splits['validation'])]
            if val.empty:continue
            model=family(name);model.fit(train[cols],train.target_outcome.eq('up').astype(int))
            p=score(model,val[cols]);policy=select_cutoff(tuning_frame(val),p,5)
            called=p>=policy['cutoff'];vm={**stats(val,p,called),**day_precision(val,called)};econ=returns(val,called)
            passes_economic=bool(econ['resolved']>0 and econ['mean_net_50bps_pct']>0 and econ['worst_pct']>=-10)
            approved=bool(policy['qualified'] and passes_economic)
            ident=segment+'__'+feature_set
            artifact=dict(estimator=model,features=cols,segment=segment,feature_set=feature_set,policy=policy,approved_for_forward_candidate=approved,
                entry_et='10:05',feature_cutoff_et='09:45',scale='existing prior sigma20',target='up-first within 60 minutes, +/-1 sigma barriers')
            joblib.dump(artifact,OUT/'models'/f'{ident}.joblib',compress=3)
            selections.append(dict(id=ident,segment=segment,feature_set=feature_set,model=name,oof_metrics=m,policy=policy,validation_metrics=vm,validation_returns=econ,economic_gate=passes_economic,approved=approved))
            frame=oof[KEYS+['target_outcome','score','fold','top20']].copy();frame['candidate']=ident;oof_frames.append(frame)
    if not selections:raise RuntimeError('No candidate had supported chronological OOF evidence')
    primary=max(selections,key=lambda x:(x['approved'],np.nan_to_num(x['validation_metrics']['wilson_low'],nan=-1),x['validation_metrics']['calls']))['id']
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (OUT/'frozen_selection.json').write_text(json.dumps(dict(primary=primary,candidates=selections,skipped=skipped),indent=2)+'\n')
    pd.DataFrame(development).to_csv(OUT/'development_comparison.csv',index=False)
    pd.concat(oof_frames,ignore_index=True).to_parquet(OUT/'development_predictions.parquet',index=False,compression='zstd')
    print('Frozen primary:',primary,'; supported candidates:',len(selections),'; skipped:',len(skipped),flush=True)
    metrics=[];trade_rows=[];ranking=[];monthly=[];predictions=[];segment_rates=[]
    for segment in SEGMENTS:
        subset=d[mask(d,segment)]
        for split,dates in splits.items():
            g=subset[subset.date.isin(dates)]
            segment_rates.append(dict(segment=segment,split=split,rows=len(g),dates=g.date.nunique(),known=int(g.target_outcome.notna().sum()),up=int(g.target_outcome.eq('up').sum()),down=int(g.target_outcome.eq('down').sum()),neither=int(g.target_outcome.eq('neither').sum())))
    for selection in selections:
        ident=selection['id'];artifact=joblib.load(OUT/'models'/f'{ident}.joblib');subset=d[mask(d,selection['segment'])]
        for split,dates in splits.items():
            g=subset[subset.date.isin(dates)].copy()
            if g.empty:continue
            p=score(artifact['estimator'],g[artifact['features']]);called=p>=artifact['policy']['cutoff']
            metrics.append(dict(candidate=ident,split=split,model=selection['model'],qualified=artifact['policy']['qualified'],approved=selection['approved'],**stats(g,p,called),**day_precision(g,called)))
            for entry,prefix in [('10:05','target'),('10:15_sensitivity','late')]:trade_rows.append(dict(candidate=ident,split=split,entry=entry,**returns(g,called,prefix)))
            for fraction in (.05,.1,.2):ranking.append(dict(candidate=ident,split=split,nominal_fraction=fraction,**stats(g,p,top_calls(p,fraction))))
            g['score']=p;g['called']=called;g['month']=pd.to_datetime(g.date).dt.to_period('M').astype(str)
            for month,group in g.groupby('month'):
                monthly.append(dict(candidate=ident,split=split,month=month,**stats(group,group.score,group.called),**{'trade_'+k:v for k,v in returns(group,group.called).items()}))
            frame=g[KEYS+['target_outcome','target_reason','target_gross_return_pct','late_outcome','late_gross_return_pct','score','called']].copy();frame['candidate']=ident;frame['split']=split;predictions.append(frame)
    metrics=pd.DataFrame(metrics);trades=pd.DataFrame(trade_rows);ranks=pd.DataFrame(ranking)
    metrics.to_csv(OUT/'metrics.csv',index=False);trades.to_csv(OUT/'strategy_metrics.csv',index=False);ranks.to_csv(OUT/'ranking_precision.csv',index=False)
    pd.DataFrame(monthly).to_csv(OUT/'monthly.csv',index=False);pd.DataFrame(segment_rates).to_csv(OUT/'segment_rates.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_parquet(OUT/'predictions.parquet',index=False,compression='zstd')
    chosen=next(s for s in selections if s['id']==primary);model_path=OUT/'models'/f'{primary}.joblib'
    protocol=dict(primary_research_candidate=primary,model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),model=str(model_path.resolve()),cutoff=chosen['policy']['cutoff'],
        automatic_paper_entries_enabled=False,validation_gates_passed=chosen['approved'],status='Candidate needs forward confirmation' if chosen['approved'] else 'No validation-approved edge; research-only logging candidate',
        forward_test='At least 60 new exchange sessions and >=50 resolved signals, with unchanged model/features/cutoff and no interim retuning. Report precision, Wilson/date-bootstrap intervals, counts, costs and loss severity.',
        availability='Archive every raw headline version and first API receipt, minute receipt timestamps and decisions; only score inputs actually received before decision; precommit next-minute entries. No historical publication timestamp alone certifies availability.',
        constraints='No real broker orders, no short trades, no model replacement. News API failure is unavailable, not no catalyst. Require new input/feature schema match and risk/capital limits before enabling paper entries.')
    (OUT/'FORWARD_PROTOCOL.json').write_text(json.dumps(protocol,indent=2)+'\n')
    assert hashlib.sha256((DATA/'dataset.parquet').read_bytes()).hexdigest()==manifest['dataset_sha256']
    assert (OUT/'frozen_selection.json').stat().st_mtime<=(OUT/'predictions.parquet').stat().st_mtime
    report(primary,selections,skipped,metrics,trades,ranks,pd.DataFrame(segment_rates))
    print('SEGMENT RATES\n',pd.DataFrame(segment_rates).to_string(index=False),flush=True)
    print('TEST PRECISION\n',metrics[metrics.split.eq('test')].to_string(index=False),flush=True)
    print('PRIMARY TRADES\n',trades[trades.candidate.eq(primary)].to_string(index=False),flush=True)
    print('PRIMARY RANKING\n',ranks[ranks.candidate.eq(primary)].to_string(index=False),flush=True)


def report(primary,selections,skipped,metrics,trades,ranks,rates):
    text=['# Catalyst/setup precision-first study','',f'Frozen primary research candidate: **{primary}**.','',
        'This is exploratory reuse of previously inspected calendar test dates, not independent confirmation. News is a headline/publication-time proxy: neither original ingestion timing nor headline version history is certified. Standard Massive news is accessible; structured Benzinga earnings/news returned 403. We therefore do not claim verified earnings surprise or genuine positive fundamentals. Vendor sentiment/description fields are excluded.','',
        '## Fixed design','',
        'Features and setup flags stop before 09:45 ET. Fresh news means published after previous regular close and strictly before 09:45. Entry 10:05 ET assumes 15-minute feed delay plus five minutes of data collection/decision budget, with a hypothetical decision at 10:04. It is not an exact replay of actual REST scan latency. Independent fixed 10:15 sensitivity uses the same scores/cutoff, not retraining or selection.','',
        'Target: first +/-1 prior sigma20 log-price barrier within 60 minutes of entry; timeout if neither. Stop gaps fill at worse open, upside fills at target. Gaps before exit and ambiguous minutes remain unresolved. No actual fills/spreads/market impact are reconstructed. Failed-breakout remains a long diagnostic, not a short strategy without borrow data.','',
        'Models: regularized logistic and shallow HGB. Three expanding chronological development folds (150/200/250 initial sessions) choose family by top20% OOF conservative precision Wilson lower bound with >=20 calls across >=10 dates, requiring all three folds. AP breaks ties. Final fits use original training period only. Later original validation chooses cutoff; no final refit.','',
        'A precision-qualified policy needs >=80% conservative validation precision, >=25 calls across >=10 dates and Wilson lower >=65%. The economic gate also needs positive validation mean after 0.5% assumed round-trip costs and worst observed gross loss no worse than -10%. Unsupported groups are skipped rather than advertised as precise.','',
        '## Frozen validation candidates','',
        '| Candidate | Model | Validation calls | Precision | Wilson lower | Mean net at 0.5% cost | All gates? |','|---|---|---:|---:|---:|---:|---|']
    for s in selections:
        v=s['validation_metrics'];e=s['validation_returns']
        text.append(f"| {s['id']} | {s['model']} | {v['calls']} | {100*v['precision']:.1f}% | {100*v['wilson_low']:.1f}% | {e['mean_net_50bps_pct']:.2f}% | {s['approved']} |")
    text+=['','## Cohort coverage','', '```',rates.to_string(index=False),'```','',
        '## Reused-test precision','', '```',metrics[metrics.split.eq('test')].to_string(index=False),'```','',
        '## Primary policy trade outcomes','', '```',trades[trades.candidate.eq(primary)].to_string(index=False),'```','',
        '## Primary ranking diagnostics','', '```',ranks[ranks.candidate.eq(primary)].to_string(index=False),'```','',
        'Top fractions are descriptive ranks across the period, not deployable cutoffs known in advance. Ties are retained, so actual coverage can exceed nominal coverage. Signal counts/dates and unknowns accompany every precision. Training diagnostics are in sample; validation is tuned. Only supported frozen cutoffs are evaluated as policies. P&L means describe resolved equally weighted positions, not a portfolio, and exclude unknown returns rather than treating them as zero.','',
        '## Unsupported candidate groups','', '```',json.dumps(skipped,indent=2),'```','',
        'No classified archive headline does not mean no real catalyst. Coverage, company matching, text negation/hypotheticals, and historical revisions can change the proxy. A positive candidate is not a verified surprise; all classifications can be audited in data/headline_audit.csv. Monthly stability and all top5/10/20 diagnostics are exported.','',
        'The saved FORWARD_PROTOCOL.json disables automatic paper entries. If no candidate passes validation gates, there is no approved edge to deploy; a research-only candidate may be logged prospectively. At least 60 future sessions and 50 resolved signals, with raw first-receipt/version archives and no retuning, are required by the proposed confirmation protocol. Existing paper strategy/model remain unchanged.','',
        'Files: manifest.json, frozen_selection.json, development_comparison.csv, development_predictions.parquet, metrics.csv, strategy_metrics.csv, ranking_precision.csv, monthly.csv, segment_rates.csv, predictions.parquet, FORWARD_PROTOCOL.json and models/.']
    (OUT/'REPORT.md').write_text('\n'.join(text)+'\n')

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
