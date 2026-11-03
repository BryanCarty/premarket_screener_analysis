"""Descriptive exit comparison on previously viewed regularized-model test signals."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import exchange_calendars as xcals
from runner.api import API
from runner.collect import rest_day, save
from .analyze import simulate

OUT=Path('modeling/results_regularized_5pct_close/exit_comparison')
PATHS=Path('modeling/exit_study/paths')
POLICIES=[('close_only',None,None,'close'),('tp5_or_close',5,None,'close')]
for stop in (2,3,5,10,15,20):
    POLICIES.extend([(f'tp5_sl{stop}_close',5,stop,'close'),(f'sl{stop}_or_close',None,stop,'close')])


def statistics(d,all_dates):
    x=d.gross_pct.to_numpy()
    wins=x[x>1e-10];losses=x[x < -1e-10]
    day=d.groupby('date').gross_pct.agg(['sum','size']).reindex(all_dates,fill_value=0).to_numpy()
    rng=np.random.default_rng(731)
    totals=day[rng.integers(0,len(day),size=(3000,len(day)))].sum(axis=1)
    valid=totals[:,1]>0
    ci=np.quantile(totals[valid,0]/totals[valid,1],[.025,.975]) if valid.any() else [np.nan,np.nan]
    return dict(trades=len(x),wins=len(wins),losses=len(losses),breakeven=len(x)-len(wins)-len(losses),win_rate_pct=100*len(wins)/len(x) if len(x) else np.nan,
        average_gain_pct=wins.mean() if len(wins) else np.nan,average_loss_pct=losses.mean() if len(losses) else np.nan,
        mean_gross_pct=x.mean() if len(x) else np.nan,median_gross_pct=np.median(x) if len(x) else np.nan,
        max_gain_pct=x.max() if len(x) else np.nan,max_loss_pct=x.min() if len(x) else np.nan,
        profit_factor=wins.sum()/(-losses.sum()) if len(losses) else np.nan,
        mean_net_50bps_pct=x.mean()-.5 if len(x) else np.nan,
        mean_net_100bps_pct=x.mean()-1 if len(x) else np.nan,
        mean_net_200bps_pct=x.mean()-2 if len(x) else np.nan,
        mean_gross_ci_low=ci[0],mean_gross_ci_high=ci[1],
        ambiguous_bars=int(d.ambiguous_bar.sum()),stop_gap_exits=int(d.reason.eq('stop_gap').sum()))


def main():
    OUT.mkdir(parents=True,exist_ok=True);PATHS.mkdir(parents=True,exist_ok=True)
    pred=pd.read_parquet('modeling/results_regularized_5pct_close/predictions.parquet')
    signals=pred[pred.split.eq('test')&pred.predicted_runner].copy()
    signals[['date','ticker','identity','score']].to_csv(OUT/'signals.csv',index=False)
    (OUT/'plan.json').write_text(json.dumps(dict(policies=POLICIES,model='regularized_5pct_runner',signals=len(signals),
        rule='09:45 entry, static TP/SL, otherwise regular-session close; gap stops at bar open; stop first if both barriers touched in one minute',
        warning='Descriptive comparison on an already viewed test period; no policy chosen or deployed. Newly fetched historical data may contain vendor revisions.',
        main_cohort='Same verifiable stock-days across every exit policy; policy-specific verifiable cohorts and observed-path sensitivity are reported separately.'),indent=2)+'\n')
    api=API(OUT/'request_cache');cal=xcals.get_calendar('XNYS');rows=[];errors=[];sources=[]
    for i,s in enumerate(signals.itertuples(),1):
        file=PATHS/f'{s.date}_{s.ticker}.parquet'
        try:
            cached=file.exists()
            if cached:b=pd.read_parquet(file)
            else:
                b=rest_day(s.date,[s.ticker],api);b=b[b.minute.ge(585)].copy();save(b,file)
                for p in api.cache.glob('*.json'):p.unlink()
            close=cal.session_close(pd.Timestamp(s.date)).tz_convert('America/New_York')
            close_min=close.hour*60+close.minute
            sources.append(dict(date=s.date,ticker=s.ticker,cached=cached,path=str(file),mtime_ns=file.stat().st_mtime_ns))
            for policy,tp,sl,horizon in POLICIES:
                result=simulate(b,tp,sl,horizon,close_min)
                rows.append(dict(date=s.date,ticker=s.ticker,identity=s.identity,policy=policy,**result))
        except Exception as exc:
            errors.append(dict(date=s.date,ticker=s.ticker,error=type(exc).__name__))
            for policy,*_ in POLICIES:
                rows.append(dict(date=s.date,ticker=s.ticker,identity=s.identity,policy=policy,entry=np.nan,exit=np.nan,gross_pct=np.nan,reason='api_failure',exit_minute=np.nan,ambiguous_bar=False,missing_before_exit=True))
        if i%25==0 or i==len(signals):print(f'{i}/{len(signals)} signals processed; API failures {len(errors)}',flush=True)
    trades=pd.DataFrame(rows)
    trades['verifiable']=trades.gross_pct.notna()&~trades.missing_before_exit
    common=trades.groupby(['date','ticker','identity']).verifiable.all().rename('common_verifiable').reset_index()
    trades=trades.merge(common,on=['date','ticker','identity'],validate='many_to_one')
    trades.to_parquet(OUT/'trades.parquet',index=False,compression='zstd')
    trades.to_csv(OUT/'trades.csv',index=False)
    pd.DataFrame(sources).to_csv(OUT/'path_provenance.csv',index=False)
    (OUT/'errors.json').write_text(json.dumps(errors,indent=2)+'\n')
    results=[];dates=sorted(signals.date.unique())
    for policy,group in trades.groupby('policy',sort=False):
        for cohort,subset in [('matched_verifiable',group[group.common_verifiable]),('policy_verifiable',group[group.verifiable]),('observed_path_sensitivity',group[group.gross_pct.notna()])]:
            results.append(dict(policy=policy,cohort=cohort,signals=len(signals),unresolved=len(group)-len(subset),missing_path_cases_used=int(subset.missing_before_exit.sum()),**statistics(subset,dates)))
    results=pd.DataFrame(results);results.to_csv(OUT/'metrics.csv',index=False)
    # Reconcile the no-stop baseline against the labels used in the earlier calculation.
    old=pd.read_csv('modeling/results_regularized_5pct_close/tp5_or_close/trades.csv')
    old=old[['date','ticker','identity','strategy_return_pct']]
    baseline=trades[trades.policy.eq('tp5_or_close')].merge(old,on=['date','ticker','identity'],validate='one_to_one')
    baseline['previous_resolved']=baseline.strategy_return_pct.notna()
    baseline['return_changed']=baseline.previous_resolved&baseline.verifiable&~np.isclose(baseline.strategy_return_pct,baseline.gross_pct,atol=1e-7,equal_nan=True)
    baseline.to_csv(OUT/'baseline_reconciliation.csv',index=False)
    reconciliation=dict(previous_resolved=int(baseline.previous_resolved.sum()),now_verifiable=int(baseline.verifiable.sum()),changed_known_returns=int(baseline.return_changed.sum()),previous_resolved_now_unverifiable=int((baseline.previous_resolved&~baseline.verifiable).sum()))
    (OUT/'baseline_reconciliation.json').write_text(json.dumps(reconciliation,indent=2)+'\n')
    report=['# Regularized +5% model: exploratory exit policies on test signals','',
        f'{len(signals)} frozen test signals, July 15 through October 28, 2026. Entry is the observed 09:45 minute open. This comparison uses the historical research entry, not the later next-minute entry used in forward paper trading.','',
        'This is a descriptive analysis on an already inspected test period. No exit policy is selected or deployed. Stops and targets are percentages of entry. At a gap below the stop, exit at the worse minute open. If both TP and SL occur inside the same minute, assume stop first. Static target fills are capped at the target. No costs are embedded in gross results; cost sensitivities subtract fixed percentage points.','',
        'The primary table uses exactly the same stock-days with verifiable exits under every compared policy. This avoids ranking exits on different missing-data cohorts, but the shared cohort is selected for observability and may not generalize to all signals. Policy-specific verifiable and observed-path results are exported separately. Observed-path results may use gaps and are sensitivity analyses only.','',
        '## Same-cohort gross results','',
        '| Exit policy | Trades | Wins | Losses | Win rate | Mean return | Average win | Average loss | Best | Worst | Ambiguous bars |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in results[results.cohort.eq('matched_verifiable')].itertuples():
        report.append(f'| {r.policy} | {r.trades} | {r.wins} | {r.losses} | {r.win_rate_pct:.2f}% | {r.mean_gross_pct:.2f}% | {r.average_gain_pct:.2f}% | {r.average_loss_pct:.2f}% | {r.max_gain_pct:.2f}% | {r.max_loss_pct:.2f}% | {r.ambiguous_bars} |')
    report+=['','TP5 = take profit at +5%; SLn = fixed stop at -n%; all untriggered positions exit at close. No trailing stops. Stop prices are not guaranteed fills, especially during halts, gaps or thin markets. Positive gross returns are not evidence of executable net profit.','',
        '## Data reconciliation','',json.dumps(reconciliation),'',
        'Different API retrieval times and vendor revisions can change observed histories. Path provenance and a trade-level comparison with the earlier no-stop calculation are saved. Mean returns equally weight trades; they are not compounded or portfolio returns. Confidence intervals resample signal dates and do not capture all model-selection/regime uncertainty.','',
        'Files: metrics.csv, trades.csv/parquet, signals.csv, path_provenance.csv, baseline_reconciliation.csv, errors.json and plan.json.']
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n')
    assert len(trades)==len(signals)*len(POLICIES)
    assert not trades.duplicated(['date','ticker','identity','policy']).any()
    assert results[results.cohort.eq('matched_verifiable')].trades.nunique()==1
    assert (results.wins+results.losses+results.breakeven).eq(results.trades).all()
    print(results[results.cohort.eq('matched_verifiable')].to_string(index=False),flush=True)
    print('POLICY SPECIFIC\n',results[results.cohort.eq('policy_verifiable')].to_string(index=False),flush=True)
    print('Reconciliation:',reconciliation,flush=True)
    print(OUT/'REPORT.md',flush=True)

if __name__=='__main__':main()
