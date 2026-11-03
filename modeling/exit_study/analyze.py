"""Exit-policy research on the frozen high-precision model's validation/test signals."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import exchange_calendars as xcals
from threadpoolctl import threadpool_limits
from modeling.core import cluster_interval
from runner.api import API
from runner.collect import rest_day,save

ROOT=Path('modeling/exit_study')
POLICIES=[('close_only',None,None,'close'),('tp5_or_close',5,None,'close'),('tp10_or_close',10,None,'close'),('tp15_or_close',15,None,'close'),('tp20_or_close',20,None,'close'),
          ('tp5_sl5_close',5,5,'close'),('tp5_sl10_close',5,10,'close'),('tp10_sl5_close',10,5,'close'),('tp10_sl10_close',10,10,'close'),
          ('tp15_sl5_close',15,5,'close'),('tp20_sl10_close',20,10,'close'),('time60_only',None,None,60),('time120_only',None,None,120),
          ('tp5_sl5_time60',5,5,60),('tp10_sl5_time120',10,5,120)]

def simulate(b,tp,sl,horizon,close_min):
    entry=b[b.minute.eq(585)]
    base={'entry':np.nan,'exit':np.nan,'gross_pct':np.nan,'reason':'missing_entry','exit_minute':np.nan,'ambiguous_bar':False,'missing_before_exit':True}
    if entry.empty:return base
    entry=float(entry.open.iloc[0]);end=close_min if horizon=='close' else min(585+horizon,close_min)
    target=entry*(1+tp/100) if tp is not None else np.inf
    stop=entry*(1-sl/100) if sl is not None else -np.inf
    epsilon=entry*1e-10
    w=b[(b.minute>=585)&(b.minute<end)].sort_values('minute')
    seen=set();price=None;reason=None;amb=False;minute=None
    for r in w.itertuples():
        seen.add(int(r.minute))
        # The bar open occurs before its intraminute extremes. Stop gaps fill worse.
        if r.open<=stop+epsilon:price=float(r.open);reason='stop_gap';minute=int(r.minute);break
        if r.open>=target-epsilon:price=float(target);reason='target';minute=int(r.minute);break
        hitstop=r.low<=stop+epsilon;hittarget=r.high>=target-epsilon
        if hitstop or hittarget:
            amb=bool(hitstop and hittarget)
            price=float(stop if hitstop else target);reason='stop' if hitstop else 'target';minute=int(r.minute);break
    if price is None:
        if end<close_min:
            closing=b[b.minute.eq(end)]
            price=float(closing.open.iloc[0]) if len(closing) else np.nan
        else:
            closing=b[b.minute.eq(end-1)]
            price=float(closing.close.iloc[0]) if len(closing) else np.nan
        reason='timed_exit' if pd.notna(price) else 'missing_timed_exit';minute=end
    # For unconditional time exits, interior gaps cannot change the exit rule.
    # A static take-profit without a stop has the same modeled exit price
    # even if it was touched earlier in an unobserved minute.
    path_matters=sl is not None or (tp is not None and reason!='target')
    missing=bool(path_matters and len(seen)<min(minute+1,end)-585)
    return {'entry':entry,'exit':price,'gross_pct':100*(price/entry-1),'reason':reason,'exit_minute':minute,'ambiguous_bar':amb,'missing_before_exit':missing}

def main():
    ROOT.mkdir(exist_ok=True)
    model=joblib.load('modeling/results/models/hclose_gain5.joblib')
    manifest=json.loads(Path('modeling/results/manifest.json').read_text())
    features=pd.read_parquet('data/master/features.parquet');features=features[features.opening_missing_minutes.eq(0)].copy()
    selected=[]
    for period in ['threshold','test']:
        f=features[features.date.isin(manifest['split_dates'][period])].copy()
        score=model['estimator'].predict_proba(f[model['features']].replace([np.inf,-np.inf],np.nan))[:,1]
        f=f[score>=model['policy']['cutoff']].copy();f['period']=period
        selected.append(f[['date','ticker','identity','period']])
    signals=pd.concat(selected,ignore_index=True)
    signals.to_csv(ROOT/'signals.csv',index=False)
    (ROOT/'policy_plan.json').write_text(json.dumps({'model':'hclose_gain5','cutoff':model['policy']['cutoff'],'policies':POLICIES,'selection':'Validation signals only; maximize mean net return at 0.5% round-trip cost on one common set of verifiable paths. Bootstrap uncertainty and observed-path sensitivity are disclosed.','test_warning':'The model-study test period has already been viewed. Exit comparisons remain exploratory; no fresh independent strategy holdout.','assumptions':'Entry at 09:45 bar open; static target filled at target when touched; stop gap at bar open, otherwise stop level; if both barriers crossed within one minute, stop first. Time exit at end-minute open or final regular-session close. No order-book, halt, spread or market-impact reconstruction.'},indent=2))
    api=API(ROOT/'request_cache');cal=xcals.get_calendar('XNYS');paths=ROOT/'paths';paths.mkdir(exist_ok=True)
    rows=[]
    for i,s in signals.iterrows():
        filename=paths/f"{s.date}_{s.ticker}.parquet"
        if filename.exists():b=pd.read_parquet(filename)
        else:
            b=rest_day(s.date,[s.ticker],api);b=b[b.minute.ge(585)].copy();save(b,filename)
            for p in api.cache.glob('*.json'):p.unlink()
        close=cal.session_close(pd.Timestamp(s.date)).tz_convert('America/New_York');close_min=close.hour*60+close.minute
        for name,tp,sl,horizon in POLICIES:
            rows.append({**s.to_dict(),'policy':name,**simulate(b,tp,sl,horizon,close_min)})
        print(f'{i+1}/{len(signals)} {s.date} {s.ticker}',flush=True)
    trades=pd.DataFrame(rows)
    trades['verifiable']=trades.gross_pct.notna()&~trades.missing_before_exit
    trades.to_parquet(ROOT/'trade_results.parquet',index=False,compression='zstd')
    # Matched cohorts remove differing-data-coverage effects from policy ranking.
    common=trades.groupby(['period','date','ticker']).verifiable.all().rename('common_verifiable').reset_index()
    trades=trades.merge(common,on=['period','date','ticker'])
    results=[]
    for (period,policy),d in trades.groupby(['period','policy']):
        for cohort,subset in [('observed_paths',d[d.gross_pct.notna()]),('matched_verifiable',d[d.common_verifiable])]:
            x=subset.gross_pct.to_numpy();net=x-.5
            rng=np.random.default_rng(731)
            day=subset.assign(net=net).groupby('date').agg(total=('net','sum'),n=('net','size'))
            if len(day):
                vals=day.to_numpy();sums=vals[rng.integers(0,len(vals),size=(5000,len(vals)))].sum(axis=1);boot=sums[:,0]/sums[:,1];ci=np.quantile(boot,[.025,.975])
            else:ci=[np.nan,np.nan]
            results.append({'period':period,'policy':policy,'cohort':cohort,'signals':len(d),'trades_evaluated':len(x),'unverifiable_trades':int((~d.verifiable).sum()),'missing_path_trades_used':int(subset.missing_before_exit.sum()),'ambiguous_bars':int(subset.ambiguous_bar.sum()),'mean_gross_pct':float(np.mean(x)) if len(x) else np.nan,'mean_net_50bps_pct':float(np.mean(net)) if len(x) else np.nan,'median_net_50bps_pct':float(np.median(net)) if len(x) else np.nan,'win_rate_net':float((net>0).mean()) if len(x) else np.nan,'worst_net_pct':float(np.min(net)) if len(x) else np.nan,'day_bootstrap_low':ci[0],'day_bootstrap_high':ci[1], 'mean_net_100bps_pct':float(np.mean(x-1)) if len(x) else np.nan,'mean_net_200bps_pct':float(np.mean(x-2)) if len(x) else np.nan})
    result=pd.DataFrame(results);result.to_csv(ROOT/'exit_comparison.csv',index=False)
    eligible=result[(result.period=='threshold')&result.cohort.eq('matched_verifiable')]
    winner=eligible.sort_values('mean_net_50bps_pct',ascending=False).iloc[0]
    (ROOT/'selected_on_validation.json').write_text(json.dumps(winner.to_dict(),indent=2))
    print('Validation-selected exit:',winner.policy,flush=True)
    report(result,signals,winner)

def report(result,signals,winner):
    text=['# Exploratory exit-policy study','',f"Frozen entry model: Extra Trees, +5% by close, cutoff 0.905710. Validation signals: {sum(signals.period.eq('threshold'))}; previously viewed test signals: {sum(signals.period.eq('test'))}.",
          '', '**This is an observed-minute-bar simulation, not an executable backtest.** Original model precision measures threshold touches, not trade profits. The model-study holdout was already viewed before this new exit experiment. Exits are ranked on the earlier threshold-validation period; a fresh future paper-trading period is still required.',
          '', f'Validation-selected policy on the common verifiable cohort: **{winner.policy}**. The common cohort requires that every compared policy can be evaluated without a missing pre-exit minute; this reduces the sample sharply and introduces liquidity selection. The larger observed-path cohort is a sensitivity analysis only.',
          '', 'Round-trip cost assumption: 0.50 percentage points per position, subtracted from gross return. This is a scenario, not measured execution cost. Also saved: 1% and 2% round-trip cost sensitivities.',
          '', '| Period | Cohort | Exit rule | Trades | Mean gross | Mean net (0.5% cost) | Median net | Net win rate | Worst net | 95% mean net interval |', '|---|---|---|---:|---:|---:|---:|---:|---:|---|']
    for _,r in result.sort_values(['period','cohort','policy']).iterrows():
        text.append(f'| {r.period} | {r.cohort} | {r.policy} | {r.trades_evaluated} / {r.signals} | {r.mean_gross_pct:.2f}% | {r.mean_net_50bps_pct:.2f}% | {r.median_net_50bps_pct:.2f}% | {100*r.win_rate_net:.1f}% | {r.worst_net_pct:.2f}% | {r.day_bootstrap_low:.2f} to {r.day_bootstrap_high:.2f}% |')
    text+=['','Policy naming: `tp5` = +5% take profit, `sl5` = -5% fixed stop, `time60` = exit at 10:45 ET, `time120` = 11:45 ET, `close` = regular-session close. Target and stop levels are relative to entry; no overnight positions.',
           '', 'Gap handling: when a bar opens below the stop, the model exits at that worse opening price. Both barriers inside one minute are resolved stop-first. Missing entry or final exit prices remain unknown. Trades with missing minutes before a conditional exit cannot establish barrier ordering and are flagged. Minute timestamps do not reproduce live order latency or queue priority.',
           '', 'Interpretation: mean trade return equally weights positions; it is not a portfolio return, annualized return, or capacity estimate. No live bid/ask quotes, broker fees, liquidity, halts, partial fills, or market impact were modeled. Few signals and many unverified paths prevent a claim that one exit is best.',
           '', 'Reproduce: `python -m modeling.exit_study.analyze`. Small targeted minute paths are cached locally as compressed Parquet to avoid repeat downloads; broad raw market files are not retained.']
    (ROOT/'REPORT.md').write_text('\n'.join(text)+'\n')

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
