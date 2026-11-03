"""Rebuild an opening-price occupancy cohort without opening-window selection leakage."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib,json,os,shutil,time
from pathlib import Path
import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from runner.collect import COLUMNS,save

OUT=Path('modeling/open_occupancy/data')
DAILY=Path('data/master/daily')
CUTS={'delayed_0914':554,'realtime_0929':569}
KEYS=['date','ticker','identity']


def aggregate_window(b,lo,hi):
    s=b[b.minute.ge(lo)&b.minute.lt(hi)].sort_values(['ticker','window_start']).copy()
    if s.empty:return pd.DataFrame()
    s['dollars']=(s.high+s.low+s.close)/3*s.volume
    s['up_volume']=s.volume.where(s.close>s.open,0.)
    s['rv2']=np.log(s.close).groupby(s.ticker).diff().pow(2)
    g=s.groupby('ticker',sort=False).agg(volume=('volume','sum'),transactions=('transactions','sum'),bars=('minute','size'),
        first=('open','first'),last=('close','last'),high=('high','max'),low=('low','min'),last_minute=('minute','last'),
        dollars=('dollars','sum'),up_volume=('up_volume','sum'),rv2=('rv2','sum'),invalid=('invalid','sum'))
    g['return_pct']=100*(g['last']/g['first']-1);g['range_pct']=100*(g.high/g.low-1)
    g['rv_pct']=100*np.sqrt(g.rv2);g['vwap']=g.dollars/g.volume.replace(0,np.nan)
    g['close_to_vwap_pct']=100*(g['last']/g.vwap-1)
    g['close_location']=(g['last']-g.low)/(g.high-g.low).replace(0,np.nan)
    g['up_volume_fraction']=g.up_volume/g.volume.replace(0,np.nan)
    g['last_age_minutes']=hi-g.last_minute
    g['shares_per_transaction']=g.volume/g.transactions.replace(0,np.nan)
    return g.drop(columns=['up_volume','rv2','last_minute'])


def opening_targets(b):
    s=b[b.minute.ge(570)&b.minute.lt(585)].sort_values(['ticker','window_start']).copy()
    if s.empty:return pd.DataFrame()
    entry=s[s.minute.eq(570)].drop_duplicates('ticker',keep=False).set_index('ticker').open
    s['entry']=s.ticker.map(entry)
    s['above']=s.close>s.entry;s['entire_above']=s.low>s.entry
    g=s.groupby('ticker').agg(observed=('minute','nunique'),rows=('minute','size'),above=('above','sum'),entire_above=('entire_above','sum'),last_close=('close','last'),invalid=('invalid','sum'))
    g['entry_price']=entry
    exact=g.observed.eq(15)&g.rows.eq(15)&g.entry_price.notna()&g.invalid.eq(0)
    g['above_pct']=np.where(exact,100*g.above/15,np.nan)
    g['entire_above_pct']=np.where(exact,100*g.entire_above/15,np.nan)
    g['end_return_pct']=np.where(exact,100*(g.last_close/g.entry_price-1),np.nan)
    return g


def summarize(b,master,day,close_min):
    out=master[['ticker','identity','reg_close','reg_volume']].rename(columns={'reg_close':'official_close','reg_volume':'official_volume'}).set_index('ticker')
    reg=b[b.minute.ge(570)&b.minute.lt(close_min)]
    coverage=reg.groupby('ticker').agg(reg_rows=('window_start','size'),reg_unique=('window_start','nunique'),reg_invalid=('invalid','sum'),
        reg_open=('open','first'),reg_last=('close','last'),reg_high=('high','max'),reg_low=('low','min'),reg_first_minute=('minute','min'),reg_last_minute=('minute','max'))
    coverage['reg_return_pct']=(100*(coverage.reg_last/coverage.reg_open-1)).where(coverage.reg_first_minute.eq(570)&coverage.reg_last_minute.eq(close_min-1))
    coverage['reg_range_pct']=100*(coverage.reg_high/coverage.reg_low-1)
    coverage['reg_expected']=close_min-570
    coverage['reg_complete']=coverage.reg_unique.eq(close_min-570)&coverage.reg_rows.eq(close_min-570)&coverage.reg_invalid.eq(0)
    out=out.join(coverage)
    for phase,cut in CUTS.items():
        for name,lo in [('pm',240),('last30',cut-30),('last15',cut-15),('last5',cut-5)]:
            stats=aggregate_window(b,lo,cut)
            out=out.join(stats.add_prefix(phase+'_'+name+'_'))
        for name in ('pm','last30','last15','last5'):
            for field in ('volume','bars','transactions','dollars'):
                col=phase+'_'+name+'_'+field
                if col in out:out[col]=out[col].fillna(0.)
    target=opening_targets(b).add_prefix('y_')
    out=out.join(target).copy();out['date']=day
    return out.reset_index()


def day_job(day,cache,s3,close_min):
    file=cache/f'{day}.parquet'
    if file.exists():return day
    master=pd.read_parquet(DAILY/f'{day}.parquet')
    key=f'us_stocks_sip/minute_aggs_v1/{day[:4]}/{day[5:7]}/{day}.csv.gz'
    meta=s3.head_object(Bucket='flatfiles',Key=key)
    if shutil.disk_usage(cache).free<max(2_000_000_000,8*meta['ContentLength']):raise RuntimeError('Insufficient disk space')
    raw=cache/f'{day}.csv.gz'
    if not raw.exists() or raw.stat().st_size!=meta['ContentLength']:
        s3.download_file('flatfiles',key,str(raw),Config=TransferConfig(use_threads=False))
    b=pd.read_csv(raw,usecols=COLUMNS,keep_default_na=False)
    b=b[b.ticker.isin(set(master.ticker)|{'SPY','QQQ'})].copy()
    midnight=pd.Timestamp(day,tz='America/New_York').value
    b['minute']=((b.window_start-midnight)//60_000_000_000).astype(int)
    b=b.sort_values(['ticker','window_start'])
    b['invalid']=((b.window_start%60_000_000_000!=0)|(b[['open','close','high','low']].le(0).any(axis=1))|b.volume.lt(0)|
        b.high.lt(b[['open','close','low']].max(axis=1))|b.low.gt(b[['open','close','high']].min(axis=1)))
    # Retain benchmark rows in the compact summary, but never in the stock opportunity set.
    extra=pd.DataFrame({'ticker':['SPY','QQQ'],'identity':['benchmark_SPY','benchmark_QQQ'],'reg_close':[np.nan]*2,'reg_volume':[np.nan]*2})
    summary=summarize(b,pd.concat([master,extra],ignore_index=True),day,close_min)
    save(summary,file)
    provenance=dict(day=day,key=key,etag=meta.get('ETag'),bytes=meta['ContentLength'],retrieved_unix=time.time())
    file.with_suffix('.json').write_text(json.dumps(provenance,indent=2)+'\n');raw.unlink()
    return day


def assemble(days,cache,splits,target_dates):
    result=[];audit=[];history=[]
    for day in days:
        current=pd.read_parquet(cache/f'{day}.parquet')
        if day in target_dates and len(history)==10:
            prior=history[-1][1].set_index(['ticker','identity'])
            for phase,cut in CUTS.items():
                prefix=phase+'_'
                stack=[]
                for hd,frame in history:
                    h=frame.copy()
                    events=splits[(splits.execution_date>hd)&(splits.execution_date<=day)] if not splits.empty else pd.DataFrame()
                    if not events.empty:
                        factor=h.ticker.map((events.split_to/events.split_from).groupby(events.ticker).prod()).fillna(1.)
                    else:factor=pd.Series(1.,index=h.index)
                    h[prefix+'pm_volume']*=factor;h['official_volume']*=factor;h['official_close']/=factor
                    stack.append(h)
                hist=pd.concat(stack,ignore_index=True)
                hist['log_pm_volume']=np.log1p(hist[prefix+'pm_volume'])
                grouped=hist.groupby(['ticker','identity'])
                baseline=grouped.agg(history_count=('date','size'),prior10_pm_volume=(prefix+'pm_volume','mean'),
                    prior10_log_pm_volume_mean=('log_pm_volume','mean'),prior10_log_pm_volume_std=('log_pm_volume','std'),
                    prior10_pm_rv_mean=(prefix+'pm_rv_pct','mean'),prior10_pm_rv_std=(prefix+'pm_rv_pct','std'),
                    prior10_daily_volume=('official_volume','mean'),prior10_regular_return_mean=('reg_return_pct','mean'),
                    prior10_regular_return_std=('reg_return_pct','std'),prior10_opening_above_mean=('y_above_pct','mean'),prior10_opening_above_count=('y_above_pct','count'))
                prev=stack[-1].set_index(['ticker','identity'])
                candidates=current.set_index(['ticker','identity']).join(baseline)
                candidates=candidates.join(prev[['official_close','reg_complete','reg_return_pct','reg_range_pct','y_above_pct']].add_suffix('_prior'))
                eligible=candidates.reg_complete_prior.eq(True)&candidates.history_count.eq(10)&candidates.prior10_pm_volume.gt(0)&candidates.official_close_prior.gt(0)
                gap=100*(candidates[prefix+'pm_last']/candidates.official_close_prior-1)
                rvol=candidates[prefix+'pm_volume']/candidates.prior10_pm_volume
                eligible&=gap.ge(10-1e-10)&rvol.ge(5-1e-10)
                eligible&=candidates[prefix+'pm_invalid'].eq(0)
                eligible&=~candidates.index.get_level_values('ticker').isin(['SPY','QQQ'])
                selected=candidates[eligible]
                audit.append(dict(date=day,phase=phase,universe=len(current)-2,eligible=len(selected),known_targets=int(selected.y_above_pct.notna().sum())))
                market=current.set_index('ticker')
                for (ticker,identity),row in selected.iterrows():
                    values={'date':day,'ticker':ticker,'identity':identity,'phase':phase,'feature_cutoff_utc':(pd.Timestamp(day,tz='America/New_York')+pd.Timedelta(minutes=cut)).tz_convert('UTC').isoformat(),
                        'pm_gap_pct':float(gap.loc[(ticker,identity)]),'pm_rvol':float(rvol.loc[(ticker,identity)]),'prior_close':row.official_close_prior,
                        'prior_regular_return_pct':row.reg_return_pct_prior,'prior_regular_range_pct':row.reg_range_pct_prior,'prior_opening_above_pct':row.y_above_pct_prior}
                    for col in current:
                        if col.startswith(prefix):values[col[len(prefix):]]=row[col]
                    for col in baseline:
                        if col!='history_count':values[col]=row[col]
                    values['pm_volume_z']=(np.log1p(row[prefix+'pm_volume'])-row.prior10_log_pm_volume_mean)/row.prior10_log_pm_volume_std if row.prior10_log_pm_volume_std>1e-12 else np.nan
                    values['pm_rv_ratio']=row[prefix+'pm_rv_pct']/row.prior10_pm_rv_mean if row.prior10_pm_rv_mean>0 else np.nan
                    values['pm_last5_volume_fraction']=row[prefix+'last5_volume']/row[prefix+'pm_volume'] if row[prefix+'pm_volume']>0 else np.nan
                    for benchmark in ('SPY','QQQ'):
                        for win in ('pm','last30','last15','last5'):
                            value=market.loc[benchmark,prefix+win+'_return_pct']
                            values[f'market_{benchmark}_{win}_return_pct']=value
                            values[f'market_excess_{benchmark}_{win}_return_pct']=values.get(win+'_return_pct',np.nan)-value
                    for col in ('above_pct','entire_above_pct','entry_price','observed','end_return_pct','above'):
                        values['target_'+col]=row['y_'+col]
                    result.append(values)
        history.append((day,current));history=history[-10:]
    return pd.DataFrame(result).replace([np.inf,-np.inf],np.nan),pd.DataFrame(audit)


def main():
    import exchange_calendars as xcals
    from runner.api import API
    OUT.mkdir(parents=True,exist_ok=True);cache=OUT/'daily';cache.mkdir(exist_ok=True)
    calendar=xcals.get_calendar('XNYS');reference=json.loads(Path('modeling/results_regularized_5pct_close/manifest.json').read_text())
    dates=set(sum(reference['split_dates'].values(),[]))|set(reference['embargo_dates'])
    days=sorted(p.stem for p in DAILY.glob('*.parquet') if p.stem<=max(dates))
    load_dotenv();s3=boto3.client('s3',endpoint_url='https://files.massive.com',aws_access_key_id=os.environ['MASSIVE_FLATFILES_ACCESS_KEY_ID'],aws_secret_access_key=os.environ['MASSIVE_FLATFILES_SECRET_ACCESS_KEY'],config=Config(retries={'max_attempts':8,'mode':'adaptive'},max_pool_connections=2))
    start=time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(day_job,day,cache,s3,(lambda c:c.hour*60+c.minute)(calendar.session_close(pd.Timestamp(day)).tz_convert('America/New_York'))):day for day in days}
        for i,f in enumerate(as_completed(futures),1):
            f.result()
            if i%10==0 or i==len(days):print(f'Pre-open rebuild {i}/{len(days)} sessions; elapsed {(time.monotonic()-start)/60:.1f} min',flush=True)
    api=API(OUT/'requests');events=api.pages('/stocks/v1/splits',**{'execution_date.gte':min(days),'execution_date.lte':max(days),'limit':1000})
    splits=pd.DataFrame(events);(OUT/'split_events.json').write_text(json.dumps(events,indent=2)+'\n')
    data,audit=assemble(days,cache,splits,dates);save(data,OUT/'dataset.parquet');audit.to_csv(OUT/'eligibility_audit.csv',index=False)
    manifest=dict(rows=len(data),phases=data.groupby('phase').agg(rows=('ticker','size'),known=('target_above_pct','count'),tickers=('ticker','nunique')).reset_index().to_dict('records'),
        target='100 * number of closes strictly above 09:30 minute open / 15, in [09:30,09:45). Exact label requires every candle and observed 09:30 open. Opening price is label-only, never an input.',
        eligibility='Same-cutoff premarket gap >=10%, premarket volume >=5 times previous10 matched sessions at same cutoff; same identity and complete prior regular-session grid. No current opening turnover/coverage condition selects stocks.',
        phases_explained={'delayed_0914':'Pre-09:14 completed candles; 15-minute feed delay leaves approximately one minute for pre-open decision. Hypothetical timing, actual arrival/broker fill not reconstructed.','realtime_0929':'Pre-09:29 completed candles; requires timely real-time data. Not usable at open from a 15-minute delayed feed.'},
        split_dates=reference['split_dates'],embargo_dates=reference['embargo_dates'],raw_sessions=len(days),note='Original first15-minute volume filters were removed to avoid selection leakage. Historical reference data and bars can contain retrospective corrections. All original research test calendar dates have already been inspected.')
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2),flush=True)

if __name__=='__main__':main()
