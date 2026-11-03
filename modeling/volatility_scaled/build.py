"""Prior-only same-window activity baselines and volatility-scaled excursion targets."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from urllib.parse import quote
import time
import numpy as np
import pandas as pd
import exchange_calendars as xcals
from runner.api import API
from runner.collect import COLUMNS,normalize,save
from runner.engine import window

OUT=Path('modeling/volatility_scaled/data')
SOURCE=Path('data/master_open15_1m_prior_complete')
KEYS=['date','ticker','identity']


def history_dates(day,cal,n=20):
    return [str(cal.session_offset(pd.Timestamp(day),-i).date()) for i in range(n,0,-1)]


def window_stats(b,lo,hi):
    s=window(b,lo,hi)
    if s.empty:return dict(volume=0.,rv_pct=np.nan,range_pct=np.nan,bars=0)
    rv=100*np.sqrt(np.log(s.close).diff().pow(2).sum()) if len(s)>=2 else np.nan
    return dict(volume=float(s.volume.sum()),rv_pct=float(rv),range_pct=float(100*(s.high.max()/s.low.min()-1)),bars=len(s))


def day_stats(b,day,cal):
    c=cal.session_close(pd.Timestamp(day)).tz_convert('America/New_York');end=c.hour*60+c.minute
    d={'date':day,'observed_day_bars':len(b)}
    for name,lo,hi in [('pm',240,570),('pm_last30',540,570),('open15',570,585)]:
        d.update({f'{name}_{k}':v for k,v in window_stats(b,lo,hi).items()})
    entry=b[b.minute.eq(585)];last=b[b.minute.eq(end-1)];post=window(b,585,end)
    d['entry_close_log_return']=float(np.log(last.close.iloc[0]/entry.open.iloc[0])) if len(entry) and len(last) else np.nan
    d['post_complete']=bool(len(entry) and len(post)==end-585)
    return d


def prior_features(row,hist):
    """Every row of hist must precede the observation; no expanding current-day fit."""
    if len(hist) and not hist.date.lt(row['date']).all():raise ValueError('Current/future session in historical baseline')
    output={'history_sessions_matched':len(hist)}
    for name in ('pm','pm_last30','open15'):
        for metric,current_field in [('volume','volume'),('rv_pct','realized_vol_pct'),('range_pct','range_pct')]:
            values=pd.to_numeric(hist[f'{name}_{metric}'],errors='coerce').dropna()
            current=float(row[f'{name}_{current_field}'])
            prefix=f'hist20_{name}_{metric}'
            output[prefix+'_count']=len(values)
            mean=values.mean() if len(values) else np.nan
            std=values.std(ddof=1) if len(values)>=2 else np.nan
            output[prefix+'_mean']=float(mean)
            output[prefix+'_ratio']=float(current/mean) if mean>0 and len(values)>=15 else np.nan
            # log-volume z-score limits domination by a few extreme historical days.
            transform=np.log1p if metric=='volume' else lambda x:x
            transformed=transform(values);transformed_std=transformed.std(ddof=1)
            output[prefix+'_z']=float((transform(current)-transformed.mean())/transformed_std) if len(values)>=15 and transformed_std>1e-12 else np.nan
        output[f'hist20_{name}_mean_bars']=float(hist[f'{name}_bars'].mean()) if len(hist) else np.nan
    returns=hist.entry_close_log_return.dropna()
    sigma=returns.std(ddof=1)
    output['historical_return_count']=len(returns)
    output['sigma20_log']=float(sigma) if len(returns)>=15 and sigma>1e-6 else np.nan
    output['historical_return_mean_log']=float(returns.mean()) if len(returns) else np.nan
    return output


def adjust_volume_history(hist,events,asof):
    hist=hist.copy()
    for index,row in hist.iterrows():
        factor=np.prod([float(e['split_to'])/float(e['split_from']) for e in events if row.date < e['execution_date'] <= asof])
        for name in ('pm','pm_last30','open15'):
            hist.loc[index,f'{name}_volume'] *= factor
    return hist


def main():
    OUT.mkdir(parents=True,exist_ok=True);cache=OUT/'history_parts';cache.mkdir(exist_ok=True)
    api=API(OUT/'request_cache');cal=xcals.get_calendar('XNYS')
    f=pd.read_parquet(SOURCE/'features.parquet');labels=pd.read_parquet(SOURCE/'labels.parquet');labels=labels[labels.horizon.eq('close')]
    dates={d:history_dates(d,cal) for d in f.date.unique()}
    needed=defaultdict(set)
    for r in f.itertuples():needed[r.ticker].update(dates[r.date])
    identities={}
    wanted=set(f.ticker)
    for p in Path('data/master/daily').glob('*.parquet'):
        h=pd.read_parquet(p,columns=['ticker','identity']);h=h[h.ticker.isin(wanted)]
        identities.update({(p.stem,r.ticker):r.identity for r in h.itertuples()})
    missing_ref={}
    for r in f.itertuples():
        for day in dates[r.date]:
            if (day,r.ticker) not in identities:
                if (Path('data/master/daily')/f'{day}.parquet').exists():
                    identities[(day,r.ticker)]=None
                else:
                    missing_ref[(day,r.ticker)]=None
    for (day,ticker) in missing_ref:
        try:
            result=api.get(f'/v3/reference/tickers/{quote(ticker,safe="")}',date=day).get('results',{})
        except RuntimeError as exc:
            if 'HTTP 404' not in str(exc):raise
            result={}
        identities[(day,ticker)]=next((result[k] for k in ('share_class_figi','composite_figi') if result.get(k)),ticker) if result else None
    jobs=[]
    for ticker,days in sorted(needed.items()):
        part=[]
        for day in sorted(days):
            if part and len(cal.sessions_in_range(part[0],day))>20:
                jobs.append((ticker,part));part=[]
            part.append(day)
        if part:jobs.append((ticker,part))
    summaries=[];errors=[];started=time.monotonic()
    print(f'{len(f)} target rows, {len(needed)} tickers, {len(jobs)} historical requests; {len(missing_ref)} historical reference lookups',flush=True)
    for i,(ticker,days) in enumerate(jobs,1):
        token=hashlib.sha256((ticker+'|'+'|'.join(days)).encode()).hexdigest()[:24]
        file=cache/f'{token}.parquet'
        if file.exists():summary=pd.read_parquet(file)
        else:
            try:
                response=api.pages(f'/v2/aggs/ticker/{quote(ticker,safe="")}/range/1/minute/{days[0]}/{days[-1]}',adjusted='false',sort='asc',limit=50000)
                if len(response)>=50000:raise ValueError('Aggregate response may be truncated')
                records=[dict(ticker=ticker,volume=x['v'],open=x['o'],close=x['c'],high=x['h'],low=x['l'],window_start=int(x['t'])*1000000,transactions=x.get('n',np.nan)) for x in response]
                b=normalize(pd.DataFrame(records,columns=COLUMNS))
                b['date']=pd.to_datetime(b.window_start,unit='ns',utc=True).dt.tz_convert('America/New_York').dt.strftime('%Y-%m-%d')
                rows=[]
                for day in days:
                    bars=b[b.date.eq(day)]
                    # A wholly empty response cannot establish a zero-volume trading day.
                    if bars.empty:continue
                    rows.append(dict(ticker=ticker,identity=identities.get((day,ticker)),**day_stats(bars,day,cal)))
                summary=pd.DataFrame(rows) if rows else pd.DataFrame(columns=['ticker','identity','date'])
                save(summary,file)
                for p in api.cache.glob('*.json'):p.unlink()
            except Exception as exc:
                errors.append(dict(ticker=ticker,first=days[0],last=days[-1],error=type(exc).__name__));continue
        summaries.append(summary)
        if i%25==0 or i==len(jobs):print(f'History {i}/{len(jobs)}; failures {len(errors)}; elapsed {(time.monotonic()-started)/60:.1f} min',flush=True)
    history=pd.concat(summaries,ignore_index=True).drop_duplicates(['date','ticker'])
    save(history,OUT/'history_summaries.parquet')
    finalize(f,labels,history,errors,api,cal)


def finalize(f,labels,history,errors,api,cal):
    dates={d:history_dates(d,cal) for d in f.date.unique()}
    splits=api.pages('/stocks/v1/splits',**{'execution_date.gte':min(history.date),'execution_date.lte':str(f.date.max()),'limit':1000})
    (OUT/'split_events.json').write_text(json.dumps(splits,indent=2)+'\n')
    split_by_ticker=defaultdict(list)
    for event in splits:split_by_ticker[event['ticker']].append(event)
    grouped={t:g for t,g in history.groupby('ticker')};rows=[]
    for row in f.to_dict('records'):
        hist=grouped.get(row['ticker'],pd.DataFrame(columns=history.columns))
        hist=hist[hist.date.isin(dates[row['date']])&hist.identity.eq(row['identity'])].copy()
        hist=adjust_volume_history(hist,split_by_ticker[row['ticker']],row['date'])
        rows.append({**row,**prior_features(row,hist)})
    enriched=pd.DataFrame(rows);enriched=enriched.replace([np.inf,-np.inf],np.nan)
    d=enriched.merge(labels,on=KEYS,validate='one_to_one')
    known=d.entry_available & d.missing_minutes.eq(0) & d.sigma20_log.notna()
    d['up_sigma']=np.where(known,np.log1p(d.max_gain_pct/100)/d.sigma20_log,np.nan)
    d['down_sigma']=np.where(known,np.log1p(d.max_decline_pct/100)/d.sigma20_log,np.nan)
    save(enriched,OUT/'features.parquet');save(d[KEYS+['sigma20_log','up_sigma','down_sigma','max_gain_pct','max_decline_pct','end_return_pct','runner_5pct','entry_available','missing_minutes']],OUT/'targets.parquet')
    manifest=dict(source=str(SOURCE.resolve()),source_sha256={name:hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() for name in ('features.parquet','labels.parquet')},rows=len(f),rows_with_scale=int(enriched.sigma20_log.notna().sum()),complete_excursion_targets=int(known.sum()),history_sessions=20,minimum_valid_history=15,
        scale='Sample standard deviation of prior same-stock 09:45-open to regular-close log returns; exact entry/final candles required, interior gaps do not affect terminal return.',
        targets='up_sigma = log(maximum post-09:45 high / entry) / sigma20; down_sigma = log(minimum post-09:45 low / entry) / sigma20. Only complete future paths establish exact excursions.',
        features='Prior-only same-window means, ratios, z-scores; volume z-scores use log1p. Windows PM 04:00–09:30, last30 09:00–09:30, open15 09:30–09:45 ET.',
        limitations='Missing historical candles can understate observed activity; coverage counts are explicit. Sparse/zero-variance baselines produce NaN, not fabricated z-scores. Historical API responses can be retrospectively revised. +/-1 sigma is a scale unit, not a normality or probability guarantee.',errors=errors)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2),flush=True)

if __name__=='__main__':main()
