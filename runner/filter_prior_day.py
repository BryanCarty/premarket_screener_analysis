"""Reprocess Massive minute files to require a complete immediately prior regular session."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from dotenv import load_dotenv
import exchange_calendars as xcals
import numpy as np
import pandas as pd
from .collect import save
from .validate import validate


def atomic(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def count_day(day,tickers,opening,closing,cache,s3):
    report=cache/f'{day}.json'
    signature=hashlib.sha256('\n'.join(sorted(tickers)).encode()).hexdigest()
    if report.exists():
        result=json.loads(report.read_text())
        if result['ticker_signature']!=signature:raise ValueError('Coverage cache belongs to a different source population')
        return result
    key=f'us_stocks_sip/minute_aggs_v1/{day[:4]}/{day[5:7]}/{day}.csv.gz'
    meta=s3.head_object(Bucket='flatfiles',Key=key)
    if shutil.disk_usage(cache).free < max(2_000_000_000,meta['ContentLength']*8):raise RuntimeError('Insufficient free disk space')
    raw=cache/f'{day}.csv.gz'
    if not raw.exists() or raw.stat().st_size!=meta['ContentLength']:
        s3.download_file('flatfiles',key,str(raw),Config=TransferConfig(use_threads=False))
    lo,hi=opening.value,closing.value
    seen={t:set() for t in tickers};duplicates={t:0 for t in tickers};misaligned={t:0 for t in tickers}
    for chunk in pd.read_csv(raw,usecols=['ticker','window_start'],chunksize=500000):
        chunk=chunk[chunk.ticker.isin(tickers)&chunk.window_start.ge(lo)&chunk.window_start.lt(hi)]
        for ticker,g in chunk.groupby('ticker',sort=False):
            stamps=g.window_start.astype('int64').to_numpy()
            misaligned[ticker]+=int((stamps%60_000_000_000!=0).sum())
            values=set(map(int,stamps));duplicates[ticker]+=len(stamps)-len(values)+len(seen[ticker]&values)
            seen[ticker].update(values)
    result=dict(day=day,opening_utc=opening.isoformat(),closing_utc=closing.isoformat(),expected_minutes=int((hi-lo)//60_000_000_000),ticker_signature=signature,
        counts={t:len(v) for t,v in seen.items()},duplicates=duplicates,misaligned=misaligned,
        provenance=dict(key=key,etag=meta.get('ETag'),bytes=meta['ContentLength']),retrieved_unix=time.time())
    atomic(report,result);raw.unlink()
    return result


def run(args):
    source=Path(args.source);out=Path(args.output)
    if (out/'features.parquet').exists():raise ValueError('Completed derived dataset already exists')
    out.mkdir(parents=True,exist_ok=True);cache=out/'prior_day_coverage';cache.mkdir(exist_ok=True)
    f=pd.read_parquet(source/'features.parquet');l=pd.read_parquet(source/'labels.parquet')
    cal=xcals.get_calendar('XNYS');audit=f[['date','ticker','identity']].copy()
    previous={date:str(cal.previous_session(pd.Timestamp(date)).date()) for date in sorted(f.date.unique())}
    audit['previous_session']=audit.date.map(previous)
    load_dotenv()
    s3=boto3.client('s3',endpoint_url='https://files.massive.com',aws_access_key_id=os.environ['MASSIVE_FLATFILES_ACCESS_KEY_ID'],aws_secret_access_key=os.environ['MASSIVE_FLATFILES_SECRET_ACCESS_KEY'],config=Config(retries={'max_attempts':8,'mode':'adaptive'},max_pool_connections=2))
    groups=list(audit.groupby('previous_session',sort=True));results={};start=time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(count_day,day,set(g.ticker),cal.session_open(pd.Timestamp(day)),cal.session_close(pd.Timestamp(day)),cache,s3):day for day,g in groups}
        for i,future in enumerate(as_completed(futures),1):
            day=futures[future];results[day]=future.result()
            if i%10==0 or i==len(groups):print(f'Coverage {i}/{len(groups)} prior sessions; elapsed {(time.monotonic()-start)/60:.1f} min',flush=True)
    records=[]
    for day,g in groups:
        result=results[day]
        # Security identity must exist unchanged on the immediately prior session.
        daily=pd.read_parquet(Path(args.history)/f'{day}.parquet')
        identities=set(zip(daily.ticker,daily.identity))
        for r in g.itertuples():
            observed=result['counts'][r.ticker];expected=result['expected_minutes']
            identity_ok=(r.ticker,r.identity) in identities
            records.append(dict(date=r.date,ticker=r.ticker,identity=r.identity,previous_session=day,
                expected_minutes=expected,observed_minutes=observed,missing_minutes=expected-observed,
                duplicate_bars=result['duplicates'][r.ticker],misaligned_bars=result['misaligned'][r.ticker],
                previous_identity_matches=identity_ok,passes=bool(identity_ok and observed==expected and result['duplicates'][r.ticker]==0 and result['misaligned'][r.ticker]==0)))
    audit=pd.DataFrame(records);audit.to_csv(out/'prior_day_audit.csv',index=False)
    keys=['date','ticker','identity'];keep=audit[audit.passes][keys]
    selected=f.merge(keep,on=keys,validate='one_to_one');labels=l.merge(keep,on=keys,validate='many_to_one')
    save(selected,out/'features.parquet');save(labels,out/'labels.parquet');validate(out)
    manifest=dict(source=str(source.resolve()),source_sha256={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in ('features.parquet','labels.parquet')},rule='All regular-session minute candles on the immediately preceding exchange trading session; no current-day future coverage filter',window='09:30 to exchange regular close (exclusive), America/New_York; early closes honored; extended hours excluded',source_rows=len(f),retained_rows=len(selected),retained_percent=100*len(selected)/len(f),unique_tickers=int(selected.ticker.nunique()),retained_sessions=int(selected.date.nunique()),source_sessions=int(f.date.nunique()),label_rows=len(labels),prior_sessions=len(groups),identity_mismatch_rows=int((~audit.previous_identity_matches).sum()),warning='Historical minute files can contain retrospective vendor corrections; event-time prior-day screen, not archived as-received historical data.')
    atomic(out/'manifest.json',manifest)
    (out/'DATASET_SUMMARY.md').write_text('# Prior-session complete-candle filtered dataset\n\n```\n'+json.dumps(manifest,indent=2)+'\n```\n\nOriginal complete opening candles and >=$1m opening turnover requirements remain. Prior-day audit and source file provenance are retained. Features are unchanged; only rows are filtered. No outcome on the selected trading day participates in this filter.\n')
    print(json.dumps(manifest,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',default='data/master_open15_1m');p.add_argument('--output',default='data/master_open15_1m_prior_complete');p.add_argument('--history',default='data/master/daily');run(p.parse_args())
