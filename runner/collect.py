"""Daily-batch collection; compact warm-up summaries survive raw-file deletion."""
import argparse
import json
import os
import shutil
import time
from pathlib import Path
from urllib.parse import quote
import boto3
from botocore.config import Config
from boto3.s3.transfer import TransferConfig
import exchange_calendars as xcals
import numpy as np
import pandas as pd
from .api import API
from .engine import features, labels, ratio, window

COLUMNS = ['ticker','volume','open','close','high','low','window_start','transactions']
SAMPLE = 'FUBO,RGTI,QBTS,QUBT,KULR,LUNR,RKLB,IONQ,RCAT,SOUN,BBAI,OPTT,HOLO,MVST,RR,LAES,AAPL,MSFT,NVDA,TSLA,PLTR,DJT'

def save(df, path):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix('.tmp'); df.to_parquet(tmp,index=False,compression='zstd'); tmp.replace(path)

def normalize(df):
    if df.empty: return pd.DataFrame(columns=COLUMNS+['minute'])
    df = df[COLUMNS].copy()
    for col in COLUMNS[1:]: df[col] = pd.to_numeric(df[col],errors='raise')
    if df.duplicated(['ticker','window_start']).any(): raise ValueError('Duplicate ticker/minute bars')
    if ((df[['open','close','high','low']]<=0).any(axis=1) | (df.volume<0) |
        (df.high < df[['open','close','low']].max(axis=1)) |
        (df.low > df[['open','close','high']].min(axis=1))).any():
        raise ValueError('Invalid OHLC/volume')
    ts = pd.to_datetime(df.window_start,unit='ns',utc=True).dt.tz_convert('America/New_York')
    df['minute'] = ts.dt.hour*60+ts.dt.minute
    return df.sort_values(['ticker','window_start']).reset_index(drop=True)

def flat_day(day, root, s3):
    raw = root/'temporary'/f'{day}.csv.gz'; raw.parent.mkdir(parents=True,exist_ok=True)
    key = f'us_stocks_sip/minute_aggs_v1/{day[:4]}/{day[5:7]}/{day}.csv.gz'
    meta = s3.head_object(Bucket='flatfiles',Key=key)
    if shutil.disk_usage(root).free < max(2_000_000_000,meta['ContentLength']*8):
        raise RuntimeError('Insufficient local disk space')
    if not raw.exists() or raw.stat().st_size != meta['ContentLength']:
        tmp = raw.with_suffix('.part')
        s3.download_file('flatfiles',key,str(tmp),Config=TransferConfig(max_concurrency=1,use_threads=False))
        tmp.replace(raw)
    return normalize(pd.read_csv(raw,usecols=COLUMNS)),raw,{'key':key,'bytes':meta['ContentLength'],'etag':meta.get('ETag')}

def rest_day(day, tickers, api):
    rows=[]
    for ticker in tickers:
        bars=api.pages(f'/v2/aggs/ticker/{quote(ticker,safe="")}/range/1/minute/{day}/{day}',adjusted='false',sort='asc',limit=50000)
        for x in bars:
            rows.append({'ticker':ticker,'volume':x['v'],'open':x['o'],'close':x['c'],'high':x['h'],'low':x['l'],
                         'window_start':int(x['t'])*1_000_000,'transactions':x.get('n',np.nan)})
    return normalize(pd.DataFrame(rows,columns=COLUMNS))

def universe(api, day):
    # Historical, effective-dated universe includes securities that subsequently delisted.
    return pd.DataFrame(api.pages('/v3/reference/tickers',market='stocks',locale='us',date=day,active='true',limit=1000))

def summarize_day(bars, master, close_min):
    pm=window(bars,240,570).groupby('ticker',sort=False).agg(pm_volume=('volume','sum'),pm_last=('close','last'))
    reg=window(bars,570,close_min).groupby('ticker',sort=False).agg(reg_volume=('volume','sum'),reg_close=('close','last'),reg_last_minute=('minute','last'))
    out=master[['ticker']].copy()
    out['identity']=[next((str(r.get(k)) for k in ('share_class_figi','composite_figi') if pd.notna(r.get(k)) and r.get(k)),r.ticker) for _,r in master.iterrows()]
    out=out.merge(pm,on='ticker',how='left').merge(reg,on='ticker',how='left')
    out['pm_volume']=out.pm_volume.fillna(0.)
    out['reg_volume']=out.reg_volume.fillna(0.)
    return out

def run(args):
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    api=API(root,pace=args.pace)
    cal=xcals.get_calendar('XNYS')
    start=pd.Timestamp(args.start);end=pd.Timestamp(args.end)
    target=cal.sessions_in_range(start,end)
    if not len(target):raise ValueError('No trading sessions requested')
    first=cal.session_offset(target[0],-10)
    sessions=cal.sessions_in_range(first,target[-1])
    scope={'source':args.source,'tickers':args.tickers if args.source=='rest-sample' else None,'schema_version':1,'start':args.start}
    config=root/'scope.json'
    if config.exists() and json.loads(config.read_text())!=scope:raise ValueError('Use a separate output directory for different universes/sources')
    config.write_text(json.dumps(scope,indent=2))
    split_rows=api.pages('/stocks/v1/splits',**{'execution_date.gte':str(first.date()),'execution_date.lte':str(target[-1].date()),'limit':1000})
    splits=pd.DataFrame(split_rows)
    s3=None
    if args.source=='flat':
        required=['MASSIVE_FLATFILES_ACCESS_KEY_ID','MASSIVE_FLATFILES_SECRET_ACCESS_KEY']
        if not all(os.getenv(k) for k in required):raise RuntimeError('Add separate Massive S3 credentials to .env; see .env.example')
        s3=boto3.client('s3',endpoint_url='https://files.massive.com',aws_access_key_id=os.environ[required[0]],aws_secret_access_key=os.environ[required[1]],config=Config(retries={'max_attempts':8,'mode':'adaptive'},max_pool_connections=1))
    history=[]
    for session in sessions:
        day=str(session.date()); checkpoint=root/'checkpoints'/f'{day}.json'; summary_path=root/'daily'/f'{day}.parquet'
        close=cal.session_close(session).tz_convert('America/New_York');close_min=close.hour*60+close.minute
        if checkpoint.exists():
            history.append((day,pd.read_parquet(summary_path)));history=history[-10:]
            print(day,'resumed',flush=True);continue
        started=time.monotonic()
        master=universe(api,day)
        # Common stocks plus ADR common shares; ETFs/warrants/units excluded explicitly.
        master=master[master.type.isin(['CS','ADRC'])].copy()
        if args.source=='rest-sample':master=master[master.ticker.isin(args.tickers.split(','))]
        raw=None;provenance={}
        if args.source=='flat':bars,raw,provenance=flat_day(day,root,s3)
        else:bars=rest_day(day,master.ticker.tolist(),api)
        daily=summarize_day(bars,master,close_min)
        # Official regular-session closes, rather than potentially stale minute closes.
        official=api.get(f'/v2/aggs/grouped/locale/us/market/stocks/{day}',adjusted='false',include_otc='false').get('results',[])
        official={r['T']:r for r in official}
        daily['reg_close']=[official.get(t,{}).get('c',np.nan) for t in daily.ticker]
        daily['reg_volume']=[official.get(t,{}).get('v',np.nan) for t in daily.ticker]
        fs=[];ls=[];audit={'universe':len(master),'missing_history':0,'zero_baseline':0,'price_pass':0,'eligible':0}
        if session in target and len(history)==10:
            adjusted=[]
            for hd,h in history:
                h=h.copy()
                if not splits.empty:
                    events=splits[(splits.execution_date>hd)&(splits.execution_date<=day)].copy()
                    events['factor']=events.split_to/events.split_from
                    factors=events.groupby('ticker').factor.prod()
                    factor=h.ticker.map(factors).fillna(1.)
                    h['pm_volume']*=factor;h['reg_volume']*=factor;h['reg_close']/=factor
                adjusted.append(h)
            stack=pd.concat(adjusted,ignore_index=True)
            reference=stack.groupby(['ticker','identity'],sort=False).agg(history_count=('pm_volume','size'),baseline=('pm_volume','mean'))
            prior=adjusted[-1][['ticker','identity','reg_close']]
            candidates=daily.merge(reference,on=['ticker','identity'],how='left').merge(prior,on=['ticker','identity'],how='left',suffixes=('','_previous'))
            known=candidates.history_count.eq(10)&candidates.reg_close_previous.notna()
            audit['missing_history']=int((~known).sum())
            audit['zero_baseline']=int((known&candidates.baseline.le(0)).sum())
            price_pass=known&candidates.baseline.gt(0)&((candidates.pm_last/candidates.reg_close_previous-1)>=.10-1e-12)
            audit['price_pass']=int(price_pass.sum())
            candidates=candidates[price_pass&((candidates.pm_volume/candidates.baseline)>=5-1e-12)]
            grouped={t:g for t,g in bars[bars.ticker.isin(candidates.ticker)].groupby('ticker',sort=False)}
            for _,row in candidates.iterrows():
                hist=stack[(stack.ticker==row.ticker)&(stack.identity==row.identity)]
                base=float(row.baseline);prev=float(row.reg_close_previous)
                audit['eligible']+=1
                b=grouped[row.ticker];cutoff=pd.Timestamp(day+' 09:45',tz='America/New_York').tz_convert('UTC')
                key={'date':day,'ticker':row.ticker,'identity':row.identity}
                f=features(b[b.minute<585],prev,base,hist,cutoff);fs.append({**key,**f})
                ls.extend({**key,**r} for r in labels(b,close_min))
        save(daily,summary_path)
        # Always write empty-day outputs too; checkpoint is committed last.
        save(pd.DataFrame(fs) if fs else pd.DataFrame(columns=['date','ticker','identity']),root/'features'/f'{day}.parquet')
        save(pd.DataFrame(ls) if ls else pd.DataFrame(columns=['date','ticker','identity','horizon']),root/'labels'/f'{day}.parquet')
        checkpoint.parent.mkdir(exist_ok=True)
        audit.update(day=day,target=bool(session in target),seconds=round(time.monotonic()-started,2),source=args.source,provenance=provenance)
        tmp=checkpoint.with_suffix('.tmp');tmp.write_text(json.dumps(audit,indent=2));tmp.replace(checkpoint)
        if raw:raw.unlink()
        # REST responses contain raw bars; delete only after successful checkpoint.
        # Reference responses are small but share this cache; clearing all avoids raw accumulation.
        for cachefile in api.cache.glob('*.json'):cachefile.unlink()
        history.append((day,daily));history=history[-10:]
        print(day,audit,flush=True)
    report(root)

def report(root):
    root=Path(root)
    def read(folder):
        frames=[pd.read_parquet(p) for p in sorted((root/folder).glob('*.parquet'))]
        return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
    f=read('features');l=read('labels')
    save(f,root/'features.parquet');save(l,root/'labels.parquet')
    audits=[json.loads(p.read_text()) for p in sorted((root/'checkpoints').glob('*.json'))]
    prevalence=[]
    if 'runner_5pct' in l:
        for h,g in l.groupby('horizon'):
            for t in (5,10,15,20,30,50):
                s=g[f'runner_{t}pct'];pos=int((s==True).sum());neg=int((s==False).sum())
                prevalence.append({'horizon':h,'threshold_pct':t,'positive':pos,'negative':neg,'unknown':int(s.isna().sum()),'positive_fraction_known':pos/(pos+neg) if pos+neg else None})
    summary={'scope':json.loads((root/'scope.json').read_text()),'qualifying_rows':len(f),'unique_tickers':int(f.ticker.nunique()) if len(f) else 0,
             'target_days_completed':sum(a['target'] for a in audits),'prevalence':prevalence,
             'missing_features':{c:int(f[c].isna().sum()) for c in f.columns},
             'disk_bytes':sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),
             'final_parquet_bytes':sum((root/p).stat().st_size for p in ('features.parquet','labels.parquet')),
             'daily_audits':audits}
    (root/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    from .research import write_report
    write_report(root)
    print('Summary:',root/'summary.json','qualifying rows:',len(f),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--start',required=True);p.add_argument('--end',required=True)
    p.add_argument('--source',choices=['flat','rest-sample','rest-full'],default='flat')
    p.add_argument('--tickers',default=SAMPLE);p.add_argument('--output',default='data/master')
    p.add_argument('--pace',type=float,default=.15)
    run(p.parse_args())
