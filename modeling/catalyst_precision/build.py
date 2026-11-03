"""Archived headline proxies and delayed-entry targets for a predeclared study."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import exchange_calendars as xcals
from runner.api import API
from runner.collect import save,COLUMNS,normalize
from modeling.barrier_study.build import barrier_label
from .news import news_features

OUT=Path('modeling/catalyst_precision/data')
SOURCE=Path('modeling/barrier_study/data/dataset.parquet')
KEYS=['date','ticker','identity']


def main():
    OUT.mkdir(parents=True,exist_ok=True);parts=OUT/'news_parts';parts.mkdir(exist_ok=True);paths=OUT/'paths';paths.mkdir(exist_ok=True)
    api=API(OUT/'requests');cal=xcals.get_calendar('XNYS');source=pd.read_parquet(SOURCE)
    rows=[];audits=[];errors=[]
    for i,row in enumerate(source.to_dict('records'),1):
        day,ticker=row['date'],row['ticker'];token=hashlib.sha256((day+'|'+ticker).encode()).hexdigest()[:24]
        cutoff=pd.Timestamp(day+' 09:45',tz='America/New_York').tz_convert('UTC')
        prev=cal.previous_session(pd.Timestamp(day));prior_close=cal.session_close(prev)
        start=min(prior_close,cutoff-pd.Timedelta(hours=72));file=parts/f'{token}.json'
        try:
            if file.exists():archive=json.loads(file.read_text())
            else:
                reference=api.get(f'/v3/reference/tickers/{ticker}',date=day).get('results',{})
                identity=next((reference[k] for k in ('share_class_figi','composite_figi') if reference.get(k)),ticker)
                articles=api.pages('/v2/reference/news',**{'ticker':ticker,'published_utc.gte':start.isoformat(),'published_utc.lt':cutoff.isoformat(),'limit':1000,'sort':'published_utc','order':'asc'})
                allowed=['id','title','published_utc','tickers','article_url','last_updated_utc','updated_utc']
                archive=dict(ticker=ticker,date=day,company_name=reference.get('name',''),identity_match=identity==row['identity'],received_at=pd.Timestamp.now(tz='UTC').isoformat(),articles=[{k:a[k] for k in allowed if k in a} for a in articles])
                tmp=file.with_suffix('.tmp');tmp.write_text(json.dumps(archive,indent=2)+'\n');tmp.replace(file)
                for p in api.cache.glob('*.json'):p.unlink()
            features,audit=news_features(archive['articles'] if archive['identity_match'] else [],ticker,archive['company_name'],cutoff,prior_close)
            features['news_api_ok']=1;features['news_identity_match']=int(archive['identity_match'])
            audits.extend({**{k:row[k] for k in KEYS},**a} for a in audit)
        except Exception as exc:
            errors.append(dict(date=day,ticker=ticker,stage='news',error=type(exc).__name__))
            features={'news_api_ok':0,'news_identity_match':0}
        path=paths/f'{token}.parquet'
        try:
            if path.exists():b=pd.read_parquet(path)
            else:
                lo=pd.Timestamp(day+' 10:05',tz='America/New_York');hi=pd.Timestamp(day+' 11:15',tz='America/New_York')
                values=api.pages(f'/v2/aggs/ticker/{ticker}/range/1/minute/{int(lo.timestamp()*1000)}/{int(hi.timestamp()*1000)-1}',adjusted='false',sort='asc',limit=50000)
                b=normalize(pd.DataFrame([dict(ticker=ticker,volume=x['v'],open=x['o'],close=x['c'],high=x['h'],low=x['l'],window_start=int(x['t'])*1000000,transactions=x.get('n',np.nan)) for x in values],columns=COLUMNS))
                save(b,path)
                for p in api.cache.glob('*.json'):p.unlink()
            target=barrier_label(b,row['sigma20_log'],start=605,end=665)
            late=barrier_label(b,row['sigma20_log'],start=615,end=675)
        except Exception as exc:
            errors.append(dict(date=day,ticker=ticker,stage='path',error=type(exc).__name__))
            target={'outcome':None,'reason':'api_failure'};late=target.copy()
        flags={'setup_continuation':int(row.get('path_close_vs_pm_high_pct',np.nan)>=0 and row.get('path_last5_return_pct',np.nan)>0 and row.get('path_last5_above_running_vwap_fraction',np.nan)>=.6),
            'setup_failed_breakout':int(row.get('path_failed_pm_breakout',0)==1 and row.get('path_last5_return_pct',np.nan)<0 and row.get('path_last5_above_running_vwap_fraction',np.nan)<=.4)}
        rows.append({**row,**features,**flags,**{'target_'+k:v for k,v in target.items()},**{'late_'+k:v for k,v in late.items()}})
        if i%50==0 or i==len(source):print(f'Catalyst/delayed paths {i}/{len(source)}; errors {len(errors)}',flush=True)
    data=pd.DataFrame(rows).replace([np.inf,-np.inf],np.nan)
    save(data,OUT/'dataset.parquet');pd.DataFrame(audits).to_csv(OUT/'headline_audit.csv',index=False)
    manifest=dict(source=str(SOURCE.resolve()),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),plan_sha256=hashlib.sha256(Path('modeling/catalyst_precision/plan.json').read_bytes()).hexdigest(),
        rows=len(data),news_api_success=int(data.news_api_ok.sum()),fresh_company_news_rows=int(data.news_fresh_relevant.fillna(0).gt(0).sum()),
        classified_fresh_event_rows=int(data.news_classified_fresh.fillna(0).gt(0).sum()),positive_candidate_rows=int(data.news_positive_candidate.fillna(0).sum()),
        primary_outcomes=data.target_outcome.fillna('unknown').value_counts().to_dict(),entry_et='10:05',sensitivity_entry_et='10:15',feature_cutoff_et='09:45',
        news_caveat='Headline event/polarity proxies, not verified earnings surprises. Historical publication time is known; ingestion and version history are not. No vendor sentiment, description, revised earnings actuals or estimate fields used.',
        errors=errors)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2),flush=True)

if __name__=='__main__':main()
