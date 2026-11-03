"""Causal opening-path inputs and explicit first-touch 60-minute labels."""
import hashlib,json
from pathlib import Path
from urllib.parse import quote
import numpy as np
import pandas as pd
from runner.api import API
from runner.collect import COLUMNS,normalize,save
from runner.engine import window

OUT=Path('modeling/barrier_study/data')
SOURCE=Path('modeling/volatility_scaled/data/features.parquet')
KEYS=['date','ticker','identity']


def intraday_features(b):
    """Every stock feature is formed strictly before the 09:45 cutoff."""
    b=b[b.minute.lt(585)].copy();pm=window(b,240,570);op=window(b,570,585)
    d={'path_opening_bars':len(op),'path_premarket_bars':len(pm)}
    if op.empty:return d
    for label,lo,hi in [('first5',570,575),('middle5',575,580),('last5',580,585),('open15',570,585)]:
        s=window(op,lo,hi)
        d[f'path_{label}_return_pct']=100*(s.close.iloc[-1]/s.open.iloc[0]-1) if len(s)==hi-lo else np.nan
        d[f'path_{label}_turnover']=float((((s.high+s.low+s.close)/3)*s.volume).sum()) if len(s)==hi-lo else np.nan
    d['path_return_acceleration_pct']=d['path_last5_return_pct']-d['path_first5_return_pct']
    first=d['path_first5_turnover'];last=d['path_last5_turnover']
    d['path_turnover_acceleration']=last/first if first>0 else np.nan
    close=float(op.close.iloc[-1]);high=float(op.high.max());low=float(op.low.min())
    d['path_pullback_from_high_pct']=100*(close/high-1)
    d['path_bounce_from_low_pct']=100*(close/low-1)
    d['path_close_drawdown_pct']=float(100*(op.close/op.close.cummax()-1).min())
    wick=(op.high-op[['open','close']].max(axis=1))/(op.high-op.low).replace(0,np.nan)
    d['path_upper_wick_fraction']=float(wick.mean())
    d['path_upper_wick_last5_fraction']=float(wick[op.minute.ge(580)].mean())
    proxy=((op.high+op.low+op.close)/3*op.volume).cumsum()/op.volume.cumsum().replace(0,np.nan)
    above=op.close>proxy
    d['path_vwap_crossings']=int(above.ne(above.shift()).iloc[1:].sum())
    d['path_last5_above_running_vwap_fraction']=float(above[op.minute.ge(580)].mean())
    d['path_observed_open15_volume']=float(op.volume.sum())
    if len(pm):
        pm_high=float(pm.high.max());above_pm=op.close>pm_high
        d['path_minutes_above_pm_high']=int(above_pm.sum())
        d['path_pm_high_crossings']=int((above_pm&~above_pm.shift(fill_value=False)).sum())
        d['path_failed_pm_breakout']=int(op.high.max()>pm_high and close<pm_high)
        d['path_close_vs_pm_high_pct']=100*(close/pm_high-1)
    return d


def barrier_label(b,sigma,end=645,start=585):
    """First barrier touched; gaps before exit and same-minute ties are unresolved."""
    if not np.isfinite(sigma) or sigma<=0:raise ValueError('A positive past-based scale is required')
    result=dict(outcome=None,reason='missing_entry',entry_price=np.nan,upper_price=np.nan,lower_price=np.nan,
                exit_price=np.nan,gross_return_pct=np.nan,exit_minute=np.nan,observed_minutes=0)
    entry=b[b.minute.eq(start)]
    if len(entry)!=1:return result
    price=float(entry.open.iloc[0]);upper=price*np.exp(sigma);lower=price*np.exp(-sigma)
    result.update(entry_price=price,upper_price=upper,lower_price=lower)
    s=window(b,start,end).sort_values('minute');expected=start
    for bar in s.itertuples():
        if int(bar.minute)!=expected:return {**result,'reason':'missing_before_exit'}
        expected+=1;result['observed_minutes']+=1
        if bar.open>=upper-price*1e-12:
            return {**result,'outcome':'up','reason':'up_gap','exit_price':upper,'gross_return_pct':100*(upper/price-1),'exit_minute':int(bar.minute)}
        if bar.open<=lower+price*1e-12:
            return {**result,'outcome':'down','reason':'stop_gap','exit_price':float(bar.open),'gross_return_pct':100*(bar.open/price-1),'exit_minute':int(bar.minute)}
        up=bar.high>=upper-price*1e-12;down=bar.low<=lower+price*1e-12
        if up and down:return {**result,'reason':'ambiguous_same_minute','exit_minute':int(bar.minute)}
        if up:return {**result,'outcome':'up','reason':'target','exit_price':upper,'gross_return_pct':100*(upper/price-1),'exit_minute':int(bar.minute)}
        if down:return {**result,'outcome':'down','reason':'stop','exit_price':lower,'gross_return_pct':100*(lower/price-1),'exit_minute':int(bar.minute)}
    if expected!=end:return {**result,'reason':'missing_before_timeout'}
    last=float(s.close.iloc[-1])
    return {**result,'outcome':'neither','reason':'timeout','exit_price':last,'gross_return_pct':100*(last/price-1),'exit_minute':end}


def get_bars(api,ticker,day,start_minute):
    begin=pd.Timestamp(day,tz='America/New_York')+pd.Timedelta(minutes=start_minute)
    end=pd.Timestamp(day,tz='America/New_York')+pd.Timedelta(minutes=645)
    rows=api.pages(f'/v2/aggs/ticker/{quote(ticker,safe="")}/range/1/minute/{int(begin.timestamp()*1000)}/{int(end.timestamp()*1000)-1}',adjusted='false',sort='asc',limit=50000)
    return normalize(pd.DataFrame([dict(ticker=ticker,volume=x['v'],open=x['o'],close=x['c'],high=x['h'],low=x['l'],window_start=int(x['t'])*1000000,transactions=x.get('n',np.nan)) for x in rows],columns=COLUMNS))


def main():
    OUT.mkdir(parents=True,exist_ok=True);parts=OUT/'paths';parts.mkdir(exist_ok=True)
    source=pd.read_parquet(SOURCE);source=source[source.sigma20_log.notna()].copy()
    api=API(OUT/'request_cache');rows=[];errors=[];benchmark={}
    dates=sorted(source.date.unique())
    for i,day in enumerate(dates,1):
        for ticker in ('SPY','QQQ'):
            file=parts/f'{day}_{ticker}_benchmark.parquet'
            try:
                if file.exists():b=pd.read_parquet(file)
                else:
                    b=get_bars(api,ticker,day,570);save(b,file)
                    for p in api.cache.glob('*.json'):p.unlink()
                values=intraday_features(b)
                benchmark[(day,ticker)]={f'market_{ticker}_{w}_return_pct':values.get(f'path_{w}_return_pct',np.nan) for w in ('first5','middle5','last5','open15')}
            except Exception as exc:errors.append(dict(day=day,ticker=ticker,error=type(exc).__name__))
        if i%50==0 or i==len(dates):print(f'Benchmarks {i}/{len(dates)} dates',flush=True)
    for i,row in enumerate(source.to_dict('records'),1):
        day,ticker=row['date'],row['ticker'];token=hashlib.sha256((day+'|'+ticker).encode()).hexdigest()[:24];file=parts/f'{token}.parquet'
        try:
            if file.exists():b=pd.read_parquet(file)
            else:
                b=get_bars(api,ticker,day,240);save(b,file)
                for p in api.cache.glob('*.json'):p.unlink()
            features=intraday_features(b)
            for etf in ('SPY','QQQ'):
                values=benchmark.get((day,etf),{})
                features.update(values)
                for w in ('first5','middle5','last5','open15'):
                    features[f'market_excess_{etf}_{w}_return_pct']=features.get(f'path_{w}_return_pct',np.nan)-values.get(f'market_{etf}_{w}_return_pct',np.nan)
            label=barrier_label(b,float(row['sigma20_log']))
            rows.append({**row,**features,**{f'label_{k}':v for k,v in label.items()}})
        except Exception as exc:
            errors.append(dict(day=day,ticker=ticker,error=type(exc).__name__))
            rows.append({**row,'label_outcome':None,'label_reason':'api_failure'})
        if i%50==0 or i==len(source):print(f'Stock paths {i}/{len(source)}; failures {len(errors)}',flush=True)
    data=pd.DataFrame(rows).replace([np.inf,-np.inf],np.nan)
    save(data,OUT/'dataset.parquet')
    data[KEYS+['label_outcome','label_reason']].to_csv(OUT/'label_audit.csv',index=False)
    manifest=dict(source=str(SOURCE.resolve()),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),rows=len(data),known_outcomes=int(data.label_outcome.notna().sum()),
        outcomes=data.label_outcome.fillna('unknown').value_counts().to_dict(),reasons=data.label_reason.value_counts().to_dict(),
        rule='First +1 or -1 sigma20 log-price barrier reached in [09:45,10:45); neither if full path reaches neither. Earlier missing minutes or unresolved same-minute ties remain unknown. Stop gaps exit at worse open; upside fills capped at target.',
        scale='Existing prior-20-session 09:45-to-close log-return standard deviation, not 60-minute standard deviation; frozen before today.',
        feature_cutoff='Strictly before 09:45 ET. Market returns use only SPY/QQQ bars ending by cutoff. All labels separate under label_ prefix.',
        limitations='Historical vendor corrections/as-received timing are not reconstructed. Historical 09:45 price proxy differs from the delayed-feed paper entry. No quote access (403). No order flow or catalyst features are claimed.',errors=errors)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2),flush=True)

if __name__=='__main__':main()
