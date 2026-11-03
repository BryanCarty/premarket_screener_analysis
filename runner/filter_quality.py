"""Create a derived dataset with complete opening candles and minimum turnover."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from .collect import save
from .validate import validate

KEYS = ['date', 'ticker', 'identity']

def run(source, output, minimum):
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError('Output must differ from source')
    if minimum <= 0:
        raise ValueError('Minimum turnover must be positive')
    if output.exists():
        raise ValueError('Use a new output directory to preserve existing datasets')
    f = pd.read_parquet(source/'features.parquet')
    l = pd.read_parquet(source/'labels.parquet')
    turnover = f.open15_volume * f.open15_vwap_proxy
    complete = f.opening_missing_minutes.eq(0) & f.opening_minutes_observed.eq(15)
    liquid = turnover.ge(minimum)
    selected = f.loc[complete & liquid].copy()
    # Keep the feature schema compatible with downstream consumers; diagnostics are separate.
    labels = l.merge(selected[KEYS], on=KEYS, how='inner', validate='many_to_one')
    output.mkdir(parents=True)
    save(selected, output/'features.parquet')
    save(labels, output/'labels.parquet')
    validate(output)
    diagnostic = selected[KEYS+['open15_volume','open15_transactions','history_daily_volume_mean_10d','pm_volume','pm_last30_volume']].copy()
    diagnostic['open15_turnover_proxy'] = selected.open15_volume * selected.open15_vwap_proxy
    diagnostic.to_csv(output/'liquidity.csv', index=False)
    quantiles = diagnostic.select_dtypes('number').quantile([.1,.25,.5,.75,.9]).T
    quantiles.columns = ['p10','p25','median','p75','p90']
    quantiles.to_csv(output/'liquidity_quantiles.csv')
    bands = pd.cut(diagnostic.open15_turnover_proxy, [minimum,5e6,20e6,float('inf')], right=False, labels=['$1m–$5m','$5m–$20m','≥$20m']) if minimum == 1e6 else pd.cut(diagnostic.open15_turnover_proxy, 5)
    activity = bands.value_counts(sort=False).rename_axis('turnover').reset_index(name='stock_days')
    activity['percent'] = 100*activity.stock_days/len(selected)
    activity.to_csv(output/'turnover_distribution.csv', index=False)
    daily = f.groupby('date').size().rename('source_stock_days').to_frame()
    daily['retained_stock_days'] = selected.groupby('date').size().reindex(daily.index,fill_value=0)
    daily.to_csv(output/'daily_counts.csv')
    coverage = labels.assign(incomplete=labels.missing_minutes.gt(0)).groupby('horizon').agg(rows=('ticker','size'),missing_windows=('incomplete','sum'),entry_missing=('entry_available',lambda x:(~x).sum()))
    coverage['missing_percent'] = 100*coverage.missing_windows/coverage.rows
    coverage.to_csv(output/'label_coverage.csv')
    prevalence=[]
    for horizon,g in labels.groupby('horizon'):
        for threshold in (5,10,15,20,30,50):
            s=g[f'runner_{threshold}pct']
            prevalence.append(dict(horizon=horizon,threshold_pct=threshold,positive=int(s.eq(True).sum()),negative=int(s.eq(False).sum()),unknown=int(s.isna().sum())))
    pd.DataFrame(prevalence).to_csv(output/'label_prevalence.csv',index=False)
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(),source=str(source.resolve()),source_sha256={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in ('features.parquet','labels.parquet')},filters=dict(opening_minutes_observed=15,opening_missing_minutes=0,minimum_open15_turnover_proxy=minimum,turnover_formula='open15_volume * open15_vwap_proxy',window='09:30 inclusive to 09:45 exclusive America/New_York'),source_rows=len(f),complete_opening_rows=int(complete.sum()),turnover_pass_rows=int(liquid.sum()),retained_rows=len(selected),retained_percent=100*len(selected)/len(f),unique_tickers=int(selected.ticker.nunique()),label_rows=len(labels),source_sessions=len(daily),retained_sessions=int(daily.retained_stock_days.gt(0).sum()),first_date=str(selected.date.min()),last_date=str(selected.date.max()),missing_entry_rows=int(labels.loc[labels.horizon.eq('15'),'entry_available'].eq(False).sum()))
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def table(df):
        return df.to_string(index=False,float_format=lambda x:f'{x:,.2f}')
    report=f'''# Opening-quality filtered dataset

Source: {source}. Retains complete 09:30–09:44 ET minute grids and estimated opening turnover >=${minimum:,.0f}. Turnover uses minute typical price times volume, not exact traded notional. No extra transaction-count, historical-volume, price, market-cap, or exchange filters were applied. Feature and label schemas are preserved.

Retained {len(selected):,} of {len(f):,} stock-days ({manifest['retained_percent']:.2f}%), across {manifest['unique_tickers']:,} ticker strings and {manifest['retained_sessions']} of {len(daily)} source sessions. Dates: {manifest['first_date']} through {manifest['last_date']}. Removed {len(f)-len(selected):,} rows. Of {int(complete.sum()):,} complete-opening rows, {int((complete & ~liquid).sum()):,} fail the turnover requirement. Labels: {len(labels):,}, seven horizons per retained stock-day.

## Opening turnover distribution

```
{table(activity)}
```

## Liquidity quantiles

```
{table(quantiles.reset_index(names='metric'))}
```

## Subsequent label-window coverage

```
{table(coverage.reset_index())}
```

Complete opening coverage does not guarantee complete 09:45 entry or later candles. {manifest['missing_entry_rows']:,} retained stock-days lack the 09:45 entry candle. Original nullable labels remain unchanged in meaning; no missing values were filled.

Selection is available at the 09:45 feature cutoff, before subscription/publication delays. Applying it to earlier entries would introduce lookahead. The parent collection documents a 15-minute feed delay. This remains a selected premarket-gap/high-relative-volume population, not the full market. Exact 08:30–09:30 completeness, historical market caps, and exchange proportions remain unavailable. Complete bars and turnover do not establish spread, depth, company reputation, or execution capacity. Prior daily-volume statistics are share volume, not dollar volume.

Source hashes and rules: manifest.json. Per-stock-day diagnostics: liquidity.csv. Daily counts include zero-retention source sessions. Label coverage and prevalence are exported separately. Parent data was not modified.
'''
    (output/'DATASET_SUMMARY.md').write_text(report)
    print(json.dumps(manifest,indent=2))
    print(report)

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default='data/master')
    parser.add_argument('--output',default='data/master_open15_1m')
    parser.add_argument('--minimum-turnover',type=float,default=1e6)
    args=parser.parse_args()
    run(args.source,args.output,args.minimum_turnover)
