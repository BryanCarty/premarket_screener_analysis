"""Descriptive dataset report; no model fitting or hypothesis testing."""
import json
from pathlib import Path
import pandas as pd

def write_report(root):
    root=Path(root);s=json.loads((root/'summary.json').read_text())
    f=pd.read_parquet(root/'features.parquet');l=pd.read_parquet(root/'labels.parquet')
    text=['# Dataset summary','',f"Scope: {s['scope']['source']}. Completed target sessions: {s['target_days_completed']}.",
          f"Eligible stock-days: {s['qualifying_rows']}; distinct tickers: {s['unique_tickers']}.",
          f"Final compressed Parquet: {s['final_parquet_bytes']:,} bytes. Total logical stored bytes at report time: {s['disk_bytes']:,}.",'',
          'Natural class distribution is preserved. Unknown targets are excluded from the positive fraction, and never counted as negatives.','',
          '| Horizon (minutes) | Upside threshold | Positive | Negative | Unknown | Positive % of known |',
          '|---|---:|---:|---:|---:|---:|']
    for r in s['prevalence']:
        p=r['positive_fraction_known'];p='—' if p is None else f'{100*p:.2f}'
        text.append(f"| {r['horizon']} | {r['threshold_pct']}% | {r['positive']} | {r['negative']} | {r['unknown']} | {p} |")
    if not f.empty:
        text+=['','## Opening behavior: descriptive comparison','', 'At the 60-minute, +10% definition. Medians below are exploratory associations; they do not establish predictive value. No model has been trained. Small classes and repeated tickers can make these comparisons unstable.','',
               '| Feature | Runner median | Non-runner median |','|---|---:|---:|']
        merged=f.merge(l[l.horizon.eq('60')][['date','ticker','identity','runner_10pct']],on=['date','ticker','identity'])
        for c in ('open15_return_pct','open15_gain_retention','open15_vs_pm_volume','open15_close_vs_pm_high_pct','pm_rvol'):
            vals=[]
            for flag in (True,False):
                x=merged.loc[merged.runner_10pct.eq(flag),c].dropna()
                vals.append(f'{x.median():.4f}' if len(x) else '—')
            text.append(f'| {c} | {vals[0]} | {vals[1]} |')
        text+=['','## Missing predictive fields','','| Field | Null rows |','|---|---:|']
        text.extend(f'| {c} | {n} |' for c,n in s['missing_features'].items() if n)
    audits=[a for a in s['daily_audits'] if a['target']]
    text+=['','## Eligibility audit','',f"Missing ten-day matching history or prior close: {sum(a['missing_history'] for a in audits)} security-days.",
           f"Zero historical PM baseline: {sum(a['zero_baseline'] for a in audits)} security-days.",
           '', 'See README.md for universe scope, historical identity limitations, minute-data coverage, and entry execution assumptions. More sessions and chronological out-of-sample testing are required before deciding whether a viable ML strategy exists.']
    (root/'DATASET_SUMMARY.md').write_text('\n'.join(text)+'\n')
