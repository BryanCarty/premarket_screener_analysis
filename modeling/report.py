"""Build the precision-first report and charts from frozen holdout results."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .core import HORIZONS,THRESHOLDS

def percent(x):return '—' if pd.isna(x) else f'{100*x:.1f}%'

def build_report(out):
    out=Path(out);r=pd.read_csv(out/'test_results.csv');v=pd.read_csv(out/'validation_model_comparison.csv')
    manifest=json.loads((out/'manifest.json').read_text());frozen=json.loads((out/'frozen_selection.json').read_text())
    predictions=pd.read_parquet(out/'test_predictions.parquet')
    pri=frozen['preselected_priority_definitions'];indexed=r.set_index('definition')
    text=['# Runner prediction: precision-first evaluation','',
          f"Compared four model families across {len(r)} supported runner definitions. {len(frozen['skipped'])} definitions were skipped for insufficient positive/negative support.",
          '', '**Primary metric:** observed threshold hits divided by every stock flagged, including unknown outcomes in the denominator. This is a conservative lower bound on precision; unknown outcomes are not relabeled negative during model training.',
          '', '**Model selection and decision thresholds were frozen before test predictions.** The priorities below were chosen on validation/threshold data, not on final test results.','',
          '## Chronological data split','','| Period | First session | Last session | Sessions | Scorable stock-days |','|---|---|---|---:|---:|']
    for k in ['train','validation','threshold','test']:
        s=manifest['split_summary'][k];text.append(f"| {k} | {s['first']} | {s['last']} | {s['sessions']} | {s['cohort_rows']:,} |")
    text+=['','An entire session is embargoed between periods. All securities from a date stay together. Model refitting uses train + model-validation periods only; threshold-validation and test periods never enter fitting.','',
           '## Preselected priorities: final held-out results','',
           '| Definition | Model | Threshold validation precision | Test precision (conservative) | Test precision among known | Test calls | Hits / failures / unknown | Day-bootstrap 95% interval | Met validation target? |',
           '|---|---|---:|---:|---:|---:|---|---|---|']
    for name in pri:
        x=indexed.loc[name]
        text.append(f"| {name} | {x.model} | {percent(x.threshold_precision)} ({int(x.threshold_calls)} calls) | {percent(x.precision_conservative)} | {percent(x.precision_known)} | {int(x.calls)} | {int(x.hit)} / {int(x.confirmed_failure)} / {int(x.unknown)} | {percent(x.bootstrap_day_ci_low)}–{percent(x.bootstrap_day_ci_high)} | {'Yes' if x.threshold_qualified else 'No — research fallback'} |")
    qualified=r[r.threshold_qualified]
    text+=['','The validation target is 80% conservative precision with at least 25 calls across ten dates and a Wilson lower confidence bound of at least 65%. Passing this gate does not guarantee 80% precision on new data. Thresholds for definitions failing the gate are exploratory fallbacks, not approved trading signals.',
           '',f"{len(qualified)} definitions met the threshold-validation target; {int((qualified.precision_conservative>=.8).sum())} of those achieved at least 80% conservative precision on the final test period (signal counts and confidence intervals still matter).",'',
           '## All frozen definitions','','| Horizon | Gain | Selected family | Test precision lower–upper bound | Known-label precision | Calls | Complete-window recall | Baseline observed hit fraction | Validation-qualified |',
           '|---|---:|---|---|---:|---:|---:|---:|---|']
    rank={h:i for i,h in enumerate(HORIZONS)}
    rr=r.assign(order=r.horizon.astype(str).map(rank)).sort_values(['order','gain_threshold'])
    for _,x in rr.iterrows():
        text.append(f"| {x.horizon} | +{int(x.gain_threshold)}% | {x.model} | {percent(x.precision_conservative)}–{percent(x.precision_upper_bound)} | {percent(x.precision_known)} | {int(x.calls)} | {percent(x.recall_complete)} | {percent(x.baseline_hit_fraction)} | {'Yes' if x.threshold_qualified else 'No'} |")
    text+=['','Bounds treat all unknown calls as failures (lower) or hits (upper). They are not statistical confidence intervals. Recall and average precision are measured only on fully observed windows; those cases are more liquid and not representative of every eligible stock.',
           '', '## Selection-bias and execution limitations','',
           '- The master dataset remains unchanged. Scoring is restricted to stocks with all 15 opening minute bars, a fixed rule available before 09:45. Stocks outside this cohort are not covered by these results.',
           '- Training excludes incomplete outcome windows for positives and negatives equally. Using all observed positives while dropping only unknown negatives would create outcome-dependent selection bias. Evaluation still scores every stock in the pre-cutoff cohort, and reports unknown selected outcomes.',
           '- Unobserved outcomes are informative missingness, not missing at random. Complete-case metrics alone can overstate real-world performance; prefer the conservative precision and unknown counts.',
           '- The same ticker may appear in past and future periods. Symbols and identifiers are not input features. This measures predicting future stock-days, not exclusively unseen companies.',
           '- The day-cluster bootstrap resamples test dates, including dates with no calls. It does not fully capture ticker persistence or regime changes. Wilson intervals are supplemental binomial approximations.',
           '- Many correlated runner definitions were explored. No definition or cutoff was selected using test scores, but choosing a new favorite after reading all test results would reuse the holdout and require a fresh future evaluation.',
           '- A runner label means a high was reached, not an executed exit or a profitable strategy. Bid/ask spreads, slippage, fees, halts, and intraminute ordering are not modeled. No accuracy-based optimization, oversampling, or test-period probability calibration was used.',
           '- Historical bars are revised event-time observations, not exact as-received feed snapshots. The Developer feed is delayed; live 09:45 execution would require an appropriate timely feed.',
           '- Validation precision and model scores are empirical research estimates, not guarantees or necessarily calibrated probabilities.',
           '', '## Files','',
           '- `manifest.json`: source hashes, features, split dates, fixed evaluation rules and software versions.',
           '- `validation_model_comparison.csv`: all four families compared on the model-validation period.',
           '- `frozen_selection.json`: model/threshold selections and global priorities saved before holdout scoring.',
           '- `test_results.csv`, `test_predictions.parquet`: final metrics and stock-level held-out predictions.',
           '- `test_monthly.csv`: month-by-month signal outcomes for the frozen policies.',
           '- `models/*.joblib`: fitted pipelines, feature order, and operating thresholds. Load only trusted local artifacts.',
           '- `precision_overview.png` and `.pdf`: precision/support overview; `precision_recall.png`: curves for preselected priorities on complete test windows.',
           '', '## Reproduce','',
           'From the project root: `python -m modeling.train --output modeling/results_NEW`. The script refuses to overwrite completed test results. Repeated experimentation must disclose that the existing holdout has already been viewed.',
           '', 'Metric references: [scikit-learn precision–recall](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html) and [threshold tuning](https://scikit-learn.org/1.8/modules/classification_threshold.html).']
    (out/'REPORT.md').write_text('\n'.join(text)+'\n')
    monthly=predictions.copy();monthly['month']=pd.to_datetime(monthly.date).dt.to_period('M').astype(str)
    monthly['hit']=monthly.predicted_runner&monthly.label.fillna(False).astype(bool)
    monthly['unknown']=monthly.predicted_runner&monthly.label.isna()
    m=monthly.groupby(['definition','month']).agg(calls=('predicted_runner','sum'),hits=('hit','sum'),unknown=('unknown','sum')).reset_index()
    m['precision_conservative']=m.hits/m.calls.replace(0,np.nan);m.to_csv(out/'test_monthly.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,(ax,bx)=plt.subplots(1,2,figsize=(17,8),gridspec_kw={'width_ratios':[1.2,1]},facecolor='#f6f8fc')
    values=np.full((6,7),np.nan)
    for _,x in r.iterrows():values[THRESHOLDS.index(int(x.gain_threshold)),HORIZONS.index(str(x.horizon))]=x.precision_conservative*100
    im=ax.imshow(np.ma.masked_invalid(values),vmin=0,vmax=100,cmap='YlGnBu',aspect='auto')
    for _,x in r.iterrows():
        i=THRESHOLDS.index(int(x.gain_threshold));j=HORIZONS.index(str(x.horizon));txt=f'{100*x.precision_conservative:.0f}%\nn={int(x.calls)}'+(' *' if x.threshold_qualified else '')
        ax.text(j,i,txt,ha='center',va='center',fontsize=9,color='white' if x.precision_conservative>.6 else '#172b46')
    ax.set_xticks(range(7),['5m','15m','30m','60m','90m','120m','Close']);ax.set_yticks(range(6),[f'+{x}%' for x in THRESHOLDS]);ax.set_xlabel('Horizon after 09:45 ET');ax.set_ylabel('Runner gain threshold');ax.set_title('Frozen policy: conservative test precision',loc='left',pad=15)
    fig.colorbar(im,ax=ax,orientation='horizontal',pad=.14,fraction=.045,label='Observed hits / all flagged stocks (%)')
    chosen=indexed.loc[pri].iloc[::-1];y=np.arange(len(chosen))
    vals=chosen.precision_conservative.to_numpy()*100;lo=chosen.bootstrap_day_ci_low.to_numpy()*100;hi=chosen.bootstrap_day_ci_high.to_numpy()*100
    bx.errorbar(vals,y,xerr=[np.maximum(0,vals-lo),np.maximum(0,hi-vals)],fmt='o',color='#197d9c',capsize=4)
    for i,(_,x) in enumerate(chosen.iterrows()):bx.text(2,i+.17,f'{int(x.hit)} hits / {int(x.calls)} calls; {int(x.unknown)} unknown',fontsize=9,color='#536781')
    bx.set_yticks(y,chosen.index);bx.set_xlim(0,103);bx.axvline(80,color='#a76b00',ls='--',lw=1,label='80% target');bx.set_xlabel('Conservative test precision (%)');bx.set_title('Priorities chosen BEFORE the test',loc='left',pad=15);bx.grid(axis='x',alpha=.15);bx.legend(loc='lower right',frameon=False)
    fig.suptitle('Runner prediction — precision before accuracy',x=.06,ha='left',fontsize=22,weight='bold',color='#172b46')
    fig.subplots_adjust(left=.065,right=.96,top=.83,bottom=.22,wspace=.35)
    fig.text(.065,.115,'* Met the 80% threshold-validation gate. Unstarred policies are research-only fallbacks. Blank cells: insufficient training support.',fontsize=10,color='#536781')
    fig.text(.065,.085,'n = test signals, not total samples. Unknown selected outcomes count against precision. Error bars: 95% bootstrap intervals by date.',fontsize=10,color='#536781')
    fig.text(.065,.055,'Different definitions have different base rates. Threshold hits do not establish executable trading returns. Final test was not used for selection.',fontsize=10,color='#536781')
    for ext in ['png','pdf']:fig.savefig(out/f'precision_overview.{ext}',dpi=170)
    plt.close(fig)
    from sklearn.metrics import precision_recall_curve
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    for ax,name in zip(axes,pri[:3]):
        q=predictions[predictions.definition.eq(name)&predictions.complete_outcome].copy()
        pr,re,_=precision_recall_curve(q.label.astype(int),q.score)
        ax.step(re,pr,where='post',color='#197d9c');ax.axhline(q.label.mean(),ls=':',color='#888',label='Complete-case base rate');ax.set(xlabel='Recall',ylabel='Precision',ylim=(0,1.03),xlim=(0,1),title=name);ax.grid(alpha=.15);ax.legend(fontsize=8)
    fig.suptitle('Preselected priorities — complete-window test curves (coverage-biased subset)',fontsize=13)
    fig.tight_layout();fig.savefig(out/'precision_recall.png',dpi=170);plt.close(fig)
    (out/'COMPLETE.txt').write_text('Model selection frozen before test; held-out scoring and reports complete.\n')
    print(out/'REPORT.md',flush=True)

if __name__=='__main__':
    import sys
    build_report(sys.argv[1] if len(sys.argv)>1 else 'modeling/results')
