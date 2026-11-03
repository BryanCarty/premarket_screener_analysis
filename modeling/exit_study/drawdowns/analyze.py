"""Per-trade excursion/drawdown bounds and paired high-vs-low distribution tests."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('modeling/exit_study/drawdowns')
PARENT=ROOT.parent

def route_drawdown(prices,peak):
    worst=0.
    for price in prices:
        peak=max(peak,float(price));worst=max(worst,100*(1-float(price)/peak))
    return worst,peak

def ohlc_bounds(bars,entry,exit_price,exit_minute,target):
    """Tight OHLC-order bounds on OBSERVED prices; missing bars invalidate true bounds."""
    peak=entry;ddlo=ddhi=0.;lowlo=lowhi=entry;high=entry
    prior=bars[(bars.minute>=585)&(bars.minute<exit_minute)]
    for r in prior.itertuples():
        a,p1=route_drawdown([r.open,r.high,r.low,r.close],peak)
        z,p2=route_drawdown([r.open,r.low,r.high,r.close],peak)
        ddlo=max(ddlo,min(a,z));ddhi=max(ddhi,max(a,z));peak=max(p1,p2)
        lowlo=min(lowlo,r.low);lowhi=min(lowhi,r.low);high=max(high,r.high)
    if target:
        row=bars[bars.minute.eq(exit_minute)]
        if len(row)!=1:raise ValueError('Missing modeled target exit bar')
        r=row.iloc[0]
        if r.open>=exit_price-entry*1e-10:
            # Conservative legacy target-price fill at a favorable opening gap.
            best=[exit_price];worst=[exit_price]
        else:
            best=[float(r.open),exit_price]
            worst=[float(r.open),float(r.low),exit_price]
            lowlo=min(lowlo,r.open)
            lowhi=min(lowhi,r.open,r.low)
        a,_=route_drawdown(best,peak);z,_=route_drawdown(worst,peak)
        ddlo=max(ddlo,a);ddhi=max(ddhi,z);high=max(high,exit_price)
    else:
        # exit_minute is the exclusive regular-session boundary; last full bar is already included.
        a,_=route_drawdown([exit_price],peak);ddlo=max(ddlo,a);ddhi=max(ddhi,a)
    return {'mfe_pct':100*(high/entry-1),'mae_lower_pct':100*(1-lowlo/entry),'mae_upper_pct':100*(1-lowhi/entry),
            'peak_to_trough_lower_pct':ddlo,'peak_to_trough_upper_pct':ddhi}

def clusters(frame):
    # Link same-session trades AND repeated symbols, rather than pretending all rows independent.
    n=len(frame);parent=list(range(n))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i in range(n):
        for j in range(i):
            if frame.iloc[i].date==frame.iloc[j].date or frame.iloc[i].ticker==frame.iloc[j].ticker:parent[find(i)]=find(j)
    roots=[find(i) for i in range(n)];lookup={r:k for k,r in enumerate(sorted(set(roots)))}
    return np.array([lookup[r] for r in roots])

def paired_distribution_test(a,b,group):
    """Exact cluster-preserving paired label-swap test using a KS-style ECDF distance."""
    # Remove machine-roundoff distinctions between identical +5% target fills.
    a=np.round(np.asarray(a,float),8);b=np.round(np.asarray(b,float),8);group=np.asarray(group)
    grid=np.unique(np.r_[a,b]);delta=(a[:,None]<=grid).astype(float)-(b[:,None]<=grid).astype(float)
    unique=np.unique(group);contrib=np.array([delta[group==g].sum(axis=0) for g in unique]);g=len(unique)
    observed=float(np.max(np.abs(contrib.sum(axis=0)))/len(a));extreme=0;total=2**g
    for start in range(0,total,4096):
        ids=np.arange(start,min(total,start+4096),dtype=np.uint64)
        signs=((ids[:,None]>>np.arange(g,dtype=np.uint64))&1).astype(float)*2-1
        values=np.max(np.abs(signs@contrib),axis=1)/len(a)
        extreme+=int((values>=observed-1e-12).sum())
    return {'ecdf_distance':observed,'p_value':extreme/total,'clusters':g,'exact_permutations':total}

def cluster_bootstrap_difference(a,b,groups):
    d=np.asarray(a)-np.asarray(b);unique=np.unique(groups)
    block=np.array([[d[groups==g].sum(),(groups==g).sum()] for g in unique])
    rng=np.random.default_rng(731);sample=block[rng.integers(0,len(block),size=(10000,len(block)))].sum(axis=1)
    return list(map(float,np.quantile(sample[:,0]/sample[:,1],[.025,.975])))

def summaries(a):
    return {k:float(v) for k,v in zip(['minimum','q25','median','q75','maximum'],np.quantile(a,[0,.25,.5,.75,1]))}|{'mean':float(np.mean(a))}

def main():
    ROOT.mkdir(exist_ok=True)
    (ROOT/'analysis_plan.json').write_text(json.dumps({'scope':'The same 19 later-period stock-day signals, TP +5% or session close, no stop. No winner-only filtering.',
      'primary_comparison':'Within-entry-to-exit MFE (maximum gain from entry) vs absolute MAE, using conservative exit-minute MAE upper bound; only complete pre-exit minute grids.',
      'secondary_comparison':'Uncapped 09:45-to-close MFE vs absolute MAE; only complete full-session minute grids. No post-exit observations included in actual-trade drawdowns.',
      'test':'Two-sided exact paired, cluster-preserving label-swap test of a KS-style empirical CDF distance. Clusters connect either shared trading date or ticker. Holm adjust the two planned distribution tests.',
      'diagnostics':'Cluster-bootstrap mean-difference interval; exact paired sign-test diagnostic (unadjusted, treats rows independent); observed-bar and missing-path sensitivities explicitly descriptive.',
      'limits':'Exit is selected by reaching +5%, mechanically censoring favorable excursions. Tests are descriptive, not evidence of independent strategy edge. Low/high order inside minutes is unknown. Missing paths are observed estimates only.'},indent=2))
    trades=pd.read_csv(PARENT/'tp5_or_close_test_trades.csv');rows=[]
    for r in trades.itertuples():
        b=pd.read_parquet(PARENT/'paths'/f'{r.date}_{r.ticker}.parquet').sort_values('minute')
        target=r.reason=='target';end=int(r.exit_minute)+(1 if target else 0)
        path=b[b.minute.ge(585)&b.minute.lt(end)]
        full=b[b.minute.ge(585)&b.minute.lt(960)]
        x=ohlc_bounds(b,float(r.entry),float(r.exit),int(r.exit_minute),target)
        fullx=ohlc_bounds(b,float(r.entry),float(full.close.iloc[-1]),960,False)
        rows.append({'date':r.date,'ticker':r.ticker,'entry':r.entry,'exit':r.exit,'exit_reason':r.reason,'exit_minute':r.exit_minute,
                     'preexit_missing_minutes':end-585-len(path),'full_session_missing_minutes':375-len(full),
                     **x,**{'full_'+k:v for k,v in fullx.items()},'gross_return_pct':r.gross_pct})
    data=pd.DataFrame(rows);data.to_csv(ROOT/'trade_drawdowns.csv',index=False)
    results={}
    for name,mask,mfe,mae in [('trade_window',data.preexit_missing_minutes.eq(0),'mfe_pct','mae_upper_pct'),('full_session',data.full_session_missing_minutes.eq(0),'full_mfe_pct','full_mae_upper_pct')]:
        d=data[mask].reset_index(drop=True);a=d[mfe].to_numpy();b=d[mae].to_numpy();group=clusters(d)
        test=paired_distribution_test(a,b,group)
        diff=a-b;positive=int((diff>1e-8).sum());negative=int((diff< -1e-8).sum())
        results[name]={'n_pairs':len(d),'mfe':summaries(a),'mae':summaries(b),'paired_mean_difference_pp':float(diff.mean()),'paired_median_difference_pp':float(np.median(diff)),
          'cluster_bootstrap_mean_difference_95ci_pp':cluster_bootstrap_difference(a,b,group),**test,
          'pairs_high_larger':positive,'pairs_low_larger':negative,'pairs_tied':len(d)-positive-negative,
          'paired_sign_test_diagnostic_unadjusted':float(binomtest(positive,positive+negative,.5).pvalue) if positive+negative else 1.}
    ordered=sorted(results,key=lambda n:results[n]['p_value']);last=0
    for i,name in enumerate(ordered):
        value=min(1,results[name]['p_value']*(len(ordered)-i));last=max(last,value);results[name]['holm_adjusted_p_value']=last
    # Exit-minute uncertainty sensitivity uses the optimistic MAE bound on the same complete trade windows.
    d=data[data.preexit_missing_minutes.eq(0)].reset_index(drop=True)
    results['exit_minute_mae_lower_sensitivity']={'n_pairs':len(d),**paired_distribution_test(d.mfe_pct,d.mae_lower_pct,clusters(d))}
    results['all_observed_trade_windows']={'n_pairs':len(data),'mfe':summaries(data.mfe_pct),'mae_upper':summaries(data.mae_upper_pct)}
    results['all_observed_full_sessions']={'n_pairs':len(data),'mfe':summaries(data.full_mfe_pct),'mae':summaries(data.full_mae_upper_pct)}
    (ROOT/'statistics.json').write_text(json.dumps(results,indent=2))
    report(data,results);plot(data,results)
    print(data[['date','ticker','mae_lower_pct','mae_upper_pct','peak_to_trough_lower_pct','peak_to_trough_upper_pct','preexit_missing_minutes']].to_string(index=False))
    print(json.dumps(results,indent=2))

def report(data,results):
    lines=['# Drawdowns and high/low excursion distributions','',
      'Scope: the 19 later-period signals for the saved +5% target-or-close policy, not the entire dataset. Drawdowns are gross price movements before exit; assumed transaction costs are not included.',
      '', '**MAE** is the deepest dip below entry, expressed as a positive loss magnitude. **MFE** is the largest gain above entry. **Maximum drawdown (MDD)** is the greatest percentage fall from a prior price peak while the position is open. MDD can exceed MAE.',
      '', 'Minute OHLC bars do not identify high/low ordering. MDD bounds enumerate both admissible high-first/low-first paths in every full minute. In a target-exit minute, the low may be before or after the exit; the lower bound excludes that low, while the upper bound places it before the first target touch. Post-exit highs and lows are excluded. A missing minute invalidates true path bounds: flagged rows are observed-bar estimates only. Unobserved prices could hide either a larger dip or an earlier profit-target exit, so the estimate can differ from the actual strategy path in either direction.',
      '', '| Date | Ticker | MAE lower–upper | Peak-to-trough MDD lower–upper | Missing pre-exit minutes |','|---|---|---:|---:|---:|']
    for r in data.itertuples():lines.append(f'| {r.date} | {r.ticker} | {r.mae_lower_pct:.2f}–{r.mae_upper_pct:.2f}% | {r.peak_to_trough_lower_pct:.2f}–{r.peak_to_trough_upper_pct:.2f}% | {r.preexit_missing_minutes} |')
    lines+=['','## Paired distribution comparisons','', 'High/low price levels and signed gains/losses are trivially separated; the meaningful comparison here is **upside magnitude (MFE) versus downside magnitude (absolute MAE)** for the same stock-day and window.','',
      '| Window | Complete pairs | Connected date/ticker clusters | Mean upside | Mean downside | Median upside | Median downside | ECDF distance | Exact p | Holm-adjusted p |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in ['trade_window','full_session']:
        r=results[name];lines.append(f"| {name} | {r['n_pairs']} | {r['clusters']} | {r['mfe']['mean']:.2f}% | {r['mae']['mean']:.2f}% | {r['mfe']['median']:.2f}% | {r['mae']['median']:.2f}% | {r['ecdf_distance']:.3f} | {r['p_value']:.5f} | {r['holm_adjusted_p_value']:.5f} |")
    lines+=['', 'The exact paired label-swap permutation compares a KS-style empirical CDF distance. High/low labels are swapped jointly within connected clusters sharing a date or ticker; every sign assignment is enumerated. It tests the stronger null of within-cluster upside/downside exchangeability, not arbitrary equality of means. Clustering does not fully remove serial market/regime dependence.',
      '', 'The two planned distribution tests use Holm adjustment. Bootstrap intervals and sign-test diagnostics in statistics.json are descriptive; the sign-test diagnostic treats trade rows as independent and is not the primary cluster-aware test.',
      '', '**Interpretation limits:** +5% profit-taking mechanically caps MFE for successful actual trades. A significant pre-exit distribution difference can therefore reflect the exit rule itself, not predictive skill. The fixed 09:45-to-close comparison removes that cap, but only a small number of these 19 stocks has complete minute coverage. Missing-bar session extrema are incomplete bounds and are not used in the primary test. These signals and exits were previously examined; this is exploratory analysis, not new confirmation of an edge.',
      '', 'For each test, statistics.json also includes the paired mean/median difference, a cluster-bootstrap 95% mean-difference interval, and the counts of trades with larger upside or downside. Statistical difference in shape does not imply that mean upside exceeds mean downside.',
      '', 'Reproduce: `python -m modeling.exit_study.drawdowns.analyze`. Data are read from the existing cached minute Parquet paths; no additional API requests are needed.',
      '', 'Method reference: [SciPy paired permutation tests](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html).']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')

def plot(data,results):
    fig,axes=plt.subplots(2,2,figsize=(16,11),facecolor='#f6f8fc')
    labels=[f'{r.ticker} {r.date[5:]}' for r in data.itertuples()];y=np.arange(len(data))
    a=axes[0,0];a.barh(y,-data.mae_upper_pct,color=['#d49c41' if n else '#b66c73' for n in data.preexit_missing_minutes]);a.barh(y,data.mfe_pct,color='#168a86');a.set_yticks(y,labels,fontsize=8);a.invert_yaxis();a.axvline(0,color='#333',lw=1);a.set_xlabel('Observed pre-exit excursion from entry (%)');a.set_title('Each simulated trade: upside vs downside')
    a=axes[0,1]
    for name,mask,prefix,title in [('trade_window',data.preexit_missing_minutes.eq(0),'','Complete pre-exit paths'),('full_session',data.full_session_missing_minutes.eq(0),'full_','Complete 09:45–close paths')]:
        a=axes[0,1] if name=='trade_window' else axes[1,1]
        d=data[mask];up=d[prefix+'mfe_pct'].to_numpy();down=d[prefix+'mae_upper_pct'].to_numpy()
        for values,label,col in [(up,'Upside magnitude','#168a86'),(down,'Downside magnitude','#b66c73')]:
            values=np.sort(values);a.step(np.r_[0,values],np.r_[0,np.arange(1,len(values)+1)/len(values)],where='post',label=label,color=col,lw=2)
            a.plot(values,np.arange(1,len(values)+1)/len(values),'|',color=col,markersize=8)
        p=results[name]['holm_adjusted_p_value'];a.set_title(f'{title} (n={len(d)}; adjusted p={p:.4f})');a.set_xlabel('Excursion magnitude (%)');a.set_ylabel('Cumulative fraction of stock-days');a.set_ylim(0,1.04);a.legend(fontsize=9)
    a=axes[1,0];low=data.peak_to_trough_lower_pct.to_numpy();high=data.peak_to_trough_upper_pct.to_numpy()
    a.hlines(y,low,high,color='#536781',lw=3);a.plot(low,y,'o',color='#197d9c',label='Lower OHLC bound');a.plot(high,y,'|',color='#b66c73',markersize=10,label='Upper OHLC bound');a.set_yticks(y,labels,fontsize=8);a.invert_yaxis();a.set_xlabel('Peak-to-trough maximum drawdown (%)');a.set_title('Drawdown uncertainty within minute bars');a.legend(fontsize=8)
    for a in axes.flat:a.spines[['top','right']].set_visible(False);a.grid(alpha=.12)
    fig.suptitle('The +5% target-or-close trades — drawdowns and excursion distributions',x=.06,ha='left',fontsize=19,weight='bold',color='#172b46')
    fig.subplots_adjust(left=.10,right=.96,top=.88,bottom=.16,hspace=.38,wspace=.26)
    fig.text(.06,.09,'Orange bars / flagged CSV rows have missing pre-exit minutes; their true paths are uncertain (including possible earlier target exits).',fontsize=10,color='#536781')
    fig.text(.06,.06,'Profit-taking caps pre-exit upside at +5%, so a distribution difference is partly structural. Full-session excursions continue after the modeled exit.',fontsize=10,color='#536781')
    fig.text(.06,.03,'Tests preserve paired observations and linked date/ticker clusters; adjusted p-values cover two planned comparisons. Small, previously viewed sample.',fontsize=10,color='#536781')
    for ext in ['png','pdf']:fig.savefig(ROOT/f'drawdown_distributions.{ext}',dpi=170)
    plt.close(fig)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
