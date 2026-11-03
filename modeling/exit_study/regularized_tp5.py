"""Evaluate the requested TP5-or-close rule from existing immutable outcome summaries."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main(model_results='modeling/results_regularized_5pct_close', dataset='data/master_open15_1m'):
    source=Path(model_results)
    out=source/'tp5_or_close'
    out.mkdir(exist_ok=True)
    pred=pd.read_parquet(source/'predictions.parquet')
    calls=pred[pred.split.eq('test')&pred.predicted_runner]
    labels=pd.read_parquet(Path(dataset)/'labels.parquet')
    d=calls.merge(labels[labels.horizon.eq('close')],on=['date','ticker','identity'],suffixes=('_prediction',''),validate='one_to_one')
    hit=d.runner_5pct.eq(True).fillna(False)
    miss=d.runner_5pct.eq(False).fillna(False)
    d['strategy_return_pct']=np.nan
    d['exit_reason']='unresolved'
    d.loc[hit,'strategy_return_pct']=5.
    d.loc[hit,'exit_reason']='take_profit'
    closing=miss&d.end_return_pct.notna()
    d.loc[closing,'strategy_return_pct']=d.loc[closing,'end_return_pct']
    d.loc[closing,'exit_reason']='close'
    # An observed hit determines a fixed TP exit even if earlier bars are absent.
    # Without an observed hit, interior gaps prevent establishing that TP never triggered.
    r=d.strategy_return_pct.dropna();wins=r[r>0];losses=r[r<0]
    summary=dict(test_signals=len(d),resolved_trades=len(r),unresolved_trades=int(d.strategy_return_pct.isna().sum()),take_profit_exits=int(hit.sum()),close_exits=int(closing.sum()),wins=len(wins),losses=len(losses),breakeven=int(r.eq(0).sum()),win_rate_pct=100*(r>0).mean(),mean_return_pct=r.mean(),median_return_pct=r.median(),average_gain_pct=wins.mean(),average_loss_pct=losses.mean(),max_gain_pct=r.max(),max_loss_pct=r.min(),profit_factor=wins.sum()/(-losses.sum()),mean_dollar_pnl_per_1000=r.mean()*10)
    unknown=d[d.strategy_return_pct.isna()]
    known_bounds=unknown.entry_available&unknown.end_return_pct.notna()
    lower_sum=r.sum()+unknown.loc[known_bounds,'end_return_pct'].sum()-100*int((~known_bounds).sum())
    upper_sum=r.sum()+5*len(unknown)
    summary['all_signal_mean_lower_bound_pct']=lower_sum/len(d)
    summary['all_signal_mean_upper_bound_pct']=upper_sum/len(d)
    summary['unresolved_with_close_price']=int(known_bounds.sum())
    sensitivity=[]
    for cost in (0,.5,1,2):
        net=r-cost
        sensitivity.append(dict(round_trip_cost_percentage_points=cost,mean_net_return_pct=net.mean(),wins=int(net.gt(0).sum()),losses=int(net.lt(0).sum()),win_rate_pct=100*net.gt(0).mean()))
    d.to_csv(out/'trades.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(out/'cost_sensitivity.csv',index=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    worst=d.loc[d.strategy_return_pct.idxmin()]
    text=['# Regularized model: test signals, +5% take profit or close','',
          'Entry is 09:45 minute open. Exit at exactly entry * 1.05 if an observed high reaches the target; otherwise use final regular-session minute close only if complete bars establish no target hit. No stop loss. Assumes full target fills without spread, fees, slippage, or impact. Known target hits remain resolved with missing earlier candles because the fixed target exit price is unchanged. Unknown no-hit paths remain unresolved, even where a closing price exists.','',
          'All metrics below describe resolved trades and can be biased by nonrandom missingness. Test calendar dates were previously viewed. Equal dollar weighting per trade; mean return is not a portfolio or compounded return.','',
          '```',json.dumps(summary,indent=2),'```','',
          f"Worst resolved trade: {worst.ticker}, {worst.date}, {worst.strategy_return_pct:.2f}%; entry ${worst.entry_price_proxy:.6f}, close ${worst.entry_price_proxy*(1+worst.end_return_pct/100):.6f}.", '',
          'Full-cohort bounds assume each unknown trade could at most return +5%; if entry and final close are known, the closing return is a lower bound and +5% is the upper bound. Otherwise -100% is used as a theoretical long-position lower bound. These are missing-data bounds, not statistical confidence intervals.','',
          'Cost sensitivities for resolved trades:','```',pd.DataFrame(sensitivity).to_string(index=False),'```']
    (out/'REPORT.md').write_text('\n'.join(text)+'\n')
    assert len(d)==len(calls) and len(r)+len(unknown)==len(d)
    assert len(wins)+len(losses)+int(r.eq(0).sum())==len(r)
    assert r.le(5+1e-10).all()
    assert d.loc[closing,'missing_minutes'].eq(0).all()
    print(json.dumps(summary,indent=2));print('Worst:',worst[['date','ticker','strategy_return_pct']].to_dict())
    print(pd.DataFrame(sensitivity).to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-results',default='modeling/results_regularized_5pct_close')
    p.add_argument('--dataset',default='data/master_open15_1m')
    args=p.parse_args()
    main(args.model_results,args.dataset)
