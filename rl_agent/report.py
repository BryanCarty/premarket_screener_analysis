"""Transparent reports for the coarse, delayed-execution RL experiment."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def pct(x):return '—' if pd.isna(x) else f'{100*x:.1f}%'
def number(x):return '—' if pd.isna(x) else f'{x:+.2f}%'

def report(out):
    out=Path(out);m=json.loads((out/'manifest.json').read_text());frozen=json.loads((out/'frozen_selection.json').read_text());r=pd.read_csv(out/'test_metrics.csv');episodes=pd.read_parquet(out/'test_episodes.parquet');v=pd.read_csv(out/'validation_metrics.csv')
    chosen=r[r.policy.eq('SELECTED')].iloc[0];matched=r[r.policy.eq('SELECTED_complete_subset')].iloc[0]
    text=['# Offline RL: entry and exit decisions','',
          f"Algorithm: **finite-horizon fitted Q-learning** with tree ensembles. Six agents were trained: three reward definitions × two random seeds. Selected agent: **{frozen['selected_agent']}**, entry advantage margin **{frozen['selected_margin']}**.",
          '', '**This is a coarse horizon-landmark research simulator, not minute-level or live trading.** It uses existing saved feature/label Parquet without further downloads. Future horizon outcomes are used for training rewards and transitions, never as observations before the corresponding horizon has elapsed.',
          '', '## Main held-out-period result','',
          f"Across {int(chosen.episodes):,} scorable stock-days: {int(chosen.closed_trades)} completed trades, {int(chosen.wins)} wins, {int(chosen.losses)} non-winning trades, {int(chosen.no_trade)} confirmed no-trades, and {int(chosen.unresolved_episodes)} unresolved episodes.",
          f"Known completed-trade win precision: **{pct(chosen.win_precision_known)}**. Conservative precision including every unresolved episode as a potential losing trade: **{pct(chosen.win_precision_conservative)}**. Mean net return per completed trade: **{number(chosen.mean_trade_net_pct)}** at assumed 0.5% round-trip cost.",
          '', f"On the common, fully observed endpoint subset ({int(matched.episodes)} stock-days), selected-agent precision was {pct(matched.win_precision_known)} across {int(matched.closed_trades)} trades. This subset is selected using future data availability and is more liquid; it is a diagnostic matched comparison, not a deployable entry filter.",
          '', f"The selected entry policy {'met' if frozen['threshold_qualified_80pct'] else 'did not meet'} the 80% conservative win-precision gate in the earlier threshold-validation period. No-trade is allowed. If an agent loses money, declines to trade, or underperforms a simple baseline, that result is retained.",
          '', '**Assessment:** the selected agent did not achieve the precision goal. Its mean known trade return was slightly negative, and its P&L contribution on the identical complete-endpoint cohort was below staying in cash. The loss-averse and precision-reward agents made no trades on fully observed test episodes. This prototype does not establish a tradable edge.',
          '', '**The chronological test dates were already viewed in earlier supervised-model and exit-policy studies.** RL agents and entry margins were selected without using their RL test metrics, but this is not a genuinely new independent strategy holdout.',
          '', '## Timeline and execution','',
          '- Decisions at 09:45, 09:50, 10:00, 10:15, 10:45, 11:15, and 11:45 Eastern. Forced flat at the actual regular-session close (13:00 on half-days).',
          '- Actions: flat → wait or enter; long → hold or exit. At most one long trade per stock-day. No shorting, leverage, sizing optimization, or re-entry.',
          '- At 09:45 the initial opening execution price is hidden from the agent and used only after its initial entry action. Initial observations consist only of pre-09:45 saved features.',
          '- Later buy/sell instructions fill at the **next saved landmark endpoint**, not the same price used in the current observation. For example: a 09:50 entry instruction fills at the saved 10:00 endpoint; a 10:15 exit instruction fills at the saved 10:45 endpoint. These delayed historical price proxies avoid same-price lookahead but do not reproduce a practical broker-order execution model.',
          '- There are no decisions inside the long 11:45-to-close interval. Horizon end prices are the preceding minute close represented at the horizon boundary, not NBBO quotes.',
          '- Round-trip execution cost is 0.50 percentage points per position; 1% and 2% sensitivities are included. Limits, stops, queue priority, market impact, and halts are not simulated.',
          '', '## Chronological splits','','| Period | First date | Last date | Sessions | Opening-complete episodes | Complete endpoint paths |','|---|---|---|---:|---:|---:|']
    for k in ['train','validation','threshold','test']:
        d=m['split_counts'][k];text.append(f"| {k} | {d['first']} | {d['last']} | {d['sessions']} | {d['episodes']} | {d['complete_endpoints']} |")
    text+=['','All stock-days from one exchange date remain in the same period. One full session is embargoed at each boundary. Static imputation is fit on complete training episodes only. Agents are not refit using validation, threshold, or test observations.',
           '', '## Six-agent comparison on model validation','','| Agent | Completed trades | Unknown episodes | Known win precision | Conservative precision | Mean net per known trade |','|---|---:|---:|---:|---:|---:|']
    for _,x in v.iterrows():text.append(f'| {x.agent} | {int(x.closed_trades)} | {int(x.unresolved_episodes)} | {pct(x.win_precision_known)} | {pct(x.win_precision_conservative)} | {number(x.mean_trade_net_pct)} |')
    text+=['','## Final test: same fully observed endpoint cohort','','No rows are silently dropped from the full test report. The table below separately compares all policies on the identical complete-endpoint subset, where every required decision/exit price is known. Precision means a **net-profitable completed trade**, not the original +5% target-touch label.','',
           '| Policy | Stock-days | Trades | Win precision | Mean net / trade | Known net contribution / opportunity | Worst trade | Mean net at 1% cost | Mean net at 2% cost |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    subset=r[r.policy.str.endswith('_complete_subset')]
    for _,x in subset.iterrows():text.append(f'| {x.policy.removesuffix("_complete_subset")} | {int(x.episodes)} | {int(x.closed_trades)} | {pct(x.win_precision_known)} | {number(x.mean_trade_net_pct)} | {number(x.known_net_sum_per_opportunity_pct)} | {number(x.worst_trade_net_pct)} | {number(x.mean_trade_net_100bps_pct)} | {number(x.mean_trade_net_200bps_pct)} |')
    text+=['','## Full cohort and missing-data uncertainty','','| Policy | Trades | Wins | Losses / flat exits | Unknown episodes | Never entered | Conservative precision | Known precision | Mean known trade net |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for _,x in r[~r.policy.str.endswith('_complete_subset')].iterrows():text.append(f'| {x.policy} | {int(x.closed_trades)} | {int(x.wins)} | {int(x.losses)} | {int(x.unresolved_episodes)} | {int(x.no_trade)} | {pct(x.win_precision_conservative)} | {pct(x.win_precision_known)} | {number(x.mean_trade_net_pct)} |')
    text+=['','Unresolved means a price needed for the next decision, mark, or execution was absent. It does not mean zero P&L. Unresolved flat episodes could have entered later, so they also join the conservative precision denominator. Known net contribution/opportunity is only the sum of observed completed P&L divided by all opportunities; it is not a complete portfolio-return estimate when episodes are unresolved.',
           '', '## Reinforcement learning design','',
           '- Training is finite-horizon fitted Q-learning: construct feasible wait/enter/hold/exit transitions from historical price paths; learn terminal rewards and bootstrap the maximum valid next-state action value backward through time. This is not a classifier trained on a runner label.',
           '- The market is assumed unaffected by these hypothetical small trades. Counterfactual actions are available in the historical simulator, but market impact and realistic order execution are not.',
           '- The return reward uses clipped terminal net return [-50,+50] percentage points. Loss-averse reward triples negative clipped returns. Precision reward pays +1 for a net-positive trade and -2 for a non-positive trade. A no-trade episode earns zero. Reported test returns are un-clipped.',
           '- Gamma is 1.0 because decision intervals are irregular and the horizon is finite. Learning uses all valid position/entry states, including counterfactual entry points. Invalid actions are masked.',
           '- Six fitted agents (three rewards × two seeds) are compared on earlier model-validation dates. A separate threshold period chooses an entry advantage margin from [0,0.1,0.25,0.5] in the agent reward units. Choices are frozen before RL test scoring.',
           '- Entry filtering aims for 80% conservative precision with at least 25 closed trades across ten dates and a 65% Wilson lower bound. If no supported policy meets it, the best supported fallback is disclosed as research-only.',
           '', '## Limits and next confirmation','',
           'The dataset has only two years, one pre-market eligibility population, coarse endpoints, retrospectively revised bars, and no executable quotes. Complete-path training can create liquidity selection bias. Reward clipping, extra-trees approximation, maximization bias, and selection over several agents can affect results. Two random seeds are a small stability check, not exhaustive robustness testing.',
           '', 'Each stock-day is an independent fixed-notional hypothetical opportunity. Multiple simultaneous opportunities are not consolidated into a capital-constrained portfolio. There is no annualized return, Sharpe-ratio claim, account equity curve, or live-trading recommendation. A genuinely later paper-trading period and minute-level execution-aware data are needed before deployment.',
           '', '## Artifacts and reproduction','',
           '- `agents/*.joblib`: fitted Q functions for every agent; `preprocessing.joblib`: train-only static imputer.',
           '- `manifest.json`: split dates, source hashes, observations, reward and execution rules.',
           '- `frozen_selection.json`: agent/margin selected before RL test scoring.',
           '- `test_metrics.csv`, `test_episodes.parquet`, `test_actions.parquet`: results, stock-level trades/no-trades/unknowns, and action traces.',
           '- `rl_overview.png` and `.pdf`: matched-cohort comparisons and selected-agent decisions.',
           '', 'Rebuild from project root: `python -m rl_agent.train --output rl_agent/results_NEW`. Original feature/label files are read only. A completed test result cannot be overwritten silently. Reusing this previously viewed test period for further selection is not a new independent evaluation.',
           '', 'Method: [Ernst, Geurts & Wehenkel (2005), Tree-Based Batch Mode Reinforcement Learning](https://jmlr.org/papers/v6/ernst05a.html).']
    (out/'REPORT.md').write_text('\n'.join(text)+'\n')
    graph(out,r,episodes)
    (out/'COMPLETE.txt').write_text('Training, frozen agent selection, chronological evaluation, and report generation complete.\n')
    print(out/'REPORT.md',flush=True)


def graph(out,r,episodes):
    names=['SELECTED_complete_subset','buy945_exit60_complete_subset','buy945_exitclose_complete_subset','cash_complete_subset']
    labels=['Selected RL','Buy 09:45 → 10:45','Buy 09:45 → close','Stay in cash']
    s=r.set_index('policy').loc[names]
    fig,axes=plt.subplots(2,2,figsize=(15,10),facecolor='#f6f8fc');colors=['#168a86','#5e7fa6','#8798ad','#b7c3d1']
    a=axes[0,0];a.bar(labels,s.known_net_sum_per_opportunity_pct,color=colors);a.axhline(0,color='#333',lw=1);a.set_ylabel('Net % per equal-notional opportunity');a.set_title('Identical complete-endpoint cohort: P&L contribution');a.tick_params(axis='x',rotation=13)
    a=axes[0,1];a.bar(labels[:3],s.win_precision_known.iloc[:3]*100,color=colors[:3]);a.set_ylim(0,105);a.set_ylabel('Profitable completed trades (%)');a.set_title('Precision after 0.5% assumed round-trip cost')
    for i,(_,x) in enumerate(s.iloc[:3].iterrows()):a.text(i,float(x.win_precision_known*100)+2,f'n={int(x.closed_trades)}',ha='center')
    selected=episodes[episodes.policy.eq('SELECTED')];closed=selected[selected.status.eq('closed')]
    a=axes[1,0]
    if len(closed):
        a.hist(closed.net_pct.clip(-50,50),bins=25,color='#168a86',edgecolor='white');a.axvline(0,color='#a76b00',ls='--')
    a.set_xlabel('Selected-agent known trade net % (display clipped ±50%)');a.set_ylabel('Trades');a.set_title('Full-cohort known trade outcomes')
    a=axes[1,1];counts=[int(selected.status.eq(x).sum()) for x in ['closed','no_trade','unresolved']];a.bar(['Closed trade','No trade','Unresolved'],counts,color=['#168a86','#5e7fa6','#e6b55b'])
    for i,n in enumerate(counts):a.text(i,n+max(counts)*.02,f'{n:,}',ha='center')
    a.set_ylim(0,max(counts)*1.15 if max(counts)>0 else 1);a.set_title('Selected agent: every test opportunity accounted for');a.set_ylabel('Stock-days')
    for a in axes.flat:a.spines[['top','right']].set_visible(False);a.grid(axis='y',alpha=.12)
    fig.suptitle('Offline RL — causal entry / exit decisions at horizon landmarks',x=.06,ha='left',fontsize=19,weight='bold',color='#172b46')
    fig.subplots_adjust(left=.07,right=.96,top=.88,bottom=.16,hspace=.5,wspace=.28)
    fig.text(.07,.085,'Later orders fill at the NEXT saved endpoint. One long trade maximum per stock-day; forced flat by close. No minute-level execution model.',fontsize=10,color='#536781')
    fig.text(.07,.055,'Matched-cohort comparison conditions on future endpoint availability. Full-cohort unknowns remain unresolved. Historical test dates were previously viewed.',fontsize=10,color='#536781')
    fig.text(.07,.025,'P&L is a simulation per independent opportunity, not a capital-constrained portfolio return or a live profitability estimate.',fontsize=10,color='#536781')
    for ext in ['png','pdf']:fig.savefig(out/f'rl_overview.{ext}',dpi=170)
    plt.close(fig)

if __name__=='__main__':
    import sys
    report(Path(sys.argv[1] if len(sys.argv)>1 else 'rl_agent/results'))
