# Offline RL entry/exit experiment

This folder contains a separate reinforcement-learning experiment. It reads the existing dataset; it does not retrain or replace the supervised models, change source Parquet, or place trades.

## Reproduce

From the project root:

```bash
source /home/bryan/.cache/premarket-screener-venv/bin/activate
uv pip install --python "$VIRTUAL_ENV/bin/python" -r rl_agent/requirements.txt
python -m pytest -q rl_agent/test_environment.py
python -m rl_agent.train --output rl_agent/results
python -m rl_agent.verify_results
```

Finished runs refuse to overwrite test results. Intermediate fitted agents are resumable; resume only with identical data, code and parameters. `results/COMPLETE.txt` marks completed reporting. `train.log` records the original run. Full environment versions are captured in `environment.lock.txt`.

## What the agent learns

The agent chooses **wait / enter** while flat and **hold / exit** while long. There is at most one long trade per stock-day. It may never enter. No forced participation, shorting, leverage, position sizing, re-entry, or overnight holding is introduced.

This is **finite-horizon fitted Q-learning**, an offline RL method using tree-ensemble Bellman value regressions. Historical price paths supply feasible counterfactual state/action/reward transitions under a small price-taking order assumption. Q values are learned backward through time with gamma=1, rather than training a classifier to predict a label. This is suitable for the existing small tabular dataset and does not require a GPU or repeated online interaction.

Six agents are trained: return, loss-averse return, and precision rewards, each with two fixed random seeds. Return rewards clip terminal net returns to ±50 percentage points for learning stability; evaluation reports un-clipped returns. The loss-averse reward triples negative rewards. The precision reward pays +1 for a net-profitable trade and -2 for a non-positive trade. Never entering earns zero in every objective.

## Causal observations and coarse fills

Static state: the 43 predictive features available before 09:45, excluding identifiers, dates, and outcomes. Dynamic state: elapsed time, position, returns and observed extrema from horizons that have already ended, last interval return, held-position return/drawdown, and holding time. The first execution price is hidden from the initial observation.

Decisions happen at 09:45, 09:50, 10:00, 10:15, 10:45, 11:15 and 11:45 ET. Any remaining position closes at the actual session close. At half-days, that is 13:00. There are no intrainterval decisions.

- An initial entry action fills at the historical 09:45 minute-open proxy, after the agent chooses from pre-09:45 information.
- Later entry/exit orders fill at the **next saved endpoint**. Example: an entry instruction at 09:50 fills at the saved 10:00 endpoint; an exit instruction at 10:15 fills at the saved 10:45 endpoint. This avoids observing a price and assuming a fill at that identical price. It also introduces potentially large delays, so this is not a realistic minute-level broker execution model.
- Saved endpoint prices are the final minute closes of the corresponding windows. These are historical observations, not executable bid/ask quotes. Initial and later fills have different conventions, documented explicitly.
- Forced liquidation occurs by the close, including any open position whose last decision was at 11:45. Entering at the final decision is disallowed because it would fill at the close with no trading horizon left.

Completed-horizon max gain / max decline become public dynamic observations only after their horizon ends. Values from future horizons never enter a current observation. Tests perturb future paths to check this property.

Round-trip costs are assumed to be 0.5% per position; 1% and 2% sensitivities are reported without changing the frozen policy. Intrabar targets/stops, spreads, market impact, queue priority, halts, and feed arrival latency are not modeled.

## Data split and evaluation

The same chronological exchange-date split as the earlier study is reused for comparability, with all stock-days from a date kept together and one embargo session between periods. Only date splitting and Wilson metric helpers are shared with `modeling.core`; no supervised prediction scores enter the RL observations.

1. Train: first 300 sessions. Training requires all saved endpoint prices, producing 4,399 training episodes.
2. Model validation: next 74 non-embargo sessions. Compare six agents, favoring conservative precision with minimum signal support and positive known net contribution.
3. Entry-margin validation: next 49 non-embargo sessions. Choose from four fixed entry-advantage margins in the agent's reward units. Require 80% conservative precision, 25 completed trades across ten dates, and Wilson lower bound ≥65% when achievable; otherwise disclose a research-only fallback.
4. Test: last 75 sessions, 2026-07-15 through 2026-10-28. Freeze agent and margin first. Evaluate all predeclared agents, the selected policy, staying in cash, and fixed buy-at-09:45 exits at 10:45 or the close. Never select a different favorite using test scores.

**These test dates were already viewed in previous supervised and exit studies. This is not a new independent strategy holdout.** A later untouched/paper-trading period is needed for confirmation. Further experiments on the same dates must disclose test reuse.

The pre-cutoff cohort requires all 15 opening minute bars. Training further conditions on complete future endpoints; that creates liquidity/observability selection. Evaluation does not silently remove incomplete episodes: replay stops at the first price necessary for a decision/mark/fill that is missing, marks it unresolved, and leaves P&L unknown. A known exit before later missing prices stays valid. Even an unresolved flat episode could enter later; every unresolved episode joins the conservative precision denominator as a possible loss. This is conservative but does not eliminate selection bias.

Complete-endpoint comparisons are also reported on the identical 922 test stock-days for every policy. That matched subset is selected retrospectively and is not a deployable entry filter. Compare policies on both cohorts.

Here, **precision means a net-profitable completed trade**, not the prior supervised +5%-touch label. Realized gross exits ≥5% are counted separately. Conditional mean returns exclude unresolved episodes, so they are not complete real-world estimates. Independent stock-day opportunities are not a capital-constrained portfolio; no annualized account return or Sharpe ratio is claimed.

## Outputs

- `results/REPORT.md`: findings, baselines, limitations and reproducibility.
- `results/agents/*.joblib`: six fitted agents, each containing action-value functions at all decision stages.
- `results/preprocessing.joblib`: train-only static imputation and feature ordering.
- `results/manifest.json`: data hashes, split dates, feature definitions, reward and execution rules.
- `results/frozen_selection.json`: selected agent and entry margin saved before RL test scoring.
- `results/test_metrics.csv`: individual agents, selected policy, baselines, matched subsets, costs and confidence intervals.
- `results/test_episodes.parquet`: trades, abstentions and unresolved outcomes for every policy/stock-day.
- `results/test_actions.parquet`: stage-by-stage decisions, entry state, and fill indices.
- `results/rl_overview.png`, `.pdf`: summary graphics.

Joblib artifacts are trusted local Python objects; do not load untrusted downloaded models.

Reference: [Ernst, Geurts & Wehenkel (2005), Tree-Based Batch Mode Reinforcement Learning](https://jmlr.org/papers/v6/ernst05a.html).
