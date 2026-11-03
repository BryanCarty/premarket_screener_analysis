# First-touch barrier target and opening-path feature experiment

This is the requested follow-up: test a shorter, trade-aligned target with opening-path and market-relative features. It does not replace the existing paper strategy or model.

## Fixed target

At the historical **09:45 minute-open entry proxy**, set an upper barrier `entry * exp(sigma20)` and lower barrier `entry * exp(-sigma20)`. Predict which is reached first in **09:45 inclusive to 10:45 exclusive**:

- **up:** upper barrier first;
- **down:** lower barrier first;
- **neither:** neither barrier touched in a complete 60-minute path.

`sigma20` is the already constructed, prior-only standard deviation of same-stock **09:45-to-close** log returns over the previous 20 sessions, requiring at least 15 valid identity-matched returns. It is **not a 60-minute volatility estimate** and not a probability guarantee. Keeping this scale fixed isolates the effect of the new first-touch target.

A missing minute before the deciding event or an unresolved minute touching both barriers gives an unknown label. An opening gap beyond a barrier establishes which was crossed at that open. Stop gaps fill at the worse minute open; take-profits are capped at target. Missing minutes after an established exit do not change the result. A timeout exits at the 10:44 bar close. No additional stop/target/horizon search was performed.

## Features

All feature calculations stop strictly before 09:45. The same eligible stock-days are scored for every variant:

1. **Base:** earlier inputs plus shared prior volatility scale and return history.
2. **Opening path:** adds three five-minute price/estimated-turnover windows, acceleration, pullback from high, bounce from low, close-based drawdown, upper-wick fractions, running VWAP crossings, minutes holding above premarket high, breakout crossings and failed breakouts.
3. **Path + market:** adds contemporaneous SPY/QQQ first/middle/last-five-minute and opening-15 returns, plus stock excess returns over those same windows.
4. **Full:** also adds prior-window relative volume/volatility/range features from the previous study.

A historical quote-access probe returned **HTTP 403**. No spread, bid/ask depth, aggressor classification, order-flow imbalance or news/catalyst features were included. Bar-based pressure measures are proxies, not actual order flow. New minute bars can include vendor revisions; this is not an as-received historical replay.

Original cohort rules remain: premarket gap >=10%, relative premarket volume >=5, complete opening 15 minutes, >=$1m opening turnover and all prior regular-session candles. A sufficient historical volatility scale is also required. The comparison covers qualified stock-days, **not the previous model's 19 signals**.

## Training and selection

Original calendar splits and embargo dates are reused. Eligible rows: training 513, validation 207, test 90. Three training outcomes are unknown; all validation/test outcomes resolve. No balancing, oversampling or class weights are used.

Each feature set compares a constant training-prior baseline, regularized logistic regression, and three fixed shallow gradient-boosting settings. The minimum known-label validation multiclass log loss chooses the model. Models fit training only; there is no refit after selection.

For long-entry signals, the same validation period selects a probability cutoff using the original supported 80% conservative precision policy; if no cutoff qualifies, a research fallback is retained. The global primary feature set is also frozen by validation log loss before test scoring. Test scores never select the primary model. Because validation is tuned, it is not an independent evaluation.

**Previously viewed calendar test dates are reused. This remains exploratory follow-up, not independent confirmation.**

## Results

### Class frequencies

| Split | Upside first | Downside first | Neither | Unknown |
|---|---:|---:|---:|---:|
| Training | 179 | 222 | 109 | 3 |
| Validation | 78 | 83 | 46 | 0 |
| Test | **27 (30.0%)** | **38 (42.2%)** | **25 (27.8%)** | **0** |

Downside-first is more common here, but the majority class is not enough to establish an edge. Every outcome is mutually exclusive under this target, unlike independently counting eventual upside/downside excursions.

### Identical-cohort test comparison

| Inputs | Selected model | Log loss (lower better) | Upside AP (higher better) |
|---|---|---:|---:|
| Constant training frequencies | Prior baseline | 1.094 | 0.300 |
| **Base** | **Logistic regression** | **1.043** | **0.398** |
| Opening path | Shallow boosting | 1.064 | 0.407 |
| Path + market | Shallow boosting | 1.073 | 0.380 |
| Full | Shallow boosting | 1.112 | 0.301 |

Validation selected **base/logistic** as the primary model. The opening-path model has slightly higher test upside AP, but was not selected on that basis; its validation/test multiclass probability loss is worse than base. Added features did not improve the primary metric. Full enriched inputs performed worse than the simple prior baseline on test log loss.

### Primary frozen policy: bracket or timeout

The validation-selected cutoff was 0.3991170193324285. It did **not** pass the 80% validation precision gate (validation upside-first precision 46.5%); this is a research fallback.

On test:

- 44 signals, all resolved; 16 upside-first outcomes, 28 other outcomes.
- 20 profitable trades, 24 losing trades (a timeout can be profitable without reaching the upper barrier).
- Mean gross return **-0.24%**; median **-0.63%**.
- Mean after assumed 0.5% round-trip cost **-0.74%**; after 1% cost **-1.24%**.
- Average win **+2.27%**, average loss **-2.33%**.
- Best **+6.03%**, worst **-6.27%**; barrier sizes vary by each stock's historical scale.
- Day-bootstrap 95% interval for mean gross return: approximately **-0.97% to +0.57%**.

The other feature policies also have negative mean test returns after 0.5% assumed costs. Their signals differ because cutoffs were selected independently; compare identical-cohort probability metrics for feature contributions, not returns across different signal subsets.

Returns equally weight hypothetical positions and are not portfolio returns. No spreads, fills, halts, commissions, capital constraints or market impact are reconstructed. Historical 09:45 entries differ from the live delayed-feed paper runner's future-minute entries. These results do not establish a profitable edge.

## Interpretation

This target is more directly aligned with an executable bracket decision and controls the large loss tails of the former no-stop rule. It did **not** reveal a reliable profitable strategy. Some probability ranking information remains, but on this sample opening-path/market/relative-activity additions did not outperform the simpler inputs on the predeclared model-selection metric. More features are not automatically more informative.

## Artifacts and verification

- `data/dataset.parquet`: base/enriched inputs and `label_`-prefixed outcomes.
- `data/paths/`: archived stock and benchmark minute paths through 10:45.
- `data/label_audit.csv`, `manifest.json`: coverage/outcome reasons and input provenance.
- `results/REPORT.md`, `metrics.csv`: train/validation/test classifier results.
- `results/strategy_metrics.csv`: per-policy trade counts, returns and cost sensitivities.
- `results/validation_comparison.csv`, `frozen_selection.json`, `models/`: candidate comparisons and frozen artifacts.
- `results/predictions.parquet`: all eligible predictions, probabilities, calls and exit outcomes.
- `results/confusion_matrices.csv`, `base_rates.csv`: imbalance diagnostics.
- `results/overview.png` and `.pdf`: standalone results figure.

```bash
python -m modeling.barrier_study.build
python -m modeling.barrier_study.evaluate
python -m modeling.barrier_study.verify
python -m pytest -q modeling/barrier_study/test_barrier.py
```

Use the existing project Python environment. Path downloads are resumable. Completed evaluations refuse overwrite. Eleven relevant tests passed. Independent checks reproduced opening/benchmark features and labels from 82 archived paths, source hashes, model probabilities/cutoffs, test log loss/returns, split membership and freeze ordering. A future independent period remains necessary before any deployment decision.
