# Market-open prediction: percentage of opening candles above entry

The requested analysis is complete. **Premarket data is the primary input. None of the fitted models beat the training-average forecast on validation RMSE, and no high-occupancy policy met the supported 80% precision gate.**

## Exact target

Reference entry is the **09:30 one-minute bar open**. Count the closes strictly above that price in **09:30–09:44 inclusive**, then divide by 15 and multiply by 100. Twelve qualifying closes means 80%. This measures 15 closing snapshots, not time continuously above entry or profit magnitude.

All 15 bars and the 09:30 open must exist to establish the exact target. Missing opening data leaves the label unknown; the stock remains in the scoring cohort. No observed-only denominator, zero return or stale entry is substituted. An auxiliary whole-candle-low-above-entry percentage is saved separately and was not optimized.

## Eligibility rebuilt to prevent selection leakage

The previous dataset required opening turnover >=$1m and complete current opening candles. Those conditions are unavailable before the open and therefore **were removed** for this experiment. Reusing the earlier filtered dataset would leak future information into selection.

Historical full-market minute files were reprocessed for all 511 target/warmup sessions. Universe: historically active common stocks/common ADRs from the existing date-specific reference universe. At the feature cutoff require:

- last observed PM price >=10% above previous official regular close;
- PM volume >=5 times mean volume over the previous ten sessions **at the same cutoff**;
- ten matched historical security identities;
- the same stock/identity has every regular-session candle on the immediately preceding trading session (390 normally, 210 on early closes), with no duplicates/invalid bars.

No current opening price, volume, turnover or completeness participates in opportunity selection or model inputs. Split adjustment uses only execution dates up to the observation date. Current full-session fields retained in daily caches are never current-day model features; prior-session versions are already known and can be used.

## Two timing assumptions

| Case | Latest included bar | Reason |
|---|---|---|
| **Delayed feed** | 09:13 bar, ending 09:14 | A 15-minute delay leaves approximately one minute before the research opening entry |
| **Real-time assumption** | 09:28 bar, ending 09:29 | Allows a pre-open decision; requires timely data unavailable on the current delayed entitlement |

These are separate opportunity sets, not test-selected alternatives. Historical arrival times, revisions, collection speed, auction deadlines and broker fills are not reconstructed. The exact minute-open fill is a research price proxy, not a guaranteed market-on-open execution. The delayed case assumes inputs and the candidate watchlist can actually be available by 09:29; the existing sequential live scanner is not asserted to meet that deadline.

## Features

All current inputs end at the phase cutoff:

- PM gap, relative volume, volume/turnover/transaction counts and average trade size;
- observed realized volatility, high/low range, position in range, approximate VWAP and close-to-VWAP;
- final 30/15/5-minute price momentum, volume and volume concentration, plus last-bar age;
- prior-day regular return/range and prior opening occupancy;
- prior-ten-session PM baselines, volume/volatility comparisons, daily liquidity, regular returns and opening-occupancy mean/count;
- pre-cutoff SPY/QQQ returns and stock-relative returns.

No current opening-window indicator or actual opening price is an input. Historical opening behaviour belongs to prior sessions and is available before today's open. Observed minute volatility/volume can understate activity where historical bars are missing; historical coverage/observation features help expose that limitation, without filling nonexistent bars.

## Models and frozen selection

Mean and median references, regularized ridge, fractional-logit regression, and shallow gradient boosting with absolute/squared loss were compared. The fractional model estimates expected occupancy without claiming that the 15 candles are independent evidence. Predictions are clipped to 0–100.

The original chronological train/validation/test calendar split and two embargo dates were reused. Models fit exact **training** targets only; all eligible validation/test rows receive scores. No refit follows selection. Primary model/feature selection minimizes validation **RMSE**, with MAE also reported. A separate precision-oriented companion maximizes the validation top20% Wilson lower bound for high occupancy (>=12/15 closes above entry), then selects a supported cutoff on validation only.

The precision gate requires >=80% conservative precision, >=25 calls over >=10 dates and Wilson lower >=65%. Unknown selected targets count against precision. Otherwise a research fallback is reported. A forecast of 80% occupancy is **not** an 80% probability of exceeding the high-occupancy threshold.

The calendar test dates have been viewed repeatedly in earlier studies. This is exploratory research, not independent confirmation. Validation itself is tuned and is not a fresh score.

## Dataset

| Case | Total stock-days | Unique tickers | Exact targets |
|---|---:|---:|---:|
| Delayed | 975 | 566 | 948 |
| Real-time assumption | 1,001 | 594 | 964 |

| Case / split | Eligible rows | Exact targets |
|---|---:|---:|
| Delayed training | 613 | 593 |
| Delayed validation | 246 | 241 |
| Delayed test | **114** | **112** |
| Real-time training | 635 | 610 |
| Real-time validation | 249 | 241 |
| Real-time test | **115** | **111** |

Two stock-days per timing case lie on embargo dates. The source first-15 turnover restriction was removed, so these are not the earlier 19 signals or 90 test stock-days. A common-cohort diagnostic uses 102 test opportunities, 100 with exact targets, shared between timing cases.

## Prediction results

Validation selected the **constant training mean** for the primary percentage forecast in both timing cases. The fitted models did not beat it on validation RMSE, so there is no demonstrated useful stock-specific percentage forecast.

| Timing case | Constant forecast | Actual mean in exact test rows | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|---:|---:|
| Delayed | 44.68% | 39.88% | **35.25 points** | 38.62 points | -0.016 |
| Real-time assumption | 45.07% | 42.34% | **35.79 points** | 39.04 points | -0.005 |

An MAE of 35 percentage points is about **5.3 of the 15 candles**, not 35% trading accuracy. A constant forecast around 45% is a dataset average and cannot distinguish individual stocks. Test R² near/below zero reflects this lack of explanatory power. The common-cohort test RMSE is 38.82 points for delayed and 38.86 for real-time; differing opportunity sets do not explain away the weak results.

Previous-day occupancy was also a weak reference (test MAE 43.69/43.18 points); prior-ten-day mean MAE was 34.71/36.12 points, without a consistent improvement on RMSE. The training-median reference sometimes has lower test MAE, but was not retroactively selected after reading test results.

## Precision companion

The delayed precision-oriented selection is also the constant mean: all scores tie, so it does **not** provide selective ranking. Nominal top5/10/20% diagnostics therefore include all tied scores and must not be interpreted as selective signals.

The real-time precision-oriented selection is ridge regression on premarket inputs:

| Split | Calls | >=12/15 hits | Unknown calls | Conservative precision |
|---|---:|---:|---:|---:|
| Training (in sample) | 113 | 64 | 6 | 56.6% |
| Validation (tuned) | 49 | 29 | 1 | 59.2% |
| Reused test | **26** | **7** | **1** | **26.9%** |

The exact test-label base rate was 28.8%, so this did not improve on the base rate. No candidate met the supported 80% validation precision gate. Current test unknowns remain in the conservative precision denominator.

## Artifacts and verification

- `plan.json`: predeclared target, eligibility, timings, features and model-selection objectives.
- `data/daily/`: compact market-wide summaries and S3 object provenance; downloaded raw files were deleted after successful summarization.
- `data/dataset.parquet`, `eligibility_audit.csv`, `manifest.json`, `split_events.json`: reconstructed source cohort and targets.
- `results/REPORT.md`, `regression_metrics.csv`, `precision_metrics.csv`, `ranking_precision.csv`: full train/validation/test results.
- `results/baselines.csv`, `calibration.csv`, `target_distribution.csv`, `matched_cohort_metrics.csv`: reference forecasts, calibration and shared-cohort diagnostics.
- `results/frozen_selection.json`, `manifest.json`, `models/`, `predictions.parquet`: frozen settings, fitted artifacts and all forecasts.
- `results/overview.png` and `.pdf`: standalone results figure.
- `results/SERIALIZATION_REPAIR.json`: one custom estimator's module path was repaired for reliable loading; fitted weights, cutoffs and all saved predictions were preserved, without retraining or reselection.

```bash
python -m modeling.open_occupancy.build
python -m modeling.open_occupancy.evaluate
python -m modeling.open_occupancy.verify
python -m pytest -q modeling/open_occupancy/test_open.py
```

Use the existing project Python environment. Daily source summaries resume; completed evaluated results refuse overwrite. Eleven relevant tests passed. Verification checked 51 prior-grid/cutoff/history samples, integer target counts, source/plan hashes, every saved test prediction/error/precision and frozen chronology. Tests explicitly mutate current opening prices/coverage/volume and verify that eligibility and pre-open inputs do not change; removing a prior-day candle does exclude the opportunity. Stable custom-model serialization is also tested.

This is an occupancy prediction experiment, not a profitable strategy backtest. Spending most sampled minutes above entry does not determine executable exit return, spreads, drawdowns or costs. Existing paper strategy files and model are unchanged.
