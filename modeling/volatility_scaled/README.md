# Relative-activity and volatility-scaled excursion experiment

This extends the prior-day-complete cohort without replacing the existing paper strategy. The user's selected outcome is **largest intraday rise and fall after 09:45**, not return at close.

## Inputs

Current PM (04:00–09:30), last 30 PM minutes, and opening 15-minute volume, realized volatility and high/low range are compared with their own same-stock, same-window histories over the preceding 20 exchange sessions. Features include means, current/mean ratios, z-scores and historical observed-bar counts. Volume z-scores use log1p because share volumes are skewed. Volume history is split-aligned to each target day's share basis; returns within each session are already dimensionless. Only matching historical security identities contribute. A ratio/z-score needs at least 15 valid historical window observations; otherwise it remains missing. Constant histories do not produce fabricated z-scores.

Zero PM volume is accepted only when reference identity and other minute data establish the stock/day's existence. A wholly empty minute response does not become a zero-activity day. Missing historical bars can still understate activity; observed counts disclose, but cannot eliminate, this uncertainty.

## Scale and outputs

Let `sigma20` be the sample standard deviation of historical log(close / 09:45 entry), using the preceding 20 exchange sessions with at least 15 observed endpoint returns. Each prior return needs the exact 09:45 open and final regular-session close, not a stale substitute. Prior interior gaps do not affect that endpoint return. Session lengths follow actual exchange closes, so the baseline can include shorter holiday sessions.

The two regression targets are:

```
up_sigma   = log(maximum high after 09:45 / entry) / sigma20
down_sigma = log(minimum low after 09:45 / entry) / sigma20
```

They are nonnegative/nonpositive respectively. These are **risk-scaled excursions**, not centered z-scores of historical highs/lows. One sigma is a scale unit, not a statement of normality, a confidence level, or a predicted one-standard-deviation distribution. For sigma20=0.02, +1 unit means a +2.02% excursion, and -1 unit means -1.98%. Both barriers may be hit on one day. Independent predicted extrema do not reveal ordering or strategy profit.

Exact current-day targets require complete future paths. That quality requirement affects labels and metric availability, never pre-entry scoring eligibility. All rows with valid past-based scale receive predictions, even when future extrema remain unknown.

## Comparison

Original prior-day-complete source: 930 rows. Sufficient scale history: 812 rows. Exact excursion targets: 759 rows. The original calendar train/validation/test split and embargo dates are unchanged. There are 90 eligible test rows and 86 exact-target cases, **not the 19 signals from the previous classifier**.

Both feature ablations receive the old inputs plus the historical scale/return baseline. Enriched inputs add relative window activity and observed-count features. Median reference, regularized ridge, and three shallow histogram gradient boosting settings are compared. Models fit training only; validation MAE selects settings; no refit follows. Sigma-target and raw-percent-target versions are evaluated on the same cases in common sigma units. A +5% classifier control compares both feature groups using the same original precision policy. Validation is reused for settings and cutoff selection, so its scores are optimistic after tuning. No oversampling or class balancing is used.

The 10th/50th/90th quantile models are fixed before test evaluation. Their physical signs are constrained and quantiles sorted to form ordered intervals. Actual coverage and original crossing counts are reported; intervals are not guaranteed calibrated. See [scikit-learn's quantile regression documentation](https://scikit-learn.org/stable/auto_examples/ensemble/plot_hgbt_regression.html).

## Main results

On 86 fully observed test stock-days:

- +1 sigma reached: 49 (57.0%); -1 sigma reached: 50 (58.1%); both: 19 (22.1%).
- At fixed +/-5%, 26 (30.2%) reached each side. These outcomes overlap. Closing returns were positive in 43, negative in 42, flat in one.
- Upside sigma-model MAE: base 0.940, enriched 0.943, constant training median 0.935. Enrichment did not improve upside prediction.
- Downside sigma-model MAE: base 0.993, enriched 0.984, constant median 1.010. Improvement is small; enriched test R² is 0.039.
- Upside enriched test R² is -0.019. Raw-percentage target training was competitive and sometimes better, so standardization is not a demonstrated accuracy improvement.
- +5% classifier complete-window AP changed from 0.497 to 0.501; validation AP was nearly unchanged. Operating cutoffs produced different small signal sets, so precision alone is not a controlled comparison.
- Intended 80% quantile intervals covered 76.7% of upside outcomes and 77.9% of downside outcomes, with average widths around three sigma units.

Relative activity is extreme in this selected population, but does not automatically predict positive direction. The median test stock had PM volume 21.6x its prior mean and opening volume 7.8x; realized volatility ratios were about 2.8x and 2.2x. These are descriptive selected-cohort statistics, not the general stock-market population.

The idea is useful for expressing risk consistently, but these results do not show a reliable improvement in upside prediction or a profitable strategy. Exact-case metrics remain conditional on observability. Historical API responses can contain retrospective revisions, and the calendar test dates were previously examined: this is exploratory follow-up, not fresh independent confirmation.

## Files and reproduction

- `data/features.parquet`, `targets.parquet`: enriched features and explicit excursion targets.
- `data/history_summaries.parquet`, `history_parts/`, `split_events.json`: compact historical input summaries.
- `results/REPORT.md`, `metrics.csv`, `base_rates.csv`: full results, including training/validation/test.
- `results/frozen_selection.json`, `manifest.json`, `models/`: frozen settings, source hashes and saved models.
- `results/predictions.parquet`: every feature-eligible prediction, with unknown targets retained.
- `results/classifier_metrics.csv`, `median_baselines.csv`, `interval_metrics.csv`, `feature_associations.csv`: controls and diagnostics.
- `results/overview.png` and `.pdf`: standalone comparison figure.

From the project root with the existing environment:

```bash
python -m modeling.volatility_scaled.build
python -m modeling.volatility_scaled.evaluate
python -m modeling.volatility_scaled.verify
python -m pytest -q modeling/volatility_scaled/test_features.py
```

History chunks are resumable; completed evaluation results refuse overwrite. Further experiments must use distinct output paths and disclose reuse of test dates. Nine relevant tests passed. Verification reproduced prior-only split-adjusted features on 93 source rows, saved test predictions and errors, source hashes, split dates, and outcome-count accounting.
