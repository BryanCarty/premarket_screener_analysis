# Modeling complete

Compared logistic regression, random forests, Extra Trees, and histogram gradient boosting across 38 supported runner definitions. Four definitions lacked sufficient positive-example support. All model-family selections and decision cutoffs were fixed before final held-out scoring.

Final test: 2026-07-15 through 2026-10-28, 75 sessions, 1,018 scorable stock-days with complete opening 15-minute bars. Across 38 definitions: 38,684 held-out prediction rows. Saved 38 fitted pipelines and policies.

The strongest preselected candidate was Extra Trees for +5% by close: 17 observed hits, one confirmed failure, one unknown among 19 test calls. Conservative precision 89.5%; day-bootstrap 95% interval 72.7–100%. This was the only policy that met the 80% threshold-validation gate. Signal count is small and complete-window recall was 2.6%.

Results: `results/REPORT.md`, `results/precision_overview.png`, `results/test_results.csv`, `results/test_predictions.parquet`, and `results/models/`.

Three metric/split/policy tests passed. Artifact verification passed: source hashes, test dates, frozen thresholds, totals, and reloaded-model score reproduction. See README.md for reproduction and missing-data limitations. Further experiments on the same final test period would reuse a previously viewed holdout.
