# Precision-first runner models

All modeling artifacts are separate from the original dataset. Source Parquet files are read only.

## Run

From the project root, using the project's existing environment:

```bash
source /home/bryan/.cache/premarket-screener-venv/bin/activate
uv pip install --python "$VIRTUAL_ENV/bin/python" -r modeling/requirements.txt
python -m pytest -q modeling/test_core.py
python -m modeling.train --output modeling/results
```

The final report is `modeling/results/REPORT.md`; charts, held-out predictions, metrics, source hashes, and fitted models are alongside it. `modeling/train.log` records the original run. `COMPLETE.txt` appears only after reports finish. Each supported runner definition has a checkpoint and fitted model. Repeating an interrupted command reuses completed definition checkpoints; a completed experiment refuses to overwrite its held-out results. Do not change code, data, or policies while resuming an experiment. Further experiments on this same holdout are no longer an untouched test and must be disclosed.

## Fixed experiment design

- Seven horizons × six upside thresholds are considered. A definition requires at least 30 train positives and negatives, five model-validation observed positives, and five threshold-validation observed positives. Sparse definitions are skipped rather than given unstable claims.
- Four fixed model families: regularized logistic regression, random forests, Extra Trees, and histogram gradient boosting. No broad hyperparameter search, no balancing, and no accuracy optimization. Forests use two CPU workers; numerical libraries are capped at two threads.
- All stock-days from one session stay in one chronological partition: first 60% of exchange dates for training, next 15% for model validation, next 10% for threshold tuning, last 15% for the final test. One session is embargoed at each boundary. Input feature statistics are fit only on the appropriate training rows.
- For each target, model validation chooses the family by the Wilson lower bound of conservative precision among the top-scored 10% of the model-validation cohort. Complete-window average precision breaks ties. The selected family is then refit on train + model-validation only.
- Separate later threshold-validation data chooses the largest signal set achieving 80% conservative precision, at least 25 calls across ten dates, and a Wilson lower bound of at least 65%. If no cutoff meets that policy, the strongest supported lower-bound cutoff is saved as a **research-only fallback**. Its test metrics are still disclosed, but it is not labeled as meeting the precision target.
- Threshold selection and the five priority runner definitions are frozen in `frozen_selection.json` before final test scoring. Priorities are chosen by threshold-validation qualification, lower bound, and signal count; the holdout does not select a winner. Test results for other definitions are descriptive and must not be treated as an independent test of a newly selected favorite.
- Prediction scores are ranking scores, not asserted calibrated probabilities. The default 0.5 cutoff is not assumed suitable for the user's precision goal.

## Missing outcome bias

The original labels keep an observed positive even with missing bars, but cannot declare a negative from missing bars. Training on every non-null label would therefore retain incompletely observed positives and drop similarly incomplete negatives. This experiment avoids that asymmetry by training only on complete outcome windows for both classes.

Scoring uses a fixed, pre-09:45 activity screen: all 15 opening minute bars must exist. It does **not** require future complete coverage. Every test row in that cohort is scored. Primary precision is observed hits divided by **all** flagged stock-days, including unknown outcomes in the denominator. Also reported are known-label precision, lower/upper missing-outcome bounds, unknown counts, complete-window recall/AP, and date-cluster bootstrap intervals. This is more conservative than silently dropping unknown test outcomes, but does not eliminate all liquidity/observability shift between complete training cases and the scoring population.

## Interpretation

Precision is the fraction of predicted runners that actually reach the specified upside within the horizon. Recall is the fraction of runners found. Choosing high precision generally means fewer calls and lower recall. Large confidence intervals or small signal counts cannot establish a stable edge.

A threshold hit is not a realized trading return. The entry is a historical 09:45 minute-open proxy; no spreads, slippage, executable exit timing, costs, or live feed latency are simulated. The final report records these limits. The same security may appear across dates; identifiers and dates are excluded from model inputs. This is an event-time research experiment, not a deployed trading system.

The test period should not be repeatedly mined for better settings. The next confirmation step, if results justify it, is a genuinely later chronological/paper-trading period with frozen feature, model, and decision policies.

## Batch scoring a saved model

```bash
python -m modeling.predict \
  --model modeling/results/models/h60_gain10.joblib \
  --features data/master/features.parquet \
  --output modeling/scored_example.parquet
```

This replays scoring only; it is not a new evaluation. The output preserves keys, scores, activity-screen flags, threshold crossing, and whether the threshold met its original validation gate. That gate does not promise the same precision in new periods.

Validate finished artifacts without retraining: `python -m modeling.verify_results`. This checks source hashes, held-out dates, thresholds, metrics, and score reproduction from reloaded saved models. `modeling/environment.lock.txt` captures the full environment used for this run.
