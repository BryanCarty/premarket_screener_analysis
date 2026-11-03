# Precision-first catalyst/setup study

The requested study is complete, using the connected Massive REST API. **No supported candidate met the validation precision/economic gates, and the available positive-catalyst sample is too small to test that hypothesis reliably.**

## What was tested

The existing 812 qualified stock-days were preserved. Rules still require premarket gap >=10%, premarket relative volume >=5, all opening 15-minute candles, >=$1m opening turnover, complete prior regular-session candles and sufficient historical volatility history.

Inputs stop before **09:45 ET**. The primary entry is **10:05 ET**: a predeclared assumption of 15-minute market-data delay plus five minutes for collection/decision, with a hypothetical 10:04 decision. It is not proof that the existing sequential REST scanner would finish by 10:04. We also report a fixed **10:15 entry sensitivity**, using unchanged model scores and cutoff.

The success target is **upper volatility barrier first**, before the lower barrier or a 60-minute timeout. Barriers are entry * exp(+/-sigma20), where sigma20 is the existing prior-20-session 09:45-to-close log-return standard deviation. A stop gap fills at the worse minute open. A missing minute before exit or an ambiguous same-minute touch remains unresolved. This is not an eventual +5% label. Timeout trades can be profitable without counting as target hits.

## News access and coverage

Standard `/v2/reference/news` access succeeded. Benzinga structured earnings and news returned **HTTP 403**, so verified EPS/revenue surprises were unavailable.

The study uses raw titles, publication timestamps and ticker/company matching. Fixed rules identify earnings/guidance, contracts, regulatory/trial events, financing/dilution, and a narrow explicit-positive candidate flag. Only events since the previous regular close are fresh catalysts. The query also includes a longer history for stale-news counts. Multi-company headlines require company/ticker relevance; titles are deduplicated. Known post-cutoff updates are excluded. Vendor sentiment and descriptions are **not** model inputs.

Across 812 rows:

| Coverage | Rows |
|---|---:|
| Successful news requests | 812 |
| Fresh company-relevant news found | 159 |
| Fresh classified event found | 47 |
| Strict positive-headline candidate | **7** |

Those seven positive candidates divide into **zero training, three validation, four test**. No fresh financing headline matched the fixed rules. This is insufficient evidence for dedicated positive-catalyst or financing models. It does not establish absence of real news or absence of a catalyst edge. Our highly selected gap universe plus the accessible archive/headline rules may miss many genuine events. Classifications are auditable in `data/headline_audit.csv`.

A headline-positive proxy is **not a verified positive earnings surprise or fundamental improvement**. Publication time does not certify original subscriber receipt time or headline version history. Historical annotations/revisions cannot be fully reconstructed. These are exploratory publication-time features, not a certified point-in-time news archive. [Massive news documentation](https://massive.com/docs/rest/stocks/news).

## Precision-first protocol

`plan.json` was saved before model fitting. Predeclared segments: all stocks, positive-headline candidates, financing candidates, no classified fresh event, continuation, failed breakout, and positive-catalyst continuation. Failed breakout remains a long-entry diagnostic; no short execution is simulated without borrow/locate evidence.

Two feature sets compare price/risk/setup inputs with the same inputs plus headline features. Models are regularized logistic regression and shallow histogram boosting, without balancing or oversampling.

Model selection uses **three expanding chronological development folds entirely inside the original training dates**. Initial training windows are 150/200/250 calendar sessions, with a one-session embargo and validation through session 200/250/300. Each fold ranks its top20% scores, retaining ties. The model family with the highest pooled conservative precision Wilson lower bound is selected, requiring >=20 OOF calls across >=10 dates and all three supported folds. AP breaks ties. Final fitting uses original training only.

The later original validation period selects a fixed cutoff. A qualifying policy requires:

- >=80% observed target-hit precision, counting unknown selected outcomes against it;
- >=25 calls across >=10 dates;
- Wilson lower 95% bound >=65%;
- positive mean validation return after an assumed 0.5% round-trip cost;
- no resolved validation gross loss worse than -10%.

A supported fallback is reported if precision requirements fail, but it is not a validated edge. Sparse groups are skipped rather than relaxed after viewing outcomes. Primary selection is frozen before test scoring. The test calendar was previously inspected, so it remains exploratory, not fresh independent evidence.

Four candidates had enough OOF support: all-stock and no-classified-event models, each with/without news inputs. Ten segment/feature combinations were unsupported. Dedicated continuation/failed-breakout groups were also too small for the three-fold support requirements.

## Selected research fallback

Validation selected logistic regression on **no classified fresh event**, with price and available news-context features. This means no classified event in the accessible archive, not proof of no actual catalyst. Its cutoff is **0.4548515669336637** as recorded in the saved artifact; use `frozen_selection.json` as the authoritative value rather than copying documentation values.

| Metric | Validation | Previously viewed test |
|---|---:|---:|
| Eligible rows in selected segment | 190 | 80 |
| Signals | 27 | **15** |
| Upper barrier first | 13 | **6** |
| Other resolved outcomes | 14 | **9** |
| Unknown selected outcomes | 0 | 0 |
| **Target-hit precision** | **48.1%** | **40.0%** |
| Wilson 95% interval | 30.7–66.0% | **19.8–64.3%** |
| Mean gross return | -0.53% | **+0.35%** |
| Mean after 0.5% assumed costs | -1.03% | **-0.15%** |

No candidate passed all validation gates. In test, this selected policy had nine profitable trades and six losses; only six reached the upper barrier first, since three profitable timeouts did not hit the target. Average winning return +1.91%; average losing return -1.98%; best +5.63%; worst -4.39%. Delaying the same selected signals to 10:15 produced mean gross -0.22%, or -0.72% after 0.5% costs.

## Increasing selectivity did not establish high precision

For the selected model's 80 eligible test rows:

| Descriptive ranking | Calls | Target hits | Precision |
|---|---:|---:|---:|
| Top 5% | 4 | 1 | **25.0%** |
| Top 10% | 8 | 4 | **50.0%** |
| Top 20% | 16 | 6 | **37.5%** |

These are period-wide ranking diagnostics, not cutoffs available prospectively. Counts are too small for high-precision claims; the top10% Wilson interval is approximately 21.5–78.5%. Training results are in sample, validation is tuned, and monthly outcome tables expose instability. All rankings/counts/unknowns are exported.

The same-cohort all-stock price model produced 4/13 target hits (30.8%) on test; adding news produced 6/22 (27.3%). Different signal sets prevent interpreting those precision differences as a controlled trade-profit comparison. Available headline features did not produce a supported edge.

## Conclusion and future confirmation

This run does **not** establish a profitable edge. It also cannot meaningfully accept or reject the positive-catalyst hypothesis, since the training set contains zero qualifying examples. The current data access and selected universe are the limiting evidence, not a reason to claim perfect precision on four events. An event-first sample with enough timestamped verified catalysts would be needed to test that hypothesis properly, rather than silently loosening the current rules after viewing outcomes.

`results/FORWARD_PROTOCOL.json` freezes the research artifact/hash and confirmation requirements. It explicitly disables automatic entries. The proposed confirmation needs **at least 60 new sessions AND 50 resolved signals**, unchanged features/model/cutoff, actual first-receipt/version archives, delayed-data-aware decisions and precommitted future-minute entries. No candidate has passed the validation gates; no forward-confirmed model is claimed. Existing paper strategy and frozen model are unchanged.

All P&L equally weights resolved hypothetical trades; it is not portfolio performance. Spread, halt, slippage, impact, actual commissions and capital constraints are unavailable. Assumed cost scenarios are not measured costs. Missing outcomes are not zero returns.

## Artifacts and reproduction

- `plan.json`, `access.json`: predeclared study and endpoint access checks.
- `data/dataset.parquet`, `headline_audit.csv`, `news_parts/`, `paths/`: auditable headlines, feature rows and delayed-entry paths.
- `results/manifest.json`, `frozen_selection.json`, `models/`, `development_models/`: source hashes, features, selections and fitted pipelines.
- `results/development_comparison.csv`, `development_predictions.parquet`: precision-first OOF evidence and skipped candidates.
- `results/metrics.csv`, `ranking_precision.csv`, `monthly.csv`: policy precision, intervals and stability.
- `results/strategy_metrics.csv`, `predictions.parquet`: exact signals, outcomes and timing/cost sensitivities.
- `results/FORWARD_PROTOCOL.json`: disabled-entry research protocol.
- `results/overview.png` and `.pdf`: coverage, precision uncertainty and timing/cost figure.

```bash
python -m modeling.catalyst_precision.build
python -m modeling.catalyst_precision.evaluate
python -m modeling.catalyst_precision.verify
python -m pytest -q modeling/catalyst_precision/test_catalysts.py
```

Use the existing project environment. Data download chunks resume; completed evaluations refuse overwrite. Fifteen relevant tests passed. Verification reproduced 82 headline/delayed-target samples, all chosen OOF probabilities and chronological memberships, source/plan hashes, frozen test probabilities/precision/returns and validation gates.
