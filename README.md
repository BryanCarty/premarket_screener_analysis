# Pre-market runner research dataset

Local, reproducible proof of concept. No model is trained. See `STATUS.md` for the latest collection state. Credentials stay in `.env`.

## Run

This external drive does not support Python environment symlinks. The environment lives on the system drive; all market data lives here.

```bash
uv venv /home/bryan/.cache/premarket-screener-venv
uv pip install --python /home/bryan/.cache/premarket-screener-venv/bin/python -r requirements.txt
source /home/bryan/.cache/premarket-screener-venv/bin/activate
python -m pytest -q
# Full-market pilot; automatically collects ten preceding exchange sessions.
python -m runner.collect --source flat --start 2025-01-06 --end 2025-01-06 --output data/pilot
# Two years, ending on the last complete session before 2026-10-29.
python -m runner.collect --source flat --start 2024-10-29 --end 2026-10-28 --output data/master
python -m runner.validate data/master
```

Repeat the same collection command after interruption. Checkpoints are committed only after a day's summaries, features, and labels are written. A failed day's downloaded file is reused. Successful raw downloads and raw REST responses are removed. Final files are rebuilt from daily partitions. Keep `daily/`, `checkpoints/`, and `scope.json` to resume. Keep the same start date and source; the end date may be extended. To intentionally rebuild with changed calculations, use a fresh output directory. Do not run two writers against the same directory.

`--source rest-full` uses the date-specific complete universe and makes one minute request per stock per day. This is a much slower fallback. `--source rest-sample --tickers FUBO,AAPL,...` is a **selected-universe engineering sample**, unsuitable for population prevalence estimates. REST development samples are not the master dataset.

## Definitions and quality policy

- Universe: historically active US stock-market securities classified `CS` or `ADRC` on each date. Includes subsequently delisted stocks where Massive retains them. Excludes ETFs, warrants, preferreds, units, OTC, and unclassified reference records. Effective-dated reference data is maintained retrospectively by the vendor, not an archived as-of snapshot.
- Pre-market window: **04:00 <= time < 09:30 America/New_York**, identical for each of ten prior XNYS sessions. Empty pre-market windows count as zero volume when a security exists in the historical universe. A stock must have ten matched history records; new listings, unresolved symbol/identity changes, unavailable prior close, and zero-volume baselines cannot establish eligibility and are counted separately.
- Eligibility: last observed pre-market minute close / previous official regular-session close - 1 >= 10%; current pre-market volume / mean prior ten pre-market volumes >= 5. No post-open price or volume participates in selection. No positive/negative balancing.
- Splits: raw prices and volumes; historical closes divided and volumes multiplied by the product of `split_to / split_from` for execution dates after the observation through the target date. Execution-day pre-market is already on the new share basis. Corporate-action corrections made later by the vendor cannot be reconstructed as originally published. Dividends are not adjusted, matching raw tradable prices.
- Identity: share-class FIGI, then composite FIGI, then ticker fallback. Renames are not stitched across symbols: these stocks generally lose up to ten days of eligibility history. Ticker reuse without a vendor identity cannot be fully resolved.
- Features: only bars with start time < 09:45; includes 09:30–09:44. The last 09:44 bar is assumed available at 09:45. Publication latency and subsequent corrections cannot be modeled from historical bars. This is event-time research: Developer's documented 15-minute feed delay would prevent live use of these bars at 09:45 on that subscription. No current float, market cap, financials, or company metadata enters predictive features.
- Entry: **09:45 minute open**, an observed price proxy, not an executable quote. If that bar is missing, entry and labels are unknown; no delayed substitute. Quotes/spread/slippage/fees are unavailable in this proof of concept. Later backtests must model execution costs. Features with missing opening bars remain in the dataset with quality flags.
- Labels: windows [09:45, 09:45+horizon), including the entry minute. Close horizon ends at the exchange's actual session close, including half-days. High/low are minute-level excursions; time-to-hit means the start of the first minute bucket, not exact seconds. Within-bar ordering is unknown. A known observed threshold hit is positive even with missing bars; a negative requires a complete minute grid. Otherwise labels are null. End return requires the final minute, never a stale substitute. Missing minutes can be genuine inactivity or missing data; this conservative policy overstates unknowns for illiquid stocks.
- Minute aggregate volume and OHLC follow Massive trade-condition rules. `vwap_proxy` uses typical price times volume, not trade VWAP. Shares/transaction and transaction counts provide inexpensive trade-activity proxies. Raw ticks were deliberately deferred: trade-size distributions/aggressor classification need more data and condition handling; quote-based aggressor side is outside Developer's documented quote entitlement.

## Outputs

- `features.parquet`: one eligible security/day row, including rows with unavailable opening data.
- `labels.parquet`: seven horizon rows per eligible security/day; join on `date,ticker,identity`.
- `DATASET_SUMMARY.md`: readable prevalence tables and exploratory opening-behavior comparisons; no model fitting.
- `summary.json`: qualifying counts, positives/negatives/unknowns at every horizon/threshold, feature null counts, storage, daily eligibility audit.
- `features/`, `labels/`: compressed daily Parquet partitions; `daily/`: compact market-wide warm-up summaries.
- `checkpoints/`: committed provenance including S3 object key, ETag, compressed size and processing time. A local ETag records the version downloaded; reruns do not automatically refresh revised vendor objects.
- `docs/FEATURE_DICTIONARY.md`: predictive and outcome field definitions.

Use chronological splits and group related same-day observations when training later. Multiple horizons are correlated; no inference about predictive utility follows from feature existence alone.

## Source documentation checked 2026-10-29

[Pricing](https://massive.com/pricing?product=stocks): Developer documents 10 years, minute bars, trades, unlimited REST calls, and flat files; quotes are on Advanced. Actual minute, reference, splits, and S3 access were probed locally. No request limit is treated as permission for unbounded concurrency: requests are sequential, default pacing 0.15 seconds, with retries and Retry-After handling.

[Flat-file setup](https://massive.com/docs/flat-files/quickstart), [raw split policy](https://massive.com/docs/flat-files/stocks/overview), [historical universe](https://massive.com/docs/rest/stocks/tickers/all-tickers), [splits](https://massive.com/docs/rest/stocks/corporate-actions/splits).

## Storage plan

82 GiB free at initial inspection. A measured 2025-01-06 minute flat file is 22,152,665 compressed bytes. At 500 sessions, that size implies roughly 11 GB transferred, **not permanently stored**. Actual dates vary; pilot checkpoint sizes and final dataset bytes provide measured estimates. Only one compressed raw day is held at a time; in-memory processing needs several GB. A 2 GB minimum free-space guard runs before every download. Daily summaries are retained for resumability; no tick files are downloaded. Final Parquet files plus daily partitions duplicate the compact derived output for convenience. The partial REST engineering sample was removed after full-market pilot validation.

The durable full-run wrapper is `python -m runner.job`. It updates `STATUS.md` every 30 seconds, writes `data/master.log`, validates the completed dataset automatically, and marks status COMPLETE only after success. It runs independently of the chat when launched as a background process.

Rebuild the visual overview: `python -m runner.plot_summary`. PNG, PDF, and SVG files are written to `data/master/charts/`. Heatmap percentages use all eligible rows, so missing coverage does not inflate the displayed hit rate.
