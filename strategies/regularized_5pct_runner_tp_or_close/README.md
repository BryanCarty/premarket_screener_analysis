# Regularized +5% runner — take profit or session close

This folder packages the frozen histogram gradient boosting model, a forward-only Massive paper-trading runner, and an automatic end-of-day report. It submits **no real broker orders**. Runtime output stays under this folder's `runtime/`; the research datasets and original models are read only.

## Strategy

1. Universe: date-specific US common stocks and common ADRs, with matching identities across the preceding ten exchange sessions. A complete prior history and positive premarket-volume baseline are required. An optional ticker list narrows coverage and is recorded in the frozen daily universe.
2. Premarket eligibility, using **04:00–09:30 America/New_York**: last observed premarket close >=10% above the previous official regular-session close, and premarket volume >=5 times mean premarket volume over the preceding ten exchange sessions. Adjust prior prices/volumes for splits executed through today. No current-session daily aggregate or post-cutoff bar enters eligibility or features.
3. Opening quality: all 15 candles in **09:30–09:44** must exist. Estimated turnover during that window must be **at least $1 million**. Turnover is volume times the saved typical-price VWAP proxy, not exact trade notional.
4. Calculate the original feature functions using only bars starting before **09:45** and fully completed by receipt time. Apply the bundled frozen model, with score cutoff **0.688697686467451**. Scores are not calibrated probabilities.
5. When a ticker crosses the cutoff, persist the decision and commit a $1,000 fractional-share paper buy for the **next full minute after actual scoring**. Entry uses that future minute's open when subsequently observable. Skip orders outside the entry window, or cancel if persistence misses the planned minute. There is one order per ticker per session. Missing entry candles are unresolved; no substitute price is invented.
6. From that entry, take profit at exactly **+5%** if an observed completed minute high touches the target. Otherwise exit at the last regular-session minute's close. **No stop loss.** A closing exit is established only when the entire post-entry candle grid is present. Otherwise retain an unresolved result. An observed fixed target hit resolves at +5% even with prior interior gaps because no competing stop exists.
7. Use the exchange calendar for holidays, daylight saving and early closes. The default entry deadline is **10:30 ET**. No new entries are created by starting the program after that deadline.

The runner does not refit, tune thresholds, read research outcome labels, or select stocks using future prices. Orders, scores and feature rows are saved as they are observed. Completed trades are not re-evaluated from corrected later bars. Recovery after downtime only settles **already committed** positions; it never reconstructs missed signals. Late-recovery fills are explicitly tagged and may use revised historical observations.

## Delayed-feed timing changes the historical strategy

The existing project documents a Massive Developer feed with **15-minute delay**. The default is therefore `--feed-delay-seconds 900`, plus a ten-second candle-completion grace. A 09:45 feature window becomes usable around **10:00:10 ET**, and scanning many tickers takes additional time. The program commits entries after actual computation; **it never books a 09:45 entry from a later signal**.

Consequently this is a forward test of the model with operational entry timing, **not an exact reproduction of historical 09:45 entries**. The research performance does not establish profitability at these later entries. Even real-time REST collection cannot produce a truly instantaneous 09:45 entry from a fully completed 09:44 bar.

Only set `--feed-delay-seconds 0` after confirming a real-time entitlement. If Massive explicitly returns `DELAYED` under a zero-delay configuration, the scan aborts rather than silently using it. The configured delay is an assumption, not automatic entitlement verification. Missing candles, unknown latency and vendor corrections prevent a blanket guarantee of perfectly reconstructing the information available historically; archived receipt times establish the inputs actually observed in this forward run.

Massive references: [custom minute aggregates](https://massive.com/docs/rest/stocks/aggregates/custom-bars), [feed timing and DELAYED status](https://massive.com/knowledge-base/article/how-quickly-can-i-access-massives-market-date), [missing aggregate bars](https://massive.com/knowledge-base/article/why-are-there-missing-aggregates-in-massives-data), [flat-file availability](https://massive.com/knowledge-base/article/what-time-are-massives-flat-files-ready-each-day).

## Installation and credentials

Run commands from the **project root**:

```bash
cd '/media/bryan/External HD/premarket_screener'
source /home/bryan/.cache/premarket-screener-venv/bin/activate
```

Use the existing project and modeling dependencies (`requirements.txt` and `modeling/requirements.txt`). No new dependency is required. The trusted bundled joblib uses the project's scikit-learn version (1.9.1); load only trusted model files.

The project `.env` needs:

```dotenv
MASSIVE_API_KEY=...
MASSIVE_FLATFILES_ACCESS_KEY_ID=...
MASSIVE_FLATFILES_SECRET_ACCESS_KEY=...
```

The REST key is used for reference records, corporate actions and live minute aggregates. Flat-file credentials fill missing **prior-session** market-wide warmup summaries. Existing prior summaries are copied into this strategy's history cache when available. Full market-wide history preparation needs several GB of free space temporarily and can take time; raw downloaded files are deleted after summaries are committed. Flat files may not be available until the following morning. If required prior history is unavailable the runner does not trade with an incomplete baseline.

Check REST access without creating orders:

```bash
python strategies/regularized_5pct_runner_tp_or_close/paper_trade.py check
```

On a trading date, optionally prepare the ten preceding sessions before the decision window:

```bash
python strategies/regularized_5pct_runner_tp_or_close/paper_trade.py prepare
```

## Run one day or continuously

Start before the feature window, preferably by **08:00 ET**, so history and reference preparation finish early:

```bash
python -u strategies/regularized_5pct_runner_tp_or_close/paper_trade.py run
```

This processes today's session and finishes after the close plus feed delay/grace. It refuses historical date arguments. If started late, it refuses retrospective entries. A restart resumes committed positions and skips previously scanned tickers; it never resets a daily signal decision. API failures skip the affected symbol and are disclosed in the summary. A single-writer lock prevents two runners using the same state directory.

For daily unattended operation:

```bash
python -u strategies/regularized_5pct_runner_tp_or_close/paper_trade.py daemon
```

The daemon waits through closed sessions, prepares each trading day from 08:00 ET, runs it, and writes the daily summary automatically. The computer must stay on, the external drive mounted, and the network connected. To survive terminal closure, use the provided systemd user service template or a process supervisor. The template is **not installed or started automatically**.

To install and start the supplied user service yourself:

```bash
mkdir -p ~/.config/systemd/user
cp strategies/regularized_5pct_runner_tp_or_close/paper-strategy.service ~/.config/systemd/user/
chmod 644 ~/.config/systemd/user/paper-strategy.service
systemctl --user daemon-reload
systemctl --user enable --now paper-strategy.service
journalctl --user -u paper-strategy.service -f
```

Stop with `systemctl --user stop paper-strategy.service`; disable automatic starts with `systemctl --user disable paper-strategy.service`. A user service depends on the user's service manager staying active; it does not by itself guarantee operation after logout or computer shutdown.

Optional parameters:

```bash
python -u strategies/regularized_5pct_runner_tp_or_close/paper_trade.py run \
  --notional 1000 --round-trip-cost-pct 0.5 \
  --feed-delay-seconds 900 --poll-seconds 60 --entry-deadline 630
```

`--entry-deadline` is minutes after New York midnight; 630 is 10:30. Cost is in percentage points: 0.5 means a 0.5% round-trip scenario, not $0.50. Defaults assume fixed-dollar fractional positions and **unlimited aggregate paper capital**; there is no account balance, margin, exposure cap or position-size liquidity model. Scanning is sequential and paced; a full-universe scan may not finish before the deadline. The summary reports scan coverage. `--tickers AAPL,TSLA,...` can restrict scanning, but changes the strategy's coverage. `--once` performs one iteration only and is for diagnostics, not a complete trading day.

No notification/email/chat integration is configured. The end-of-day summary is written locally and printed to the process log.

## End-of-day results

For each New York session date:

```text
runtime/sessions/YYYY-MM-DD/
  SUMMARY.md          readable end-of-day results (interim while running)
  summary.json        machine-readable metrics and completion flag
  trades.csv          decisions, planned entries, fills, exits, status
  state.json          resumable decisions and positions
  universe.json       frozen daily universe and prior histories
  features/*.parquet  inputs actually used to score eligible tickers
  observations/*.json timestamped REST responses, without API credentials
```

The report includes signals, resolved/unresolved trades, net wins/losses, average winning/losing return, mean gross/net return, max gain/loss, resolved P&L, exit reasons, scan coverage and error counts. **Unresolved trades are excluded from return averages and remain visible in the denominator counts.** Cost-adjusted outcomes use the configured cost scenario. Reported average returns are not portfolio returns.

Inspect `SUMMARY.md` or `summary.json` while the daemon runs. With no runner holding the lock, regenerate today's summary:

```bash
python strategies/regularized_5pct_runner_tp_or_close/paper_trade.py summary
```

Intraday API responses are intentionally not cached for reuse: a fresh request is made for each observation. Observation archives can grow; archive old completed sessions manually. Do not delete the current session's state while running.

## Model provenance and research results

`artifacts/model.joblib` is an unchanged copy of `modeling/results_regularized_5pct_close/model.joblib`, verified by SHA-256 before running. `artifacts/training_manifest.json` records model settings, features, source hashes and chronological splits. `artifacts/bundle.json` records the frozen artifact hash and cutoff.

The model has three leaves per tree, maximum depth two, at least 100 training rows per leaf, 120 iterations, learning rate 0.03 and L2 penalty 30. It was fitted on 2,041 complete training outcome windows. Configuration was chosen using validation data. The original historical test dates were already examined, and the policy did not meet its 80% validation precision target.

Historical 09:45-entry TP5-or-close simulation on the selected test signals: 340 wins / 57 losses / 77 unresolved, average resolved gross return +1.43%, average loss -19.89%, worst loss -54.45%. These are not forecasts for this forward runner and assume fills without execution costs. Take-profit touches do not guarantee actual fills; no quotes, order book, broker fills or halts are modeled. The forward runner is an observable paper simulation, not a broker-connected execution system.

## Verification

```bash
python -m pytest -q strategies/regularized_5pct_runner_tp_or_close/test_paper_trade.py
```

Tests cover completed-candle and feature cutoffs, future-only entry commitments, rejection of pre-entry price action, delayed visibility, take-profit handling with gaps, unresolved no-hit paths, early closes, and summary accounting. Run the access check separately; unit tests use no network or credentials.

Initial verification on October 30, 2026: eight tests passed; live REST access succeeded and explicitly returned delayed status; model hash/cutoff matched the original; service configuration validation passed. A late-start smoke check refused retrospective entries. No forward session or unattended service was started during setup.
