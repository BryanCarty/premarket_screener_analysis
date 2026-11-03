# Exploratory exit-policy study

Frozen entry model: Extra Trees, +5% by close, cutoff 0.905710. Validation signals: 29; previously viewed test signals: 19.

**This is an observed-minute-bar simulation, not an executable backtest.** Original model precision measures threshold touches, not trade profits. The model-study holdout was already viewed before this new exit experiment. Exits are ranked on the earlier threshold-validation period; a fresh future paper-trading period is still required.

Validation-selected policy on the common verifiable cohort: **time60_only**. The common cohort requires that every compared policy can be evaluated without a missing pre-exit minute; this reduces the sample sharply and introduces liquidity selection. The larger observed-path cohort is a sensitivity analysis only.

Round-trip cost assumption: 0.50 percentage points per position, subtracted from gross return. This is a scenario, not measured execution cost. Also saved: 1% and 2% round-trip cost sensitivities.

| Period | Cohort | Exit rule | Trades | Mean gross | Mean net (0.5% cost) | Median net | Net win rate | Worst net | 95% mean net interval |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| test | matched_verifiable | close_only | 10 / 19 | 1.82% | 1.32% | 0.41% | 50.0% | -34.42% | -12.02 to 14.13% |
| test | matched_verifiable | time120_only | 10 / 19 | 5.82% | 5.32% | 2.59% | 70.0% | -10.09% | -1.08 to 12.31% |
| test | matched_verifiable | time60_only | 10 / 19 | 12.04% | 11.54% | 7.71% | 90.0% | -1.87% | 4.86 to 18.47% |
| test | matched_verifiable | tp10_or_close | 10 / 19 | 7.01% | 6.51% | 9.50% | 90.0% | -20.36% | -0.45 to 9.50% |
| test | matched_verifiable | tp10_sl10_close | 10 / 19 | 8.00% | 7.50% | 9.50% | 90.0% | -10.50% | 2.83 to 9.50% |
| test | matched_verifiable | tp10_sl5_close | 10 / 19 | 4.00% | 3.50% | 9.50% | 60.0% | -5.50% | -1.41 to 9.50% |
| test | matched_verifiable | tp10_sl5_time120 | 10 / 19 | 4.00% | 3.50% | 9.50% | 60.0% | -5.50% | -1.41 to 9.50% |
| test | matched_verifiable | tp15_or_close | 10 / 19 | 11.51% | 11.01% | 14.50% | 90.0% | -20.36% | 2.88 to 14.50% |
| test | matched_verifiable | tp15_sl5_close | 10 / 19 | 7.00% | 6.50% | 14.50% | 60.0% | -5.50% | -0.05 to 14.50% |
| test | matched_verifiable | tp20_or_close | 10 / 19 | 16.01% | 15.51% | 19.50% | 90.0% | -20.36% | 6.21 to 19.50% |
| test | matched_verifiable | tp20_sl10_close | 10 / 19 | 17.00% | 16.50% | 19.50% | 90.0% | -10.50% | 9.50 to 19.50% |
| test | matched_verifiable | tp5_or_close | 10 / 19 | 2.51% | 2.01% | 4.50% | 90.0% | -20.36% | -3.79 to 4.50% |
| test | matched_verifiable | tp5_sl10_close | 10 / 19 | 3.50% | 3.00% | 4.50% | 90.0% | -10.50% | -0.50 to 4.50% |
| test | matched_verifiable | tp5_sl5_close | 10 / 19 | 2.00% | 1.50% | 4.50% | 70.0% | -5.50% | -1.06 to 4.50% |
| test | matched_verifiable | tp5_sl5_time60 | 10 / 19 | 2.00% | 1.50% | 4.50% | 70.0% | -5.50% | -1.06 to 4.50% |
| test | observed_paths | close_only | 18 / 19 | -0.57% | -1.07% | -1.12% | 50.0% | -34.42% | -8.55 to 6.78% |
| test | observed_paths | time120_only | 19 / 19 | 0.28% | -0.22% | -1.18% | 47.4% | -19.60% | -5.03 to 4.77% |
| test | observed_paths | time60_only | 19 / 19 | 6.73% | 6.23% | 2.57% | 68.4% | -7.52% | 1.80 to 11.17% |
| test | observed_paths | tp10_or_close | 18 / 19 | 5.01% | 4.51% | 9.50% | 77.8% | -20.36% | -0.48 to 8.35% |
| test | observed_paths | tp10_sl10_close | 18 / 19 | 5.56% | 5.06% | 9.50% | 77.8% | -10.50% | 1.08 to 8.45% |
| test | observed_paths | tp10_sl5_close | 18 / 19 | 2.50% | 2.00% | 2.00% | 50.0% | -5.50% | -1.33 to 5.97% |
| test | observed_paths | tp10_sl5_time120 | 19 / 19 | 1.92% | 1.42% | -2.42% | 47.4% | -5.50% | -1.83 to 4.76% |
| test | observed_paths | tp15_or_close | 18 / 19 | 7.42% | 6.92% | 14.50% | 77.8% | -20.36% | 0.92 to 12.03% |
| test | observed_paths | tp15_sl5_close | 18 / 19 | 5.00% | 4.50% | 4.50% | 50.0% | -5.50% | 0.06 to 9.79% |
| test | observed_paths | tp20_or_close | 18 / 19 | 8.24% | 7.74% | 19.50% | 72.2% | -20.36% | 0.59 to 14.06% |
| test | observed_paths | tp20_sl10_close | 18 / 19 | 8.67% | 8.17% | 19.50% | 72.2% | -10.50% | 1.53 to 14.18% |
| test | observed_paths | tp5_or_close | 19 / 19 | 3.28% | 2.78% | 4.50% | 89.5% | -20.36% | -0.24 to 4.50% |
| test | observed_paths | tp5_sl10_close | 19 / 19 | 3.42% | 2.92% | 4.50% | 89.5% | -10.50% | 0.55 to 4.50% |
| test | observed_paths | tp5_sl5_close | 19 / 19 | 1.84% | 1.34% | 4.50% | 68.4% | -5.50% | -0.76 to 3.39% |
| test | observed_paths | tp5_sl5_time60 | 19 / 19 | 1.84% | 1.34% | 4.50% | 68.4% | -5.50% | -0.76 to 3.39% |
| threshold | matched_verifiable | close_only | 11 / 29 | 19.84% | 19.34% | 10.31% | 63.6% | -12.19% | 3.34 to 38.80% |
| threshold | matched_verifiable | time120_only | 11 / 29 | 31.64% | 31.14% | 24.46% | 90.9% | -1.91% | 16.07 to 50.09% |
| threshold | matched_verifiable | time60_only | 11 / 29 | 32.16% | 31.66% | 11.17% | 100.0% | 4.19% | 11.32 to 60.59% |
| threshold | matched_verifiable | tp10_or_close | 11 / 29 | 10.00% | 9.50% | 9.50% | 100.0% | 9.50% | 9.50 to 9.50% |
| threshold | matched_verifiable | tp10_sl10_close | 11 / 29 | 6.36% | 5.86% | 9.50% | 81.8% | -10.50% | 0.41 to 9.50% |
| threshold | matched_verifiable | tp10_sl5_close | 11 / 29 | 7.27% | 6.77% | 9.50% | 81.8% | -5.50% | 2.68 to 9.50% |
| threshold | matched_verifiable | tp10_sl5_time120 | 11 / 29 | 6.99% | 6.49% | 9.50% | 81.8% | -5.50% | 2.68 to 9.50% |
| threshold | matched_verifiable | tp15_or_close | 11 / 29 | 15.00% | 14.50% | 14.50% | 100.0% | 14.50% | 14.50 to 14.50% |
| threshold | matched_verifiable | tp15_sl5_close | 11 / 29 | 11.36% | 10.86% | 14.50% | 81.8% | -5.50% | 5.41 to 14.50% |
| threshold | matched_verifiable | tp20_or_close | 11 / 29 | 20.00% | 19.50% | 19.50% | 100.0% | 19.50% | 19.50 to 19.50% |
| threshold | matched_verifiable | tp20_sl10_close | 11 / 29 | 14.55% | 14.05% | 19.50% | 81.8% | -10.50% | 5.86 to 19.50% |
| threshold | matched_verifiable | tp5_or_close | 11 / 29 | 5.00% | 4.50% | 4.50% | 100.0% | 4.50% | 4.50 to 4.50% |
| threshold | matched_verifiable | tp5_sl10_close | 11 / 29 | 2.27% | 1.77% | 4.50% | 81.8% | -10.50% | -2.32 to 4.50% |
| threshold | matched_verifiable | tp5_sl5_close | 11 / 29 | 3.18% | 2.68% | 4.50% | 81.8% | -5.50% | -0.05 to 4.50% |
| threshold | matched_verifiable | tp5_sl5_time60 | 11 / 29 | 3.18% | 2.68% | 4.50% | 81.8% | -5.50% | -0.05 to 4.50% |
| threshold | observed_paths | close_only | 29 / 29 | 7.85% | 7.35% | -1.91% | 44.8% | -30.53% | -3.86 to 22.08% |
| threshold | observed_paths | time120_only | 26 / 29 | 13.57% | 13.07% | 5.33% | 61.5% | -16.29% | 4.79 to 24.45% |
| threshold | observed_paths | time60_only | 28 / 29 | 13.45% | 12.95% | 4.28% | 64.3% | -13.06% | 2.99 to 27.32% |
| threshold | observed_paths | tp10_or_close | 29 / 29 | 5.28% | 4.78% | 9.50% | 82.8% | -27.70% | 0.97 to 8.14% |
| threshold | observed_paths | tp10_sl10_close | 29 / 29 | 3.58% | 3.08% | 9.50% | 69.0% | -10.50% | -0.04 to 6.17% |
| threshold | observed_paths | tp10_sl5_close | 29 / 29 | 3.79% | 3.29% | 9.50% | 58.6% | -5.50% | 0.93 to 5.75% |
| threshold | observed_paths | tp10_sl5_time120 | 29 / 29 | 3.84% | 3.34% | 9.50% | 58.6% | -5.50% | 1.22 to 5.61% |
| threshold | observed_paths | tp15_or_close | 29 / 29 | 6.65% | 6.15% | 14.50% | 75.9% | -27.70% | -0.16 to 11.53% |
| threshold | observed_paths | tp15_sl5_close | 29 / 29 | 5.34% | 4.84% | 14.50% | 51.7% | -5.50% | 1.17 to 8.79% |
| threshold | observed_paths | tp20_or_close | 29 / 29 | 4.19% | 3.69% | 5.16% | 58.6% | -30.53% | -2.74 to 9.90% |
| threshold | observed_paths | tp20_sl10_close | 29 / 29 | 3.47% | 2.97% | -2.18% | 44.8% | -10.50% | -2.06 to 8.43% |
| threshold | observed_paths | tp5_or_close | 29 / 29 | 2.33% | 1.83% | 4.50% | 86.2% | -27.70% | -1.06 to 4.10% |
| threshold | observed_paths | tp5_sl10_close | 29 / 29 | 1.38% | 0.88% | 4.50% | 75.9% | -10.50% | -0.86 to 2.77% |
| threshold | observed_paths | tp5_sl5_close | 29 / 29 | 2.24% | 1.74% | 4.50% | 72.4% | -5.50% | 0.63 to 3.02% |
| threshold | observed_paths | tp5_sl5_time60 | 29 / 29 | 2.29% | 1.79% | 4.50% | 72.4% | -5.50% | 0.67 to 3.07% |

Policy naming: `tp5` = +5% take profit, `sl5` = -5% fixed stop, `time60` = exit at 10:45 ET, `time120` = 11:45 ET, `close` = regular-session close. Target and stop levels are relative to entry; no overnight positions.

Gap handling: when a bar opens below the stop, the model exits at that worse opening price. Both barriers inside one minute are resolved stop-first. Missing entry or final exit prices remain unknown. Trades with missing minutes before a conditional exit cannot establish barrier ordering and are flagged. Minute timestamps do not reproduce live order latency or queue priority.

Interpretation: mean trade return equally weights positions; it is not a portfolio return, annualized return, or capacity estimate. No live bid/ask quotes, broker fees, liquidity, halts, partial fills, or market impact were modeled. Few signals and many unverified paths prevent a claim that one exit is best.

Reproduce: `python -m modeling.exit_study.analyze`. Small targeted minute paths are cached locally as compressed Parquet to avoid repeat downloads; broad raw market files are not retained.
