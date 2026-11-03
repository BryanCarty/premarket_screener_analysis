# Drawdowns and high/low excursion distributions

Scope: the 19 later-period signals for the saved +5% target-or-close policy, not the entire dataset. Drawdowns are gross price movements before exit; assumed transaction costs are not included.

**MAE** is the deepest dip below entry, expressed as a positive loss magnitude. **MFE** is the largest gain above entry. **Maximum drawdown (MDD)** is the greatest percentage fall from a prior price peak while the position is open. MDD can exceed MAE.

Minute OHLC bars do not identify high/low ordering. MDD bounds enumerate both admissible high-first/low-first paths in every full minute. In a target-exit minute, the low may be before or after the exit; the lower bound excludes that low, while the upper bound places it before the first target touch. Post-exit highs and lows are excluded. A missing minute invalidates true path bounds: flagged rows are observed-bar estimates only. Unobserved prices could hide either a larger dip or an earlier profit-target exit, so the estimate can differ from the actual strategy path in either direction.

| Date | Ticker | MAE lower–upper | Peak-to-trough MDD lower–upper | Missing pre-exit minutes |
|---|---|---:|---:|---:|
| 2026-07-22 | SNTG | 2.92–2.92% | 4.12–4.18% | 0 |
| 2026-07-23 | AEHL | 7.64–7.64% | 9.77–9.77% | 0 |
| 2026-07-23 | WLDS | 1.14–1.14% | 2.82–2.82% | 0 |
| 2026-07-24 | WLDS | 0.00–0.00% | 2.38–2.38% | 0 |
| 2026-08-05 | ZJYL | 7.83–7.83% | 7.83–7.83% | 1 |
| 2026-08-10 | EVGN | 3.73–3.73% | 4.64–4.64% | 0 |
| 2026-08-12 | HXHX | 3.60–3.60% | 3.73–3.77% | 0 |
| 2026-08-14 | ONFO | 1.35–1.35% | 3.39–3.39% | 0 |
| 2026-08-25 | BDRX | 21.23–21.23% | 23.84–23.84% | 0 |
| 2026-08-28 | EPOW | 8.82–8.82% | 9.13–9.13% | 0 |
| 2026-08-31 | COOT | 3.10–3.10% | 7.46–7.46% | 0 |
| 2026-09-11 | SXTC | 0.65–0.65% | 0.65–1.91% | 0 |
| 2026-09-23 | LXEH | 4.67–4.67% | 8.73–8.73% | 0 |
| 2026-09-28 | ONCO | 4.61–4.61% | 4.76–4.76% | 0 |
| 2026-10-01 | MYND | 16.17–16.17% | 19.33–19.33% | 63 |
| 2026-10-07 | WETO | 1.10–1.10% | 2.61–2.61% | 0 |
| 2026-10-15 | SNES | 6.88–6.88% | 7.20–7.20% | 0 |
| 2026-10-22 | XXII | 0.00–1.05% | 0.00–1.05% | 0 |
| 2026-10-23 | FMST | 0.02–0.02% | 2.80–3.43% | 0 |

## Paired distribution comparisons

High/low price levels and signed gains/losses are trivially separated; the meaningful comparison here is **upside magnitude (MFE) versus downside magnitude (absolute MAE)** for the same stock-day and window.

| Window | Complete pairs | Connected date/ticker clusters | Mean upside | Mean downside | Median upside | Median downside | ECDF distance | Exact p | Holm-adjusted p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| trade_window | 17 | 15 | 4.91% | 4.26% | 5.00% | 3.10% | 0.706 | 0.00342 | 0.00684 |
| full_session | 8 | 6 | 47.15% | 13.32% | 37.10% | 10.07% | 0.750 | 0.18750 | 0.18750 |

The exact paired label-swap permutation compares a KS-style empirical CDF distance. High/low labels are swapped jointly within connected clusters sharing a date or ticker; every sign assignment is enumerated. It tests the stronger null of within-cluster upside/downside exchangeability, not arbitrary equality of means. Clustering does not fully remove serial market/regime dependence.

The two planned distribution tests use Holm adjustment. Bootstrap intervals and sign-test diagnostics in statistics.json are descriptive; the sign-test diagnostic treats trade rows as independent and is not the primary cluster-aware test.

**Interpretation limits:** +5% profit-taking mechanically caps MFE for successful actual trades. A significant pre-exit distribution difference can therefore reflect the exit rule itself, not predictive skill. The fixed 09:45-to-close comparison removes that cap, but only a small number of these 19 stocks has complete minute coverage. Missing-bar session extrema are incomplete bounds and are not used in the primary test. These signals and exits were previously examined; this is exploratory analysis, not new confirmation of an edge.

For each test, statistics.json also includes the paired mean/median difference, a cluster-bootstrap 95% mean-difference interval, and the counts of trades with larger upside or downside. Statistical difference in shape does not imply that mean upside exceeds mean downside.

Reproduce: `python -m modeling.exit_study.drawdowns.analyze`. Data are read from the existing cached minute Parquet paths; no additional API requests are needed.

Method reference: [SciPy paired permutation tests](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html).
