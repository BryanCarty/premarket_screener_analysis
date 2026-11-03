# Feature dictionary

Prices are raw dollars on the target day's share basis. Volume is shares. Returns ending `_pct` are percentage points (10 means 10%); fractions and ratios are unitless. Null means unavailable or undefined; zero-volume denominators yield null.

| Field | Meaning |
|---|---|
| date, ticker, identity | Local session date, effective ticker, vendor identifier/fallback; join keys, not numeric predictive inputs |
| feature_cutoff_utc | 09:45 ET expressed in UTC; metadata |
| previous_close | Prior official regular-session close, split aligned to target day |
| pm_baseline_volume_10d | Mean 04:00–09:30 share volume over the ten preceding exchange sessions, split aligned |
| pm_rvol | Current PM volume / baseline; mandatory >=5 |
| pm_gap_pct | Last observed PM close / previous close minus one, times 100; mandatory >=10 |
| history_daily_volume_mean_10d | Mean official daily aggregate volume over prior ten sessions |
| history_pm_active_days_10d | Number of prior ten sessions with PM volume >0 |
| pm_last_age_minutes | 09:30 minus last PM bar start, minutes; bar covers the following minute |
| opening_available | At least one opening 15-minute bar exists |
| opening_minutes_observed, opening_missing_minutes | Observed and absent bars in 09:30–09:45 |
| pm_volume_acceleration | Volume 09:00–09:30 / volume 08:30–09:00 |
| open_volume_acceleration | Volume 09:40–09:45 / volume 09:30–09:35 |
| open15_vs_pm_volume | Opening fifteen-minute volume / PM volume |
| open15_close_vs_pm_high_pct | Last opening-window close relative to PM high |
| open15_gain_retention | (Opening-window last close - previous close) / (PM last close - previous close) |
| open15_low_vs_pm_last_pct | Opening-window low relative to PM last close |

The following suffixes exist for **pm** (04:00–09:30), **pm_last30** (09:00–09:30), and **open15** (09:30–09:45):

| Suffix | Meaning |
|---|---|
| volume | Sum of bar share volumes |
| return_pct | Last close relative to first open |
| range_pct | Window high relative to window low |
| vwap_proxy | Sum(((high+low+close)/3)*volume) / sum(volume); approximate, not true trade VWAP |
| close_to_vwap_pct | Last window close relative to proxy VWAP |
| transactions | Sum of bar qualifying transaction counts |
| shares_per_transaction | Volume / transaction count; excludes no further trade-condition corrections |
| realized_vol_pct | 100 * sqrt(sum(squared log returns between observed minute closes)); gaps are not filled |
| close_location | (Last close - low)/(high - low); null for a flat window |
| up_volume_fraction | Fraction of volume in bars closing above their own open |

# Labels dictionary

Each key has seven rows: horizons `5`, `15`, `30`, `60`, `90`, `120`, `close`. Each row has six nullable binary targets.

| Field | Meaning |
|---|---|
| horizon | Duration in minutes or remainder of regular session |
| entry_price_proxy | 09:45 minute open; null if missing |
| entry_available | Whether the entry minute exists |
| interval_end_minute | Exclusive local end minute, minutes after midnight |
| observed_minutes, expected_minutes, missing_minutes | Outcome-window grid coverage |
| end_price_observed | Final minute is present |
| max_gain_pct | Maximum observed high / entry minus one, times 100; partial lower bound when coverage is incomplete |
| max_decline_pct | Minimum observed low / entry minus one, times 100; generally negative; partial upper bound when incomplete |
| end_return_pct | Final interval minute close / entry minus one, times 100; null if final minute missing |
| minutes_to_max_gain | Minute bucket offset of first highest observed high |
| runner_{5,10,15,20,30,50}pct | True if observed high reaches entry*(1+threshold/100); false only with complete minute coverage; otherwise null |
| minutes_to_{5,10,15,20,30,50}pct | First threshold-hit minute bucket offset; null when unobserved |

Outcomes are descriptive minute-bar measurements. They do not assert that highs/lows could have been executed at size. Only labels contain post-cutoff information.
