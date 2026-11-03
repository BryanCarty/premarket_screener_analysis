"""Pure calculations. All windows are left-closed/right-open, in New York time."""
import numpy as np
import pandas as pd

THRESHOLDS = (5, 10, 15, 20, 30, 50)
HORIZONS = (5, 15, 30, 60, 90, 120, 'close')

def ratio(a, b):
    return float(a / b) if pd.notna(b) and b > 0 else np.nan

def window(bars, lo, hi):
    return bars[(bars.minute >= lo) & (bars.minute < hi)]

def session_stats(b):
    if b.empty: return {'volume': 0.0, 'last': np.nan, 'high': np.nan, 'low': np.nan}
    return {'volume': float(b.volume.sum()), 'last': float(b.close.iloc[-1]),
            'high': float(b.high.max()), 'low': float(b.low.min())}

def features(b, prev_close, baseline, hist, cutoff):
    pm = window(b, 240, 570); op = window(b, 570, 585)
    p = session_stats(pm)
    f = {'pm_rvol': ratio(p['volume'], baseline), 'pm_gap_pct': 100*(ratio(p['last'], prev_close)-1),
         'previous_close': prev_close, 'pm_baseline_volume_10d': baseline,
         'history_daily_volume_mean_10d': float(hist.reg_volume.mean()),
         'history_pm_active_days_10d': int((hist.pm_volume > 0).sum()),
         'pm_last_age_minutes': 570-int(pm.minute.iloc[-1]),
         'opening_minutes_observed': len(op), 'opening_missing_minutes': 15-len(op),
         'feature_cutoff_utc': cutoff.isoformat(), 'opening_available': not op.empty}
    for name, s in [('pm', pm), ('open15', op), ('pm_last30', window(b, 540, 570))]:
        if s.empty:
            for k in ('volume','return_pct','range_pct','vwap_proxy','close_to_vwap_pct','transactions','shares_per_transaction','realized_vol_pct','close_location','up_volume_fraction'):
                f[name+'_'+k] = np.nan
            continue
        volume = s.volume.sum(); proxy = ((s.high+s.low+s.close)/3*s.volume).sum()/volume if volume else np.nan
        transactions = s.transactions.sum(min_count=1)
        f.update({name+'_volume': float(volume), name+'_return_pct':100*(ratio(s.close.iloc[-1], s.open.iloc[0])-1),
                  name+'_range_pct':100*(ratio(s.high.max(),s.low.min())-1), name+'_vwap_proxy':float(proxy),
                  name+'_close_to_vwap_pct':100*(ratio(s.close.iloc[-1],proxy)-1), name+'_transactions':float(transactions),
                  name+'_shares_per_transaction':ratio(volume, transactions),
                  name+'_realized_vol_pct':float(np.sqrt(np.square(np.log(s.close).diff()).sum())*100),
                  name+'_close_location':ratio(s.close.iloc[-1]-s.low.min(),s.high.max()-s.low.min()),
                  name+'_up_volume_fraction':ratio(s.loc[s.close>s.open,'volume'].sum(),volume)})
    f['pm_volume_acceleration'] = ratio(window(b,540,570).volume.sum(), window(b,510,540).volume.sum())
    f['open_volume_acceleration'] = ratio(window(b,580,585).volume.sum(), window(b,570,575).volume.sum())
    f['open15_vs_pm_volume'] = ratio(op.volume.sum(), p['volume'])
    f['open15_close_vs_pm_high_pct'] = 100*(ratio(op.close.iloc[-1],p['high'])-1) if len(op) else np.nan
    f['open15_gain_retention'] = ratio(op.close.iloc[-1]-prev_close,p['last']-prev_close) if len(op) else np.nan
    f['open15_low_vs_pm_last_pct'] = 100*(ratio(op.low.min(),p['last'])-1) if len(op) else np.nan
    return f

def labels(b, close_minute):
    """Entry proxy = 09:45 minute open; no delayed entries. No implied executable fill."""
    entrybar = b[b.minute == 585]
    entry = float(entrybar.open.iloc[0]) if len(entrybar) else np.nan
    records = []
    for horizon in HORIZONS:
        end = close_minute if horizon == 'close' else min(close_minute,585+horizon)
        s = window(b,585,end)
        valid = bool(pd.notna(entry) and entry>0 and len(s))
        boundary = bool(valid and int(s.minute.iloc[-1]) == end-1)
        r = {'horizon':str(horizon), 'entry_price_proxy':entry, 'entry_available':pd.notna(entry),
             'interval_end_minute':end, 'observed_minutes':len(s), 'expected_minutes':end-585,
             'end_price_observed':boundary, 'missing_minutes':end-585-len(s),
             'max_gain_pct':np.nan,'max_decline_pct':np.nan,'end_return_pct':np.nan,
             'minutes_to_max_gain':np.nan}
        if valid:
            r.update(max_gain_pct=100*(s.high.max()/entry-1),max_decline_pct=100*(s.low.min()/entry-1),
                     end_return_pct=100*(s.close.iloc[-1]/entry-1) if boundary else np.nan,
                     minutes_to_max_gain=int(s.loc[s.high.idxmax(),'minute'])-585)
        for threshold in THRESHOLDS:
            hit = valid and s.high.max() >= entry*(1+threshold/100)-entry*1e-12
            # Missing bars cannot establish a true negative; observed hits remain positives.
            r[f'runner_{threshold}pct'] = bool(hit) if hit or (valid and len(s)==end-585) else None
            hits = s[s.high >= entry*(1+threshold/100)-entry*1e-12] if valid else s.iloc[:0]
            r[f'minutes_to_{threshold}pct'] = int(hits.minute.iloc[0])-585 if len(hits) else np.nan
        records.append(r)
    return records
