"""Forward-only, persisted Massive paper execution. No broker orders are submitted."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode, quote

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import boto3
from botocore.config import Config
from dotenv import load_dotenv
import exchange_calendars as xcals
import joblib
import numpy as np
import pandas as pd
import requests
from threadpoolctl import threadpool_limits
from runner.collect import COLUMNS, flat_day, normalize, save, summarize_day, universe
from runner.engine import features, window

HERE = Path(__file__).resolve().parent
ET = 'America/New_York'


def now():
    return pd.Timestamp.now(tz='UTC')


def stamp(day, minute):
    return pd.Timestamp(day, tz=ET) + pd.Timedelta(minutes=minute)


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + '\n')
    tmp.replace(path)


def redact(value):
    if isinstance(value, dict):
        return {k: redact(v) for k, v in value.items() if k.lower() not in ('apikey','api_key','authorization')}
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, str) and value.startswith('https://'):
        p = urlsplit(value)
        return urlunsplit((p.scheme,p.netloc,p.path,urlencode([(k,v) for k,v in parse_qsl(p.query) if k.lower() not in ('apikey','api_key')]),p.fragment))
    return value


def safe_features(bars, cutoff, received_at):
    """Freeze the input window; drop incomplete candles and any returned future bars."""
    starts = pd.to_datetime(bars.window_start, unit='ns', utc=True)
    return bars[(starts < cutoff) & (starts + pd.Timedelta(minutes=1) <= received_at)].copy()


def future_entry(decision_at):
    return decision_at.floor('min') + pd.Timedelta(minutes=1)


def evaluate_order(order, bars, close_at, watermark):
    """Replay only a previously committed paper order, never invent an earlier order."""
    entry_at = pd.Timestamp(order['entry_at'])
    if entry_at <= pd.Timestamp(order['decision_at']):
        raise ValueError('Entry must be strictly after the persisted decision')
    starts = pd.to_datetime(bars.window_start, unit='ns', utc=True)
    # The configured watermark gates candle completion, even if API returns newer bars.
    b = bars[(starts >= entry_at) & (starts < close_at) & (starts + pd.Timedelta(minutes=1) <= watermark)].copy()
    starts = pd.to_datetime(b.window_start, unit='ns', utc=True)
    entry = b.loc[starts.eq(entry_at)]
    final = watermark >= close_at
    if entry.empty:
        return {'status': 'unresolved', 'reason': 'missing_entry'} if final else {'status': 'pending'}
    price = float(entry.open.iloc[0])
    target = price * 1.05
    hit = b[b.high.ge(target - price * 1e-12)]
    base = {'entry_price': price, 'target_price': target, 'shares': order['notional'] / price}
    if len(hit):
        first = hit.iloc[0]
        return {**base, 'status': 'closed', 'reason': 'target', 'exit_price': target,
                'exit_at': pd.Timestamp(int(first.window_start), unit='ns', tz='UTC').isoformat(),
                'gross_return_pct': 5.0}
    if not final:
        return {**base, 'status': 'open'}
    last_at = close_at - pd.Timedelta(minutes=1)
    last = b.loc[starts.eq(last_at)]
    expected = int((close_at - entry_at).total_seconds() / 60)
    if len(last) != 1 or len(b) != expected or b.window_start.nunique() != expected:
        return {**base, 'status': 'unresolved', 'reason': 'missing_path_or_close'}
    exit_price = float(last.close.iloc[0])
    return {**base, 'status': 'closed', 'reason': 'close', 'exit_price': exit_price,
            'exit_at': close_at.isoformat(), 'gross_return_pct': 100 * (exit_price / price - 1)}


class Massive:
    """No response reuse: each live observation has its own receipt timestamp/archive."""
    def __init__(self, root, pace=.2):
        load_dotenv(ROOT / '.env')
        self.key = os.environ.get('MASSIVE_API_KEY')
        if not self.key:
            raise RuntimeError('Set MASSIVE_API_KEY in the project .env')
        self.root = Path(root)
        self.session = requests.Session()
        self.pace, self.last = pace, 0.
        self.last_received = now()
        self.delayed_seen = False

    def get(self, path, **params):
        parts = urlsplit(path if path.startswith('https://') else 'https://api.massive.com' + path)
        if parts.hostname != 'api.massive.com':
            raise ValueError('Unexpected API host')
        query = dict(parse_qsl(parts.query)); query.pop('apiKey', None); query.update(params)
        url = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ''))
        response = None
        for attempt in range(4):
            time.sleep(max(0, self.last + self.pace - time.monotonic()))
            try:
                response = self.session.get(url, headers={'Authorization': 'Bearer ' + self.key}, timeout=(10, 35))
                self.last = time.monotonic()
            except requests.RequestException:
                if attempt == 3:
                    raise RuntimeError('Massive connection failed; authenticated URL withheld') from None
                time.sleep(2 ** attempt)
                continue
            if response.status_code == 429 or response.status_code >= 500:
                time.sleep(min(30, 2 ** attempt))
                continue
            if response.status_code != 200:
                raise RuntimeError(f'Massive HTTP {response.status_code} on {parts.path}')
            data = response.json()
            if data.get('status') in ('ERROR', 'NOT_AUTHORIZED'):
                raise RuntimeError(f'Massive rejected request on {parts.path}')
            self.last_received = now()
            self.delayed_seen |= data.get('status') == 'DELAYED'
            name = f'{time.time_ns()}_{hashlib.sha256(url.encode()).hexdigest()[:12]}.json'
            # Store exact response, response time and parameters; never store an API key.
            atomic(self.root / 'observations' / name, {'received_at': self.last_received.isoformat(),
                   'path': parts.path, 'parameters': query, 'response': redact(data)})
            return data
        raise RuntimeError('Massive retry budget exhausted')

    def pages(self, path, **params):
        rows = []
        while path:
            result = self.get(path, **params)
            rows.extend(result.get('results', []))
            path, params = result.get('next_url'), {}
        return rows

    def bars(self, ticker, start, end):
        # Millisecond bounds avoid requesting today's later bars as inputs.
        rows = self.pages(f'/v2/aggs/ticker/{quote(ticker,safe="")}/range/1/minute/{int(start.timestamp()*1000)}/{int(end.timestamp()*1000)-1}',
                          adjusted='false', sort='asc', limit=50000)
        records = [{'ticker': ticker, 'volume': r['v'], 'open': r['o'], 'close': r['c'],
                    'high': r['h'], 'low': r['l'], 'window_start': int(r['t'])*1000000,
                    'transactions': r.get('n', np.nan)} for r in rows]
        return normalize(pd.DataFrame(records, columns=COLUMNS))


def prior_sessions(day, cal):
    session = pd.Timestamp(day)
    return [str(cal.session_offset(session, offset).date()) for offset in range(-10, 0)]


def prepare(day, root, api, cal):
    """Own copy of ten fully finished sessions; never extend the research dataset."""
    if day != str(now().tz_convert(ET).date()):
        raise ValueError('Preparation must target the current New York date')
    load_dotenv(ROOT / '.env')
    for hd in prior_sessions(day, cal):
        path = root / 'history' / f'{hd}.parquet'
        if path.exists():
            continue
        if cal.session_close(pd.Timestamp(hd)) >= now():
            raise ValueError('History must be from a closed prior session')
        existing = ROOT / 'data/master/daily' / f'{hd}.parquet'
        if existing.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(existing, path)
            continue
        required = ['MASSIVE_FLATFILES_ACCESS_KEY_ID', 'MASSIVE_FLATFILES_SECRET_ACCESS_KEY']
        if not all(os.getenv(k) for k in required):
            raise RuntimeError('History warmup needs Massive flat-file credentials in .env')
        s3 = boto3.client('s3', endpoint_url='https://files.massive.com',
                         aws_access_key_id=os.environ[required[0]], aws_secret_access_key=os.environ[required[1]],
                         config=Config(retries={'max_attempts': 4, 'mode': 'adaptive'}))
        master = universe(api, hd)
        master = master[master.type.isin(['CS', 'ADRC'])]
        bars, raw, provenance = flat_day(hd, root / 'history_download', s3)
        close = cal.session_close(pd.Timestamp(hd)).tz_convert(ET)
        daily = summarize_day(bars, master, close.hour*60+close.minute)
        official = api.get(f'/v2/aggs/grouped/locale/us/market/stocks/{hd}', adjusted='false', include_otc='false').get('results', [])
        if not official:
            raise RuntimeError(f'No official daily data for {hd}')
        official = {r['T']: r for r in official}
        daily['reg_close'] = [official.get(t, {}).get('c', np.nan) for t in daily.ticker]
        daily['reg_volume'] = [official.get(t, {}).get('v', np.nan) for t in daily.ticker]
        save(daily, path)
        atomic(path.with_suffix('.json'), {'day': hd, 'prepared_at': now().isoformat(), 'provenance': provenance})
        raw.unlink()


def history_stack(day, ticker, identity, splits, root, cal):
    rows = []
    for hd in prior_sessions(day, cal):
        h = pd.read_parquet(root / 'history' / f'{hd}.parquet')
        h = h[h.ticker.eq(ticker) & h.identity.eq(identity)].copy()
        if len(h) != 1:
            return None
        events = [e for e in splits if e.get('ticker') == ticker and hd < e['execution_date'] <= day]
        factor = math.prod(float(e['split_to']) / float(e['split_from']) for e in events)
        h['pm_volume'] *= factor; h['reg_volume'] *= factor; h['reg_close'] /= factor
        rows.append(h)
    return pd.concat(rows, ignore_index=True)


def summarize(state, folder):
    orders = list(state.get('orders', {}).values())
    closed = [o for o in orders if o.get('status') == 'closed']
    gross = np.array([o['gross_return_pct'] for o in closed])
    net = gross - state['round_trip_cost_pct']
    wins, losses = net[net > 0], net[net < 0]
    def value(x):
        return float(x) if np.isfinite(x) else None
    report = {'date': state['date'], 'finished': state.get('finished', False),
              'mode': 'forward paper orders; next-minute entries after actual scoring',
              'configured_feed_delay_seconds': state['feed_delay_seconds'],
              'scanned': len(state.get('scanned', [])), 'universe_size': len(state.get('universe', [])),
              'opening_cohort': len(state.get('scores', {})), 'signals': len(orders),
              'closed': len(closed), 'unresolved_or_pending': len(orders)-len(closed),
              'cancelled': sum(o.get('status') == 'cancelled' for o in orders),
              'scan_incomplete': len(state.get('scanned', [])) < len(state.get('universe', [])),
              'wins_net': len(wins), 'losses_net': len(losses), 'breakeven_net': int((net == 0).sum()),
              'win_rate_net_pct': value(100*len(wins)/len(net)) if len(net) else None,
              'mean_gross_pct': value(gross.mean()) if len(gross) else None,
              'mean_net_pct': value(net.mean()) if len(net) else None,
              'average_win_net_pct': value(wins.mean()) if len(wins) else None,
              'average_loss_net_pct': value(losses.mean()) if len(losses) else None,
              'max_gain_net_pct': value(net.max()) if len(net) else None,
              'max_loss_net_pct': value(net.min()) if len(net) else None,
              'total_resolved_net_pnl_dollars': value(sum(o['notional']*r/100 for o,r in zip(closed,net))),
              'target_exits': sum(o.get('reason') == 'target' for o in closed),
              'close_exits': sum(o.get('reason') == 'close' for o in closed),
              'errors': state.get('errors', []), 'updated_at': now().isoformat()}
    atomic(folder / 'summary.json', report)
    if orders:
        pd.DataFrame(orders).to_csv(folder / 'trades.csv', index=False)
    text = ['# End-of-day paper strategy summary', '',
            f"Session: {state['date']}. Finished: {report['finished']}.", '',
            f"Signals: {len(orders)}; resolved: {len(closed)}; unresolved/pending: {len(orders)-len(closed)}.",
            f"Net wins: {len(wins)}; net losses: {len(losses)}.", '', '```', json.dumps(report, indent=2), '```', '',
            'Equal fixed-dollar paper positions, fractional shares, no capital cap. P&L totals describe resolved trades only. Missing outcomes are not zero returns. Entry is after the actual decision, not backdated to 09:45. Target touch assumes a full fill. Costs are a configured scenario, not measured costs. No broker orders were submitted.']
    (folder / 'SUMMARY.md').write_text('\n'.join(text)+'\n')
    return report


def recover_previous(args, cal):
    """Only settle previously saved orders after downtime; never reconstruct signals."""
    today = str(now().tz_convert(ET).date())
    for file in sorted((args.state / 'sessions').glob('*/state.json')):
        state = json.loads(file.read_text())
        if state['date'] >= today or state.get('finished'):
            continue
        api = Massive(file.parent)
        close = cal.session_close(pd.Timestamp(state['date']))
        state['recovered_after_session'] = True
        for ticker, order in state['orders'].items():
            if order['status'] in ('closed','unresolved','cancelled'): continue
            try:
                bars = api.bars(ticker, pd.Timestamp(order['entry_at']), close)
                order.update(evaluate_order(order,bars,close,now()))
                order['late_recovery'] = True
            except Exception as exc:
                order.update(status='unresolved',reason='recovery_api_failure')
                state['errors'].append({'ticker':ticker,'error':type(exc).__name__,'at':now().isoformat()})
        state['finished'] = True
        atomic(file,state)
        summarize(state,file.parent)


def run_day(args, cal):
    day = str(now().tz_convert(ET).date())
    session = pd.Timestamp(day)
    if not cal.is_session(session):
        print('No exchange session today.', flush=True)
        return
    folder = args.state / 'sessions' / day
    folder.mkdir(parents=True, exist_ok=True)
    file = folder / 'state.json'
    state = json.loads(file.read_text()) if file.exists() else None
    close = cal.session_close(session)
    cutoff = stamp(day, 585).tz_convert('UTC')
    deadline = min(stamp(day, args.entry_deadline).tz_convert('UTC'), close - pd.Timedelta(minutes=1))
    if state and state.get('finished'):
        summarize(state, folder); return
    if state is None and now() >= deadline:
        print('Entry window has ended; refusing retrospective paper entries.', flush=True)
        return
    api = Massive(folder)
    artifact_path = HERE / 'artifacts/model.joblib'
    digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    expected = json.loads((HERE / 'artifacts/bundle.json').read_text())['model_sha256']
    if digest != expected:
        raise ValueError('Bundled model hash mismatch')
    artifact = joblib.load(artifact_path)
    if state is None:
        prepare(day, args.state, api, cal)
        master = universe(api, day)
        master = master[master.type.isin(['CS','ADRC'])].copy()
        # Freeze identity and reference universe before scanning. No price-based live shortlist.
        master['identity'] = [next((str(r.get(k)) for k in ('share_class_figi','composite_figi') if pd.notna(r.get(k)) and r.get(k)), r.ticker) for _,r in master.iterrows()]
        if args.tickers:
            master = master[master.ticker.isin(args.tickers.split(','))]
        splits = api.pages('/stocks/v1/splits', **{'execution_date.gte': prior_sessions(day,cal)[0], 'execution_date.lte': day, 'limit':1000})
        # Precompute ten-session identity-matched history once; no current-session aggregates here.
        histories = {}
        for hd in prior_sessions(day, cal):
            h = pd.read_parquet(args.state/'history'/f'{hd}.parquet')
            events = pd.DataFrame([e for e in splits if hd < e['execution_date'] <= day])
            if not events.empty:
                events['factor'] = events.split_to / events.split_from
                factors = h.ticker.map(events.groupby('ticker').factor.prod()).fillna(1.)
                h['pm_volume'] *= factors; h['reg_volume'] *= factors; h['reg_close'] /= factors
            for r in h.to_dict('records'):
                histories.setdefault((r['ticker'], r['identity']), []).append(r)
        ready = []
        for r in master[['ticker','identity']].to_dict('records'):
            hist = histories.get((r['ticker'],r['identity']), [])
            if len(hist) != 10 or not np.isfinite(hist[-1]['reg_close']) or hist[-1]['reg_close'] <= 0 or sum(x['pm_volume'] for x in hist) <= 0:
                continue
            r['history'] = hist
            ready.append(r)
        atomic(folder / 'universe.json', ready)
        state = {'date': day, 'created_at': now().isoformat(), 'model_sha256': digest,
                 'cutoff': artifact['policy']['cutoff'], 'feed_delay_seconds': args.feed_delay_seconds,
                 'round_trip_cost_pct': args.round_trip_cost_pct, 'notional': args.notional,
                 'entry_deadline': deadline.isoformat(), 'universe': [{'ticker':r['ticker'],'identity':r['identity']} for r in ready], 'scanned': [], 'scores': {},
                 'orders': {}, 'errors': [], 'finished': False}
        atomic(file, state)
    if state['model_sha256'] != digest:
        raise ValueError('Existing session belongs to another model')
    if any(state[k] != v for k,v in [('feed_delay_seconds',args.feed_delay_seconds),('round_trip_cost_pct',args.round_trip_cost_pct),('notional',args.notional)]):
        raise ValueError('Resume with the same delay, costs and position size')
    deadline = pd.Timestamp(state['entry_deadline'])
    ready = json.loads((folder / 'universe.json').read_text())
    last_monitor = 0.
    while True:
        current = now()
        watermark = current - pd.Timedelta(seconds=args.feed_delay_seconds + args.bar_grace_seconds)
        if watermark < cutoff:
            if args.once: return
            time.sleep(min(30, max(1,(cutoff-watermark).total_seconds())))
            continue
        remaining = [r for r in ready if r['ticker'] not in state['scanned']]
        if remaining and current < deadline:
            r = remaining[0]; ticker = r['ticker']
            try:
                bars = api.bars(ticker, stamp(day,240), cutoff)
                received = now()
                if args.feed_delay_seconds == 0 and api.delayed_seen:
                    raise RuntimeError('Delayed response with zero-delay configuration; refusing entries')
                bars = safe_features(bars, cutoff, received)
                pm = window(bars,240,570); op=window(bars,570,585)
                hist = pd.DataFrame(r['history']); baseline=float(hist.pm_volume.mean()); prev=float(hist.reg_close.iloc[-1])
                if not pm.empty and len(op)==15 and pm.close.iloc[-1]/prev-1 >= .10-1e-12 and pm.volume.sum()/baseline >=5-1e-12:
                    f = features(bars,prev,baseline,hist,cutoff)
                    if f['open15_volume']*f['open15_vwap_proxy'] >= 1e6:
                        x=pd.DataFrame([f])[artifact['features']].replace([np.inf,-np.inf],np.nan)
                        score=float(artifact['estimator'].predict_proba(x)[0,1])
                        decision=now(); entry=future_entry(decision)
                        save(pd.DataFrame([{'date':day,'ticker':ticker,'identity':r['identity'],**f}]),folder/'features'/f'{ticker}.parquet')
                        state['scores'][ticker]={'score':score,'decision_at':decision.isoformat(),'feature_cutoff':cutoff.isoformat()}
                        if score >= state['cutoff'] and entry < deadline and entry < close:
                            state['orders'][ticker]={'ticker':ticker,'identity':r['identity'],'score':score,
                                'decision_at':decision.isoformat(),'entry_at':entry.isoformat(),
                                'notional':args.notional,'status':'pending'}
                state['scanned'].append(ticker)
                # Persist orders before the committed entry minute. Never submit an order saved too late.
                atomic(file,state)
                if ticker in state['orders'] and now() >= pd.Timestamp(state['orders'][ticker]['entry_at']):
                    state['orders'][ticker].update(status='cancelled',reason='missed_order_persistence_deadline')
                    atomic(file,state)
            except Exception as exc:
                state['errors'].append({'at':now().isoformat(),'ticker':ticker,'error':type(exc).__name__})
                # Failure skips a symbol; no later retrospective reconstruction of its decision.
                state['scanned'].append(ticker)
                atomic(file,state)
                if args.feed_delay_seconds == 0 and api.delayed_seen:
                    raise RuntimeError('Delayed feed detected; use --feed-delay-seconds 900') from None
        if time.monotonic()-last_monitor >= args.poll_seconds or watermark >= close or args.once:
            for ticker, order in state['orders'].items():
                if order['status'] in ('closed','unresolved','cancelled'): continue
                if watermark < pd.Timestamp(order['entry_at']) + pd.Timedelta(minutes=1): continue
                try:
                    bars=api.bars(ticker,pd.Timestamp(order['entry_at']),min(now(),close))
                    receipt=now()
                    order.update(evaluate_order(order,bars,close,receipt-pd.Timedelta(seconds=args.feed_delay_seconds+args.bar_grace_seconds)))
                    order['last_observed_at']=receipt.isoformat()
                except Exception as exc:
                    state['errors'].append({'at':now().isoformat(),'ticker':ticker,'error':type(exc).__name__})
            last_monitor=time.monotonic()
            atomic(file,state)
            summarize(state,folder)
        if watermark >= close:
            # Permit transient close-data failures for 10 minutes, then report unresolved explicitly.
            pending=[o for o in state['orders'].values() if o['status'] in ('pending','open')]
            if not pending or watermark >= close+pd.Timedelta(minutes=10):
                for o in pending: o.update(status='unresolved',reason='api_unavailable_at_finalization')
                state['finished']=True
                atomic(file,state)
                summary=summarize(state,folder)
                print(json.dumps(summary,indent=2),flush=True)
                print(f'End-of-day summary: {folder / "SUMMARY.md"}',flush=True)
                return
        if args.once:return
        # During scanning do not sleep except API pacing. Otherwise stay responsive to shutdown.
        if not remaining or now() >= deadline:time.sleep(min(30,args.poll_seconds))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','run','daemon','summary','check'])
    parser.add_argument('--state',type=Path,default=HERE/'runtime')
    parser.add_argument('--feed-delay-seconds',type=int,default=900)
    parser.add_argument('--bar-grace-seconds',type=int,default=10)
    parser.add_argument('--poll-seconds',type=int,default=60)
    parser.add_argument('--notional',type=float,default=1000.)
    parser.add_argument('--round-trip-cost-pct',type=float,default=.5)
    parser.add_argument('--entry-deadline',type=int,default=630,help='Minutes after New York midnight; 630 = 10:30 ET')
    parser.add_argument('--tickers',help='Optional explicit restricted universe; default full CS/ADRC universe')
    parser.add_argument('--once',action='store_true',help='One iteration for diagnostics; not an unattended trading session')
    args=parser.parse_args()
    if args.notional<=0 or args.feed_delay_seconds<0 or args.bar_grace_seconds<0 or args.poll_seconds<1 or args.round_trip_cost_pct<0 or not 586<=args.entry_deadline<=959:
        parser.error('Invalid timing, size or cost configuration')
    args.state.mkdir(parents=True,exist_ok=True)
    # One writer per state root; lock is released automatically after a crash.
    with (args.state/'runner.lock').open('a') as lock, threadpool_limits(limits=2):
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('A runner already owns this state directory')
        cal=xcals.get_calendar('XNYS')
        day=str(now().tz_convert(ET).date())
        if args.command=='summary':
            file=args.state/'sessions'/day/'state.json'
            if not file.exists():raise SystemExit('No recorded forward session today; no retrospective trades will be generated')
            print(json.dumps(summarize(json.loads(file.read_text()),file.parent),indent=2));return
        if args.command=='check':
            api=Massive(args.state/'checks')
            end=now();bars=api.bars('AAPL',end-pd.Timedelta(days=1),end)
            latest=pd.to_datetime(bars.window_start,unit='ns',utc=True).max() if len(bars) else None
            print(json.dumps({'api_access':True,'delayed_status_seen':api.delayed_seen,'latest_bar_start':str(latest),'configured_delay_seconds':args.feed_delay_seconds,'model_bundle_present':(HERE/'artifacts/model.joblib').exists()},indent=2));return
        if args.command=='prepare':
            if not cal.is_session(pd.Timestamp(day)):raise SystemExit('Today is not a trading session')
            prepare(day,args.state,Massive(args.state/'preparation'/day),cal);print('Prior-session history ready.');return
        if args.command=='run':
            recover_previous(args,cal)
            run_day(args,cal);return
        while True:
            recover_previous(args,cal)
            today=now().tz_convert(ET)
            if cal.is_session(pd.Timestamp(str(today.date()))) and today.hour>=8:
                file=args.state/'sessions'/str(today.date())/'state.json'
                finished=file.exists() and json.loads(file.read_text()).get('finished')
                if not finished and today.hour*60+today.minute<args.entry_deadline:
                    try:run_day(args,cal)
                    except Exception as exc:
                        print(f'{now().isoformat()} session error: {type(exc).__name__}; retry in 30s',flush=True)
                elif file.exists() and not finished:
                    # Resume committed positions even after the entry deadline.
                    try:run_day(args,cal)
                    except Exception as exc:print(f'Resume error: {type(exc).__name__}',flush=True)
            time.sleep(30)


if __name__=='__main__':main()
