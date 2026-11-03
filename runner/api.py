"""Sequential, paced, retrying Massive client. Never log authenticated URLs."""
import hashlib
import json
import os
import random
import time
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlsplit, parse_qsl, urlencode, urlunsplit
import requests
from dotenv import load_dotenv

class API:
    def __init__(self, root, pace=0.15):
        load_dotenv()
        self.key = os.environ['MASSIVE_API_KEY']
        self.cache = Path(root) / 'api_cache'
        self.cache.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.pace = pace
        self.last = 0
        self.requests = 0

    def get(self, path, **params):
        url = path if path.startswith('https:') else 'https://api.massive.com' + path
        parts = urlsplit(url)
        if parts.hostname != 'api.massive.com':
            raise ValueError('Unexpected pagination host')
        query = dict(parse_qsl(parts.query)); query.pop('apiKey', None); query.update(params)
        url = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(sorted(query.items())), ''))
        file = self.cache / (hashlib.sha256(url.encode()).hexdigest() + '.json')
        if file.exists():
            return json.loads(file.read_text())
        for attempt in range(8):
            time.sleep(max(0, self.last + self.pace - time.monotonic()))
            try:
                r = self.session.get(url, headers={'Authorization': 'Bearer ' + self.key}, timeout=(15, 120))
                self.last = time.monotonic(); self.requests += 1
            except requests.RequestException:
                if attempt == 7: raise RuntimeError('API connection failed (URL redacted)') from None
                time.sleep(min(60, 2 ** attempt) + random.random()); continue
            if r.status_code == 429 or r.status_code >= 500:
                retry = r.headers.get('Retry-After', '')
                try: delay = float(retry)
                except ValueError:
                    try: delay = max(0, parsedate_to_datetime(retry).timestamp() - time.time())
                    except (ValueError, TypeError): delay = 2 ** attempt
                time.sleep(max(delay, min(60, 2 ** attempt)) + random.random()); continue
            if r.status_code != 200:
                raise RuntimeError(f'Massive HTTP {r.status_code} on {parts.path}; credentials and URL redacted')
            result = r.json()
            if result.get('status') in ('ERROR', 'NOT_AUTHORIZED'):
                raise RuntimeError('Massive rejected request')
            tmp = file.with_suffix('.tmp'); tmp.write_text(json.dumps(result)); tmp.replace(file)
            return result
        raise RuntimeError('Massive retries exhausted')

    def pages(self, path, **params):
        out = []
        while path:
            d = self.get(path, **params); out.extend(d.get('results', []))
            path = d.get('next_url'); params = {}
        return out
