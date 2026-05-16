import json
import time
from datetime import date
from functools import lru_cache

import requests

from irp.core.config import config

CACHE_DIR = config.data.root_dir / 'sec' / 'submissions'
HEADERS = {'User-Agent': 'irp data-quality (admin@local)'}
EDGAR_DOC_URL = 'https://www.sec.gov/Archives/edgar/data/{cik}/{acc_no_dashes}/{primary_doc}'
CACHE_TTL_SECONDS = 86400 * 7


def _fetch_submissions(cik: int) -> dict:
    cik_str = f'{cik:010d}'
    cache = CACHE_DIR / f'{cik_str}.json'
    if cache.exists() and (time.time() - cache.stat().st_mtime) < CACHE_TTL_SECONDS:
        return json.loads(cache.read_text())
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    r = requests.get(
        f'https://data.sec.gov/submissions/CIK{cik_str}.json',
        headers=HEADERS,
        timeout=10,
    )
    r.raise_for_status()
    cache.write_text(r.text)
    time.sleep(0.11)  # SEC limit: 10 req/sec
    return r.json()


def _parse(d: str) -> date | None:
    try:
        return date.fromisoformat(d)
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=10000)
def filing_url(cik: int | None, report_date: str | None, period: str, tol_days: int = 10) -> str | None:
    """Resolve EDGAR URL by Report Date match with +/- tol_days tolerance
    (SimFin uses 09-30, SEC has actual fiscal end e.g. 09-28).
    period='A' -> 10-K, 'Q' -> 10-Q. report_date format 'YYYY-MM-DD'.
    """
    target = _parse(report_date) if report_date else None
    if not cik or target is None:
        return None
    form = '10-K' if period == 'A' else '10-Q'
    try:
        data = _fetch_submissions(int(cik))
    except Exception:
        return None
    recent = data.get('filings', {}).get('recent', {})
    rows = list(zip(
        recent.get('form', []),
        recent.get('accessionNumber', []),
        recent.get('primaryDocument', []),
        recent.get('reportDate', []),
    ))
    best = None
    best_delta = tol_days + 1
    for f, acc, doc, rdate in rows:
        if f != form or not rdate or not doc:
            continue
        rd = _parse(rdate)
        if rd is None:
            continue
        delta = abs((rd - target).days)
        if delta <= tol_days and delta < best_delta:
            best = (acc, doc)
            best_delta = delta
    if best is None:
        return None
    acc, doc = best
    return EDGAR_DOC_URL.format(
        cik=int(cik), acc_no_dashes=acc.replace('-', ''), primary_doc=doc
    )
