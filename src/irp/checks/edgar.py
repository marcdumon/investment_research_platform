import json
import logging
import time
from datetime import date
from functools import lru_cache

import requests

from irp.core.config import config

logger = logging.getLogger(__name__)

CACHE_DIR = config.data.root_dir / 'sec' / 'submissions'
HEADERS = {'User-Agent': 'irp data-quality (admin@local)'}
EDGAR_DOC_URL = 'https://www.sec.gov/Archives/edgar/data/{cik}/{acc_no_dashes}/{primary_doc}'
CACHE_TTL_SECONDS = 86400 * 7

# CIKs for companies that reorganised under a new SEC entity.
# Maps current CIK → list of predecessor CIKs to try when the current CIK
# has no filings old enough to match the target date.
_LEGACY_CIKS: dict[int, list[int]] = {
    1652044: [1288776],  # Alphabet Inc. → Google Inc. (pre-2015 filings)
}


def _fetch_chunk(filename: str) -> dict:
    """Fetch one older-filings chunk listed in `filings.files[]`.

    SEC partitions per-CIK filings: the master JSON's `filings.recent`
    holds ~1000 most-recent filings, with everything older referenced
    in `filings.files[]` as additional chunk file names (e.g.
    `CIK0001326801-submissions-001.json`). Each chunk has the same
    parallel-arrays schema as `recent` (no nesting). Disk-cached for
    `CACHE_TTL_SECONDS`.
    """
    cache = CACHE_DIR / filename
    if cache.exists() and (time.time() - cache.stat().st_mtime) < CACHE_TTL_SECONDS:
        logger.debug(f'EDGAR chunk cache hit: {filename}')
        return json.loads(cache.read_text())
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f'GET https://data.sec.gov/submissions/{filename}')
    r = requests.get(
        f'https://data.sec.gov/submissions/{filename}',
        headers=HEADERS,
        timeout=10,
    )
    r.raise_for_status()
    cache.write_text(r.text)
    time.sleep(0.11)
    return r.json()


def _fetch_submissions(cik: int) -> dict:
    """Fetch the SEC submissions JSON for a CIK with full history.

    Hits `https://data.sec.gov/submissions/CIK{cik:010d}.json` for the
    master JSON, then iterates `filings.files[]` to fetch every older
    chunk and `list.extend()`-merges them into `filings.recent`. After
    return, `data['filings']['recent']` spans the full filing history
    for the company, exposed via the same parallel-arrays schema.

    Both master and chunk files are disk-cached. Subsequent calls within
    `CACHE_TTL_SECONDS` issue zero SEC requests.
    """
    cik_str = f'{cik:010d}'
    cache = CACHE_DIR / f'{cik_str}.json'
    if cache.exists() and (time.time() - cache.stat().st_mtime) < CACHE_TTL_SECONDS:
        logger.debug(f'EDGAR submissions cache hit: CIK {cik}')
        data = json.loads(cache.read_text())
    else:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f'GET https://data.sec.gov/submissions/CIK{cik_str}.json')
        r = requests.get(
            f'https://data.sec.gov/submissions/CIK{cik_str}.json',
            headers=HEADERS,
            timeout=10,
        )
        r.raise_for_status()
        cache.write_text(r.text)
        time.sleep(0.11)  # SEC limit: 10 req/sec
        data = r.json()

    filings = data.get('filings', {})
    recent = {k: list(v) for k, v in filings.get('recent', {}).items()}
    for finfo in filings.get('files', []):
        chunk = _fetch_chunk(finfo['name'])
        for key in recent:
            if key in chunk:
                recent[key].extend(chunk[key])
    data['filings']['recent'] = recent
    return data


def _parse(d: str) -> date | None:
    """Parse an ISO date string to `date`, return None on bad input."""
    try:
        return date.fromisoformat(d)
    except (TypeError, ValueError):
        return None


def _search_submissions(cik: int, target, forms: set[str], tol_days: int) -> tuple[str, str, int] | None:
    """Scan submissions for one CIK; return (accessionNumber, primaryDoc, cik) or None."""
    try:
        data = _fetch_submissions(cik)
    except (requests.RequestException, ValueError) as e:
        logger.debug(f'EDGAR fetch failed for CIK {cik}: {type(e).__name__}: {e}')
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
        if f not in forms or not rdate or not doc:
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
    return acc, doc, cik


@lru_cache(maxsize=10000)
def filing_url(cik: int | None, report_date: str | None, period: str, tol_days: int = 10) -> str | None:
    """Build the direct EDGAR document URL for one (cik, report_date, period).

    Picks the form by period ('A' -> 10-K, 'Q' -> 10-Q) and scans the
    SEC submissions JSON for a filing whose `reportDate` is within
    `tol_days` of `report_date`, returning the row with the smallest
    delta. The tolerance handles SimFin's month-end normalisation
    (e.g. 2024-09-30) versus SEC's actual fiscal end (e.g. 2024-09-28).

    For companies that reorganised under a new SEC entity (e.g. Alphabet/Google),
    falls back to `_LEGACY_CIKS` if the primary CIK has no matching filing.

    Returns a URL of the form
        https://www.sec.gov/Archives/edgar/data/{cik}/{acc_no_dashes}/{primary_doc}
    or None if the CIK/report_date is missing, the fetch fails, or no
    filing matches within the tolerance.

    `report_date` must be 'YYYY-MM-DD'. Memoised per (cik, report_date,
    period, tol_days) for the process lifetime; the underlying submissions
    JSON is disk-cached separately by `_fetch_submissions`.
    """
    target = _parse(report_date) if report_date else None
    if not cik or target is None:
        return None
    # Accept historical form variants (10-K405, 10-KSB used pre-2002; 10-QSB for small filers)
    forms = {'10-K', '10-K405', '10-KSB'} if period == 'A' else {'10-Q', '10-QSB'}

    ciks_to_try = [int(cik)] + _LEGACY_CIKS.get(int(cik), [])
    for try_cik in ciks_to_try:
        result = _search_submissions(try_cik, target, forms, tol_days)
        if result is not None:
            acc, doc, matched_cik = result
            return EDGAR_DOC_URL.format(
                cik=matched_cik, acc_no_dashes=acc.replace('-', ''), primary_doc=doc
            )
    return None
