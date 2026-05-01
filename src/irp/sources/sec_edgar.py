import functools
import logging
from datetime import datetime

import requests

logger = logging.getLogger(__name__)

SEC_HEADERS = {
    "User-Agent": "marc Duon marcdumon@msn.com",
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}

TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSION_URL = "https://data.sec.gov/submissions/CIK{:010d}.json"
ARCHIVE = "https://www.sec.gov/Archives/edgar/data"


@functools.lru_cache(maxsize=1)
def _load_ticker_map() -> dict[str, int]:
    """Download SEC ticker→CIK map once per process."""
    r = requests.get(TICKER_URL, headers={"User-Agent": SEC_HEADERS["User-Agent"]})
    r.raise_for_status()
    return {v["ticker"].upper(): int(v["cik_str"]) for v in r.json().values()}


def _ticker_to_cik(ticker: str) -> int:
    mapping = _load_ticker_map()
    ticker = ticker.upper()
    if ticker not in mapping:
        raise ValueError("ticker not found")
    return mapping[ticker]


@functools.lru_cache(maxsize=512)
def _load_submissions(cik: int) -> dict:
    """Fetch recent filings for a CIK — cached per CIK per process."""
    r = requests.get(SUBMISSION_URL.format(cik), headers=SEC_HEADERS)
    r.raise_for_status()
    return r.json()["filings"]["recent"]


def _parse_period(period: str) -> tuple[int, str, int | None]:
    year = int(period[:4])
    if period.endswith("A") or period.endswith("Q4"):
        return year, "A", None
    q = int(period[-1])
    return year, "Q", q


def _pick_filing(
    filings: dict,
    year: int,
    kind: str,
    quarter: int | None,
    *,
    publish_date: str | None = None,
) -> tuple[str, str, str]:
    annual_forms = {"10-K", "20-F"}
    quarterly_forms = {"10-Q", "6-K"}
    valid_forms = annual_forms if kind == "A" else quarterly_forms
    rows = list(zip(
        filings["form"], filings["filingDate"],
        filings["accessionNumber"], filings["primaryDocument"],
    ))

    if publish_date:
        # Filter by form type only — date uniquely identifies the filing
        # regardless of fiscal year alignment
        candidates = [(d, a, doc) for f, d, a, doc in rows if f in valid_forms]
        if not candidates:
            raise ValueError("no matching filing")
        target = datetime.strptime(publish_date, "%Y-%m-%d")
        candidates.sort(key=lambda x: abs((datetime.strptime(x[0], "%Y-%m-%d") - target).days))
        return candidates[0]

    # Fallback: calendar year/quarter matching
    candidates = []
    for form, date, acc, doc in rows:
        y = int(date[:4])
        if kind == "A" and form in annual_forms and y == year:
            candidates.append((date, acc, doc))
        elif kind == "Q" and form in quarterly_forms and y == year:
            if quarter is None:
                candidates.append((date, acc, doc))
            else:
                m = int(date[5:7])
                if (m - 1) // 3 + 1 == quarter:
                    candidates.append((date, acc, doc))

    if not candidates:
        raise ValueError("no matching filing")
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0]


def sec_filing_url(ticker: str, period: str, publish_date: str | None = None) -> str:
    """Return a direct URL to the primary document of the SEC filing.

    Args:
        ticker: Stock ticker (e.g. 'MSFT').
        period: Reporting period string (e.g. '2023A', '2023Q1').
        publish_date: Optional filing date hint (YYYY-MM-DD) for precise matching.

    Returns:
        URL to the primary filing document on SEC EDGAR.

    Raises:
        ValueError: Ticker not found in SEC map, or no matching filing.
        requests.HTTPError: Network / HTTP error.
    """
    cik = _ticker_to_cik(ticker)
    year, kind, quarter = _parse_period(period)
    filings = _load_submissions(cik)
    _, acc, doc = _pick_filing(filings, year, kind, quarter, publish_date=publish_date)
    acc_nodash = acc.replace("-", "")
    return f"{ARCHIVE}/{cik}/{acc_nodash}/{doc}"
