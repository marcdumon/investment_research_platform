"""Provider-agnostic ticker universe + the market-priority dedup.

Stooq's `markets.csv` lists one row per `<region>/<market>` ticker file. Several
rows can collapse to the same canonical ticker (e.g. `t.us` = AT&T on NYSE and
`t.v` = a crypto, both -> `T`). The old seed kept the first occurrence, so a
crypto could shadow a real stock — and since crypto is excluded from Yahoo
fetch, the stock then had no prices. Here, dedup is by market *tier*: any
stock/ETF outranks everything, and crypto/fx/bonds sit in the lowest tier, so
the equity always wins regardless of row order.
"""
import re
from pathlib import Path

import pandas as pd

from dataload.context import IngestContext

# Markets banned outright — never seeded, never normalized into prices, never stored.
BANNED_MARKETS = {'cryptocurrencies'}

# Lowest-tier markets: real but never preferred when a ticker collides with equity.
_LOW_TIER_MARKETS = {'cryptocurrencies', 'currencies', 'money market', 'bonds', 'commodities'}


def _market_rank(market: str) -> int:
    """0 = stocks/ETFs (preferred), 1 = other/unknown, 2 = crypto/fx/bonds (last)."""
    m = market.lower()
    if 'stocks' in m or 'etf' in m:
        return 0
    if m in _LOW_TIER_MARKETS:
        return 2
    return 1


def _yahoo_ticker(ticker: str, market: str) -> str | None:
    """Map a canonical ticker to its Yahoo Finance symbol (None if no equivalent)."""
    m = market.lower()
    if m == 'stooq stocks indices':
        return None
    if m == 'currencies':
        return f'{ticker}=X' if re.fullmatch(r'[A-Z]{6}', ticker) else None
    match = re.fullmatch(r'([A-Z0-9]+)_([A-Z])', ticker)
    if match:
        return f'{match.group(1)}-P{match.group(2)}'
    return ticker


def seed_from_markets(markets_csv: Path, out_csv: Path) -> int:
    """Build universe.csv from Stooq markets.csv, deduped by market tier.

    Returns the number of canonical tickers written.
    """
    raw = pd.read_csv(markets_csv, dtype=str).fillna('')
    raw = raw[~raw['market'].str.lower().isin(BANNED_MARKETS)]
    raw['stooq_ticker'] = raw['ticker'].str.upper()
    raw['Ticker'] = raw['stooq_ticker'].str.split('.').str[0]
    raw['_rank'] = raw['market'].map(_market_rank)
    raw = raw.sort_values('_rank', kind='stable').drop_duplicates('Ticker', keep='first')

    raw['Market'] = raw['market']
    raw['yahoo_ticker'] = [_yahoo_ticker(t, m) for t, m in zip(raw['Ticker'], raw['Market'], strict=True)]
    out = raw[['Ticker', 'Market', 'stooq_ticker', 'yahoo_ticker']].sort_values('Ticker')
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return len(out)


def seed(ctx: IngestContext) -> int:
    """Rebuild universe.csv from the Stooq provider's markets.csv."""
    return seed_from_markets(ctx.raw_dir('stooq') / 'markets.csv', ctx.data_root / 'universe.csv')


def refresh(ctx: IngestContext) -> int:
    """Load universe.csv into the `universe` DB table. Returns the row count."""
    df = pd.read_csv(ctx.data_root / 'universe.csv')
    with ctx.connect() as con:
        con.register('_universe_new', df)
        con.execute('CREATE OR REPLACE TABLE universe AS SELECT * FROM _universe_new')
        con.unregister('_universe_new')
        count: int = con.execute('SELECT COUNT(*) FROM universe').fetchone()[0]  # type: ignore[index]
    return count
