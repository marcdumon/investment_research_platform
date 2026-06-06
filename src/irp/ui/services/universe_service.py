"""Universe + sector lookups for the UI layer.

Wraps `irp.query.universe` and `irp.query.simfin.companies`. Owns the
market/sector/watchlist ticker-filter logic that was previously
duplicated in `backtest.py` and `screener.py`.
"""

from typing import Literal

import pandas as pd

from irp.query.simfin import companies as _companies, sector_map as _sector_map, statement as _statement
from irp.query.universe import universe as _universe
from irp.ui.services.watchlist_service import load_watchlist


def _get_universe(tickers: str | list[str] | None = None) -> pd.DataFrame:
    """Editable universe table (Ticker, Market, stooq_ticker, yahoo_ticker).

    Optionally filter to one or more tickers; None = full table.
    """
    return _universe(tickers) if tickers else _universe()


def _get_companies(tickers: str | list[str] | None = None) -> pd.DataFrame:
    """SimFin companies metadata (Ticker, Sector, ISIN, ...).

    Optionally filter to one or more tickers; None = full table.
    """
    return _companies(tickers) if tickers else _companies()


def _get_sectors() -> list[str]:
    """Sorted unique sectors from the SimFin companies table."""
    return sorted(_sector_map().dropna().unique().tolist())


def _get_ticker_cik(ticker: str) -> int | None:
    """CIK for a ticker from the SimFin companies table, or None."""
    df = _get_companies([ticker])
    if df.empty or 'CIK' not in df.columns:
        return None
    val = df.iloc[0].get('CIK')
    try:
        return int(val) if pd.notna(val) else None
    except (TypeError, ValueError):
        return None


def _get_period_report_dates(ticker: str, kind: Literal['income', 'balance', 'cashflow']) -> dict[str, str]:
    """Return {period_str: 'YYYY-MM-DD'} for all filings of a ticker/statement."""
    from irp.query.simfin import fundamentals
    parts = [fundamentals(ticker, kind, v) for v in ('A', 'Q')]
    df = pd.concat(parts, ignore_index=True)
    if df.empty or 'Report Date' not in df.columns:
        return {}
    df = df.copy()
    df.loc[df['Period'] == 'A', 'Fiscal Period'] = 'FY'
    df['_period'] = [
        f'{int(fy)}FY' if p == 'A' else f'{int(fy)}{fp}'
        for fy, fp, p in zip(df['Fiscal Year'], df['Fiscal Period'], df['Period'], strict=False)
    ]
    df = df.drop_duplicates('_period', keep='first')
    return {
        row['_period']: str(row['Report Date'])[:10]
        for _, row in df.iterrows()
        if row.get('Report Date')
    }


def _get_statement(
    ticker: str, kind: Literal['income', 'balance', 'cashflow']
) -> pd.DataFrame:
    """Fundamental statement for one ticker (all available periods).

    `kind` ∈ {'income', 'balance', 'cashflow'}. Variant filtering happens
    at the caller via column-name suffix ('FY' = annual, else quarterly).
    """
    return _statement(ticker, kind)


def _filter_tickers(
    market: str | None = None,
    sector: str | None = None,
    watchlist: str | None = None,
) -> list[str] | None:
    """Resolve a market/sector/watchlist combination to a ticker list.

    Returns None when no filter is active (caller treats as "all tickers").
    Returns [] when filters produce an empty set (caller still respects this
    distinct from None).
    """
    if not (market or sector or watchlist):
        return None

    if watchlist:
        wl = load_watchlist(watchlist)
        return list(wl) if wl else []

    u = _get_universe()[['Ticker', 'Market']]
    c = _get_companies()[['Ticker', 'Sector']]
    df = u.merge(c, on='Ticker', how='left')
    if market:
        df = df[df['Market'].str.lower().str.contains(market.lower(), na=False)]
    if sector:
        df = df[df['Sector'] == sector]
    tickers = df['Ticker'].dropna().unique().tolist()
    return tickers if tickers else None
