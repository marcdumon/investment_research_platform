import duckdb
import pandas as pd
from typing import Literal

from irp.core.config import config

_con: duckdb.DuckDBPyConnection | None = None


def _db() -> duckdb.DuckDBPyConnection:
    global _con
    if _con is None:
        _con = duckdb.connect(str(config.database.path), read_only=True)
    return _con


def _ticker_filter(tickers: str | list[str] | None) -> tuple[str, list]:
    if tickers is None:
        return '', []
    if isinstance(tickers, str):
        tickers = [tickers]
    placeholders = ', '.join('?' * len(tickers))
    return f'Ticker IN ({placeholders})', list(tickers)


def _date_int(date: str) -> int:
    return int(date.replace('-', ''))


def prices(
    tickers: str | list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    src: str | None = None,
) -> pd.DataFrame:
    """OHLCV prices. start/end as 'YYYY-MM-DD'."""
    filters, params = [], []
    clause, vals = _ticker_filter(tickers)
    if clause:
        filters.append(clause)
        params.extend(vals)
    if start is not None:
        filters.append('Date >= ?')
        params.append(_date_int(start))
    if end is not None:
        filters.append('Date <= ?')
        params.append(_date_int(end))
    if src is not None:
        filters.append('Src = ?')
        params.append(src)
    where = f'WHERE {" AND ".join(filters)}' if filters else ''
    return _db().execute(
        f'SELECT * FROM prices {where} ORDER BY Ticker, Date', params
    ).df()


def fundamentals(
    tickers: str | list[str] | None = None,
    statement: Literal['income', 'balance', 'cashflow'] = 'income',
    variant: Literal['A', 'Q'] = 'A',
) -> pd.DataFrame:
    """Financial statement data. variant: 'A' annual, 'Q' quarterly."""
    filters, params = ['Period = ?'], [variant]
    clause, vals = _ticker_filter(tickers)
    if clause:
        filters.append(clause)
        params.extend(vals)
    where = f'WHERE {" AND ".join(filters)}'
    return _db().execute(
        f'SELECT * FROM fundamentals_{statement} {where} ORDER BY Ticker, "Fiscal Year" DESC',
        params,
    ).df()


def companies(tickers: str | list[str] | None = None) -> pd.DataFrame:
    """Company metadata: name, sector, industry, market, ISIN."""
    clause, params = _ticker_filter(tickers)
    where = f'WHERE {clause}' if clause else ''
    return _db().execute(
        f'SELECT * FROM companies {where} ORDER BY Ticker', params
    ).df()


def universe(
    sector: str | None = None,
    industry: str | None = None,
    market: str | None = None,
) -> list[str]:
    """Return tickers matching sector/industry/market filters."""
    filters, params = [], []
    if sector is not None:
        filters.append('Sector = ?')
        params.append(sector)
    if industry is not None:
        filters.append('Industry = ?')
        params.append(industry)
    if market is not None:
        filters.append('Market = ?')
        params.append(market)
    where = f'WHERE {" AND ".join(filters)}' if filters else ''
    rows = _db().execute(
        f'SELECT DISTINCT Ticker FROM companies {where} ORDER BY Ticker', params
    ).fetchall()
    return [r[0] for r in rows]
