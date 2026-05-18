"""Row accessors for Yahoo tables (`yahoo_prices`, `dividends`, `splits`)."""
import pandas as pd

from irp.data._common import db, ticker_filter


def prices(
    tickers: str | list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Auto-adjusted OHLCV prices from Yahoo Finance. `start` / `end` as 'YYYY-MM-DD'."""
    filters, params = [], []
    clause, vals = ticker_filter(tickers)
    if clause:
        filters.append(clause)
        params.extend(vals)
    if start is not None:
        filters.append('Date >= ?')
        params.append(start)
    if end is not None:
        filters.append('Date <= ?')
        params.append(end)
    where = f'WHERE {" AND ".join(filters)}' if filters else ''
    return (
        db()
        .execute(f'SELECT * FROM yahoo_prices {where} ORDER BY Ticker, Date', params)
        .df()
    )


def dividends(
    tickers: str | list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Cash dividend events. `start` / `end` as 'YYYY-MM-DD'."""
    filters, params = [], []
    clause, vals = ticker_filter(tickers)
    if clause:
        filters.append(clause)
        params.extend(vals)
    if start is not None:
        filters.append('Date >= ?')
        params.append(start)
    if end is not None:
        filters.append('Date <= ?')
        params.append(end)
    where = f'WHERE {" AND ".join(filters)}' if filters else ''
    return (
        db()
        .execute(f'SELECT * FROM dividends {where} ORDER BY Ticker, Date', params)
        .df()
    )


def splits(
    tickers: str | list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Stock split events. `start` / `end` as 'YYYY-MM-DD'."""
    filters, params = [], []
    clause, vals = ticker_filter(tickers)
    if clause:
        filters.append(clause)
        params.extend(vals)
    if start is not None:
        filters.append('Date >= ?')
        params.append(start)
    if end is not None:
        filters.append('Date <= ?')
        params.append(end)
    where = f'WHERE {" AND ".join(filters)}' if filters else ''
    return (
        db()
        .execute(f'SELECT * FROM splits {where} ORDER BY Ticker, Date', params)
        .df()
    )
