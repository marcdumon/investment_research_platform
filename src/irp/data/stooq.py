"""Row accessors for Stooq tables (`prices`, `markets`)."""
import pandas as pd

from irp.data._common import db, ticker_filter


def prices(
    tickers: str | list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    src: str | None = None,
) -> pd.DataFrame:
    """OHLCV prices. `start` / `end` as 'YYYY-MM-DD'."""
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
    if src is not None:
        filters.append('Src = ?')
        params.append(src)
    where = f'WHERE {" AND ".join(filters)}' if filters else ''
    return (
        db()
        .execute(f'SELECT * FROM prices {where} ORDER BY Ticker, Date', params)
        .df()
    )


def markets(tickers: str | list[str] | None = None) -> pd.DataFrame:
    """Ticker → market mapping. `Market` is Stooq's lowercase category folder."""
    clause, params = ticker_filter(tickers)
    where = f'WHERE {clause}' if clause else ''
    return (
        db()
        .execute(f'SELECT * FROM markets {where} ORDER BY Ticker', params)
        .df()
    )
