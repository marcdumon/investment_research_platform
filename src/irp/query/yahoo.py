"""Row accessors for Yahoo data: `prices` (a `src='yahoo'` view of the unified
`prices` table) plus the `dividends` and `splits` tables."""
import pandas as pd

from irp.query._common import build_where, db
from irp.query.prices import prices as _prices


def _select(table: str, tickers, start, end) -> pd.DataFrame:
    where, params = build_where(tickers, start, end)
    return (
        db()
        .execute(f'SELECT * FROM {table} {where} ORDER BY Ticker, Date', params)
        .df()
    )


def prices(
    tickers: str | list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Auto-adjusted OHLCV prices from Yahoo Finance. `start` / `end` as 'YYYY-MM-DD'."""
    return _prices(tickers, start=start, end=end, src='yahoo')


def dividends(
    tickers: str | list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Cash dividend events. `start` / `end` as 'YYYY-MM-DD'."""
    return _select('dividends', tickers, start, end)


def splits(
    tickers: str | list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Stock split events. `start` / `end` as 'YYYY-MM-DD'."""
    return _select('splits', tickers, start, end)
