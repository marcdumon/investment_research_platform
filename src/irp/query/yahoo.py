"""Row accessors for Yahoo tables (`yahoo_prices`, `dividends`, `splits`)."""
import pandas as pd

from irp.query._common import build_where, db


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
    return _select('yahoo_prices', tickers, start, end)


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
