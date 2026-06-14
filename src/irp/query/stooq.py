"""Stooq-sourced OHLCV — a `src='stooq'` view of the unified `prices` table."""
import pandas as pd

from irp.query.prices import prices as _prices


def prices(
    tickers: str | list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """OHLCV for Stooq-sourced rows. Columns: Ticker, Date, Open..Volume, Src, SrcId."""
    return _prices(tickers, start=start, end=end, src='stooq')
