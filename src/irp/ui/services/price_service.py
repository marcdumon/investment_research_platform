"""Price + corporate-action data for the UI layer.

Wraps `irp.query.stooq` and `irp.query.yahoo` so pages don't deal with:
- two different source adapters
- different column-name conventions (Yahoo's `Dividends` vs canonical `Amount`)
- repeated empty-DataFrame defensive checks

All getters return well-typed pandas DataFrames; never None.
"""
import pandas as pd

from irp.query.stooq import prices as _stooq_prices
from irp.query.yahoo import dividends as _divs, prices as _yahoo_prices, splits as _splits

PRICE_COLS = ('Date', 'Open', 'High', 'Low', 'Close', 'Volume')


def _get_prices(
    ticker: str,
    source: str = 'stooq',
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """OHLCV for one ticker from the chosen source.

    `source` is 'stooq' or 'yahoo'. Empty DataFrame on miss.
    """
    fn = _stooq_prices if source == 'stooq' else _yahoo_prices
    df = fn([ticker], start=start, end=end)
    if df is None or df.empty:
        return pd.DataFrame(columns=list(PRICE_COLS))
    return df


def _get_dividends(ticker: str) -> pd.DataFrame:
    """Dividend events for one ticker. Columns: [Date, Amount].

    Normalises the legacy `Dividends` column name from Yahoo to canonical
    `Amount` so callers don't need fallback logic.
    """
    df = _divs([ticker])
    if df is None or df.empty:
        return pd.DataFrame(columns=['Date', 'Amount'])
    if 'Amount' not in df.columns and 'Dividends' in df.columns:
        df = df.rename(columns={'Dividends': 'Amount'})
    return df


def _get_splits(ticker: str) -> pd.DataFrame:
    """Split events for one ticker. Columns: [Date, Ratio].

    Normalises Yahoo's `Stock Splits` to canonical `Ratio`.
    """
    df = _splits([ticker])
    if df is None or df.empty:
        return pd.DataFrame(columns=['Date', 'Ratio'])
    if 'Ratio' not in df.columns and 'Stock Splits' in df.columns:
        df = df.rename(columns={'Stock Splits': 'Ratio'})
    return df
