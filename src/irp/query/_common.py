"""Shared helpers for `irp.query.*` accessors."""
from typing import Any

from irp.core.db import db

__all__ = ['build_where', 'db', 'ticker_filter']


def ticker_filter(
    tickers: str | list[str] | None, col: str = 'Ticker',
) -> tuple[str, list[str]]:
    if tickers is None:
        return '', []
    if isinstance(tickers, str):
        tickers = [tickers]
    placeholders = ', '.join('?' * len(tickers))
    return f'{col} IN ({placeholders})', list(tickers)


def build_where(
    tickers: str | list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    ticker_col: str = 'Ticker',
    date_col: str = 'Date',
) -> tuple[str, list[Any]]:
    """Build a `WHERE …` clause + parameter list from the common filter args.

    Returns `('', [])` when no filters are active. Otherwise returns
    `('WHERE A AND B AND …', [params...])`. Used by every query accessor
    to keep the filter-building pattern in one place.
    """
    filters: list[str] = []
    params: list[Any] = []

    clause, vals = ticker_filter(tickers, ticker_col)
    if clause:
        filters.append(clause)
        params.extend(vals)
    if start is not None:
        filters.append(f'{date_col} >= ?')
        params.append(start)
    if end is not None:
        filters.append(f'{date_col} <= ?')
        params.append(end)

    where = f'WHERE {" AND ".join(filters)}' if filters else ''
    return where, params
