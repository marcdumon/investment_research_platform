"""Shared helpers for `irp.query.simfin` and `irp.query.stooq` accessors."""
from irp.core.db import db

__all__ = ['db', 'ticker_filter']


def ticker_filter(tickers: str | list[str] | None) -> tuple[str, list[str]]:
    if tickers is None:
        return '', []
    if isinstance(tickers, str):
        tickers = [tickers]
    placeholders = ', '.join('?' * len(tickers))
    return f'Ticker IN ({placeholders})', list(tickers)
