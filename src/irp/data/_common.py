"""Shared helpers for `irp.data.simfin` and `irp.data.stooq` accessors."""
import duckdb

from irp.core.config import config

_con: duckdb.DuckDBPyConnection | None = None


def db() -> duckdb.DuckDBPyConnection:
    global _con
    if _con is None:
        _con = duckdb.connect(str(config.database.path), read_only=True)
    return _con


def ticker_filter(tickers: str | list[str] | None) -> tuple[str, list]:
    if tickers is None:
        return '', []
    if isinstance(tickers, str):
        tickers = [tickers]
    placeholders = ', '.join('?' * len(tickers))
    return f'Ticker IN ({placeholders})', list(tickers)


def date_int(date: str) -> int:
    return int(date.replace('-', ''))
