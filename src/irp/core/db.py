"""Process-wide read-only DuckDB connection singleton for notebooks and data accessors.

Sources (yahoo, stooq, simfin) open their own short-lived read-write connections
via `duckdb.connect(config.database.path)` for writes. They must NOT call `db()`
before calling `store()` in the same process — DuckDB rejects mixing read-only
and read-write connections within one process. `_load_target_tickers()` in
yahoo.py uses a local connection for this reason.
"""
import duckdb

from irp.core.config import config

_con: duckdb.DuckDBPyConnection | None = None


def db() -> duckdb.DuckDBPyConnection:
    global _con
    if _con is None:
        _con = duckdb.connect(str(config.database.path), read_only=True)
    return _con
