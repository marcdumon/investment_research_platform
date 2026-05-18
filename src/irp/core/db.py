"""Process-wide read-only DuckDB connection singleton.

Sources (yahoo, stooq, simfin) use their own write connections via
`duckdb.connect(config.database.path)`. Anything that only reads should
go through `db()` to avoid opening a fresh connection per query.

The singleton is opened lazily on first call and reused for the process
lifetime. DuckDB handles concurrent read-only handles fine; the singleton
just keeps query overhead down for notebook and inspection workflows.
"""
import duckdb

from irp.core.config import config

_con: duckdb.DuckDBPyConnection | None = None


def db() -> duckdb.DuckDBPyConnection:
    global _con
    if _con is None:
        _con = duckdb.connect(str(config.database.path), read_only=True)
    return _con
