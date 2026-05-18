"""Process-wide DuckDB connection singleton.

Sources (yahoo, stooq, simfin) use their own short-lived write connections via
`duckdb.connect(config.database.path)`. Anything that only reads should
go through `db()` to avoid opening a fresh connection per query.

The singleton is opened lazily on first call and reused for the process
lifetime. `read_only=False` is intentional: DuckDB disallows mixing read-only
and read-write connections to the same file, and sources run in the same
process as this singleton.
"""
import duckdb

from irp.core.config import config

_con: duckdb.DuckDBPyConnection | None = None


def db() -> duckdb.DuckDBPyConnection:
    global _con
    if _con is None:
        _con = duckdb.connect(str(config.database.path))
    return _con
