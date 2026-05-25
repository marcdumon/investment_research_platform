"""Per-thread read-only DuckDB connections for notebooks and data accessors.

Each thread gets its own connection so concurrent Dash callbacks don't
share state. A global registry lets db_close() drain all connections
before a write connection is opened (DuckDB requires exclusive access
for writes).

Sources open their own short-lived read-write connections via
duckdb.connect(config.database.path). Call db_close() before any store
step in the same process.
"""
import threading

import duckdb

from irp.core.config import config

_local = threading.local()
_registry_lock = threading.Lock()
_registry: set[duckdb.DuckDBPyConnection] = set()


def db() -> duckdb.DuckDBPyConnection:
    con = getattr(_local, 'con', None)
    if con is None:
        con = duckdb.connect(str(config.database.path), read_only=True)
        _local.con = con
        with _registry_lock:
            _registry.add(con)
    return con


def db_close() -> None:
    """Close all read-only connections so a write connection can be opened.

    Increments a generation counter so threads that try to use a stale
    connection will reopen on next db() call.
    """
    import logging
    log = logging.getLogger(__name__)
    with _registry_lock:
        for con in list(_registry):
            try:
                con.close()
            except Exception as exc:
                log.debug(f'db_close: ignoring error closing stale connection: {exc}')
        _registry.clear()
    _local.con = None
