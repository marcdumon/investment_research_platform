"""Short-lived read-only DuckDB connections for notebooks and data accessors.

Each call to db() opens a fresh read-only connection. In CPython, the
connection closes immediately when the caller's expression completes
(reference count drops to zero) releasing the file lock. This avoids the
long-lived lock that would block write_session() from getting exclusive
access.

write_session() clears the gate (so db() callers block), then retries
duckdb.connect until in-flight read-only connections have been released.
"""
import time
import threading
from contextlib import contextmanager
from typing import Generator

import duckdb

from irp.core.config import config

# Cleared while a write is in progress; db() callers block until set.
_write_gate = threading.Event()
_write_gate.set()


def db() -> duckdb.DuckDBPyConnection:
    """Return a fresh read-only connection. Blocks during active writes."""
    _write_gate.wait(timeout=60)
    return duckdb.connect(str(config.database.path), read_only=True)


# Kept for backwards compat; no-op now that connections are not cached.
def _db_close() -> None:
    pass


@contextmanager
def write_session() -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """Exclusive write connection.

    Blocks new db() callers, then retries until all in-flight read-only
    connections have been released (CPython drops them quickly after each
    db().execute().df() expression completes).
    """
    import logging
    log = logging.getLogger(__name__)
    _write_gate.clear()
    try:
        con = None
        for attempt in range(40):   # up to 10 s (40 × 0.25 s)
            try:
                con = duckdb.connect(str(config.database.path))
                break
            except duckdb.IOException:
                if attempt == 39:
                    raise
                log.debug('write_session: waiting for read locks to release (attempt %d)', attempt + 1)
                time.sleep(0.25)
        try:
            yield con
        finally:
            con.close()
    finally:
        _write_gate.set()
