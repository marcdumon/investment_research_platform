from contextlib import contextmanager

import duckdb
import pandas as pd

import irp.config as _config

_ConOrPath = duckdb.DuckDBPyConnection | str | None


def _db_path() -> str:
    return str(_config.load()["store"]["db_path"])


@contextmanager
def _conn(con_or_path: _ConOrPath):
    """Yield an existing connection as-is, or open+close one from a path."""
    if isinstance(con_or_path, duckdb.DuckDBPyConnection):
        yield con_or_path
    else:
        with duckdb.connect(con_or_path or _db_path()) as c:
            yield c


def init(con_or_path: _ConOrPath = None) -> None:
    """Create ticker_sets table and query macros. Safe to call repeatedly."""
    with _conn(con_or_path) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS ticker_sets (
                set_name VARCHAR,
                ticker   VARCHAR,
                added_at TIMESTAMP DEFAULT now(),
                PRIMARY KEY (set_name, ticker)
            )
        """)
        con.execute("""
            CREATE OR REPLACE MACRO prices_for_set(s) AS TABLE
                SELECT p.* FROM prices p
                JOIN ticker_sets ts ON p.ticker = ts.ticker
                WHERE ts.set_name = s
        """)
        con.execute("""
            CREATE OR REPLACE MACRO income_for_set(s) AS TABLE
                SELECT i.* FROM income i
                JOIN ticker_sets ts ON i."Ticker" = ts.ticker
                WHERE ts.set_name = s
        """)
        con.execute("""
            CREATE OR REPLACE MACRO balance_for_set(s) AS TABLE
                SELECT b.* FROM balance b
                JOIN ticker_sets ts ON b."Ticker" = ts.ticker
                WHERE ts.set_name = s
        """)
        con.execute("""
            CREATE OR REPLACE MACRO cashflow_for_set(s) AS TABLE
                SELECT c.* FROM cashflow c
                JOIN ticker_sets ts ON c."Ticker" = ts.ticker
                WHERE ts.set_name = s
        """)


def save_set(
    set_name: str,
    tickers: list[str],
    con_or_path: _ConOrPath = None,
    replace: bool = True,
) -> int:
    """Upsert tickers into named set. replace=True clears existing tickers first."""
    if not tickers:
        return 0
    with _conn(con_or_path) as con:
        if replace:
            con.execute("DELETE FROM ticker_sets WHERE set_name = ?", [set_name])
        df = pd.DataFrame({"set_name": set_name, "ticker": tickers})
        con.register("_ts", df)
        con.execute("""
            INSERT OR REPLACE INTO ticker_sets (set_name, ticker)
            SELECT set_name, ticker FROM _ts
        """)
    return len(tickers)


def delete_set(set_name: str, con_or_path: _ConOrPath = None) -> None:
    """Delete all tickers in a named set."""
    with _conn(con_or_path) as con:
        con.execute("DELETE FROM ticker_sets WHERE set_name = ?", [set_name])


def list_sets(con_or_path: _ConOrPath = None) -> list[tuple[str, int]]:
    """Return [(set_name, ticker_count), ...] sorted by set_name."""
    with _conn(con_or_path) as con:
        rows = con.execute("""
            SELECT set_name, COUNT(*) AS n
            FROM ticker_sets
            GROUP BY set_name
            ORDER BY set_name
        """).fetchall()
    return rows


def get_set(set_name: str, con_or_path: _ConOrPath = None) -> list[str]:
    """Return sorted list of tickers in a set."""
    with _conn(con_or_path) as con:
        rows = con.execute(
            "SELECT ticker FROM ticker_sets WHERE set_name = ? ORDER BY ticker",
            [set_name],
        ).fetchall()
    return [r[0] for r in rows]
